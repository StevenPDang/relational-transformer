use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedSemType, ArchivedTableType,
    Offsets, TableInfo,
};
use clap::Parser;
#[cfg(feature = "vecdb")]
use faiss::index::io::{IoFlags, read_index_with_flags};
#[cfg(feature = "vecdb")]
use faiss::index::{Index as FaissIndex, IndexImpl};
use half::bf16;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::izip;
use memmap2::{Mmap, MmapOptions};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDictMethods};
use pyo3::{pyclass, pyfunction, pymethods};
use rand::prelude::*;
use rand::seq::index;
use rayon::prelude::*;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use std::alloc;
use std::collections::{HashMap, HashSet};

use std::fs;
use std::io::{BufReader, Read};
use std::str;
#[cfg(feature = "vecdb")]
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Allocate a Vec<T> of `len` elements initialized to zero bytes.
/// Uses alloc_zeroed (calloc) for lazy zero-page optimization,
/// avoiding eager memset for large allocations.
///
/// # Safety
/// Caller must ensure all-zero-bits is a valid representation of T.
unsafe fn alloc_zeroed_vec<T>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    let layout = alloc::Layout::array::<T>(len).unwrap();
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) }
}

const MAX_F2P_NBRS: usize = 5;

fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push('_');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

/// Expected reasons an item cannot produce a training sequence.
/// The training path silently retries with a new item; the eval path
/// propagates these as a panic via `.unwrap()`, since eval items are
/// expected to be pre-validated.
#[derive(Debug)]
enum BuildError {
    MissingTargetCol,
    NanTargetValue,
    SqlProvider(pyo3::PyErr),
}

/// Trace-only legacy policy preserves the pre-alignment baseline for comparison.
#[derive(Default)]
struct BuildTrace {
    legacy: bool,
    step_seed: u64,
    local_ctx_size: usize,
    bfs_width: usize,
    prefer_latest: bool,
    balance_labels: bool,
    mask_prob: f64,
    visits: Vec<(i32, usize)>,
    candidates: Vec<i32>,
    fallback_candidates: Vec<i32>,
    collections: Vec<CollectionTrace>,
    cells: Vec<(i32, usize, i32, i32, i32)>,
}

#[derive(serde::Serialize)]
struct CollectionTrace {
    target: i32,
    target_timestamp: Option<i32>,
    seed: i32,
    cutoff: Option<i32>,
    local_ctx_size: usize,
    bfs_width: usize,
    remaining_cells: usize,
    visited_before: Vec<(i32, usize)>,
    rows: Vec<(i32, usize)>,
}

fn sorted_visits(visits: &HashMap<i32, usize>) -> Vec<(i32, usize)> {
    let mut rows: Vec<_> = visits.iter().map(|(&idx, &count)| (idx, count)).collect();
    rows.sort_unstable();
    rows
}

fn within_cutoff(timestamp: Option<i32>, cutoff: Option<i32>) -> bool {
    timestamp.is_none() || cutoff.is_none() || timestamp <= cutoff
}

fn candidate_cmp(
    a: (usize, Option<i32>, u64, i32),
    b: (usize, Option<i32>, u64, i32),
    prefer_latest: bool,
    legacy: bool,
) -> std::cmp::Ordering {
    let score = b.0.cmp(&a.0);
    let timestamp = if prefer_latest {
        b.1.cmp(&a.1)
    } else {
        std::cmp::Ordering::Equal
    };
    (if legacy {
        timestamp.then(score)
    } else {
        score.then(timestamp)
    })
    .then_with(|| a.2.cmp(&b.2))
    .then_with(|| a.3.cmp(&b.3))
}

#[inline]
fn check_deadline(deadline: Instant) {
    if Instant::now() >= deadline {
        panic!("timeout_per_item exceeded");
    }
}

/// Stride (power of two) for amortized deadline checks inside hot inner
/// loops with small bodies. Per-iteration overhead becomes a single
/// `i & (N-1) == 0` predicted-not-taken branch (~1 ns); the actual
/// `Instant::now()` cost is paid once every N iters. Picked so that the
/// worst-case overshoot between checks (N × body ≈ 30 µs at body=30 ns)
/// is comfortably below realistic timeouts (≥ 1 ms).
const DEADLINE_CHECK_EVERY: usize = 1024;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    timestamps: Vec<i32>,
    // Visualization-only metadata: which seed node owns this cell's BFS
    // shell, and at what depth from that seed it was visited. -1 / 0 for
    // padding slots and the primary target cell respectively. Not consumed
    // by the model — it's free side-info for tools like ctx_viz.
    seed_node_idxs: Vec<i32>,
    bfs_depths: Vec<i32>,
    // Per-slot validity mask (length = bs). `true` for real items, `false`
    // for dummy/phantom slots (e.g., last-batch overshoot on higher ranks
    // when num_items isn't a multiple of bs*world_size). Downstream gathers
    // stay fixed-size and apply this mask after gather on rank 0.
    batch_mask: Vec<bool>,
    seq_len: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
    timestamps: &'a mut [i32],
    seed_node_idxs: &'a mut [i32],
    bfs_depths: &'a mut [i32],
}

impl Vecs {
    fn new(bs: usize, seq_len: usize, d_text: usize) -> Self {
        let l = bs * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: unsafe { alloc_zeroed_vec(l * d_text) },
            sem_types: vec![0; l],
            number_values: unsafe { alloc_zeroed_vec(l) },
            text_values: unsafe { alloc_zeroed_vec(l * d_text) },
            datetime_values: unsafe { alloc_zeroed_vec(l) },
            boolean_values: unsafe { alloc_zeroed_vec(l) },
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            timestamps: vec![i32::MIN; l],
            seed_node_idxs: vec![-1; l],
            bfs_depths: vec![-1; l],
            // Default-true: the common case (training, full eval batches) has
            // every slot real. Callers flip specific slots to false only when
            // they know that slot is a phantom (e.g., last-batch overshoot).
            batch_mask: vec![true; bs],
            seq_len,
        }
    }

    fn chunks_exact_mut(
        &mut self,
        seq_len: usize,
        d_text: usize,
    ) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
            self.timestamps.chunks_exact_mut(seq_len),
            self.seed_node_idxs.chunks_exact_mut(seq_len),
            self.bfs_depths.chunks_exact_mut(seq_len)
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
                seed_node_idxs,
                bfs_depths,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
                seed_node_idxs,
                bfs_depths,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("timestamps", PyArray1::from_vec(py, self.timestamps))
                .into_py_any(py)
                .unwrap(),
            (
                "seed_node_idxs",
                PyArray1::from_vec(py, self.seed_node_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("bfs_depths", PyArray1::from_vec(py, self.bfs_depths))
                .into_py_any(py)
                .unwrap(),
            ("batch_mask", PyArray1::from_vec(py, self.batch_mask))
                .into_py_any(py)
                .unwrap(),
            ("seq_len", self.seq_len).into_py_any(py).unwrap(),
        ])
    }
}

/// Per-table FAISS index + memory-mapped row vectors. Loaded only when
/// `vector_db_path` is set on the Sampler. The index is wrapped in a Mutex
/// because FAISS's `search` mutates internal scratch state, but IndexImpl is
/// not Sync. Vectors are mmap'd as raw f32, row-major, indexed by local
/// (table-relative) node index.
#[cfg(feature = "vecdb")]
struct VectorDbEntry {
    node_idx_offset: i32,
    dim: usize,
    num_rows: usize,
    vectors_mmap: Mmap,
    index: Mutex<IndexImpl>,
}

// SAFETY: IndexImpl is not Sync but FAISS's search is internally
// thread-safe-via-mutex; we serialize access through `index: Mutex<…>`.
#[cfg(feature = "vecdb")]
unsafe impl Send for VectorDbEntry {}
#[cfg(feature = "vecdb")]
unsafe impl Sync for VectorDbEntry {}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Option<Mmap>,
    offsets: Vec<i64>,
    table_info: HashMap<String, TableInfo>,
    // Opt-in neighborhoods materialized by task-specific, non-recursive SQL.
    sql_contexts: Option<HashMap<i32, Vec<(i32, usize)>>>,
    sql_seed_order: Option<HashMap<i32, Vec<i32>>>,
    // On-demand Stage 2 only. Never used by Stage 1 or eager/raw SQL contexts.
    sql_neighbor_provider: Option<PyObject>,
    // When `Some`, a deterministic bijection over this database's column
    // indices used to ablate schema semantics: at col_name_values lookup
    // time, the original col_name_idx is replaced by col_name_perm[orig]
    // before fetching the text embedding. The map's domain == range == set
    // of column indices in column_index.json, so cells that originally
    // shared a column name still share one after the shuffle, and cells
    // that originally differed still differ.
    col_name_perm: Option<HashMap<i32, i32>>,
    /// Per-table FAISS lookup, keyed by table name. `None` when
    /// `vector_db_path` is not set on the Sampler. When present, every
    /// table referenced by this dataset's task tuples must have an entry.
    #[cfg(feature = "vecdb")]
    vector_db: Option<HashMap<String, VectorDbEntry>>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
    table_name: String,
}

#[pyclass]
pub struct Sampler {
    global_rank: usize,
    local_rank: usize,
    world_size: usize,
    datasets: HashMap<String, Dataset>,
    items: Vec<Item>,
    local_ctx_sizes: Vec<usize>, // Maximum cells per BFS collection; sampled uniformly per item
    bfs_widths: Vec<usize>, // Maximum number of DB nodes per BFS level; sampled uniformly per item
    // Number of random walks used to compute same-table visit counts. When 0,
    // similar-node selection skips the walk and falls through directly to the
    // unvisited same-table tier — that's the "random_same_table" mode.
    num_walks: usize,
    walk_length: usize, // Maximum length of each random walk
    // Sort key for walk-visited same-table seeds; sampled uniformly per
    // item from this list. Both branches end with a step_seed-driven
    // random tiebreak so equal-key seeds don't cluster by HashMap
    // iteration order.
    // True:  (count desc, ts desc, random)
    // False: (count desc, random)
    // None ts ranks below any Some (std Option ord).
    prefer_latest: Vec<bool>,
    mask_prob_max: f64, // Maximum probability of masking cells
    step: u64,
    stride: u64,
    d_text: usize,
    // Drives only `self.items.shuffle()` and per-task subsampling in
    // `create_items` — i.e. *which* items end up in the dataset.
    shuffle_seed: u64,
    // Drives context-construction randomness: train-mode batch item
    // index sampling, and `step_seed` in `seq_build` (which in turn
    // feeds mask_prob, local_ctx_size pick, bfs_width pick,
    // walk visit counts, fallback same-table sampling,
    // bfs_collect_nodes, and cell masking).
    context_seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
    items_per_task: i64,
    dataset_tuples: Vec<(String, String, i32, i32)>, // (db_name, table_name, node_idx_offset, num_nodes) for each dataset
    table_ranges: Vec<(i32, i32)>, // (range_start, range_end) per dataset tuple, cached for fallback same-table sampling
    quiet: bool,
    // When true, cells whose sem_type is Text are not added to the context
    // during BFS expansion. The target cell is always emitted first and is
    // unaffected. Skipping text cells frees slots for non-text cells, so a
    // given ctx_len ends up covering more rows than it would otherwise.
    skip_text_cols: bool,
    // When true, conceptually keep two priority lists of similar same-table
    // seeds — one with target-column value < 0 and one with >= 0 — and pick
    // the next seed from each list with 50% probability (falling back to the
    // non-empty list when one runs dry). The original priority ordering
    // within each tier is preserved inside both lists. Sampled uniformly
    // per item from this list.
    balance_labels: Vec<bool>,
    // Per-item wall-clock budget for one `seq_build` attempt. Enforced by
    // the training path (`seq`); eval bypasses it. Cooperative — checked
    // at the top of each iteration of the inner BFS-collection and
    // random-walk loops so the bound is tight even when one helper call
    // dominates the runtime.
    timeout_per_item: f64,
    // When set, Tier 1 same-table seed selection switches from random
    // walks to FAISS-similarity lookups against `<vector_db_path>/<db>/
    // <table>.index` (with row vectors in `<…>_vectors.bin`). The lookup
    // is streamed: we do an initial small search and progressively
    // double the search size if the consumer asks for more seeds, so we
    // never pre-fetch a fixed huge top-k upfront. When `None`, behavior
    // is unchanged (random walk + same-table fallback).
    #[cfg_attr(not(feature = "vecdb"), allow(dead_code))]
    vector_db_path: Option<String>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dataset_tuples: Vec<(String, String, i32, i32)>,
        global_rank: usize,
        local_rank: usize,
        world_size: usize,
        local_ctx_sizes: Vec<usize>,
        bfs_widths: Vec<usize>,
        num_walks: usize,
        walk_length: usize,
        prefer_latest: Vec<bool>,
        mask_prob_max: f64,
        embedding_model: &str,
        pre_dir: String,
        d_text: usize,
        shuffle_seed: u64,
        context_seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        items_per_task: i64,
        quiet: bool,
        ignore_data_errors: bool,
        num_prev_skipped: usize,
        skip_text_cols: bool,
        mmap_populate: bool,
        balance_labels: Vec<bool>,
        timeout_per_item: f64,
        ablate_schema_semantics: bool,
        vector_db_path: Option<String>,
        train_only_fallback: bool,
    ) -> Self {
        py.allow_threads(|| {
            Self::new_impl(
                dataset_tuples,
                global_rank,
                local_rank,
                world_size,
                local_ctx_sizes,
                bfs_widths,
                num_walks,
                walk_length,
                prefer_latest,
                mask_prob_max,
                embedding_model,
                pre_dir,
                d_text,
                shuffle_seed,
                context_seed,
                target_columns,
                columns_to_drop,
                items_per_task,
                quiet,
                ignore_data_errors,
                num_prev_skipped,
                skip_text_cols,
                mmap_populate,
                balance_labels,
                timeout_per_item,
                ablate_schema_semantics,
                vector_db_path,
                train_only_fallback,
            )
        })
    }

    #[getter]
    fn num_items(&self) -> usize {
        self.items.len()
    }

    fn set_sql_contexts_py(
        &mut self,
        db_name: &str,
        contexts: HashMap<i32, Vec<(i32, usize)>>,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyValueError;
        let dataset = self
            .datasets
            .get_mut(db_name)
            .ok_or_else(|| PyValueError::new_err("Unknown SQL context database"))?;
        let valid = |idx: i32| idx >= 0 && (idx as usize) < dataset.offsets.len() - 1;
        if contexts
            .iter()
            .any(|(&seed, nodes)| !valid(seed) || nodes.iter().any(|&(idx, _)| !valid(idx)))
        {
            return Err(PyValueError::new_err("SQL context node index out of range"));
        }
        if dataset.sql_neighbor_provider.is_some() {
            return Err(PyValueError::new_err(
                "Clear SQL neighbor provider before installing eager contexts",
            ));
        }
        // Every possible same-table seed needs an entry; never silently fall
        // back to BFS when a materialized SQL neighborhood is missing.
        for ((db, _, _, _), &(start, end)) in
            self.dataset_tuples.iter().zip(self.table_ranges.iter())
        {
            if db == db_name && (start..end).any(|idx| !contexts.contains_key(&idx)) {
                return Err(PyValueError::new_err(
                    "SQL contexts missing same-table seeds",
                ));
            }
        }
        dataset.sql_contexts = Some(contexts);
        Ok(())
    }

    /// Replace walk ranking with a SQL-ordered list of historical task rows.
    fn set_sql_seed_order_py(
        &mut self,
        db_name: &str,
        orders: HashMap<i32, Vec<i32>>,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyValueError;
        let dataset = self
            .datasets
            .get_mut(db_name)
            .ok_or_else(|| PyValueError::new_err("Unknown SQL context database"))?;
        let contexts = dataset
            .sql_contexts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Set SQL contexts before SQL seed order"))?;
        if contexts.keys().any(|idx| !orders.contains_key(idx)) {
            return Err(PyValueError::new_err("SQL seed order missing seeds"));
        }
        for (&seed, candidates) in &orders {
            if !contexts.contains_key(&seed) {
                return Err(PyValueError::new_err(
                    "SQL seed order contains unknown seed",
                ));
            }
            let source = get_node(dataset, seed);
            for &idx in candidates {
                if !contexts.contains_key(&idx) {
                    return Err(PyValueError::new_err("SQL candidate missing context"));
                }
                let node = get_node(dataset, idx);
                if idx == seed
                    || node.table_name_idx != source.table_name_idx
                    || node.timestamp >= source.timestamp
                {
                    return Err(PyValueError::new_err(
                        "SQL candidates must be earlier same-table rows",
                    ));
                }
            }
        }
        dataset.sql_seed_order = Some(orders);
        Ok(())
    }

    /// Install a worker-local callable, or clear it with None. The callable
    /// returns (f2p, p2f) ordered node-index lists filtered by target cutoff.
    /// Duplicate edges must be preserved. Forward order follows the encoded
    /// column/list order; reverse order is (timestamp None first, node_idx).
    /// Rust verifies identity/order, then consumes SQL IDs with graph metadata.
    #[pyo3(signature = (db_name, provider))]
    fn set_sql_neighbor_provider_py(
        &mut self,
        py: Python<'_>,
        db_name: &str,
        provider: Option<PyObject>,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyValueError;
        let dataset = self
            .datasets
            .get_mut(db_name)
            .ok_or_else(|| PyValueError::new_err("Unknown SQL neighbor database"))?;
        if let Some(ref provider) = provider {
            if !provider.bind(py).is_callable() {
                return Err(PyValueError::new_err(
                    "SQL neighbor provider must be callable",
                ));
            }
            if dataset.p2f_adj_mmap.is_none()
                || dataset.sql_contexts.is_some()
                || dataset.sql_seed_order.is_some()
            {
                return Err(PyValueError::new_err(
                    "SQL neighbor provider requires a graph and no eager SQL contexts/seed order",
                ));
            }
        }
        dataset.sql_neighbor_provider = provider;
        Ok(())
    }

    /// Deterministic single-item build, with no retries, substitution, or step mutation.
    /// `legacy` is trace-only and reproduces the pre-alignment sampling policy.
    #[pyo3(signature = (dataset_idx, node_idx, ctx_size, step=0, legacy=false))]
    fn trace_py(
        &self,
        py: Python<'_>,
        dataset_idx: usize,
        node_idx: i32,
        ctx_size: usize,
        step: u64,
        legacy: bool,
    ) -> PyResult<PyObject> {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};
        use pyo3::types::PyDict;
        let (db, table, _, _) = self
            .dataset_tuples
            .get(dataset_idx)
            .ok_or_else(|| PyValueError::new_err("dataset_idx out of range"))?;
        let (start, end) = self.table_ranges[dataset_idx];
        if node_idx < start
            || node_idx >= end
            || ctx_size == 0
            || !self.local_ctx_sizes.iter().any(|&l| l <= ctx_size)
        {
            return Err(PyValueError::new_err("Invalid trace node or context size"));
        }
        if legacy && self.datasets[db].sql_neighbor_provider.is_some() {
            return Err(PyValueError::new_err(
                "Legacy trace requires BFS, not SQL neighbor provider",
            ));
        }
        let item = Item {
            dataset_idx: dataset_idx as i32,
            node_idx,
            table_name: table.clone(),
        };
        let mut trace = BuildTrace {
            legacy,
            ..Default::default()
        };
        let mut vecs = Vecs::new(1, ctx_size, self.d_text);
        let result = py.allow_threads(|| {
            let mut slices = vecs.chunks_exact_mut(ctx_size, self.d_text).next().unwrap();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.seq_build_traced(
                    &item,
                    &mut slices,
                    step,
                    ctx_size,
                    Instant::now() + Duration::from_secs_f64(self.timeout_per_item),
                    Some(&mut trace),
                )
            }))
        });
        match result {
            Ok(Ok(())) => (),
            Ok(Err(BuildError::SqlProvider(err))) => return Err(err),
            Ok(Err(err)) => {
                return Err(PyValueError::new_err(format!(
                    "Trace build failed: {err:?}"
                )));
            }
            Err(_) => return Err(PyRuntimeError::new_err("Trace build panicked or timed out")),
        }
        let out = PyDict::new(py);
        out.set_item("policy", if legacy { "legacy" } else { "algorithm1" })?;
        // Fourth baseline alignment: reverse timestamp ties are canonical by
        // child identity, independent of pre.rs's cross-table HashMap order.
        out.set_item(
            "reverse_neighbor_order",
            if legacy {
                "stored"
            } else if self.datasets[db].sql_contexts.is_some() {
                "not_applicable_eager_sql"
            } else {
                "timestamp_none_first_then_node_idx"
            },
        )?;
        out.set_item("target", node_idx)?;
        out.set_item("step", step)?;
        out.set_item("step_seed", trace.step_seed)?;
        out.set_item("context_seed", self.context_seed)?;
        out.set_item("num_walks", self.num_walks)?;
        out.set_item("walk_length", self.walk_length)?;
        out.set_item("local_ctx_size", trace.local_ctx_size)?;
        out.set_item("bfs_width", trace.bfs_width)?;
        out.set_item("prefer_latest", trace.prefer_latest)?;
        out.set_item("balance_labels", trace.balance_labels)?;
        out.set_item("mask_prob", trace.mask_prob)?;
        let dataset = &self.datasets[db];
        out.set_item(
            "collector",
            if dataset.sql_neighbor_provider.is_some() {
                "sql_neighbors"
            } else if dataset.sql_seed_order.is_some() {
                "raw_sql"
            } else if dataset.sql_contexts.is_some() {
                "eager_sql"
            } else {
                "bfs"
            },
        )?;
        out.set_item("visits", trace.visits)?;
        out.set_item("candidate_order", trace.candidates)?;
        out.set_item("fallback_candidates", trace.fallback_candidates)?;
        // JSON round-trip is confined to this diagnostic API, not the sampler hot path.
        let collections = serde_json::to_string(&trace.collections)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        out.set_item(
            "collections",
            py.import("json")?.call_method1("loads", (collections,))?,
        )?;
        out.set_item("cells", trace.cells)?;
        out.set_item("sequence", vecs.into_pyobject(py)?)?;
        Ok(out.into_any().unbind())
    }

    fn batch_py(
        &mut self,
        py: Python<'_>,
        batch_idx: Option<usize>,
        bs: usize,
        ctx_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let vecs = py.allow_threads(|| match batch_idx {
            Some(idx) => self.batch(Some(idx), 0, bs, ctx_size),
            None => {
                let step = self.step;
                let r = self.batch(None, step, bs, ctx_size);
                if r.is_ok() {
                    self.step += self.stride;
                }
                r
            }
        })?;
        vecs.into_pyobject(py)
    }

    /// Build contexts for explicit node indices (one context per node).
    /// Used by Rel2Tab to build local contexts for each task node.
    fn batch_for_nodes_py(
        &self,
        py: Python<'_>,
        node_idxs: Vec<i32>,
        dataset_idx: usize,
        ctx_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let bs = node_idxs.len();
        let table_name = &self.dataset_tuples[dataset_idx].1;
        let mut vecs = Vecs::new(bs, ctx_size, self.d_text);
        // batch_mask defaults to all true — every slot is real here.

        py.allow_threads(|| {
            vecs.chunks_exact_mut(ctx_size, self.d_text)
                .enumerate()
                .par_bridge()
                .try_for_each(|(i, slices)| {
                    let item = Item {
                        dataset_idx: dataset_idx as i32,
                        node_idx: node_idxs[i],
                        table_name: table_name.clone(),
                    };
                    self.seq(&item, i, slices, 0, ctx_size)
                })
        })?;

        vecs.into_pyobject(py)
    }

    fn set_step_py(&mut self, step: u64) {
        self.step = step;
    }

    fn set_stride_py(&mut self, stride: u64) {
        self.stride = stride;
    }

    fn set_mask_prob_max_py(&mut self, mask_prob_max: f64) {
        self.mask_prob_max = mask_prob_max;
    }

    #[getter]
    fn local_ctx_size(&self) -> usize {
        self.local_ctx_sizes[0]
    }

    #[getter]
    fn d_text(&self) -> usize {
        self.d_text
    }
}

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        dataset_tuples: Vec<(String, String, i32, i32)>,
        global_rank: usize,
        local_rank: usize,
        world_size: usize,
        local_ctx_sizes: Vec<usize>,
        bfs_widths: Vec<usize>,
        num_walks: usize,
        walk_length: usize,
        prefer_latest: Vec<bool>,
        mask_prob_max: f64,
        embedding_model: &str,
        pre_dir: String,
        d_text: usize,
        shuffle_seed: u64,
        context_seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        items_per_task: i64,
        quiet: bool,
        ignore_data_errors: bool,
        num_prev_skipped: usize,
        skip_text_cols: bool,
        mmap_populate: bool,
        balance_labels: Vec<bool>,
        timeout_per_item: f64,
        ablate_schema_semantics: bool,
        vector_db_path: Option<String>,
        train_only_fallback: bool,
    ) -> Self {
        let embedding_model_ref = embedding_model;

        // Collect unique db_names and their associated table_names for deduplication.
        let mut db_to_tables: HashMap<String, Vec<String>> = HashMap::new();
        for (db_name, table_name, _, _) in dataset_tuples.iter() {
            let entry = db_to_tables.entry(db_name.clone()).or_default();
            if !entry.iter().any(|t| t == table_name) {
                entry.push(table_name.clone());
            }
        }

        let mut mmap_opts = MmapOptions::new();
        if mmap_populate {
            mmap_opts.populate();
        }

        let pb = make_pb(
            db_to_tables.len() as u64,
            "loading databases",
            local_rank == 0 && !quiet,
        );

        let vector_db_path_ref = vector_db_path.as_deref();
        let datasets: HashMap<String, Dataset> = if db_to_tables.len() > 1 {
            db_to_tables
                .par_iter()
                .map(|(db_name, table_names)| {
                    let r = Self::load_dataset(
                        db_name,
                        table_names,
                        &pre_dir,
                        &mmap_opts,
                        embedding_model_ref,
                        ablate_schema_semantics,
                        shuffle_seed,
                        vector_db_path_ref,
                    );
                    pb.inc(1);
                    r
                })
                .collect()
        } else {
            db_to_tables
                .iter()
                .map(|(db_name, table_names)| {
                    let r = Self::load_dataset(
                        db_name,
                        table_names,
                        &pre_dir,
                        &mmap_opts,
                        embedding_model_ref,
                        ablate_schema_semantics,
                        shuffle_seed,
                        vector_db_path_ref,
                    );
                    pb.inc(1);
                    r
                })
                .collect()
        };
        pb.finish_and_clear();

        // Pre-compute table ranges for fallback same-table sampling.
        // When ignore_data_errors is true, catch panics per-task, print a red
        // warning, drop the offending task and its parallel arrays. When
        // false (e.g. eval), let panics propagate — eval tasks are expected
        // to be pre-validated.
        let (dataset_tuples, target_columns, columns_to_drop, table_ranges, rust_skipped) = {
            let mut kept_tuples: Vec<(String, String, i32, i32)> = Vec::new();
            let mut kept_targets: Vec<i32> = Vec::new();
            let mut kept_drops: Vec<Vec<i32>> = Vec::new();
            let mut kept_ranges: Vec<(i32, i32)> = Vec::new();
            let mut skipped: usize = 0;
            for (i, tuple) in dataset_tuples.into_iter().enumerate() {
                let (ref db_name, ref table_name, _, _) = tuple;
                let datasets_ref = &datasets;
                let db = db_name.clone();
                let table = table_name.clone();
                let compute = std::panic::AssertUnwindSafe(|| {
                    let dataset = &datasets_ref[&db];
                    let mut range_start = i32::MAX;
                    let mut range_end = i32::MIN;
                    for (key, info) in &dataset.table_info {
                        if let Some(colon_pos) = key.rfind(':')
                            && &key[..colon_pos] == table.as_str()
                            && (!train_only_fallback || &key[colon_pos + 1..] == "Train")
                        {
                            range_start = range_start.min(info.node_idx_offset);
                            range_end = range_end.max(info.node_idx_offset + info.num_nodes);
                        }
                    }
                    assert!(
                        range_start < range_end,
                        "table {} has no usable rows in db {}",
                        table,
                        db,
                    );
                    (range_start, range_end)
                });
                let result = if ignore_data_errors {
                    std::panic::catch_unwind(compute)
                } else {
                    Ok(compute())
                };
                match result {
                    Ok(range) => {
                        kept_tuples.push(tuple);
                        kept_targets.push(target_columns[i]);
                        kept_drops.push(columns_to_drop[i].clone());
                        kept_ranges.push(range);
                    }
                    Err(panic_info) => {
                        if local_rank == 0 && !quiet {
                            let msg = panic_info
                                .downcast_ref::<String>()
                                .map(|s| s.as_str())
                                .or_else(|| panic_info.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic");
                            eprintln!(
                                "\n\x1b[31mskipping task {}/{}: {}\x1b[0m",
                                db_name, table_name, msg
                            );
                        }
                        skipped += 1;
                    }
                }
            }
            (kept_tuples, kept_targets, kept_drops, kept_ranges, skipped)
        };
        assert!(
            !dataset_tuples.is_empty(),
            "All tasks were skipped due to errors, cannot proceed."
        );
        assert!(
            !local_ctx_sizes.is_empty(),
            "local_ctx_sizes must be non-empty"
        );
        assert!(!bfs_widths.is_empty(), "bfs_widths must be non-empty");
        assert!(!prefer_latest.is_empty(), "prefer_latest must be non-empty");
        assert!(
            !balance_labels.is_empty(),
            "balance_labels must be non-empty"
        );
        let mut sampler = Self {
            global_rank,
            local_rank,
            world_size,
            datasets,
            items: Vec::new(), // Will be populated by create_items
            local_ctx_sizes,
            bfs_widths,
            num_walks,
            walk_length,
            prefer_latest,
            mask_prob_max,
            step: 0,
            stride: 1,
            d_text,
            shuffle_seed: StdRng::seed_from_u64(shuffle_seed).random(),
            context_seed: StdRng::seed_from_u64(context_seed).random(),
            target_columns,
            columns_to_drop,
            items_per_task,
            dataset_tuples,
            table_ranges,
            quiet,
            skip_text_cols,
            balance_labels,
            timeout_per_item,
            vector_db_path,
        };
        // train_only_fallback is consumed by the closure that builds
        // table_ranges above; its effect is baked into the per-task
        // (range_start, range_end) and not stored on the struct.

        sampler.create_items();
        if sampler.local_rank == 0 && !sampler.quiet {
            let num_dbs = db_to_tables.len();
            let num_tasks = sampler.dataset_tuples.len();
            let num_items = sampler.items.len();
            let num_skipped = num_prev_skipped + rust_skipped;
            println!(
                "\ndata stats: \x1b[1m{}\x1b[0m dbs, \x1b[1m{}\x1b[0m tasks, \x1b[1m{}\x1b[0m items, \x1b[1m{}\x1b[0m skipped",
                fmt_thousands(num_dbs),
                fmt_thousands(num_tasks),
                fmt_thousands(num_items),
                fmt_thousands(num_skipped),
            );
        }
        sampler
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "vecdb"), allow(unused_variables))]
    fn load_dataset(
        db_name: &str,
        table_names: &[String],
        pre_dir: &str,
        mmap_opts: &MmapOptions,
        embedding_model_ref: &str,
        ablate_schema_semantics: bool,
        shuffle_seed: u64,
        vector_db_path: Option<&str>,
    ) -> (String, Dataset) {
        let pre_path = format!("{}/{}", pre_dir, db_name);

        let nodes_path = format!("{}/nodes.rkyv", pre_path);
        let file = fs::File::open(&nodes_path).unwrap();
        let mmap = unsafe { mmap_opts.map(&file).unwrap() };

        let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model_ref);
        let text_file = fs::File::open(&text_path).unwrap();
        let text_mmap = unsafe { mmap_opts.map(&text_file).unwrap() };

        let offsets_path = format!("{}/offsets.rkyv", pre_path);
        let file = fs::File::open(&offsets_path).unwrap();
        let mut bytes = Vec::new();
        BufReader::new(file).read_to_end(&mut bytes).unwrap();
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
        let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
        let offsets = offsets.offsets;

        let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
        let meta_path = format!("{}/meta.json", pre_path);
        let graph_free = if std::path::Path::new(&meta_path).is_file() {
            let meta: serde_json::Value =
                serde_json::from_reader(fs::File::open(meta_path).unwrap()).unwrap();
            meta["sampling_graph"] == false
        } else {
            false // Legacy cell stores always contain the traversal graph.
        };
        let p2f_adj_mmap = if graph_free {
            None
        } else {
            let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
            Some(unsafe { mmap_opts.map(&p2f_adj_file).unwrap() })
        };

        let table_info_path = format!("{}/table_info.json", pre_path);
        let table_info_file = fs::File::open(&table_info_path).unwrap();
        let table_info: HashMap<String, TableInfo> =
            serde_json::from_reader(BufReader::new(table_info_file)).unwrap();

        let col_name_perm = if ablate_schema_semantics {
            Some(build_col_name_perm(&pre_path, db_name, shuffle_seed))
        } else {
            None
        };

        #[cfg(not(feature = "vecdb"))]
        assert!(
            vector_db_path.is_none(),
            "vector_db_path is set but rustler was built without the 'vecdb' \
             feature; rebuild with `--features vecdb`"
        );

        #[cfg(feature = "vecdb")]
        let vector_db = vector_db_path.map(|root| {
            // Load `<root>/<db>/<table>.index` + `<root>/<db>/<table>_vectors.bin`
            // for every table referenced by this db's task tuples.
            let mut entries: HashMap<String, VectorDbEntry> = HashMap::new();
            for table in table_names {
                let index_path = format!("{}/{}/{}.index", root, db_name, table);
                let vectors_path = format!("{}/{}/{}_vectors.bin", root, db_name, table);

                // Min node_idx_offset across splits gives the (table-relative)
                // origin used to convert FAISS local labels to global indices.
                let node_idx_offset = table_info
                    .iter()
                    .filter(|(key, _)| key.rsplit_once(':').map(|(t, _)| t) == Some(table.as_str()))
                    .map(|(_, info)| info.node_idx_offset)
                    .min()
                    .unwrap_or_else(|| {
                        panic!("table {} not found in table_info for db {}", table, db_name,)
                    });

                let faiss_index =
                    read_index_with_flags(&index_path, IoFlags::MEM_MAP | IoFlags::READ_ONLY)
                        .unwrap_or_else(|e| {
                            panic!("failed to load FAISS index {}: {}", index_path, e)
                        });
                let dim = faiss_index.d() as usize;

                let vecs_file = fs::File::open(&vectors_path).unwrap_or_else(|e| {
                    panic!("failed to open vectors file {}: {}", vectors_path, e)
                });
                let vectors_mmap = unsafe { mmap_opts.map(&vecs_file).unwrap() };
                let bytes = vectors_mmap.len();
                assert!(
                    bytes % (dim * std::mem::size_of::<f32>()) == 0,
                    "vectors file {} size ({} bytes) is not a multiple of dim*4={}",
                    vectors_path,
                    bytes,
                    dim * std::mem::size_of::<f32>(),
                );
                let num_rows = bytes / (dim * std::mem::size_of::<f32>());

                entries.insert(
                    table.clone(),
                    VectorDbEntry {
                        node_idx_offset,
                        dim,
                        num_rows,
                        vectors_mmap,
                        index: Mutex::new(faiss_index),
                    },
                );
            }
            entries
        });

        (
            db_name.to_owned(),
            Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
                table_info,
                sql_contexts: None,
                sql_seed_order: None,
                sql_neighbor_provider: None,
                col_name_perm,
                #[cfg(feature = "vecdb")]
                vector_db,
            },
        )
    }

    /// Build items list with per-task random subset selection.
    fn create_items(&mut self) {
        self.items.clear();

        let pb = make_pb(
            self.dataset_tuples.len() as u64,
            "subsampling tasks",
            self.local_rank == 0 && !self.quiet,
        );

        for (i, &(_, ref table_name, node_idx_offset, num_nodes)) in
            self.dataset_tuples.iter().enumerate()
        {
            let num_to_sample = if self.items_per_task == -1 {
                num_nodes as usize
            } else {
                (num_nodes as usize).min(self.items_per_task as usize)
            };

            let rng_seed = self.shuffle_seed + (i as u64);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            let sampled_indices = index::sample(&mut rng, num_nodes as usize, num_to_sample);

            for idx in sampled_indices.iter() {
                let node_idx = node_idx_offset + idx as i32;
                self.items.push(Item {
                    dataset_idx: i as i32,
                    node_idx,
                    table_name: table_name.clone(),
                });
            }
            pb.inc(1);
        }
        pb.finish_and_clear();

        // Shuffle the entire items list
        // needed for EvalDataset as we eval only the first 1024 items
        let mut rng = StdRng::seed_from_u64(self.shuffle_seed);
        self.items.shuffle(&mut rng);
    }

    fn len(&self, bs: usize) -> usize {
        self.items.len().div_ceil(bs * self.world_size)
    }

    fn batch(
        &self,
        batch_idx: Option<usize>,
        step: u64,
        bs: usize,
        ctx_size: usize,
    ) -> PyResult<Vecs> {
        match batch_idx {
            Some(idx) => {
                // Eval sharding: each rank takes `bs` items per global batch.
                // DataLoader `__len__` is uniform across ranks (one value of
                // ceil(num_items / (bs * world_size))); higher-rank offsets on
                // the last batch may legitimately overshoot num_items. For
                // overshooting slots we leave the slot as default-padded and
                // mark batch_mask[i]=false, so the downstream fixed-size gather
                // is correct and the phantom rows get filtered on rank 0.
                let offset = self.global_rank * bs + idx * bs * self.world_size;
                let mut vecs = Vecs::new(bs, ctx_size, self.d_text);
                // Mark which slots are real before the build loop below —
                // cheap sequential pass, one bool per slot.
                for (i, m) in vecs.batch_mask.iter_mut().enumerate() {
                    *m = offset + i < self.items.len();
                }

                // Eval per-item build is bounded by timeout_per_item, exactly
                // like training. Crucially, unlike training we must NOT substitute
                // a different random item on timeout/error (that would change which
                // entity's prediction we report); instead we mark the slot as a
                // phantom (batch_mask[i]=false) and leave it padded, identical to an
                // overshoot slot. The downstream fixed-size all_gather is unchanged
                // (same row count per rank), and rank 0 filters phantom rows out of
                // labels/preds -- so a slow/pathological item is dropped from the
                // metric rather than stalling the synchronous collective across all
                // ranks. At high world_size each batch covers a far wider slice of
                // the item list, so an unbounded build on one tail item would
                // otherwise hang every rank.
                let timeout = Duration::from_secs_f64(self.timeout_per_item);
                let timed_out: Vec<bool> = vecs
                    .chunks_exact_mut(ctx_size, self.d_text)
                    .enumerate()
                    .map(|(i, mut slices)| {
                        let j = offset + i;
                        if j >= self.items.len() {
                            // Phantom slot: leave defaults (fully padded, no
                            // target). batch_mask[i] is already false.
                            return Ok(false);
                        }
                        let item = &self.items[j];
                        let deadline = Instant::now() + timeout;
                        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || self.seq_build(item, &mut slices, 0, ctx_size, deadline),
                        ));
                        let caught = match caught {
                            Ok(Err(BuildError::SqlProvider(err))) => return Err(err),
                            other => other,
                        };
                        let failed = !matches!(caught, Ok(Ok(())));
                        if failed {
                            // Reset to a clean padded slot; it becomes a phantom.
                            slices.is_targets.fill(false);
                            slices.is_padding.fill(true);
                            let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
                            eprintln!(
                                "\n\x1b[31meval: dropping item (db={}, table={}, node_idx={}): build failed/timed out\x1b[0m",
                                db_name, item.table_name, item.node_idx
                            );
                        }
                        Ok(failed)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                for (i, &failed) in timed_out.iter().enumerate() {
                    if failed {
                        vecs.batch_mask[i] = false;
                    }
                }
                Ok(vecs)
            }
            None => {
                // Distinct from the (context_seed + step) seed used to derive
                // step_seed in seq_build, so train-mode item-index sampling
                // doesn't share a stream with the per-item context randomness.
                let mut rng = StdRng::seed_from_u64(
                    (self.context_seed + step).wrapping_add(0xE0E0_E0E0_E0E0_E0E0),
                );

                let mut vecs = Vecs::new(bs, ctx_size, self.d_text);
                // Training always fills all slots; mark them all real.
                vecs.batch_mask.iter_mut().for_each(|m| *m = true);

                let global_bs = bs * self.world_size;
                let global_indices: Vec<usize> = (0..global_bs)
                    .map(|_| rng.random_range(0..self.items.len()))
                    .collect();

                vecs.chunks_exact_mut(ctx_size, self.d_text)
                    .enumerate()
                    .try_for_each(|(i, slices)| {
                        let j = self.global_rank * bs + i;
                        let item = &self.items[global_indices[j]];
                        self.seq(item, global_indices[j], slices, step, ctx_size)
                    })?;
                Ok(vecs)
            }
        }
    }

    /// Retries silently on an expected BuildError. Unexpected panics are
    /// caught, a red warning is printed, and the item is also retried. A
    /// per-item wall-clock budget bounds each `seq_build` call: when it
    /// expires the call panics with a timeout message, which falls into the
    /// red-warning-and-retry branch like any other panic.
    fn seq(
        &self,
        item: &Item,
        item_idx: usize,
        mut slices: Slices,
        step: u64,
        ctx_len: usize,
    ) -> PyResult<()> {
        let mut current_item = item;
        let mut retry_seed = item_idx as u64;
        let timeout = Duration::from_secs_f64(self.timeout_per_item);
        loop {
            let deadline = Instant::now() + timeout;
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.seq_build(current_item, &mut slices, step, ctx_len, deadline)
            }));
            match caught {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(BuildError::SqlProvider(err))) => return Err(err),
                Ok(Err(_)) => {}
                Err(panic_info) => {
                    let msg = panic_info
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic_info.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    let db_name = &self.dataset_tuples[current_item.dataset_idx as usize].0;
                    let target_column = self.target_columns[current_item.dataset_idx as usize];
                    eprintln!(
                        "\n\x1b[31mskipping item {item_idx} (db={}, table={}, target_node_idx={}, target_column_idx={}, step={step}, ctx_len={ctx_len}): {msg}\x1b[0m",
                        db_name, current_item.table_name, current_item.node_idx, target_column
                    );
                }
            }
            slices.is_targets.fill(false);
            slices.is_padding.fill(true);
            let next_idx = StdRng::seed_from_u64(retry_seed).random_range(0..self.items.len());
            current_item = &self.items[next_idx];
            retry_seed = next_idx as u64;
        }
    }

    fn seq_build(
        &self,
        item: &Item,
        slices: &mut Slices,
        step: u64,
        ctx_len: usize,
        deadline: Instant,
    ) -> Result<(), BuildError> {
        self.seq_build_traced(item, slices, step, ctx_len, deadline, None)
    }

    fn seq_build_traced(
        &self,
        item: &Item,
        slices: &mut Slices,
        step: u64,
        ctx_len: usize,
        deadline: Instant,
        mut trace: Option<&mut BuildTrace>,
    ) -> Result<(), BuildError> {
        let legacy = trace.as_ref().is_some_and(|t| t.legacy);
        check_deadline(deadline);
        let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
        let dataset = &self.datasets[db_name];
        let target_column = self.target_columns[item.dataset_idx as usize];
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];

        let target_node_idx = item.node_idx;
        let target_node = get_node(dataset, target_node_idx);

        let target_cell_i = match target_node
            .col_name_idxs
            .iter()
            .position(|&col_idx| i32::from(col_idx) == target_column)
        {
            Some(i) => i,
            None => return Err(BuildError::MissingTargetCol),
        };

        let target_value_is_nan = {
            let sem = &target_node.sem_types[target_cell_i];
            match sem {
                ArchivedSemType::Number => {
                    f32::from(target_node.number_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::DateTime => {
                    f32::from(target_node.datetime_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Boolean => {
                    f32::from(target_node.boolean_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Text => false,
            }
        };
        if target_value_is_nan {
            return Err(BuildError::NanTargetValue);
        }

        let step_seed: u64 = StdRng::seed_from_u64(self.context_seed + step).random();

        let mut seq_rng = StdRng::seed_from_u64(step_seed + target_node_idx as u64);
        let mask_prob = seq_rng.random::<f64>() * self.mask_prob_max;

        let valid_local_ctx_sizes: Vec<usize> = self
            .local_ctx_sizes
            .iter()
            .copied()
            .filter(|&l| l <= ctx_len)
            .collect();
        assert!(
            !valid_local_ctx_sizes.is_empty(),
            "No local_ctx_size in {:?} is <= ctx_len={}",
            self.local_ctx_sizes,
            ctx_len
        );
        let local_ctx_size =
            valid_local_ctx_sizes[seq_rng.random_range(0..valid_local_ctx_sizes.len())];
        let bfs_width = self.bfs_widths[seq_rng.random_range(0..self.bfs_widths.len())];
        let prefer_latest = self.prefer_latest[seq_rng.random_range(0..self.prefer_latest.len())];
        let balance_labels =
            self.balance_labels[seq_rng.random_range(0..self.balance_labels.len())];
        if let Some(t) = trace.as_deref_mut() {
            t.step_seed = step_seed;
            t.local_ctx_size = local_ctx_size;
            t.bfs_width = bfs_width;
            t.prefer_latest = prefer_latest;
            t.balance_labels = balance_labels;
            t.mask_prob = mask_prob;
        }

        // Stage 1 is fully computed before any collector can consume BFS RNG.
        // Step 1: walk visit counts (skip when num_walks == 0; that's
        // pure random_same_table mode — every same-table node falls through
        // to the unvisited-tier). Also skipped when vector_db_path is set:
        // FAISS-similarity replaces walk-based similarity entirely.
        #[cfg(feature = "vecdb")]
        let use_vector_db = self.vector_db_path.is_some();
        #[cfg(not(feature = "vecdb"))]
        let use_vector_db = false;
        assert!(
            dataset.p2f_adj_mmap.is_some()
                || (dataset.sql_contexts.is_some() && dataset.sql_seed_order.is_some()),
            "Graph-free cells require SQL contexts and SQL seed ordering"
        );
        let sql_order = dataset
            .sql_seed_order
            .as_ref()
            .map(|orders| &orders[&target_node_idx]);
        let visit_counts = if sql_order.is_some() {
            HashMap::new()
        } else if !use_vector_db && self.num_walks > 0 {
            self.compute_visit_counts(
                dataset,
                target_node_idx,
                target_node,
                self.num_walks,
                self.walk_length,
                step_seed,
                deadline,
                legacy,
            )
        } else {
            HashMap::new()
        };

        // Step 2: order visited same-table seeds. Both branches use a
        // step_seed-derived random priority as the final tiebreak so that
        // equal-key seeds don't cluster by HashMap iteration order.
        // - prefer_latest=true:  (count desc, ts desc, random)
        // - prefer_latest=false: (count desc, random)
        let mut visited_sorted: Vec<i32> = sql_order
            .cloned()
            .unwrap_or_else(|| visit_counts.keys().copied().collect());
        let priority: HashMap<i32, u64> = visited_sorted
            .iter()
            .filter(|_| sql_order.is_none())
            .map(|&n| {
                (
                    n,
                    StdRng::seed_from_u64(step_seed.wrapping_add(n as u64)).random::<u64>(),
                )
            })
            .collect();
        check_deadline(deadline);
        if sql_order.is_some() {
            // Preserve the SQL order; no graph walk or Rust ranking is needed.
        } else {
            let ts_of: HashMap<i32, Option<i32>> = visited_sorted
                .iter()
                .map(|&n| {
                    let ts = get_node(dataset, n).timestamp.as_ref().map(|t| (*t).into());
                    (n, ts)
                })
                .collect();
            visited_sorted.sort_by(|a, b| {
                candidate_cmp(
                    (visit_counts[a], ts_of[a], priority[a], *a),
                    (visit_counts[b], ts_of[b], priority[b], *b),
                    prefer_latest,
                    legacy,
                )
            });
        }
        check_deadline(deadline);

        if let Some(t) = trace.as_deref_mut() {
            t.visits = sorted_visits(&visit_counts);
            t.candidates = std::iter::once(target_node_idx)
                .chain(visited_sorted.iter().copied())
                .collect();
        }

        // Step 3: same-table fallback iterator. Lazy — only materialized when
        // the visited tier runs dry. Streams unvisited nodes from the target's
        // table in random order, applying the temporal filter inline.
        let (range_start, range_end) = self.table_ranges[item.dataset_idx as usize];
        let total_table = (range_end - range_start) as usize;
        let fallback_seed = step_seed
            .wrapping_add(target_node_idx as u64)
            .wrapping_add(0xA5A5_A5A5_A5A5_A5A5);

        let mut visited_in_ctx: HashSet<i32> = HashSet::with_capacity(ctx_len);
        let mut visited_at_depth: HashMap<i32, usize> = HashMap::with_capacity(ctx_len);
        // (node_idx, cell_i, col_idx, seed_node_idx, bfs_depth). The last two
        // are visualization metadata: the seed whose BFS shell discovered
        // the node, and the depth at which it was visited within that shell.
        let mut cells_to_add: Vec<(i32, usize, i32, i32, i32)> = Vec::with_capacity(ctx_len);
        // Distinct stream from seq_rng / cell-mask rng / compute_visit_counts
        // rng (which all derive from step_seed + target_node_idx) so each
        // consumer's draws are independent rather than correlated bytes off
        // the same ChaCha20 state.
        let mut bfs_rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0xB0B0_B0B0_B0B0_B0B0),
        );

        // Always emit the target cell first to guarantee it's in the sequence.
        // Seed = self, depth = 0.
        cells_to_add.push((
            target_node_idx,
            target_cell_i,
            target_column,
            target_node_idx,
            0,
        ));

        // Independent rng for balance_labels coin flips; unused when
        // balance_labels=false. Kept separate from bfs_rng so flipping the
        // flag doesn't perturb BFS expansion order on either branch.
        let mut balance_rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0x5A5A_5A5A_5A5A_5A5A),
        );

        // Drive the fill loop with a labeled block so each tier can break out
        // when the context is full.
        'fill_ctx: {
            // Tier 0: target node's BFS.
            if extend_with_seed_bfs(
                self,
                dataset,
                target_node_idx,
                target_node_idx,
                target_node,
                target_column,
                columns_to_drop,
                local_ctx_size,
                bfs_width,
                ctx_len,
                &mut bfs_rng,
                &mut visited_at_depth,
                &mut visited_in_ctx,
                &mut cells_to_add,
                deadline,
                legacy,
                &mut trace,
            )? {
                break 'fill_ctx;
            }

            // Tier 1: similar same-table nodes. Source is either random
            // walks (visit_counts → visited_sorted) or FAISS-similarity
            // (vector_db_path), depending on Sampler config. Seeds with
            // missing/NaN target labels are dropped — they carry no usable
            // label. When balance_labels is on, surviving seeds are routed
            // into neg / pos buckets and drained alternately.
            //
            // `tier1_seen` (random-walk path only) holds every seed Tier 1
            // considered, so Tier 2 can avoid re-issuing BFS from a node
            // Tier 1 already used. The vector_db path skips Tier 2 entirely
            // and doesn't need this set.
            let mut tier1_seen: HashSet<i32> = HashSet::new();
            if use_vector_db {
                #[cfg(feature = "vecdb")]
                {
                    let entry = dataset
                        .vector_db
                        .as_ref()
                        .expect("vector_db_path is set but dataset.vector_db was not loaded")
                        .get(item.table_name.as_str())
                        .unwrap_or_else(|| {
                            panic!(
                                "vector_db has no FAISS entry for table '{}'",
                                item.table_name
                            )
                        });
                    let mut vdb = VectorDbStream::new(entry, target_node_idx, target_node);
                    if balance_labels {
                        // Lazy variant: pull-classify-drain on demand. The 50/50
                        // alternation runs over whatever's already materialized
                        // in neg / pos, so the realized pattern depends on
                        // iterator order — distinct from the upfront-partition
                        // path but consistent with the lazy-retrieval contract.
                        let mut neg: Vec<i32> = Vec::new();
                        let mut pos: Vec<i32> = Vec::new();
                        let mut neg_i = 0;
                        let mut pos_i = 0;
                        loop {
                            if let Some(seed_node_idx) =
                                pick_balanced(&neg, &pos, &mut neg_i, &mut pos_i, &mut balance_rng)
                            {
                                check_deadline(deadline);
                                if extend_with_seed_bfs(
                                    self,
                                    dataset,
                                    seed_node_idx,
                                    target_node_idx,
                                    target_node,
                                    target_column,
                                    columns_to_drop,
                                    local_ctx_size,
                                    bfs_width,
                                    ctx_len,
                                    &mut bfs_rng,
                                    &mut visited_at_depth,
                                    &mut visited_in_ctx,
                                    &mut cells_to_add,
                                    deadline,
                                    legacy,
                                    &mut trace,
                                )? {
                                    break 'fill_ctx;
                                }
                                continue;
                            }
                            check_deadline(deadline);
                            match vdb.next(dataset) {
                                None => break,
                                Some(seed_node_idx) => {
                                    let seed_node = get_node(dataset, seed_node_idx);
                                    if seed_label_missing(seed_node, target_column) {
                                        continue;
                                    }
                                    if seed_label_is_negative(seed_node, target_column) {
                                        neg.push(seed_node_idx);
                                    } else {
                                        pos.push(seed_node_idx);
                                    }
                                }
                            }
                        }
                    } else {
                        while let Some(seed_node_idx) = vdb.next(dataset) {
                            check_deadline(deadline);
                            if seed_label_missing(get_node(dataset, seed_node_idx), target_column) {
                                continue;
                            }
                            if extend_with_seed_bfs(
                                self,
                                dataset,
                                seed_node_idx,
                                target_node_idx,
                                target_node,
                                target_column,
                                columns_to_drop,
                                local_ctx_size,
                                bfs_width,
                                ctx_len,
                                &mut bfs_rng,
                                &mut visited_at_depth,
                                &mut visited_in_ctx,
                                &mut cells_to_add,
                                deadline,
                                legacy,
                                &mut trace,
                            )? {
                                break 'fill_ctx;
                            }
                        }
                    }
                }
            } else if balance_labels {
                let mut neg: Vec<i32> = Vec::new();
                let mut pos: Vec<i32> = Vec::new();
                for (i, &n) in visited_sorted.iter().enumerate() {
                    if i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                        check_deadline(deadline);
                    }
                    let seed_node = get_node(dataset, n);
                    if seed_label_missing(seed_node, target_column) {
                        continue;
                    }
                    if seed_label_is_negative(seed_node, target_column) {
                        neg.push(n);
                    } else {
                        pos.push(n);
                    }
                }
                let mut neg_i = 0;
                let mut pos_i = 0;
                while let Some(seed_node_idx) =
                    pick_balanced(&neg, &pos, &mut neg_i, &mut pos_i, &mut balance_rng)
                {
                    check_deadline(deadline);
                    if extend_with_seed_bfs(
                        self,
                        dataset,
                        seed_node_idx,
                        target_node_idx,
                        target_node,
                        target_column,
                        columns_to_drop,
                        local_ctx_size,
                        bfs_width,
                        ctx_len,
                        &mut bfs_rng,
                        &mut visited_at_depth,
                        &mut visited_in_ctx,
                        &mut cells_to_add,
                        deadline,
                        legacy,
                        &mut trace,
                    )? {
                        break 'fill_ctx;
                    }
                }
            } else {
                for &seed_node_idx in visited_sorted.iter() {
                    check_deadline(deadline);
                    if seed_label_missing(get_node(dataset, seed_node_idx), target_column) {
                        continue;
                    }
                    if extend_with_seed_bfs(
                        self,
                        dataset,
                        seed_node_idx,
                        target_node_idx,
                        target_node,
                        target_column,
                        columns_to_drop,
                        local_ctx_size,
                        bfs_width,
                        ctx_len,
                        &mut bfs_rng,
                        &mut visited_at_depth,
                        &mut visited_in_ctx,
                        &mut cells_to_add,
                        deadline,
                        legacy,
                        &mut trace,
                    )? {
                        break 'fill_ctx;
                    }
                }
            }
            if use_vector_db || sql_order.is_some() {
                // Tier 2 (random same-table fallback) is unreachable when
                // vector_db is enabled: if the streaming iterator returned
                // None it already exhausted the table under the same
                // temporal filter, so any seed Tier 2 would draw is in
                // tier1_seen and would be skipped. SQL also exhausts its
                // strictly historical candidates; do not add same-time labels.
                break 'fill_ctx;
            }
            for &n in visit_counts.keys() {
                tier1_seen.insert(n);
            }

            // Tier 2: unvisited same-table nodes (visit count 0).
            // Lazily sampled — work happens only if Tier 0+1 haven't filled ctx.
            if total_table == 0 {
                break 'fill_ctx;
            }
            // Upper bound on seeds we could ever consume from this tier:
            // ctx_len cells minus what's already added, since each seed yields
            // ≥1 cell. Capping `amount` to total_table keeps Fisher-Yates
            // bounded; capping to remaining-cells keeps it cheap when we're
            // close to full.
            let remaining_cells = ctx_len.saturating_sub(cells_to_add.len());
            let fallback_amount = std::cmp::min(remaining_cells, total_table);
            if fallback_amount == 0 {
                break 'fill_ctx;
            }
            let mut fallback_rng = StdRng::seed_from_u64(fallback_seed);
            let fallback_offsets = index::sample(&mut fallback_rng, total_table, fallback_amount);
            if let Some(t) = trace.as_deref_mut() {
                t.fallback_candidates = fallback_offsets
                    .iter()
                    .map(|off| range_start + off as i32)
                    .collect();
            }
            check_deadline(deadline);
            if balance_labels {
                // Materialize valid candidates into neg/pos buckets, preserving
                // the random sample order, then drain alternately.
                let mut neg: Vec<i32> = Vec::new();
                let mut pos: Vec<i32> = Vec::new();
                for (i, off) in fallback_offsets.iter().enumerate() {
                    if i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                        check_deadline(deadline);
                    }
                    let seed_node_idx = range_start + off as i32;
                    if seed_node_idx == target_node_idx {
                        continue;
                    }
                    if tier1_seen.contains(&seed_node_idx) {
                        continue;
                    }
                    let seed_node = get_node(dataset, seed_node_idx);
                    if target_node.timestamp.is_some()
                        && seed_node.timestamp.is_some()
                        && seed_node.timestamp > target_node.timestamp
                    {
                        continue;
                    }
                    if seed_label_missing(seed_node, target_column) {
                        continue;
                    }
                    if seed_label_is_negative(seed_node, target_column) {
                        neg.push(seed_node_idx);
                    } else {
                        pos.push(seed_node_idx);
                    }
                }
                let mut neg_i = 0;
                let mut pos_i = 0;
                while let Some(seed_node_idx) =
                    pick_balanced(&neg, &pos, &mut neg_i, &mut pos_i, &mut balance_rng)
                {
                    check_deadline(deadline);
                    if extend_with_seed_bfs(
                        self,
                        dataset,
                        seed_node_idx,
                        target_node_idx,
                        target_node,
                        target_column,
                        columns_to_drop,
                        local_ctx_size,
                        bfs_width,
                        ctx_len,
                        &mut bfs_rng,
                        &mut visited_at_depth,
                        &mut visited_in_ctx,
                        &mut cells_to_add,
                        deadline,
                        legacy,
                        &mut trace,
                    )? {
                        break 'fill_ctx;
                    }
                }
            } else {
                for off in fallback_offsets.iter() {
                    check_deadline(deadline);
                    let seed_node_idx = range_start + off as i32;
                    if seed_node_idx == target_node_idx {
                        continue;
                    }
                    if tier1_seen.contains(&seed_node_idx) {
                        continue;
                    }
                    // Temporal: only nodes with timestamp <= target.ts (or None)
                    // can serve as similar seeds.
                    let seed_node = get_node(dataset, seed_node_idx);
                    if target_node.timestamp.is_some()
                        && seed_node.timestamp.is_some()
                        && seed_node.timestamp > target_node.timestamp
                    {
                        continue;
                    }
                    if seed_label_missing(seed_node, target_column) {
                        continue;
                    }
                    if extend_with_seed_bfs(
                        self,
                        dataset,
                        seed_node_idx,
                        target_node_idx,
                        target_node,
                        target_column,
                        columns_to_drop,
                        local_ctx_size,
                        bfs_width,
                        ctx_len,
                        &mut bfs_rng,
                        &mut visited_at_depth,
                        &mut visited_in_ctx,
                        &mut cells_to_add,
                        deadline,
                        legacy,
                        &mut trace,
                    )? {
                        break 'fill_ctx;
                    }
                }
            }
        }

        if let Some(t) = trace.as_deref_mut() {
            t.cells = cells_to_add.clone();
        }

        // Add cells to sequence. Distinct stream from seq_rng — both pull
        // f64s, so sharing a seed here would make the first cell's mask coin
        // literally equal to mask_prob / mask_prob_max instead of an
        // independent draw.
        let mut rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0xC0C0_C0C0_C0C0_C0C0),
        );
        let mut seq_i = 0;
        let mut last_node_idx: i32 = -1;
        let mut cached_node: Option<&ArchivedNode> = None;
        for &(node_idx, cell_i, _col_idx, seed_node_idx, depth) in cells_to_add.iter() {
            check_deadline(deadline);
            if seq_i >= ctx_len {
                break;
            }

            if node_idx != last_node_idx {
                cached_node = Some(get_node(dataset, node_idx));
                last_node_idx = node_idx;
            }

            self.add_single_cell(
                dataset,
                cached_node.unwrap(),
                cell_i,
                target_node_idx,
                target_column,
                &mut rng,
                &mut seq_i,
                slices,
                mask_prob,
                seed_node_idx,
                depth,
            );
        }
        Ok(())
    }

    /// Run random walks from `source_idx` and return per-(same-table,
    /// temporally-valid)-node visit counts. Excludes the source itself.
    #[allow(clippy::too_many_arguments)]
    fn compute_visit_counts(
        &self,
        dataset: &Dataset,
        source_idx: i32,
        source_node: &ArchivedNode,
        num_walks: usize,
        max_walk_length: usize,
        step_seed: u64,
        deadline: Instant,
        legacy: bool,
    ) -> HashMap<i32, usize> {
        // Distinct stream from the other (step_seed + target_node_idx)
        // RNGs in seq_build (source_idx == target_node_idx at every call
        // site today) so random walks don't share bytes with BFS / mask
        // / sizing draws.
        let mut rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(source_idx as u64)
                .wrapping_add(0xD0D0_D0D0_D0D0_D0D0),
        );

        let mut similar_node_visits: HashMap<i32, usize> = HashMap::new();

        for _ in 0..num_walks {
            check_deadline(deadline);
            let mut current_idx = source_idx;

            // Perform a random walk
            for _ in 0..max_walk_length {
                check_deadline(deadline);
                // Algorithm 1 counts the destination, including the final move.
                if !legacy {
                    match self.select_random_neighbor(dataset, current_idx, source_node, &mut rng) {
                        Some(idx) => current_idx = idx,
                        None => break,
                    }
                }
                let current_node = get_node(dataset, current_idx);

                // Count this step iff:
                // - not the source
                // - same table as the source (target)
                // - temporally valid (ts <= source.ts, or ts is None)
                if current_node.table_name_idx == source_node.table_name_idx
                    && current_idx != source_idx
                {
                    let temporally_valid = current_node.timestamp.is_none()
                        || source_node.timestamp.is_none()
                        || current_node.timestamp <= source_node.timestamp;
                    if temporally_valid {
                        *similar_node_visits.entry(current_idx).or_insert(0) += 1;
                    }
                }

                if !legacy {
                    continue;
                }
                // Legacy baseline counts before moving.
                let next_idx = match self.select_random_neighbor(
                    dataset,
                    current_idx,
                    source_node,
                    &mut rng,
                ) {
                    Some(idx) => idx,
                    None => break,
                };

                current_idx = next_idx;
            }
        }

        similar_node_visits
    }

    /// Select a random valid neighbor using binary search on sorted p2f edges.
    fn select_random_neighbor(
        &self,
        dataset: &Dataset,
        current_idx: i32,
        target_node: &ArchivedNode,
        rng: &mut StdRng,
    ) -> Option<i32> {
        let current_node = get_node(dataset, current_idx);
        let p2f_edges = get_p2f_edges(dataset, current_idx);

        // Filter p2f edges by temporal constraint (edges are sorted by timestamp)
        let valid_p2f_count = if target_node.timestamp.is_some() {
            let target_ts = target_node.timestamp;
            p2f_edges
                .as_slice()
                .partition_point(|edge| edge.timestamp.is_none() || edge.timestamp <= target_ts)
        } else {
            p2f_edges.len()
        };

        let total_valid_neighbors = current_node.f2p_edges.len() + valid_p2f_count;
        if total_valid_neighbors == 0 {
            return None;
        }

        let rand_idx = rng.random_range(0..total_valid_neighbors);
        if rand_idx < current_node.f2p_edges.len() {
            Some(current_node.f2p_edges[rand_idx].node_idx.into())
        } else {
            Some(
                p2f_edges[rand_idx - current_node.f2p_edges.len()]
                    .node_idx
                    .into(),
            )
        }
    }

    /// Add a single cell from a node to the sequence.
    #[allow(clippy::too_many_arguments)]
    fn add_single_cell(
        &self,
        dataset: &Dataset,
        node: &ArchivedNode,
        cell_i: usize,
        target_node_idx: i32,
        target_column: i32,
        rng: &mut StdRng,
        seq_i: &mut usize,
        slices: &mut Slices,
        mask_prob: f64,
        seed_node_idx: i32,
        bfs_depth: i32,
    ) {
        slices.node_idxs[*seq_i] = node.node_idx.into();

        assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
        for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
            slices.f2p_nbr_idxs[*seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
        }

        slices.table_name_idxs[*seq_i] = node.table_name_idx.into();
        slices.col_name_idxs[*seq_i] = node.col_name_idxs[cell_i].into();
        slices.class_value_idxs[*seq_i] = node.class_value_idx[cell_i].into();
        // Apply the schema-semantics ablation only at the embedding lookup:
        // the stored col_name_idx still drives same-column comparisons in
        // the model (which a bijection preserves), but the embedding the
        // model actually consumes for the column name is the shuffled one.
        let col_emb_idx = match &dataset.col_name_perm {
            Some(perm) => perm[&slices.col_name_idxs[*seq_i]],
            None => slices.col_name_idxs[*seq_i],
        };
        slices.col_name_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, col_emb_idx, self.d_text));

        slices.sem_types[*seq_i] = node.sem_types[cell_i].clone() as i32;
        slices.number_values[*seq_i] = bf16::from_f32(node.number_values[cell_i].into());

        let text_idx: i32 = node.text_values[cell_i].into();
        slices.text_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

        slices.datetime_values[*seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());
        slices.boolean_values[*seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

        // A cell's value is NaN when the underlying data is missing.
        // Never mark such cells as targets — they have no valid label.
        let value_is_nan = {
            let sem = &node.sem_types[cell_i];
            match sem {
                ArchivedSemType::Number => f32::from(node.number_values[cell_i]).is_nan(),
                ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]).is_nan(),
                ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]).is_nan(),
                ArchivedSemType::Text => false, // text uses index, not float
            }
        };

        slices.is_targets[*seq_i] = if value_is_nan {
            false
        } else if node.node_idx == target_node_idx && node.col_name_idxs[cell_i] == target_column {
            true
        } else {
            rng.random::<f64>() < mask_prob
        };

        slices.is_task_nodes[*seq_i] =
            node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
        slices.is_padding[*seq_i] = false;
        slices.timestamps[*seq_i] = match node.timestamp.as_ref() {
            Some(ts) => (*ts).into(),
            None => i32::MIN,
        };
        slices.seed_node_idxs[*seq_i] = seed_node_idx;
        slices.bfs_depths[*seq_i] = bfs_depth;

        *seq_i += 1;
    }

    /// Performs BFS to collect nodes for local context.
    #[allow(clippy::too_many_arguments)]
    fn bfs_collect_nodes(
        &self,
        dataset: &Dataset,
        start_idx: i32,
        target_idx: i32,
        legacy: bool,
        rng: &mut StdRng,
        local_ctx_size: usize,
        bfs_width: usize,
        visited_at_depth: &mut HashMap<i32, usize>,
        deadline: Instant,
    ) -> Result<Vec<(i32, usize)>, BuildError> {
        if let Some(contexts) = &dataset.sql_contexts {
            let nodes = contexts.get(&start_idx).expect("SQL context missing seed");
            let seed = get_node(dataset, start_idx);
            let mut result = Vec::new();
            let mut num_cells = 0;
            for &(idx, depth) in nodes {
                check_deadline(deadline);
                if visited_at_depth
                    .get(&idx)
                    .is_some_and(|&prev| prev <= depth)
                {
                    continue;
                }
                let node = get_node(dataset, idx);
                if node.timestamp.is_some()
                    && (seed.timestamp.is_none() || node.timestamp > seed.timestamp)
                {
                    continue;
                }
                num_cells += node.col_name_idxs.len();
                if num_cells >= local_ctx_size {
                    break;
                }
                visited_at_depth.insert(idx, depth);
                result.push((idx, depth));
            }
            return Ok(result);
        }
        let mut result: Vec<(i32, usize)> = Vec::with_capacity(128);

        let start_node = get_node(dataset, start_idx);
        let cutoff_node = get_node(dataset, if legacy { start_idx } else { target_idx });
        let cutoff: Option<i32> = cutoff_node.timestamp.as_ref().map(|t| (*t).into());
        let mut num_cells = 0;

        // Two frontier data structures:
        // f2p_ftr: stack of (depth, node_idx) for f2p edges
        // p2f_ftr: vector of vectors, one per depth level, for p2f edges
        let mut f2p_ftr: Vec<(usize, i32)> = Vec::with_capacity(64);
        let mut p2f_ftr: Vec<Vec<i32>> = vec![vec![start_idx]];
        let mut db_p2f_ftr: Vec<i32> = Vec::with_capacity(bfs_width);

        loop {
            check_deadline(deadline);
            // Select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                match p2f_ftr.iter().position(|v| !v.is_empty()) {
                    None => return Ok(result),
                    Some(depth) => {
                        let r = rng.random_range(0..p2f_ftr[depth].len());
                        let l = p2f_ftr[depth].len();
                        p2f_ftr[depth].swap(r, l - 1);
                        let node_idx = p2f_ftr[depth].pop().unwrap();
                        (depth, node_idx)
                    }
                }
            };

            // Check if node was visited at a depth <= current depth
            if let Some(&prev_depth) = visited_at_depth.get(&node_idx)
                && prev_depth <= depth
            {
                continue;
            }

            let node = get_node(dataset, node_idx);
            if !legacy && !within_cutoff(node.timestamp.as_ref().map(|t| (*t).into()), cutoff) {
                continue;
            }

            // Update number of cells collected
            num_cells += node.col_name_idxs.len();
            if num_cells >= local_ctx_size {
                return Ok(result);
            }

            // Record the depth at which this node was visited
            visited_at_depth.insert(node_idx, depth);

            result.push((node_idx, depth));

            // Query at most once per collected/popped node, never per edge.
            // The same ordered edge lists drive both collectors below.
            let (f2p_edges, p2f_edges) = if legacy {
                let p2f = get_p2f_edges(dataset, node_idx);
                let valid = p2f.as_slice().partition_point(|edge| {
                    edge.timestamp.is_none()
                        || (start_node.timestamp.is_some()
                            && edge.timestamp <= start_node.timestamp)
                });
                (
                    Stage2Edges::from_graph(node.f2p_edges.iter().collect()),
                    Stage2Edges::from_graph(p2f.as_slice()[..valid].iter().collect()),
                )
            } else {
                ordered_stage2_edges(dataset, node_idx, target_idx, start_idx, cutoff)?
            };

            // Add f2p edges to f2p frontier
            for (neighbor_idx, _) in f2p_edges.iter() {
                f2p_ftr.push((depth + 1, neighbor_idx));
            }

            // Reuse pre-allocated storage for db edges to be subsampled
            db_p2f_ftr.clear();

            for (i, (neighbor_idx, edge)) in p2f_edges.iter().enumerate() {
                if i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                    check_deadline(deadline);
                }
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != start_node.table_name_idx
                    && edge.table_type != ArchivedTableType::Db
                {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push(neighbor_idx);
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(neighbor_idx);
            }

            // Subsample DB edges based on bfs_width
            let idxs = if db_p2f_ftr.len() > bfs_width {
                index::sample(rng, db_p2f_ftr.len(), bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(db_p2f_ftr[*idx]);
            }
        }
    }
}

/// Exact adjacency contract: no omissions, extra nodes, reordering, or lost
/// duplicate edges. Checking against the graph is deliberate: Stage 1 still
/// needs that graph, and SQL data must map to the same prepared revision.
fn validate_neighbor_order(actual: &[i32], expected: &[i32]) -> PyResult<()> {
    if actual != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SQL neighbors differ from encoded graph identity/order/target cutoff",
        ));
    }
    Ok(())
}

/// Owned neighbor identities drive the frontier; graph edges supply only
/// trusted table/type/time metadata. Keeping IDs separate makes the SQL result
/// an input to traversal, rather than a verification-only side effect.
struct Stage2Edges<'a> {
    node_idxs: Vec<i32>,
    metadata: Vec<&'a ArchivedEdge>,
}

impl<'a> Stage2Edges<'a> {
    fn from_graph(metadata: Vec<&'a ArchivedEdge>) -> Self {
        let node_idxs = metadata.iter().map(|e| e.node_idx.into()).collect();
        Self {
            node_idxs,
            metadata,
        }
    }

    fn from_sql(node_idxs: Vec<i32>, metadata: Vec<&'a ArchivedEdge>) -> PyResult<Self> {
        let expected: Vec<i32> = metadata.iter().map(|e| e.node_idx.into()).collect();
        validate_neighbor_order(&node_idxs, &expected)?;
        // Exact ordered validation makes this an occurrence-by-occurrence
        // mapping, not a set/map lookup that could collapse duplicate edges.
        Ok(Self {
            node_idxs,
            metadata,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (i32, &'a ArchivedEdge)> + '_ {
        self.node_idxs
            .iter()
            .copied()
            .zip(self.metadata.iter().copied())
    }
}

fn ordered_stage2_edges<'a>(
    dataset: &'a Dataset,
    node_idx: i32,
    target_idx: i32,
    seed_idx: i32,
    cutoff: Option<i32>,
) -> Result<(Stage2Edges<'a>, Stage2Edges<'a>), BuildError> {
    let eligible = |edge: &&ArchivedEdge| {
        let neighbor_idx: i32 = edge.node_idx.into();
        within_cutoff(edge.timestamp.as_ref().map(|t| (*t).into()), cutoff)
            && within_cutoff(
                get_node(dataset, neighbor_idx)
                    .timestamp
                    .as_ref()
                    .map(|t| (*t).into()),
                cutoff,
            )
    };
    let node = get_node(dataset, node_idx);
    let f2p: Vec<_> = node.f2p_edges.iter().filter(eligible).collect();
    let reverse = get_p2f_edges(dataset, node_idx);
    // Reverse edges are timestamp-sorted; do not scan the future suffix.
    let valid = reverse.as_slice().partition_point(|edge| {
        within_cutoff(edge.timestamp.as_ref().map(|t| (*t).into()), cutoff)
    });
    let mut p2f: Vec<_> = reverse.as_slice()[..valid]
        .iter()
        .filter(eligible)
        .collect();
    // Fourth aligned-baseline policy: canonical reverse adjacency removes
    // equal/null timestamp tie order inherited from pre.rs HashMap iteration.
    // Stable sort preserves duplicate occurrences; forward order is untouched.
    p2f.sort_by_key(|edge| {
        (
            edge.timestamp.as_ref().map(|t| i32::from(*t)),
            i32::from(edge.node_idx),
        )
    });
    if let Some(provider) = &dataset.sql_neighbor_provider {
        let (actual_f2p, actual_p2f) = Python::with_gil(|py| -> PyResult<(Vec<i32>, Vec<i32>)> {
            let (actual_f2p, actual_p2f): (Vec<i32>, Vec<i32>) = provider
                .bind(py)
                .call1((node_idx, target_idx, seed_idx, cutoff))?
                .extract()?;
            Ok((actual_f2p, actual_p2f))
        })
        .map_err(BuildError::SqlProvider)?;
        return Ok((
            Stage2Edges::from_sql(actual_f2p, f2p).map_err(BuildError::SqlProvider)?,
            Stage2Edges::from_sql(actual_p2f, p2f).map_err(BuildError::SqlProvider)?,
        ));
    }
    Ok((Stage2Edges::from_graph(f2p), Stage2Edges::from_graph(p2f)))
}

/// BFS-expand around `seed_node_idx` and append cells to `cells_to_add` until
/// `ctx_len` is reached. Returns true iff the context is now full.
///
/// Free-standing because seq_build mutably borrows `bfs_rng`,
/// `visited_at_depth`, `visited_in_ctx`, `cells_to_add` simultaneously, while
/// also borrowing `&self` to call `bfs_collect_nodes` — disjoint borrows that
/// the borrow checker accepts cleanly when the helper is a function rather
/// than a method.
#[allow(clippy::too_many_arguments)]
fn extend_with_seed_bfs(
    sampler: &Sampler,
    dataset: &Dataset,
    seed_node_idx: i32,
    target_node_idx: i32,
    target_node: &ArchivedNode,
    target_column: i32,
    columns_to_drop: &[i32],
    local_ctx_size: usize,
    bfs_width: usize,
    ctx_len: usize,
    bfs_rng: &mut StdRng,
    visited_at_depth: &mut HashMap<i32, usize>,
    visited_in_ctx: &mut HashSet<i32>,
    cells_to_add: &mut Vec<(i32, usize, i32, i32, i32)>,
    deadline: Instant,
    legacy: bool,
    trace: &mut Option<&mut BuildTrace>,
) -> Result<bool, BuildError> {
    if cells_to_add.len() >= ctx_len {
        return Ok(true);
    }
    let visited_before = trace.as_ref().map(|_| sorted_visits(visited_at_depth));
    let remaining_cells = ctx_len.saturating_sub(cells_to_add.len());
    let bfs_nodes = sampler.bfs_collect_nodes(
        dataset,
        seed_node_idx,
        target_node_idx,
        legacy,
        bfs_rng,
        local_ctx_size,
        bfs_width,
        visited_at_depth,
        deadline,
    )?;
    if let Some(t) = trace.as_deref_mut() {
        let cutoff_node = get_node(
            dataset,
            if legacy || dataset.sql_contexts.is_some() {
                seed_node_idx
            } else {
                target_node_idx
            },
        );
        t.collections.push(CollectionTrace {
            target: target_node_idx,
            target_timestamp: target_node.timestamp.as_ref().map(|t| (*t).into()),
            seed: seed_node_idx,
            cutoff: cutoff_node.timestamp.as_ref().map(|t| (*t).into()),
            local_ctx_size,
            bfs_width,
            remaining_cells,
            visited_before: visited_before.unwrap(),
            rows: bfs_nodes.clone(),
        });
    }

    for (bfs_node_idx, depth) in bfs_nodes {
        check_deadline(deadline);
        if visited_in_ctx.contains(&bfs_node_idx) {
            continue;
        }
        visited_in_ctx.insert(bfs_node_idx);

        let node = get_node(dataset, bfs_node_idx);
        for cell_i in 0..node.col_name_idxs.len() {
            if cell_i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                check_deadline(deadline);
            }
            let col_idx: i32 = node.col_name_idxs[cell_i].into();

            // Skip columns to drop
            if (node.node_idx == target_node_idx && columns_to_drop.contains(&col_idx))
                || (node.timestamp == target_node.timestamp && columns_to_drop.contains(&col_idx))
            {
                continue;
            }

            // Skip target cell (already added first)
            if bfs_node_idx == target_node_idx && col_idx == target_column {
                continue;
            }

            // Skip text-typed cells when configured. The target cell was
            // already pushed unconditionally before BFS, so a text-typed
            // target is preserved.
            if sampler.skip_text_cols && matches!(node.sem_types[cell_i], ArchivedSemType::Text) {
                continue;
            }

            cells_to_add.push((bfs_node_idx, cell_i, col_idx, seed_node_idx, depth as i32));

            if cells_to_add.len() == ctx_len {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Lazy iterator over FAISS-similarity neighbors of a target node, used as
/// the Tier 1 seed source when `vector_db_path` is configured. Each
/// `next()` returns the next-most-similar same-table node not yet
/// yielded; under the hood the iterator does FAISS searches in
/// progressively larger batches (initial size 64, doubling) so we never
/// pre-fetch a fixed huge top-k upfront. Honors the temporal constraint
/// (`node.ts <= target.ts`) and excludes the target itself. The full
/// yielded set is exposed for Tier 2 dedup.
#[cfg(feature = "vecdb")]
struct VectorDbStream<'a> {
    entry: &'a VectorDbEntry,
    target_node_idx: i32,
    target_ts: Option<i32>,
    query: &'a [f32],
    /// FAISS-ordered candidates already filtered through dedup + temporal.
    buffer: Vec<i32>,
    cursor: usize,
    /// k used for the most recent FAISS search; doubles each `expand()`.
    last_k: usize,
    /// All globally-numbered candidate node_idxs the iterator has emitted
    /// (or filtered out as duplicates of an earlier emit). Used both
    /// internally for dedup across expanding searches and externally by
    /// Tier 2 to avoid redundant BFS re-expansions.
    yielded: HashSet<i32>,
    exhausted: bool,
}

/// Initial FAISS top-k requested. Subsequent expansions double this until
/// either `entry.num_rows` is reached or the consumer stops pulling.
#[cfg(feature = "vecdb")]
const VDB_INIT_K: usize = 64;

#[cfg(feature = "vecdb")]
impl<'a> VectorDbStream<'a> {
    fn new(entry: &'a VectorDbEntry, target_node_idx: i32, target_node: &ArchivedNode) -> Self {
        let local_offset = target_node_idx - entry.node_idx_offset;
        assert!(
            local_offset >= 0 && (local_offset as usize) < entry.num_rows,
            "target node {} is outside the FAISS-indexed range [{}, {})",
            target_node_idx,
            entry.node_idx_offset,
            entry.node_idx_offset + entry.num_rows as i32,
        );
        let local_idx = local_offset as usize;
        let (pref, vectors, suf) = unsafe { entry.vectors_mmap.align_to::<f32>() };
        assert!(
            pref.is_empty() && suf.is_empty(),
            "FAISS vectors mmap is not f32-aligned"
        );
        let query = &vectors[local_idx * entry.dim..(local_idx + 1) * entry.dim];
        let target_ts = target_node.timestamp.as_ref().map(|t| (*t).into());
        Self {
            entry,
            target_node_idx,
            target_ts,
            query,
            buffer: Vec::new(),
            cursor: 0,
            last_k: 0,
            yielded: HashSet::new(),
            exhausted: false,
        }
    }

    fn next(&mut self, dataset: &Dataset) -> Option<i32> {
        loop {
            if self.cursor < self.buffer.len() {
                let v = self.buffer[self.cursor];
                self.cursor += 1;
                return Some(v);
            }
            if self.exhausted {
                return None;
            }
            self.expand(dataset);
        }
    }

    fn expand(&mut self, dataset: &Dataset) {
        let new_k = if self.last_k == 0 {
            VDB_INIT_K.min(self.entry.num_rows)
        } else {
            (self.last_k * 2).min(self.entry.num_rows)
        };
        if new_k == self.last_k {
            self.exhausted = true;
            return;
        }
        let result = self
            .entry
            .index
            .lock()
            .unwrap()
            .search(self.query, new_k)
            .unwrap_or_else(|e| panic!("FAISS search failed (k={}): {}", new_k, e));
        for &label in result.labels.iter() {
            if label.is_none() {
                continue;
            }
            let local = label.to_native();
            if local < 0 || (local as usize) >= self.entry.num_rows {
                continue;
            }
            let global_idx = self.entry.node_idx_offset + local as i32;
            if global_idx == self.target_node_idx {
                continue;
            }
            // Defensive dedup: monotone FAISS top-k means a doubling
            // search re-emits the previous results, so skip what we've
            // already seen.
            if !self.yielded.insert(global_idx) {
                continue;
            }
            // Temporal constraint: only nodes with ts <= target.ts (or
            // None) are valid similar seeds.
            if let Some(target_ts) = self.target_ts {
                let node = get_node(dataset, global_idx);
                if let Some(ts) = node.timestamp.as_ref()
                    && i32::from(*ts) > target_ts
                {
                    continue;
                }
            }
            self.buffer.push(global_idx);
        }
        self.last_k = new_k;
        if new_k >= self.entry.num_rows {
            self.exhausted = true;
        }
    }
}

/// Returns true when the seed's target-column value is missing or NaN.
/// Such seeds are skipped from same-table seed sampling regardless of
/// `balance_labels` — they carry no usable label. Text targets have no
/// NaN concept and never report missing here.
fn seed_label_missing(node: &ArchivedNode, target_column: i32) -> bool {
    let cell_i = match node
        .col_name_idxs
        .iter()
        .position(|&c| i32::from(c) == target_column)
    {
        Some(i) => i,
        None => return true,
    };
    match &node.sem_types[cell_i] {
        ArchivedSemType::Number => f32::from(node.number_values[cell_i]).is_nan(),
        ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]).is_nan(),
        ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]).is_nan(),
        ArchivedSemType::Text => false,
    }
}

/// Bucket a same-table seed by label sign. Callers must have already
/// excluded missing/NaN labels via `seed_label_missing`. Panics on Text
/// targets, which have no signed numeric representation.
fn seed_label_is_negative(node: &ArchivedNode, target_column: i32) -> bool {
    let cell_i = node
        .col_name_idxs
        .iter()
        .position(|&c| i32::from(c) == target_column)
        .expect("seed_label_missing must be checked before seed_label_is_negative");
    let val = match &node.sem_types[cell_i] {
        ArchivedSemType::Number => f32::from(node.number_values[cell_i]),
        ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]),
        ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]),
        ArchivedSemType::Text => {
            panic!("balance_labels=true is not supported for Text targets")
        }
    };
    val < 0.0
}

/// Pop the next seed from a 50/50 mix of two pre-ordered lists, advancing
/// the per-list cursors in place. Falls back to whichever list is non-empty
/// when one runs out. Returns `None` once both lists are drained.
fn pick_balanced(
    neg: &[i32],
    pos: &[i32],
    neg_i: &mut usize,
    pos_i: &mut usize,
    rng: &mut StdRng,
) -> Option<i32> {
    let neg_avail = *neg_i < neg.len();
    let pos_avail = *pos_i < pos.len();
    let pick_neg = match (neg_avail, pos_avail) {
        (false, false) => return None,
        (true, false) => true,
        (false, true) => false,
        (true, true) => rng.random::<bool>(),
    };
    if pick_neg {
        let s = neg[*neg_i];
        *neg_i += 1;
        Some(s)
    } else {
        let s = pos[*pos_i];
        *pos_i += 1;
        Some(s)
    }
}

fn make_pb(len: u64, msg: &'static str, visible: bool) -> ProgressBar {
    if !visible {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg}: {percent:>3}%|\
                 {bar:10}| {pos}/{len} \
                 [{elapsed}<{eta}, {per_sec}]",
            )
            .unwrap()
            .progress_chars("██ "),
    );
    pb.set_message(msg);
    pb
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset
        .p2f_adj_mmap
        .as_ref()
        .expect("BFS requires a sampling graph")[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

/// Per-column semantic type for a preprocessed db, used to recover the
/// schema-derived "autocomplete" pretraining tasks (predict-a-masked-column)
/// that the original pipeline generated from each table's feature columns.
///
/// Returns `{ "<col> of <table>": sem_type }` where sem_type is one of
/// "Boolean" (-> clf task), "Number" (-> reg task), "DateTime", "Text".
/// Foreign-key columns are absent by construction: `pre` emits no cell for
/// them (it only records f2p edges), so they never appear in any node's
/// `col_name_idxs` and are correctly excluded as task targets -- matching the
/// original "feature_cols" (non-FK) semantics.
///
/// Implementation: for each Db table we read its first node (one rkyv access)
/// and map each cell's `col_name_idx` to its `sem_type`. sem_type is uniform
/// within a column, so a single node per table suffices. The col_name_idx is
/// resolved back to the human-readable "<col> of <table>" key via the inverse
/// of `column_index.json`.
#[pyfunction]
pub fn column_sem_types(pre_dir: String, db_name: String) -> PyResult<HashMap<String, String>> {
    let pre_path = format!("{}/{}", pre_dir, db_name);

    // offsets.rkyv
    let offsets: Vec<i64> = {
        let file = fs::File::open(format!("{}/offsets.rkyv", pre_path))?;
        let mut bytes = Vec::new();
        BufReader::new(file).read_to_end(&mut bytes)?;
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        rkyv::deserialize::<Offsets, Error>(archived)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .offsets
    };

    // nodes.rkyv (mmap)
    let nodes_file = fs::File::open(format!("{}/nodes.rkyv", pre_path))?;
    let mmap = unsafe { Mmap::map(&nodes_file)? };

    // table_info.json: table key "<table>:<split>" -> node_idx_offset
    let table_info: HashMap<String, TableInfo> = {
        let f = fs::File::open(format!("{}/table_info.json", pre_path))?;
        serde_json::from_reader(BufReader::new(f))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    // column_index.json: "<col> of <table>" -> global col_idx. Invert it.
    let column_index: HashMap<String, i32> = {
        let f = fs::File::open(format!("{}/column_index.json", pre_path))?;
        serde_json::from_reader(BufReader::new(f))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    let idx_to_name: HashMap<i32, &String> = column_index.iter().map(|(k, v)| (*v, k)).collect();

    let mut out: HashMap<String, String> = HashMap::new();
    for info in table_info.values() {
        if info.num_nodes <= 0 {
            continue;
        }
        let first_idx = info.node_idx_offset;
        let l = offsets[first_idx as usize] as usize;
        let r = offsets[(first_idx + 1) as usize] as usize;
        let node = unsafe { rkyv::access_unchecked::<ArchivedNode>(&mmap[l..r]) };
        for (cell_i, col_idx) in node.col_name_idxs.iter().enumerate() {
            let ci: i32 = (*col_idx).into();
            if let Some(name) = idx_to_name.get(&ci) {
                let sem = match node.sem_types[cell_i] {
                    ArchivedSemType::Number => "Number",
                    ArchivedSemType::Text => "Text",
                    ArchivedSemType::DateTime => "DateTime",
                    ArchivedSemType::Boolean => "Boolean",
                };
                out.insert((*name).clone(), sem.to_string());
            }
        }
    }
    Ok(out)
}

/// Build a deterministic derangement over the column indices listed in
/// `column_index.json` for this database. Seeded from `(shuffle_seed,
/// db_name)` so the same seed reproduces the same permutation, and
/// different databases get independent permutations. Sorting the keys
/// before shuffling pins the result down regardless of HashMap
/// iteration order. Uses Sattolo's algorithm (Fisher-Yates with the
/// upper bound exclusive instead of inclusive) so every column is
/// guaranteed to map to a *different* column — no fixed points, hence
/// every emitted cell's col_name embedding actually changes under
/// ablation. Requires |columns| >= 2.
fn build_col_name_perm(pre_path: &str, db_name: &str, shuffle_seed: u64) -> HashMap<i32, i32> {
    let column_index_path = format!("{}/column_index.json", pre_path);
    let file = fs::File::open(&column_index_path).unwrap();
    let column_index: HashMap<String, i32> = serde_json::from_reader(BufReader::new(file)).unwrap();

    let mut col_indices: Vec<i32> = column_index.values().copied().collect();
    col_indices.sort();
    assert!(
        col_indices.len() >= 2,
        "ablate_schema_semantics needs >= 2 columns in db {} (got {})",
        db_name,
        col_indices.len(),
    );

    let mut db_hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    db_name.hash(&mut db_hasher);
    let db_seed = shuffle_seed.wrapping_add(db_hasher.finish());

    let mut shuffled = col_indices.clone();
    let mut rng = StdRng::seed_from_u64(db_seed);
    // Sattolo's algorithm: like Fisher-Yates but the swap target is drawn
    // from `0..i` instead of `0..=i`, forcing every element to move to a
    // strictly-earlier slot. Result is a uniform random cyclic permutation
    // — no element ends up at its original index.
    for i in (1..shuffled.len()).rev() {
        let j = rng.random_range(0..i);
        shuffled.swap(i, j);
    }

    col_indices.into_iter().zip(shuffled).collect()
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(long)]
    pre_dir: String,
    #[arg(default_value = "128")]
    bs: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Adj, Edge, Node, SemType, TableType};
    use pyo3::types::PyDict;

    fn mmap(bytes: &[u8]) -> Mmap {
        let mut map = memmap2::MmapMut::map_anon(bytes.len().max(8)).unwrap();
        map[..bytes.len()].copy_from_slice(bytes);
        map.make_read_only().unwrap()
    }

    fn node(idx: i32, table: i32, timestamp: Option<i32>) -> Node {
        Node {
            node_idx: idx,
            table_name_idx: table,
            timestamp,
            is_task_node: table == 0,
            col_name_idxs: vec![0, 1],
            sem_types: vec![SemType::Number, SemType::Number],
            number_values: vec![idx as f32, 1.0],
            text_values: vec![0; 2],
            datetime_values: vec![0.0; 2],
            boolean_values: vec![0.0; 2],
            class_value_idx: vec![0; 2],
            ..Default::default()
        }
    }

    fn edge(n: &Node) -> Edge {
        Edge {
            node_idx: n.node_idx,
            table_name_idx: n.table_name_idx,
            timestamp: n.timestamp,
            table_type: if n.table_name_idx == 0 {
                TableType::Train
            } else {
                TableType::Db
            },
        }
    }

    fn fixture() -> Sampler {
        // Historical seed 1 reaches rows at 7/9/10 through timeless DB node 4.
        // Equal target time is valid; future 3/7 are not. Duplicate edge 5
        // tests visited-depth dedup; DB width sampling is exercised at node 4.
        let mut nodes = vec![
            node(0, 0, Some(10)),
            node(1, 0, Some(5)),
            node(2, 0, Some(10)),
            node(3, 0, Some(11)),
            node(4, 1, None),
            node(5, 1, Some(7)),
            node(6, 1, Some(9)),
            node(7, 1, Some(11)),
        ];
        nodes[0].f2p_edges = vec![edge(&nodes[4])];
        nodes[1].f2p_edges = vec![edge(&nodes[4])];
        // A future forward edge must also be rejected, not just reverse edges.
        nodes[2].f2p_edges = vec![edge(&nodes[7])];
        let mut adj = vec![vec![]; nodes.len()];
        adj[4] = [1, 5, 5, 6, 0, 2, 3, 7]
            .iter()
            .map(|&i| edge(&nodes[i]))
            .collect();
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        for n in &nodes {
            bytes.extend_from_slice(&rkyv::to_bytes::<Error>(n).unwrap());
            offsets.push(bytes.len() as i64);
        }
        let dataset = Dataset {
            mmap: mmap(&bytes),
            text_mmap: mmap(&[0; 8]),
            p2f_adj_mmap: Some(mmap(&rkyv::to_bytes::<Error>(&Adj { adj }).unwrap())),
            offsets,
            table_info: HashMap::new(),
            sql_contexts: None,
            sql_seed_order: None,
            sql_neighbor_provider: None,
            col_name_perm: None,
            #[cfg(feature = "vecdb")]
            vector_db: None,
        };
        Sampler {
            global_rank: 0,
            local_rank: 0,
            world_size: 1,
            datasets: HashMap::from([("fixture".into(), dataset)]),
            items: vec![Item {
                dataset_idx: 0,
                node_idx: 0,
                table_name: "task".into(),
            }],
            local_ctx_sizes: vec![8],
            bfs_widths: vec![1],
            num_walks: 12,
            walk_length: 4,
            prefer_latest: vec![true],
            mask_prob_max: 0.0,
            step: 0,
            stride: 1,
            d_text: 1,
            shuffle_seed: 0,
            context_seed: 42,
            target_columns: vec![0],
            columns_to_drop: vec![vec![0]],
            items_per_task: -1,
            dataset_tuples: vec![("fixture".into(), "task".into(), 0, 4)],
            table_ranges: vec![(0, 4)],
            quiet: true,
            skip_text_cols: false,
            balance_labels: vec![false],
            timeout_per_item: 10.0,
            vector_db_path: None,
        }
    }

    fn build(sampler: &Sampler, ctx_len: usize, legacy: bool) -> (BuildTrace, Vecs) {
        let mut trace = BuildTrace {
            legacy,
            ..Default::default()
        };
        let mut vecs = Vecs::new(1, ctx_len, sampler.d_text);
        let mut slices = vecs
            .chunks_exact_mut(ctx_len, sampler.d_text)
            .next()
            .unwrap();
        sampler
            .seq_build_traced(
                &sampler.items[0],
                &mut slices,
                0,
                ctx_len,
                Instant::now() + Duration::from_secs(10),
                Some(&mut trace),
            )
            .unwrap();
        (trace, vecs)
    }

    #[test]
    fn algorithm1_counts_after_move_including_final_destination() {
        let s = fixture();
        let d = &s.datasets["fixture"];
        // Node 4 -> task node 1 is forced by making all other reverse edges future
        // relative to seed 1. The second/final move must count node 1.
        let deadline = Instant::now() + Duration::from_secs(10);
        let legacy = s.compute_visit_counts(d, 1, get_node(d, 1), 1, 2, 42, deadline, true);
        let aligned = s.compute_visit_counts(d, 1, get_node(d, 1), 1, 2, 42, deadline, false);
        // Source itself remains excluded under both policies.
        assert!(legacy.is_empty());
        assert!(aligned.is_empty());
        // From target 0, a walk's final destination contributes after the move.
        let legacy = s.compute_visit_counts(d, 0, get_node(d, 0), 100, 2, 42, deadline, true);
        let aligned = s.compute_visit_counts(d, 0, get_node(d, 0), 100, 2, 42, deadline, false);
        assert!(legacy.is_empty());
        assert!(!aligned.is_empty());
        assert!(!aligned.contains_key(&0));
        assert!(!aligned.contains_key(&3));
        assert_eq!(
            aligned,
            s.compute_visit_counts(d, 0, get_node(d, 0), 100, 2, 42, deadline, false)
        );
    }

    #[test]
    fn legacy_trace_preserves_pre_alignment_fixture() {
        let s = fixture();
        let (trace, _) = build(&s, 12, true);
        assert_eq!(trace.visits, vec![(1, 2), (2, 2)]);
        assert_eq!(trace.candidates, vec![0, 2, 1]);
        assert_eq!(trace.fallback_candidates, vec![2]);
        assert_eq!(
            serde_json::to_value(&trace.collections).unwrap(),
            serde_json::json!([
                {"target":0,"target_timestamp":10,"seed":0,"cutoff":10,"local_ctx_size":8,"bfs_width":1,
                 "remaining_cells":11,"visited_before":[],"rows":[[0,0],[4,1],[5,2]]},
                {"target":0,"target_timestamp":10,"seed":2,"cutoff":10,"local_ctx_size":8,"bfs_width":1,
                 "remaining_cells":6,"visited_before":[[0,0],[4,1],[5,2]],"rows":[[2,0],[7,1]]},
                {"target":0,"target_timestamp":10,"seed":1,"cutoff":5,"local_ctx_size":8,"bfs_width":1,
                 "remaining_cells":3,"visited_before":[[0,0],[2,0],[4,1],[5,2],[7,1]],"rows":[[1,0]]}
            ])
        );
        assert_eq!(
            trace.cells,
            vec![
                (0, 0, 0, 0, 0),
                (0, 1, 1, 0, 0),
                (4, 0, 0, 0, 1),
                (4, 1, 1, 0, 1),
                (5, 0, 0, 0, 2),
                (5, 1, 1, 0, 2),
                (2, 1, 1, 2, 0),
                (7, 0, 0, 2, 1),
                (7, 1, 1, 2, 1),
                (1, 0, 0, 1, 0),
                (1, 1, 1, 1, 0),
            ]
        );
    }

    #[test]
    fn algorithm1_score_first_timestamp_only_breaks_ties() {
        use std::cmp::Ordering::Less;
        let older_high_score = (9, Some(5), 10, 1);
        let newer_low_score = (2, Some(10), 0, 2);
        assert_eq!(
            candidate_cmp(older_high_score, newer_low_score, true, false),
            Less
        );
        assert_eq!(
            candidate_cmp(newer_low_score, older_high_score, true, true),
            Less
        );
        assert_eq!(
            candidate_cmp((2, Some(10), 10, 2), (2, Some(5), 0, 1), true, false),
            Less
        );
        assert_eq!(
            candidate_cmp((2, Some(5), 0, 1), (2, Some(10), 10, 2), false, false),
            Less
        );
        assert_eq!(
            candidate_cmp((2, Some(5), 0, 1), (2, None, 0, 2), true, false),
            Less
        );
    }

    #[test]
    fn algorithm1_every_seed_uses_target_cutoff_and_preserves_bfs_budget() {
        let s = fixture();
        let d = &s.datasets["fixture"];
        let collect = |legacy, width, budget| {
            s.bfs_collect_nodes(
                d,
                1,
                0,
                legacy,
                &mut StdRng::seed_from_u64(42),
                budget,
                width,
                &mut HashMap::new(),
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap()
        };
        let aligned = collect(false, 10, 100);
        let legacy = collect(true, 10, 100);
        assert!(aligned.contains(&(2, 2))); // Same time as target, later than seed.
        assert!(!legacy.contains(&(2, 2)));
        assert!(aligned.contains(&(5, 2)) && aligned.contains(&(6, 2)));
        assert!(aligned.iter().all(|&(n, _)| n != 3 && n != 7));
        assert_eq!(aligned.iter().filter(|&&(n, _)| n == 5).count(), 1);
        let narrow = collect(false, 1, 100);
        assert_eq!(narrow.iter().filter(|&&(n, _)| n == 5 || n == 6).count(), 1);
        // Preserve the existing exclusive local cell limit: the row bringing
        // raw column count to >= local_ctx_size is not returned/marked visited.
        assert!(collect(false, 10, 2).is_empty());
        assert_eq!(collect(false, 10, 4), vec![(1, 0)]);
        assert!(within_cutoff(Some(100), None)); // No target time means unbounded.
    }

    #[test]
    fn traces_are_repeatable_target_first_masked_and_budget_bounded() {
        let mut s = fixture();
        let (a, va) = build(&s, 12, false);
        let (b, vb) = build(&s, 12, false);
        assert_eq!(a.visits, b.visits);
        assert_eq!(a.candidates, b.candidates);
        assert_eq!(a.cells, b.cells);
        assert_eq!(
            serde_json::to_value(&a.collections).unwrap(),
            serde_json::to_value(&b.collections).unwrap()
        );
        assert_eq!(va.is_targets, vb.is_targets);
        assert_eq!(a.candidates[0], 0);
        assert_eq!(a.collections[0].seed, 0);
        assert!(a.collections.iter().all(|c| c.cutoff == Some(10)));
        assert_eq!(a.cells[0], (0, 0, 0, 0, 0));
        assert!(va.is_targets[0]);
        assert!(a.cells.len() <= 12);
        let unique: HashSet<_> = a.cells.iter().map(|c| (c.0, c.1)).collect();
        assert_eq!(unique.len(), a.cells.len());
        assert!(a.cells.iter().all(|c| {
            !(c.2 == 0
                && c.0 != 0
                && get_node(&s.datasets["fixture"], c.0)
                    .timestamp
                    .as_ref()
                    .map(|t| i32::from(*t))
                    == Some(10))
        }));
        // Width/local collection changes do not perturb Stage 1 scores/order.
        s.bfs_widths = vec![10];
        s.local_ctx_sizes = vec![12];
        let (c, _) = build(&s, 12, false);
        assert_eq!(a.visits, c.visits);
        assert_eq!(a.candidates, c.candidates);
        s.num_walks = 0;
        s.local_ctx_sizes = vec![2]; // Force empty collections so fallback is used.
        let (f, _) = build(&s, 12, false);
        assert!(f.visits.is_empty());
        assert_eq!(f.candidates, vec![0]);
        assert!(!f.fallback_candidates.is_empty());
        assert!(f.collections.iter().any(|c| c.seed != 0));
    }

    #[test]
    fn aligned_reverse_order_is_canonical_across_null_and_equal_timestamp_ties() {
        let mut s = fixture();
        // Simulate cross-table/task insertion order from pre.rs's HashMap.
        let mut adj = vec![vec![]; 8];
        let d = &s.datasets["fixture"];
        adj[4] = [6, 5, 2, 0, 0]
            .iter()
            .map(|&idx| Edge {
                node_idx: idx,
                table_name_idx: if idx < 4 { 0 } else { 1 },
                table_type: if idx < 4 {
                    TableType::Train
                } else {
                    TableType::Db
                },
                timestamp: if idx < 4 { Some(10) } else { None },
            })
            .collect();
        let original: Vec<i32> = adj[4].iter().map(|e| e.node_idx).collect();
        assert_eq!(original, vec![6, 5, 2, 0, 0]);
        assert!(d.sql_neighbor_provider.is_none());
        s.datasets.get_mut("fixture").unwrap().p2f_adj_mmap = Some(mmap(
            &rkyv::to_bytes::<Error>(&Adj { adj: adj.clone() }).unwrap(),
        ));
        let d = &s.datasets["fixture"];
        let (_, reverse) = ordered_stage2_edges(d, 4, 0, 0, Some(10)).unwrap();
        let canonical: Vec<i32> = reverse.iter().map(|(idx, _)| idx).collect();
        assert_eq!(canonical, vec![5, 6, 0, 0, 2]);
        // The stored archive used by legacy sampling must not be rewritten.
        assert_eq!(
            get_p2f_edges(d, 4)
                .iter()
                .map(|e| i32::from(e.node_idx))
                .collect::<Vec<_>>(),
            original
        );
        let collect = |s: &Sampler| {
            s.bfs_collect_nodes(
                &s.datasets["fixture"],
                0,
                0,
                false,
                &mut StdRng::seed_from_u64(42),
                100,
                1,
                &mut HashMap::new(),
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap()
        };
        let bfs_rows = collect(&s);
        install_provider(&mut s);
        assert_eq!(collect(&s), bfs_rows); // Canonical SQL order is accepted.
        // Permute both timestamp ties without changing time-sortedness. The
        // installed provider and aligned BFS must still yield the same rows.
        adj[4].swap(0, 1);
        adj[4][2..].reverse();
        s.datasets.get_mut("fixture").unwrap().p2f_adj_mmap =
            Some(mmap(&rkyv::to_bytes::<Error>(&Adj { adj }).unwrap()));
        assert_eq!(collect(&s), bfs_rows);
        Python::with_gil(|py| s.set_sql_neighbor_provider_py(py, "fixture", None).unwrap());
        assert_eq!(collect(&s), bfs_rows);
    }

    #[test]
    fn sql_neighbor_mapping_consumes_owned_ids_and_preserves_duplicate_metadata() {
        let s = fixture();
        let reverse = get_p2f_edges(&s.datasets["fixture"], 4);
        let metadata = vec![&reverse[1], &reverse[2], &reverse[3]];
        let first_occurrence = metadata[0] as *const ArchivedEdge;
        let second_occurrence = metadata[1] as *const ArchivedEdge;
        let sql_ids = vec![5, 5, 6];
        let sql_buffer = sql_ids.as_ptr();
        let mapped = Stage2Edges::from_sql(sql_ids, metadata).unwrap();
        // Reject implementations which validate SQL but rebuild graph IDs.
        assert_eq!(mapped.node_idxs.as_ptr(), sql_buffer);
        assert_eq!(
            mapped.iter().map(|(idx, _)| idx).collect::<Vec<_>>(),
            vec![5, 5, 6]
        );
        assert_ne!(first_occurrence, second_occurrence);
        assert_eq!(mapped.metadata[0] as *const ArchivedEdge, first_occurrence);
        assert_eq!(mapped.metadata[1] as *const ArchivedEdge, second_occurrence);
    }

    fn install_provider(s: &mut Sampler) -> PyObject {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let d = &s.datasets["fixture"];
            let adjacency: HashMap<_, _> = (0..8)
                .map(|n| {
                    let (f, p) = ordered_stage2_edges(d, n, 0, 0, Some(10)).unwrap();
                    let ids = |edges: Stage2Edges<'_>| edges.node_idxs;
                    (n, (ids(f), ids(p)))
                })
                .collect();
            let globals = PyDict::new(py);
            globals.set_item("adjacency", adjacency).unwrap();
            globals
                .set_item("calls", Vec::<(i32, i32, i32, Option<i32>)>::new())
                .unwrap();
            py.run(pyo3::ffi::c_str!("def provider(node, target, seed, cutoff):\n    calls.append((node, target, seed, cutoff))\n    return adjacency[node]\n"), Some(&globals), None).unwrap();
            let provider = globals.get_item("provider").unwrap().unwrap().unbind();
            let calls = globals.get_item("calls").unwrap().unwrap().unbind();
            s.set_sql_neighbor_provider_py(py, "fixture", Some(provider))
                .unwrap();
            calls
        })
    }

    #[test]
    fn sql_provider_preserves_stage1_bfs_rng_rows_and_final_cells() {
        let mut s = fixture();
        let (bfs, vb) = build(&s, 12, false);
        let calls = install_provider(&mut s);
        let (sql, vs) = build(&s, 12, false); // Caller has no GIL; callback reacquires it.
        assert_eq!(bfs.visits, sql.visits);
        assert_eq!(bfs.candidates, sql.candidates);
        assert_eq!(bfs.cells, sql.cells);
        assert_eq!(vb.node_idxs, vs.node_idxs);
        assert_eq!(vb.is_targets, vs.is_targets);
        assert_eq!(
            serde_json::to_value(&bfs.collections).unwrap(),
            serde_json::to_value(&sql.collections).unwrap()
        );
        Python::with_gil(|py| {
            // Same dtype registration performed by rt.data before batch export.
            py.import("ml_dtypes").unwrap();
            let calls: Vec<(i32, i32, i32, Option<i32>)> = calls.bind(py).extract().unwrap();
            assert_eq!(
                calls.len(),
                sql.collections.iter().map(|c| c.rows.len()).sum::<usize>()
            );
            assert!(calls.iter().all(|c| c.1 == 0 && c.3 == Some(10)));
            // Exercise the Rayon/GIL boundary and the actual Python-callable API.
            assert!(s.batch_for_nodes_py(py, vec![0, 0], 0, 12).is_ok());
            let trace = s.trace_py(py, 0, 0, 12, 0, false).unwrap();
            assert!(trace.bind(py).get_item("collections").is_ok());
            assert_eq!(
                trace
                    .bind(py)
                    .get_item("reverse_neighbor_order")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "timestamp_none_first_then_node_idx"
            );
            assert!(s.trace_py(py, 0, 0, 12, 0, true).is_err());
            s.set_sql_neighbor_provider_py(py, "fixture", None).unwrap();
            let legacy = s.trace_py(py, 0, 0, 12, 0, true).unwrap();
            assert_eq!(
                legacy
                    .bind(py)
                    .get_item("reverse_neighbor_order")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "stored"
            );
        });
    }

    #[test]
    fn eager_raw_sql_experiment_remains_isolated_and_unchanged() {
        let mut s = fixture();
        let contexts = (0..4).map(|n| (n, vec![(n, 0), (4, 1), (5, 2)])).collect();
        s.set_sql_contexts_py("fixture", contexts).unwrap();
        let orders = (0..4)
            .map(|n| (n, if n == 0 { vec![1] } else { vec![] }))
            .collect();
        s.set_sql_seed_order_py("fixture", orders).unwrap();
        s.datasets.get_mut("fixture").unwrap().p2f_adj_mmap = None;
        let (trace, _) = build(&s, 12, false);
        assert!(trace.visits.is_empty());
        assert_eq!(trace.candidates, vec![0, 1]);
        assert!(trace.fallback_candidates.is_empty());
        let seed = trace.collections.iter().find(|c| c.seed == 1).unwrap();
        assert_eq!(seed.cutoff, Some(5)); // Eager/raw SQL retains its old policy.
        assert!(!seed.rows.iter().any(|&(n, _)| n == 5));
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let globals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!("def provider(*args):\n    return ([], [])\n"),
                Some(&globals),
                None,
            )
            .unwrap();
            let provider = globals.get_item("provider").unwrap().unwrap().unbind();
            assert!(
                s.set_sql_neighbor_provider_py(py, "fixture", Some(provider))
                    .is_err()
            );
        });
    }

    #[test]
    fn sql_provider_rejects_bad_identity_order_cutoff_and_propagates_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert!(validate_neighbor_order(&[2, 1], &[1, 2]).is_err());
            assert!(validate_neighbor_order(&[1], &[1, 1]).is_err());
            assert!(validate_neighbor_order(&[-1], &[1]).is_err());
            assert!(validate_neighbor_order(&[1, 3], &[1]).is_err());
            let mut s = fixture();
            let globals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!("def bad(*args):\n    return ([999999], [])\n"),
                Some(&globals),
                None,
            )
            .unwrap();
            let bad = globals.get_item("bad").unwrap().unwrap().unbind();
            s.set_sql_neighbor_provider_py(py, "fixture", Some(bad))
                .unwrap();
            assert!(s.trace_py(py, 0, 0, 12, 0, false).is_err());
            assert!(s.batch_py(py, Some(0), 1, 12).is_err());
            assert!(s.batch_py(py, None, 1, 12).is_err());
            assert_eq!(s.step, 0); // Errors do not advance the sampling step.
            py.run(
                pyo3::ffi::c_str!(
                    "def broken(*args):\n    raise RuntimeError('SQL query failed')\n"
                ),
                Some(&globals),
                None,
            )
            .unwrap();
            let broken = globals.get_item("broken").unwrap().unwrap().unbind();
            s.set_sql_neighbor_provider_py(py, "fixture", Some(broken))
                .unwrap();
            let err = s.trace_py(py, 0, 0, 12, 0, false).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(err.to_string().contains("SQL query failed"));
            assert!(s.set_sql_contexts_py("fixture", HashMap::new()).is_err());
            let not_callable = 1i32.into_py_any(py).unwrap();
            assert!(
                s.set_sql_neighbor_provider_py(py, "fixture", Some(not_callable))
                    .is_err()
            );
        });
    }
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let sampler = Sampler::new_impl(
        vec![(cli.db_name.clone(), String::new(), 0, 100000)], // dataset_tuples
        0,                                                     // global_rank
        0,                                                     // local_rank
        1,                                                     // world_size
        vec![128],                                             // local_ctx_sizes
        vec![16],                                              // bfs_widths
        0,                                                     // num_walks (random_same_table)
        10,                                                    // walk_length
        vec![false],                                           // prefer_latest
        0.5,                                                   // mask_prob_max
        "all-MiniLM-L12-v2",                                   // embedding_model
        cli.pre_dir.clone(),                                   // pre_dir
        384,                                                   // d_text
        0,                                                     // shuffle_seed
        0,                                                     // context_seed
        vec![0_i32],                                           // target_columns
        vec![Vec::<i32>::new()],                               // columns_to_drop
        -1,                                                    // items_per_task (no limit)
        false,                                                 // quiet
        false,                                                 // ignore_data_errors
        0,                                                     // num_prev_skipped
        false,                                                 // skip_text_cols
        true,                                                  // mmap_populate
        vec![false],                                           // balance_labels
        1.0,                                                   // timeout_per_item
        false,                                                 // ablate_schema_semantics
        None,                                                  // vector_db_path
        false,                                                 // train_only_fallback
    );
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len(cli.bs));
        let _vecs = sampler
            .batch(Some(batch_idx), 0, cli.bs, cli.seq_len)
            .expect("Sampler batch failed");
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
