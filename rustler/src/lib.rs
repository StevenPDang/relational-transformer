use pyo3::prelude::*;

mod common;
pub mod fly;
#[cfg(feature = "pre")]
mod pre;

/// Read the stored timestamps of specific preprocessed nodes without sampling
/// contexts. The alignment check uses this to compare every task seed to the
/// timestamp returned by the RelBench API.
#[pyfunction]
fn node_timestamps(pre_dataset_dir: &str, node_idxs: Vec<usize>) -> PyResult<Vec<Option<i32>>> {
    use crate::common::{ArchivedNode, ArchivedOffsets, Offsets};
    use rkyv::rancor::Error;

    let offsets_bytes = std::fs::read(format!("{pre_dataset_dir}/offsets.rkyv"))?;
    let archived = rkyv::access::<ArchivedOffsets, Error>(&offsets_bytes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let offsets = rkyv::deserialize::<Offsets, Error>(archived)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .offsets;
    let nodes_bytes = std::fs::read(format!("{pre_dataset_dir}/nodes.rkyv"))?;
    node_idxs
        .into_iter()
        .map(|idx| {
            let range = offsets.get(idx..idx + 2).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("node index {idx} out of range"))
            })?;
            let start = usize::try_from(range[0]).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            })?;
            let end = usize::try_from(range[1]).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            })?;
            let bytes = nodes_bytes.get(start..end).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid node offset for {idx}"))
            })?;
            // The preprocessor writes each archived node back-to-back, so an
            // individual slice need not satisfy rkyv's alignment validator.
            // The sampler reads these same trusted slices this way.
            let node = unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) };
            Ok(node.timestamp.as_ref().map(|ts| (*ts).into()))
        })
        .collect()
}

/// Distinct text-embedding rows the sampler can look up for these nodes: one
/// for each cell's value (`text_values`, 0 for non-text cells) and one for each
/// cell's column name (`col_name_idxs`), returned sorted as (values, names).
/// Measures how much of `text.json` a set of contexts actually needs embedded.
#[pyfunction]
fn node_text_ids(pre_dataset_dir: &str, node_idxs: Vec<usize>) -> PyResult<(Vec<i32>, Vec<i32>)> {
    use crate::common::{ArchivedNode, ArchivedOffsets, Offsets};
    use rkyv::rancor::Error;
    use std::collections::BTreeSet;

    let offsets_bytes = std::fs::read(format!("{pre_dataset_dir}/offsets.rkyv"))?;
    let archived = rkyv::access::<ArchivedOffsets, Error>(&offsets_bytes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let offsets = rkyv::deserialize::<Offsets, Error>(archived)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .offsets;
    let nodes_bytes = std::fs::read(format!("{pre_dataset_dir}/nodes.rkyv"))?;
    let (mut values, mut names) = (BTreeSet::new(), BTreeSet::new());
    for idx in node_idxs {
        let range = offsets.get(idx..idx + 2).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("node index {idx} out of range"))
        })?;
        let (start, end) = (range[0] as usize, range[1] as usize);
        let bytes = nodes_bytes.get(start..end).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid node offset for {idx}"))
        })?;
        // Same trusted back-to-back slices as node_timestamps reads.
        let node = unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) };
        values.extend(node.text_values.iter().map(|v| i32::from(*v)));
        names.extend(node.col_name_idxs.iter().map(|v| i32::from(*v)));
    }
    Ok((values.into_iter().collect(), names.into_iter().collect()))
}

/// Preprocess a relbench-3.0.0-layout dataset dir (parquet -> rustler's on-disk
/// rkyv format). Only built into wheels with the `pre` feature. Releases the GIL.
#[cfg(feature = "pre")]
#[pyfunction]
#[pyo3(signature = (dataset_dir, out_dir, *, source=None, skip_tasks=false, skip_db=false, skip_graph=false))]
fn preprocess(
    py: Python<'_>,
    dataset_dir: String,
    out_dir: String,
    source: Option<String>,
    skip_tasks: bool,
    skip_db: bool,
    skip_graph: bool,
) -> PyResult<()> {
    py.allow_threads(move || {
        pre::main(pre::Cli {
            dataset_dir,
            out_dir,
            source,
            skip_tasks,
            skip_db,
            skip_graph,
        })
    });
    Ok(())
}

#[pymodule]
fn _rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;
    m.add_function(wrap_pyfunction!(node_timestamps, m)?)?;
    m.add_function(wrap_pyfunction!(node_text_ids, m)?)?;
    m.add_function(wrap_pyfunction!(fly::column_sem_types, m)?)?;
    #[cfg(feature = "pre")]
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;

    Ok(())
}
