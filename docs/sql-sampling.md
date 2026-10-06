# Small rel-f1 SQL sampling experiment

## Target: replace Stage 2 only

The intended translation is **Algorithm 1's Stage 2**: keep random-walk visit
scoring and same-table candidate selection in Rust (Stage 1), then replace each
`BFSCOLLECT(D, s, ℓ, b, t⋆, remaining_budget)` call with a SQL neighborhood
query. The returned rows still pass through the existing cell-budget,
deduplication, masking, and model-input code. Success means comparing SQL and
BFS expansion with the *same* Stage 1 seeds and ordering, not merely obtaining
a similar AUROC after changing both stages.

The hybrid path below is closer to this target because it retains Rust's
random-walk stage. The raw-data path is a separate feasibility experiment: it
also replaces Stage 1 with SQL ordering. Its accuracy and 54-second setup cost
therefore do not measure the performance or fidelity of a Stage-2-only design.
Keeping the current Stage 1 also keeps its graph adjacency input unless that
stage is changed in a separate experiment.
The implementation steps and verification gates are in
[the Stage 2 plan](../tasks/plan.md).

Two details need an explicit choice before claiming equivalence to the stated
algorithm. With `prefer_latest`, the current Rust code sorts by timestamp
*before* visit count, whereas Algorithm 1 says score first and timestamp only
breaks ties. Also, the pseudocode passes target time `t⋆` to `BFSCOLLECT`, while
the current Rust expansion and materialized SQL neighborhoods use the seed's
timestamp for local expansion. A Stage 2 replacement should be compared against
the chosen semantics with identical Stage 1 output and temporal cutoffs.

## Compare SQL and BFS starting from raw data

Use `--raw-dataset` instead of `--pre-dir` to include preparation in evaluation
runtime. Raw evaluation currently supports `rel-f1/driver-top3` on the complete
test split in simple mode. No hosted preprocessed graph is read by this path.

SQL encodes raw cells and their foreign-key references for the transformer's
attention masks, but creates no forward traversal edges or reverse adjacency
file. DuckDB imports the raw Parquet tables, selects local neighborhoods, and
orders historical task seeds by same driver first, then recency. It replaces
both BFS and random walks. `--num-walks` and `--walk-length` do not apply to raw
SQL. The SQL historical seeds have strictly earlier timestamps; the Rust BFS
baseline retains its original seed policy. This changes sampling behavior, so
the previous hybrid SQL AUROC does not establish raw SQL's accuracy.

Both paths run the same cell normalization and compute text embeddings from
scratch using the checkpoint's embedding model. They still produce a cell
store: avoiding traversal-graph generation does not eliminate model-input
encoding or the row-level foreign-key metadata the model requires.

Run on the same CUDA machine, using a fresh `--prepare-dir` each time:

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --raw-dataset stanford-star/relbench/rel-f1 \
  --prepare-dir data/raw_eval/sql_run1 --sampler sql \
  --tasks rel-f1/driver-top3 --out-dir eval_sql_raw_run1 \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --prefer-latest --shuffle-seed 0

pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --raw-dataset stanford-star/relbench/rel-f1 \
  --prepare-dir data/raw_eval/bfs_run1 --sampler bfs \
  --tasks rel-f1/driver-top3 --out-dir eval_bfs_raw_run1 \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --num-walks 10000 --walk-length 20 --prefer-latest --shuffle-seed 0
```

`--prepare-dir` must not exist: each run builds new artifacts rather than
silently reusing a graph, embeddings, or DuckDB import. Repeat with `run2` and
`run3`, alternating sampler order. For a reproducible comparison, resolve the
raw dataset once and pass the same local directory to both commands. Keep the
checkpoint, embedding batch size, worker count and hardware identical. Confirm
that both runs report `n=726` and compare their AUROC as well as runtime.

Each output directory gets `runtime.json`. Its `seconds` object records model
loading, raw preparation, evaluator setup, evaluation/scoring, and total time.
`preparation_seconds` breaks down raw input resolution, cell encoding (plus
graph construction for BFS), text embeddings, and DuckDB import for SQL. This
breakdown is included in `raw data preparation`, so do not add it twice.
`total` starts after CLI argument parsing and includes raw-data preparation;
Python imports and Pixi's native build/install step are outside that clock.
CUDA is synchronized at evaluation timing boundaries.

Raw SQL runs also report `setup_seconds` in `runtime.json` and print its stages:
SQL neighborhood queries and assembly, historical seed queries, and separate
Python-to-Rust transfer/validation for contexts and seed orders. `other evaluator
setup` is the remaining time in `task and evaluator setup`, including evaluator
construction and data-loader initialization. These are subdivisions of setup,
not additional time to add to `total`.

Verified locally: graph-free and graph-backed encoders produce identical model
tensors for identical SQL selections, including forward-key attention metadata.
Both raw-data paths also produced complete 726-row submissions from the cached
rel-f1 source with real preparation/embeddings/sampling and mock predictions.
That smoke check does not measure model accuracy or GPU inference performance.

## Challenges in translating sampling to SQL

- **The sampler is more than graph traversal.** The Rust path uses random walks
  to rank same-table task seeds and BFS to expand each seed. Raw SQL replaces
  both with earlier task rows ordered by same driver and recency, then fixed
  joins for local history. This changes which evidence the model sees, so SQL
  is not an exact implementation of the BFS policy. `--num-walks` and
  `--walk-length` cannot make the two paths equivalent.
- **Context budgets make row order consequential.** An early SQL version put too
  few historical labels in the model context. We moved earlier labels ahead of
  database history and interleaved qualifying, standings, and results by
  recency rank. The four-target sampling check below shows the resulting label
  counts; it is a context check, not an accuracy measurement. Rust still
  applies cell budgets, deduplication, masks, and temporal filtering after SQL
  selects rows.
- **SQL row IDs must match model node IDs.** The model consumes encoded cells
  addressed by Rust node index, while DuckDB queries identify raw rows by
  `rowid`. Joins therefore retain `rowid` explicitly and add each table's node
  offset. The hybrid path verifies raw table order and values plus task counts
  and timestamps against the preprocessed source. The raw path reads the same
  Parquet files for both encoding and DuckDB import, avoiding that separate
  source-alignment pass.
- **Removing the traversal graph does not remove preprocessing.** SQL still
  needs encoded cells, text embeddings, and foreign-key metadata for the model's
  attention masks. On the measured runs, raw preparation took about five
  seconds with either sampler; the SQL path did not gain a meaningful setup
  advantage from omitting graph construction.
- **Eager SQL work dominates this raw prototype's runtime.** Every possible
  train, validation, and test task seed needs a neighborhood because it may be
  selected as historical context. The current implementation materializes all
  2,667 neighborhoods and each seed's ordered historical candidates before
  evaluation. That repeated per-seed query work is the main cost, not
  Python-to-Rust transfer.

For the full 726-row test split, the reported AUROC was **0.909888 for raw SQL**
and **0.911293 for raw BFS** (a 0.001404 difference). The latest timed CUDA
runs, with `ctx-size=8192`, `local-ctx-size=256`, and `bfs-width=32`, reported:

| Measure | Raw SQL | Raw BFS |
| --- | ---: | ---: |
| Raw data preparation | 5.23 s | 4.94 s |
| Task and evaluator setup | 54.29 s | 0.09 s |
| Evaluation and scoring | 54.23 s | 55.89 s |
| Total | 114.67 s | 61.85 s |

The AUROC difference is small on this test split; no uncertainty estimate was
computed. The timed runs show SQL about 1.85 times slower. Earlier runs were
similar (114.53 s for SQL, 61.67 s for BFS), though they lacked the setup
breakdown. The latest SQL run's 54.29-second setup breaks down into 46.88 seconds
for neighborhood queries and assembly, 6.96 seconds for historical seed queries,
0.17 seconds for both Rust transfer/validation steps, and 0.28 seconds of other
evaluator setup. These measurements explain the raw prototype's cost. The
Stage-2-only target should measure its own SQL calls after holding Stage 1
fixed; optimizing raw SQL's historical seed ordering is outside that target.

## On-demand Stage 2 SQL collector

Graph-backed `--sql-context-db` evaluation now uses `load_stage2_sql_provider`
in `src/rt/sql_context.py`, installed through
`Sampler.set_sql_neighbor_provider_py`. Graph-free raw SQL retains the earlier
experiment and its historical-seed policy. No eager task-context map or SQL
seed-order map is created for graph-backed evaluation.

The aligned BFS baseline counts walk destinations after moving, ranks by visit
score with optional recency tie-breaking, and applies the target's timestamp to
all local seeds and both edge directions. Null cutoff means unbounded; null
neighbors and equal timestamps are eligible. Reverse-edge ties are ordered by
node index (null timestamps first). This fourth alignment is necessary because
preprocessing previously inherited hash-map iteration order across tables.
Forward keys retain source column/list order. These baseline changes affect
ordinary BFS too; old benchmark numbers are not measurements of this baseline.

`Sampler.trace_py(dataset_idx, node_idx, ctx_size, step=0, legacy=False)` returns
visits, candidate order, fallback order, collector parameters/rows, final cell
identities, and `sequence` as the usual list of `(name, value)` pairs. Set
`legacy=True` before installing a provider to capture the pre-alignment policy;
Rust tests retain a golden legacy trace.

The SQL provider returns complete ordered forward/reverse neighbor IDs for one
expanded node. Rust validates them against the encoded graph, then uses those
SQL-returned IDs with graph metadata to retain its frontier, width RNG, shared
visited depths, local/global budgets, deduplication, and masks. FK values are
**parent row positions**, not arbitrary PK values. SQL uses parameterized
cutoffs, row-position joins, and `UNION ALL` to preserve duplicate FK paths.

Source tables must match the physical DuckDB rows and encoded counts/timestamps.
Every graph table and task split must be available, including local task
manifests. Each process/thread lazily opens a read-only connection with temporary
normalized row tables and an edge view. Connections/locks are excluded from
pickling; persistent database tables are not modified. The provider still copies
source row/FK metadata per querying thread; it is not a graph-free sampler.

### Validation and performance status

`tests/test_sql_stage2.py` covers native BFS/SQL trace and batch parity, duplicate
paths, multiple child tables and all task splits, equal/future timestamps,
width-limited collection, budgets, deduplication, masks, thread/fresh-process
pickle determinism, and rejection of source/neighbor mismatches. Rust has ten
focused baseline/seam tests. Full DataLoader fork determinism and production
revision parity remain unverified.

A bounded Mac CPU spike on a 100-node synthetic graph (80 walks, length 6,
64 global cells, 32 local cells, width 1, ten warm repeats) measured:

| Collector | Cold item | Warm mean/item |
| --- | ---: | ---: |
| Aligned BFS | 0.152 ms | 0.070 ms |
| On-demand Python/DuckDB prototype | 131.3 ms | 116.4 ms |

The SQL measurements included 20 adjacency queries/item and graph validation.
This route is **not accepted as a performance solution**. It provides a working
correctness reference; plan step 4 still requires a batched/two-pass or native
worker-local execution route. Do not interpret this tiny spike as rel-f1 timing.
The cached rel-f1 source also lacks a required task manifest, so production
source validation currently fails closed. No new CUDA timing or AUROC is claimed.

### Running equal-input comparisons

After building the extension, with complete matching source data:

```bash
pixi run --environment mac python -m pytest tests/test_sql_stage2.py tests/test_sql_context.py
pixi run --environment mac python -m scripts.compare_sql_stage2 \
  --pre-dir stanford-star/relbench-preprocessed \
  --duckdb data/duckdb/rel-f1.duckdb --items 726 --repeats 3 \
  --ctx-size 8192 --local-ctx-size 256 --width 32 \
  --num-walks 10000 --walk-length 20 --context-seed 0 \
  --output compare_sql_stage2.json
```

The checkpoint-independent harness records legacy/aligned/SQL traces, setup and
per-item times, overlaps, label counts, query counts, temporal/mask/budget checks,
and exact aligned BFS/SQL Stage 1 and encoded-sequence parity. Equal-time label
counts are reported rather than rejected under the inclusive Algorithm 1 policy.
Defaults cover all 726 test targets; no model or AUROC is involved.

For model scoring on the same CUDA machine, run `scripts/eval.py` twice with the
same checkpoint, prepared data, seed and context settings, changing only
`--sql-context-db` and the output directory. Verify both score all 726 rows and
repeat end-to-end timings. Keep the raw experiment above separate.

## Archived eager hybrid SQL experiment

The first experiment replaces local BFS expansion for `rel-f1/driver-top3`.
Random walks and same-table seed selection remain in Rust. SQL supplies each
seed's neighborhood; Rust still supplies embeddings, masks, deduplication,
cell budgets, and the model input format.

Non-recursive queries select the driver's row, earlier labels for that driver,
and recent qualifying, standings and results rows at or before the seed
timestamp, with their races, constructors and circuits. Labels precede DB
history, which is interleaved by recency rank to retain all three history tables
within small budgets. This is a small task-specific neighborhood, not an
exact reproduction of the BFS sample. Raw values are retained rather than
aggregated into features.

At that time, opt-in evaluation built batches from these SQL nodes, retained
the target and budgets, and left the default sampler unchanged. This archived
policy is no longer selected by graph-backed `--sql-context-db`.

Implementation: `src/rt/sql_context.py` materializes neighborhoods once before
DataLoader workers start; `rustler/src/fly.rs` consumes them instead of BFS.
Tests live in `tests/test_sql_context.py`. Use `pixi run build-sampler` to build
the extension and `pixi run python -m pytest tests/test_sql_context.py` to test.
No persistent database tables or preprocessing artifacts are modified.

The commands below now select the on-demand implementation described above;
the following query-width, seed-cutoff, and materialization descriptions are
historical details of the eager experiment, not the current flag's behavior.

Build and inspect a small batch locally without a model or GPU:

```bash
pixi run --environment mac python -m scripts.sample_sql_context --items 4
```

This prints the target node, selected nodes, cell count, and context-label count
for each sampled test row, and checks target placement and temporal cutoffs.
Use `--num-walks 10000` to retain the normal walk-based seed ranking; the smoke
command defaults to `0` for quick iteration.

Run the full test split on a CUDA machine with the original baseline's context
settings, changing only the sampler:

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed \
  --tasks rel-f1/driver-top3 --out-dir eval_sql_matched \
  --sql-context-db data/duckdb/rel-f1.duckdb \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --num-walks 10000 --walk-length 20 --prefer-latest --shuffle-seed 0
```

Repeat the command without `--sql-context-db` and with
`--out-dir eval_bfs_matched` for the matching BFS run. Both runs must report
`n=726`, use the same checkpoint and data, and keep the same context flags.

`--bfs-width` bounds rows from each history table and historical labels per seed
in SQL mode.
`--local-ctx-size` and `--ctx-size` retain their existing cell budget semantics.
Leave `--num-walks` at its default to preserve walk-based seed selection; set it
to `0` for a faster smoke run with random same-table seeds. Omit
`--sql-context-db` to run the original BFS sampler with otherwise identical
flags. The experiment currently supports only `driver-top3`, and requires the
raw RelBench source corresponding to the preprocessed data, plus DuckDB (both
available in the Pixi environment).

Neighborhoods are materialized for all train/val/test seeds before workers
start. The SQL label-neighbor query uses earlier rows from all three splits,
matching the existing BFS sampler's split policy. The query requires strictly
earlier timestamps, so it excludes the seed's own label. Existing same-table
seed selection retains its original split and timestamp policies.
Rust also filters temporal nodes against the seed timestamp
and retains its existing target/leakage masking. Relationship depths shown in
batch visualization metadata describe the fixed joins, not a BFS traversal.

DuckDB `rowid` is retained explicitly through joins to identify raw rows; see
the [DuckDB order preservation documentation](https://duckdb.org/docs/lts/sql/dialect/order_preservation).
The loader checks raw table values/order and task counts/timestamps against the
source and preprocessed metadata. Use matching data revisions: preprocessing
does not store the task entity keys needed to prove full task-row identity.

Verified locally: SQL cutoff/order/mapping tests and native sampler
bypass/bounds/temporal tests pass. The cached real rel-f1 data produced 2,667
seed neighborhoods. On four identical test targets with 10,000 walks, the
label counts were:

| Context / local cells / width | Original BFS | Initial SQL | Revised SQL |
| --- | --- | --- | --- |
| 256 / 128 / 8 | 6–32 | 2–3 | 8–20 |
| 8192 / 256 / 32 | 789–931 | 95–232 | 457–546 |

The revised contexts include qualifying, standings and results. These are
sampling checks, not model scores. No matched *hybrid* SQL/BFS AUROC comparison
has been run; the raw-data CUDA comparison above uses a different seed policy.
The existing FlexAttention CPU compilation path is unsupported on this Mac, so
model evaluation requires the CUDA environment.
