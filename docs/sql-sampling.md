# Small rel-f1 SQL sampling experiment

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

## Existing hybrid SQL path using prepared data

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

Acceptance: an opt-in evaluation builds valid model batches using these SQL
nodes, keeps the target first, respects temporal cutoffs and cell budgets, and
fails on misaligned data. The default sampler remains unchanged.

Implementation: `src/rt/sql_context.py` materializes neighborhoods once before
DataLoader workers start; `rustler/src/fly.rs` consumes them instead of BFS.
Tests live in `tests/test_sql_context.py`. Use `pixi run build-sampler` to build
the extension and `pixi run python -m pytest tests/test_sql_context.py` to test.
No persistent database tables or preprocessing artifacts are modified.

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
sampling checks, not model scores. No new AUROC comparison has been run; the
existing FlexAttention CPU compilation path is unsupported on this Mac, so
model evaluation requires the CUDA environment.
