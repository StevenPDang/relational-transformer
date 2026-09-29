# Small rel-f1 SQL sampling experiment

The first experiment replaces local BFS expansion for `rel-f1/driver-top3`.
Random walks and same-table seed selection remain in Rust. SQL supplies each
seed's neighborhood; Rust still supplies embeddings, masks, deduplication,
cell budgets, and the model input format.

Non-recursive queries select the driver's row, recent results at or before the
seed timestamp, their races, constructors and circuits, and earlier training
labels for that driver. This is a small task-specific neighborhood, not an
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

Run a small model evaluation on a CUDA machine with:

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed \
  --tasks rel-f1/driver-top3 --out-dir eval_sql_small \
  --sql-context-db data/duckdb/rel-f1.duckdb \
  --ctx-size 256 --local-ctx-size 128 --bfs-width 8 \
  --items-per-task 16 --tokens-per-gpu 1024 --num-workers 0
```

`--bfs-width` bounds recent results and historical labels per seed in SQL mode.
`--local-ctx-size` and `--ctx-size` retain their existing cell budget semantics.
Leave `--num-walks` at its default to preserve walk-based seed selection; set it
to `0` for a faster smoke run with random same-table seeds. Omit
`--sql-context-db` to run the original BFS sampler with otherwise identical
flags. The experiment currently supports only `driver-top3`, and requires the
raw RelBench source corresponding to the preprocessed data, plus DuckDB (both
available in the Pixi environment).

Neighborhoods are materialized for all train/val/test seeds before workers
start. The SQL label-neighbor query uses only the training split and strictly
earlier timestamps. Existing same-table seed selection still uses its original
split policy. Rust also filters temporal nodes against the seed timestamp
and retains its existing target/leakage masking. Relationship depths shown in
batch visualization metadata describe the fixed joins, not a BFS traversal.

DuckDB `rowid` is retained explicitly through joins to identify raw rows; see
the [DuckDB order preservation documentation](https://duckdb.org/docs/lts/sql/dialect/order_preservation).
The loader checks raw table values/order and task counts/timestamps against the
source and preprocessed metadata. Use matching data revisions: preprocessing
does not store the task entity keys needed to prove full task-row identity.

Verified locally: four focused tests pass, including SQL cutoff/order/mapping
and native sampler bypass/bounds/temporal behavior. The cached real rel-f1 data
produced 2,667 seed neighborhoods and full 256-cell batches. No AUROC comparison
has been run; the existing FlexAttention CPU compilation path is unsupported on
this Mac, so model evaluation requires the CUDA environment.
