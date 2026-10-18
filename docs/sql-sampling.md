# SQL sampling with MetaSieve

SQL sampling replaces the Rust BFS for RelBench forecast tasks: DuckDB builds
every seed's neighborhood up front, following the relational paths that
MetaSieve (Khan & Aboulnaga, 2026) keeps for the task. Rust still supplies
embeddings, masks, deduplication, cell budgets and the model input format.
Nothing is specific to one database: each task's rule set carries the schema
and the task's entity table, key and time columns.

## 1. Import the database and select metapaths

```bash
pixi run import-duckdb --dataset stanford-star/relbench/rel-trial
pixi run --environment cuda124 python -m scripts.metasieve --source stanford-star/relbench/rel-trial
```

The import writes `data/duckdb/<db>.duckdb`. Selection runs for every
node-level classification and regression task of the dataset (`--task` picks
some), scoring every schema-valid metapath up to three hops from the task's
entity table using per-seed SQL counts and the **train** labels: mutual
information of log-count and log-rate with the label, divided by fanout cost,
plus coverage, summarized by a lower confidence bound over seed batches and
split into good and bad paths with a two-component Gaussian mixture. A path is
kept if it or any descendant is good. Rules apply to whole metapaths; unlike
the paper's PyG approximation, nothing is pruned by terminal table alone.
Large training tables are scored on at most `--max-seeds` rows (50,000 by
default, capped per label for classification).

Each task gets `runs/metasieve/<db>-<task>/<timestamp>.json` (scores, cluster
and keep decision for every path) and a `latest.json` pointing at it. On rel-f1
it keeps 28/34 paths for driver-top3, 26/34 for driver-dnf and 24/34 for
driver-position; the dropped paths fan out from a race or constructor to every
other driver's rows.

## 2. Evaluate

Both modes need `--sql-context-rules`: one task's JSON, or the `runs/metasieve`
folder, which resolves `<db>-<task>/latest.json` for each task evaluated.

**Prepared (hybrid):** hosted preprocessed data, SQL neighborhoods, and Rust
random walks still rank historical seeds. `--sql-context-db` is a DuckDB file
or a folder of `<db>.duckdb` files.

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed --tasks rel-trial/study-outcome \
  --sql-context-db data/duckdb --sql-context-rules runs/metasieve \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --num-walks 10000 --walk-length 20 --prefer-latest --shuffle-seed 0
```

Omit `--sql-context-db` and `--sql-context-rules` for the matching BFS run.

**Raw:** prepares cells, embeddings and a DuckDB import from raw Parquet inside
the timed run, with no traversal graph, then evaluates the dataset's tasks of
the checkpoint's kind (`--tasks` narrows them). Historical seeds are ordered
without random walks: strictly earlier task rows, same entity first, then most
recent, so `--num-walks` and `--walk-length` do not apply.

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/regression \
  --raw-dataset "$RAW" --sampler sql --sql-context-rules runs/metasieve \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --prefer-latest --shuffle-seed 0
```

Use `--sampler bfs` (without rules) for the raw BFS baseline. Resolve the raw
dataset once and pass the same local directory to every run, for example
`RAW=$(pixi run --environment cuda124 python -c "from relbench.hf import download_dataset_dir; print(download_dataset_dir('stanford-star/relbench/rel-trial'))" | tail -1)`.
Keep the checkpoint, embedding batch size, worker count and hardware identical
and alternate sampler order across repeats.

## What a neighborhood contains

For each seed, in order:

1. the entity row, then up to `--bfs-width` most recent strictly earlier labels
   of the same entity (never the seed's own);
2. **spine** rows: hop-1 rows of each kept path out of the entity table,
   round-robin by recency, each followed by its forward parents;
3. **fan-out** rows past a later reverse step (on rel-f1: constructor results
   and standings, other drivers' rows in the same race), which only use budget
   the spine leaves.

Reverse steps are capped at `--bfs-width` rows per seed and path; rows dated
after the seed are excluded (`<=` the seed time, like the Rust sampler).
`--local-ctx-size` and `--ctx-size` keep their cell-budget meaning.

## Scale

Neighborhoods are built for every train/val/test task row before DataLoader
workers start, because the sampler may pick any earlier row as a historical
seed. Frontier tables are built 20,000 seeds at a time, and each seed keeps at
most 1,024 candidate historical seeds (full 8,192-cell rel-f1 contexts use at
most 39 with SQL neighborhoods, 135 with BFS), so time and memory grow
linearly with task rows. On a local raw rel-f1 preparation, setup takes 3.7 s
for driver-top3 (2.7K task rows), 7.5 s for driver-position (8.7K) and 11.4 s
for driver-dnf (12.7K). The neighborhoods are held in Python until Rust copies
them, so tasks with millions of rows (rel-amazon) need more memory than that
path is designed for.

## Run folders

Without `--out-dir`, every eval run writes its submission CSVs and
`runtime.json` to a new folder named by start time and run type, grouped by the
selected tasks (the dataset name for raw runs); raw runs prepare into its
`prepare/` subfolder:

```text
runs/eval/rel-trial/
├── 20261018-183012_raw-bfs/
│   ├── prepare/
│   ├── rel-trial__study-outcome.csv
│   └── runtime.json
└── 20261018-184105_raw-sql-metasieve/
```

Run types are `raw-` or `prepared-` plus the sampler, with `-metasieve` for
rules and `-ensemble` in ensemble mode. `--out-dir` and `--prepare-dir` choose
paths yourself; `--runs-dir` moves the root.

## Timing

`runtime.json` records model loading, raw preparation, evaluator setup,
evaluation/scoring and total time. `preparation_seconds` breaks down raw input
resolution, cell encoding (plus graph construction for BFS), text embeddings
and the DuckDB import. Raw SQL runs also report `setup_seconds`: SQL
neighborhood queries and assembly, historical seed queries, Rust transfer and
validation of contexts and seed orders, and other evaluator setup. These
subdivide their parent entries; do not add them twice.

BFS does its walks and expansion per item during evaluation, while SQL does all
of its work during setup, so compare `total` (plus the selection step), not
setup alone.

## Checks and implementation

Locally, without a model or GPU, build and inspect a small batch:

```bash
pixi run --environment mac python -m scripts.sample_sql_context --db rel-f1 --task driver-dnf --items 4
```

It prints the target, selected nodes, cell count and context-label count for
each sampled test row, and checks target placement and temporal cutoffs
(`--duckdb` defaults to `data/duckdb`, `--rules` to `runs/metasieve`).

The prepared-data loader checks every table a kept path may reach for matching
row counts and key order, plus task counts and timestamps, against the source
and preprocessed metadata; DuckDB `rowid` identifies raw rows through joins (see
the [DuckDB order preservation documentation](https://duckdb.org/docs/lts/sql/dialect/order_preservation)).
Use matching data revisions: preprocessing does not store the task entity keys
needed to prove full task-row identity. Graph-free and graph-backed encoders
produce identical model tensors for identical SQL selections.

Code: `src/rt/metasieve.py` (selection, capped path rows, ordering),
`src/rt/sql_context.py` (task loaders, batched labels, seed orders),
`scripts/metasieve.py`; tests in `tests/test_metasieve.py`,
`tests/test_sql_context.py` and `tests/test_raw_eval.py`. Model evaluation
needs the CUDA environment; FlexAttention's CPU path is unsupported on macOS.
