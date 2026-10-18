# rel-f1 SQL sampling with MetaSieve

SQL sampling replaces the Rust BFS for `rel-f1/driver-top3`: DuckDB builds every
seed's neighborhood up front, following the relational paths that MetaSieve
(Khan & Aboulnaga, 2026) keeps for the task. Rust still supplies embeddings,
masks, deduplication, cell budgets and the model input format. SQL sampling
always needs a MetaSieve rule set; the hand-written fixed joins it started from
were removed once MetaSieve matched or beat them and built contexts ~40 s faster.

## 1. Select metapaths

```bash
pixi run --environment cuda124 python -m scripts.metasieve --duckdb data/duckdb/rel-f1.duckdb
```

This scores every schema-valid metapath up to three hops from the drivers
table using per-seed SQL counts and the **train** labels: mutual information of
log-count and log-rate with the label, divided by fanout cost, plus coverage,
summarized by a lower confidence bound over seed batches and split into good
and bad paths with a two-component Gaussian mixture. A path is kept if it or
any descendant is good. Rules apply to whole metapaths; unlike the paper's PyG
approximation, nothing is pruned by terminal table alone.

It writes `runs/metasieve/rel-f1-driver-top3/<timestamp>.json` (scores,
cluster and keep decision for every path) and points `latest.json` there;
`--out` picks a path instead. On rel-f1/driver-top3 it keeps 28 of 34 paths in
about 3 s; the six dropped paths all fan out from a race to every other
driver's qualifying, standings or results rows.

## 2. Evaluate

Two modes; both need `--sql-context-rules`.

**Prepared (hybrid):** hosted preprocessed data, SQL neighborhoods, and
Rust random walks still rank historical seeds.

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed --tasks rel-f1/driver-top3 \
  --sql-context-db data/duckdb/rel-f1.duckdb \
  --sql-context-rules runs/metasieve/rel-f1-driver-top3/latest.json \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --num-walks 10000 --walk-length 20 --prefer-latest --shuffle-seed 0
```

Omit `--sql-context-db` and `--sql-context-rules` for the matching BFS run.

**Raw:** prepares cells, embeddings and a DuckDB import from raw Parquet inside
the timed run, with no traversal graph. Historical seeds are ordered in SQL
(strictly earlier task rows, same driver first, then most recent) instead of
random walks, so `--num-walks` and `--walk-length` do not apply.

```bash
pixi run --environment cuda124 eval \
  --checkpoint checkpoints/rt-j/classification \
  --raw-dataset "$RAW" --sampler sql --tasks rel-f1/driver-top3 \
  --sql-context-rules runs/metasieve/rel-f1-driver-top3/latest.json \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --prefer-latest --shuffle-seed 0
```

Use `--sampler bfs` (without rules) for the raw BFS baseline. Resolve the raw
dataset once and pass the same local directory to every run, for example
`RAW=$(pixi run --environment cuda124 python -c "from relbench.hf import download_dataset_dir; print(download_dataset_dir('stanford-star/relbench/rel-f1'))" | tail -1)`.
Keep the checkpoint, embedding batch size, worker count and hardware identical,
alternate sampler order across repeats, and confirm every run reports `n=726`.

## What a neighborhood contains

For each seed, in order:

1. the driver row, then up to `--bfs-width` most recent strictly earlier labels
   (never the seed's own);
2. **spine** rows: hop-1 history (qualifying, standings, results) round-robin by
   recency, each followed by its forward parents (race, circuit, constructor);
3. **fan-out** rows past a later reverse step (constructor results and
   standings, other drivers' rows in the same race), which only use budget the
   spine leaves.

Reverse steps are capped at `--bfs-width` rows per seed and path; rows dated
after the seed are excluded (`<=` the seed time, like the Rust sampler).
`--local-ctx-size` and `--ctx-size` keep their cell-budget meaning.
Neighborhoods and seed orders for all train/val/test seeds are built in batched
queries over shared frontier tables before DataLoader workers start.

At large budgets the fan-out paths displace some historical seeds: on four test
targets at 8192/256/32, contexts held 102–267 labels against 293–398 for the
removed fixed joins, plus 411–710 constructor result and standing rows.

## Run folders

Without `--out-dir`, every eval run writes its submission CSVs and
`runtime.json` to a new folder named by start time and run type, grouped by
task; raw runs prepare into its `prepare/` subfolder:

```text
runs/eval/rel-f1-driver-top3/
├── 20261018-183012_raw-bfs/
│   ├── prepare/
│   ├── rel-f1__driver-top3.csv
│   └── runtime.json
├── 20261018-184105_raw-sql-metasieve/
└── 20261018-185230_prepared-sql-metasieve/
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
of its work during setup, so compare `total` (plus the ~3 s selection step), not
setup alone.

## Checks and implementation

Locally, without a model or GPU, build and inspect a small batch:

```bash
pixi run --environment mac python -m scripts.sample_sql_context --items 4
```

It prints the target, selected nodes, cell count and context-label count for
each sampled test row, and checks target placement and temporal cutoffs
(`--rules` defaults to `latest.json`).

The prepared-data loader checks every table a kept path may reach for matching
row values and order, plus task counts and timestamps, against the source and
preprocessed metadata; DuckDB `rowid` identifies raw rows through joins (see
the [DuckDB order preservation documentation](https://duckdb.org/docs/lts/sql/dialect/order_preservation)).
Use matching data revisions: preprocessing does not store the task entity keys
needed to prove full task-row identity. Graph-free and graph-backed encoders
produce identical model tensors for identical SQL selections.

Code: `src/rt/metasieve.py` (selection, capped path rows, ordering),
`src/rt/sql_context.py` (rel-f1 loaders, batched labels and seed orders),
`scripts/metasieve.py`; tests in `tests/test_metasieve.py`,
`tests/test_sql_context.py` and `tests/test_raw_eval.py`. Model evaluation
needs the CUDA environment; FlexAttention's CPU path is unsupported on macOS.
