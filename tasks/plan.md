# Plan: SQL replacement for Algorithm 1 Stage 2

## Goal and scope

Keep Stage 1 random-walk scoring and candidate selection in Rust. Replace the
per-seed `BFSCOLLECT` expansion with SQL-backed neighborhood collection while
retaining the cell sequence builder, budgets, deduplication, target masking,
and model-input encoding. Use the user's Algorithm 1 as the final behavioral
contract. The graph-free raw SQL experiment, which also replaces Stage 1, is
separate evidence and is not the baseline for this work.

## Where the code stands

| Algorithm 1 piece | Current implementation | Gap |
| --- | --- | --- |
| Walk visits (Stage 1) | `Sampler::compute_visit_counts` in `rustler/src/fly.rs` | Preserved by the hybrid SQL path. The loop counts before moving, unlike the pseudocode's count-after-move order. |
| Candidate ranking | `seq_build` sorts visited same-table rows | With `prefer_latest`, timestamp precedes score; the pseudocode makes it a tie-break. Rust also has target-first expansion and a zero-visit fallback omitted from the simplified algorithm. |
| Per-seed collection | `extend_with_seed_bfs` calls `bfs_collect_nodes` | There is already an opt-in SQL branch, but it reads precomputed row lists instead of querying for the selected `(target, seed)` at collection time. |
| SQL neighborhoods | `load_rel_f1_contexts` and `query_context_nodes` in `src/rt/sql_context.py` | Fixed `rel-f1/driver-top3` joins and recency ordering approximate BFS; they do not reproduce frontier order, random width sampling, or arbitrary schema traversal. |
| Time and budgets | Rust applies `local_ctx_size`, `ctx_len`, deduplication, and masking after SQL rows arrive | Both existing collectors use seed time for local expansion; Algorithm 1 passes target time `t⋆`. SQL currently materializes at one width for all seeds. |

The hybrid path is the starting point because it keeps Rust walk scoring. The
raw-data `--sampler sql` path supplies its own historical seed order and is not
used for Stage 2 comparisons. Keeping walks also keeps the graph adjacency
needed by Stage 1 unless that stage is redesigned separately.

## Implementation sequence

### 1. Freeze the behavioral contract and baseline

Record a trace for fixed target rows and RNG seeds: walk visit counts, ordered
candidate seeds (including target-first and fallback behavior), every
`BFSCOLLECT` call's parameters, returned `(node, depth)` rows, and final cells.
Reconcile three differences from Algorithm 1 in an isolated baseline change:
count-after-move, score-first tie breaking, and target-time versus seed-time
cutoff. Keep a pre-change trace so any effect on BFS accuracy is attributable
to this alignment, not to SQL.

**Acceptance:** the chosen Algorithm 1 policy is stated in tests; the BFS path
passes them; fixed seeds reproduce the same Stage 1 trace across repeated runs.

### 2. Define the Stage 2 seam and SQL data contract

Specify one collector interface for BFS and SQL: target node/time, seed node,
`ℓ`, `b`, remaining `L`, RNG state or seed, and shared visited-depth state.
The result is ordered `(node_idx, depth)` rows that the existing cell appender
consumes. Define a SQL edge/row mapping using stable encoded node indices,
PK/FK directions, table types, and timestamps. Build it from the same source
revision as the encoded cells; verify row identity before sampling.

**Acceptance:** swapping collectors leaves Stage 1 scores and seed order
unchanged; SQL results cannot refer to an unknown node or a future row.

### 3. Prove SQL collection for one selected seed

Use a small deterministic graph with forward keys, reverse keys, equal and
future timestamps, duplicate paths, and a width-limited frontier. Implement
SQL neighbor retrieval for one actual Stage 1 seed. Keep frontier ordering,
width sampling, cell-budget stopping, and visited-depth behavior explicit. If
the SQL query intentionally changes BFS's stochastic row order, record that as
a sampling-policy difference rather than claiming exact equivalence.

**Acceptance:** target placement, cutoff, width, `ℓ`, `L`, deduplication, and
masking pass focused tests; the Stage 1 trace is identical for both collectors.

### 4. Integrate all seeds without eager database-wide materialization

Call SQL only for seeds selected for an evaluation item until its `L` budget is
full. Decide the Rust-to-DuckDB execution route with a small performance spike:
either a worker-local SQL connection in Rust or a two-pass interface that first
exports Stage 1 seed plans and then supplies SQL results to Rust. Avoid a Python
callback for every frontier edge unless measured overhead is acceptable.

**Acceptance:** no full train/val/test context map or SQL historical seed-order
map is required; multi-worker evaluation is deterministic for fixed seeds and
connections are worker-safe.

### 5. Compare on equal inputs and measure the new path

Run BFS and Stage-2-only SQL on the same prepared graph-backed data, checkpoint,
test rows, `W`, `K`, `ρ`, `L`, `ℓ`, `b`, and RNG seed. Report per-item collection
time, setup time, full runtime, context overlap/label counts, and RelBench
AUROC for all 726 `rel-f1/driver-top3` test rows. Repeat timing runs on the
same CUDA machine. Keep the raw SQL experiment's numbers labeled separately.

**Acceptance:** Stage 1 traces match; every output has at most `L` cells and
passes temporal/leakage checks; both runs score all 726 rows; performance and
quality differences are attributable to Stage 2.

## Risks and decisions

- A single fixed-join query is task-specific and cannot faithfully reproduce
  random graph traversal. Favor a correct SQL-backed collector before deciding
  whether approximate fixed joins are an acceptable alternate policy.
- Stage 1 currently needs graph adjacency. SQL Stage 2 alone does not remove
  graph preparation; measure it as a separate goal if required.
- The exact cutoffs and ranking in the supplied pseudocode differ from current
  Rust. Align the BFS baseline first, then hold it fixed for the SQL comparison.
- On-demand SQL calls may move cost from setup to evaluation. Compare total
  runtime, not only evaluator setup.
