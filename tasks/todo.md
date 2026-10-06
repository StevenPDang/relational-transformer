# SQL Stage 2 checklist

- [x] Capture deterministic Stage 1, per-seed collection, and final-cell traces.
- [x] Align the BFS baseline with Algorithm 1's walk counting, ranking, and time cutoff; verify the baseline independently.
- [x] Define the BFS/SQL collector interface and stable SQL row-to-node mapping.
- [x] Implement and test SQL collection for one selected seed on a synthetic graph.
- [x] Integrate collection across selected seeds without precomputing every task row.
- [ ] Verify multi-worker determinism, temporal safety, cell budgets, and masks.
- [ ] Compare context content, 726-row AUROC, and repeated end-to-end CUDA timing with Stage 1 held fixed.

## Progress and remaining acceptance gates

- `Sampler.trace_py` captures aligned/legacy baseline traces. Fourteen Rust tests
  cover the baseline and collector seam, including the pre-change golden trace.
- On-demand SQL collection is integrated behind graph-backed `--sql-context-db`.
  Native synthetic trace/batch parity tests pass, including multiple child
  tables, all task splits, width, masks, deduplication and temporal budgets.
  Thread/fresh-process provider determinism passes; full DataLoader/fork
  determinism is still pending.
- Reverse-neighbor timestamp ties now use node-index order for aligned BFS/SQL;
  legacy tracing preserves stored order. This additional alignment removes the
  preprocessing hash-map tie-order dependency and is documented in the plan.
- The per-node route is replaced by default with item-local graph planning,
  one SQL adjacency batch, and validated replay. No eager contexts or SQL seed
  orders are used. SQL joins run once per worker connection into a temporary
  indexed edge relation. Stage 1 currently runs twice with identical RNG;
  planning still uses graph BFS and is included in timing.
- Repeated tiny-fixture measurements: scalar/view 134.86 ms warm/item;
  scalar/index 11.18 ms; batch/index 1.41 ms; BFS 0.075 ms. Batch cold time
  20.5 ms includes 19.0 ms connection/index setup. These demonstrate a local
  improvement, not production performance acceptance. The user reported a
  passing original four-target rel-f1 smoke (37.600 s SQL, 0.105 s aligned BFS).
  Rebuild and rerun those exact inputs with `--sql-route batched`; keep the
  previous JSON and report setup/memory and repeated timings.
- `scripts/compare_sql_stage2.py` provides legacy/aligned/SQL trace/content/timing
  comparisons, defaulting to all 726 test rows. It now records index setup,
  requested-node counts and Rust execution stats; `--sql-route scalar` retains
  the per-node reference. Local real SQL comparison is blocked by the cached
  source's missing `qualifying-position` manifest, while the user's matching
  source passed its smoke. Do not bypass alignment checks. CUDA/AUROC remains
  pending on a CUDA machine.
- Latest validation: 14 focused Rust tests and 11 Python SQL/raw-eval/CLI
  tests passed across focused runs. Batch tests cover request-only maps, scalar parity, concurrent items,
  errors/deadlines, thread/fresh-process workers and live DuckDB coexistence.
  The previous expanded run had two `tests/test_eval_subsets.py` failures in
  leaderboard subset handling/single-class AUROC (outside this change).
