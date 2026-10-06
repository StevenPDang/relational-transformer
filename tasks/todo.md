# SQL Stage 2 checklist

- [x] Capture deterministic Stage 1, per-seed collection, and final-cell traces.
- [x] Align the BFS baseline with Algorithm 1's walk counting, ranking, and time cutoff; verify the baseline independently.
- [x] Define the BFS/SQL collector interface and stable SQL row-to-node mapping.
- [x] Implement and test SQL collection for one selected seed on a synthetic graph.
- [ ] Integrate collection across selected seeds without precomputing every task row.
- [ ] Verify multi-worker determinism, temporal safety, cell budgets, and masks.
- [ ] Compare context content, 726-row AUROC, and repeated end-to-end CUDA timing with Stage 1 held fixed.

## Progress and remaining acceptance gates

- `Sampler.trace_py` captures aligned/legacy baseline traces. Ten Rust tests
  cover the baseline and collector seam, including the pre-change golden trace.
- On-demand SQL collection is integrated behind graph-backed `--sql-context-db`.
  Native synthetic trace/batch parity tests pass, including multiple child
  tables, all task splits, width, masks, deduplication and temporal budgets.
  Thread/fresh-process provider determinism passes; full DataLoader/fork
  determinism is still pending.
- Reverse-neighbor timestamp ties now use node-index order for aligned BFS/SQL;
  legacy tracing preserves stored order. This additional alignment removes the
  preprocessing hash-map tie-order dependency and is documented in the plan.
- Step 4 is **not accepted**: the per-expanded-node Python/DuckDB correctness
  prototype measured 116.4 ms warm/item versus 0.070 ms BFS on a tiny fixture.
  Replace it with a measured two-pass/batched or native worker-local route before
  claiming performance readiness. No eager contexts or SQL seed orders are used.
- `scripts/compare_sql_stage2.py` provides legacy/aligned/SQL trace/content/timing
  comparisons, defaulting to all 726 test rows. Real SQL comparison is blocked
  by the cached rel-f1 source's missing `qualifying-position` task manifest.
  Use a complete source revision matching the encoded graph; do not bypass
  alignment checks. CUDA/AUROC remains pending on a CUDA machine.
- Validation: 10 focused Rust tests; 8 Python SQL/raw-eval tests passed in the
  expanded run. Two existing `tests/test_eval_subsets.py` tests failed in
  leaderboard subset handling/single-class AUROC (outside this change).
