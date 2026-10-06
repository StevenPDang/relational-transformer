# SQL Stage 2 checklist

- [ ] Capture deterministic Stage 1, per-seed collection, and final-cell traces.
- [ ] Align the BFS baseline with Algorithm 1's walk counting, ranking, and time cutoff; verify the baseline independently.
- [ ] Define the BFS/SQL collector interface and stable SQL row-to-node mapping.
- [ ] Implement and test SQL collection for one selected seed on a synthetic graph.
- [ ] Integrate collection across selected seeds without precomputing every task row.
- [ ] Verify multi-worker determinism, temporal safety, cell budgets, and masks.
- [ ] Compare context content, 726-row AUROC, and repeated end-to-end CUDA timing with Stage 1 held fixed.
