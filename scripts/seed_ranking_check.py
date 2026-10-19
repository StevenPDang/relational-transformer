"""Do raw SQL seed orders retrieve useful in-context examples? (no model needed)

RT predicts a row from labeled rows placed in its context, so the historical
seeds ranked first should resemble it. For each validation and test row this
predicts the label as the mean label of its top-k candidate seeds and scores
that prediction (AUROC for classification, Spearman correlation for
regression), comparing recency ranking (same entity, then most recent) with
path ranking (rows shared through kept MetaSieve paths). A ranking near chance
here gives the model little to learn from in context.

    pixi run --environment cuda124 python -m scripts.seed_ranking_check \\
      --run runs/eval/rel-trial-study-outcome/<timestamp>_raw-sql-metasieve
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def knn_scores(orders, query_seeds, label_of, k):
    """Mean label of each query's first k labeled candidates (NaN if none)."""
    import numpy as np

    preds = []
    for seed in query_seeds:
        cands = [c for c in orders[seed] if c in label_of][:k]
        preds.append(np.mean([label_of[c] for c in cands]) if cands else np.nan)
    return np.array(preds, dtype="float64")


def score(task_type, y, pred):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    ok = ~np.isnan(pred)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan"), int(ok.sum())
    if task_type == "clf":
        return float(roc_auc_score(y[ok], pred[ok])), int(ok.sum())
    return float(spearmanr(y[ok], pred[ok]).correlation), int(ok.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="raw SQL eval run folder (reads runtime.json and prepare/)")
    parser.add_argument("--prepare-dir", help="raw preparation folder, if not --run/prepare")
    parser.add_argument("--rules", default=None, help="MetaSieve rules (default: the run's --sql-context-rules)")
    parser.add_argument("--tasks", nargs="+", default=None, help="task names (default: the run's tasks)")
    parser.add_argument("--width", type=int, default=None, help="default: the run's --bfs-width, else 32")
    parser.add_argument("--k", type=int, default=32, help="candidates averaged per query (contexts use ~30-40)")
    args = parser.parse_args()

    import duckdb
    import numpy as np
    import pandas as pd

    from rt.sql_context import (_seed_frame, _task_rows, historical_seed_orders, load_metasieve_rules,
                                metasieve_context_nodes, path_seed_orders, resolve_duckdb)

    settings = json.loads((Path(args.run).expanduser() / "runtime.json").read_text())["settings"] \
        if args.run else {}
    prep = Path(args.prepare_dir or settings.get("prepare_dir") or Path(args.run) / "prepare").expanduser()
    rules_spec = args.rules or settings.get("sql_context_rules")
    width = args.width or settings.get("bfs_width") or 32
    if not rules_spec:
        parser.error("pass --rules (the run did not record --sql-context-rules)")
    db_dir = next(d for d in prep.iterdir() if (d / "table_info.json").is_file())
    db_name = db_dir.name
    info = json.loads((db_dir / "table_info.json").read_text())
    meta = json.loads((db_dir / "meta.json").read_text())
    raw_dir = Path(meta["raw_dataset_dir"])
    tasks = args.tasks or [s.split("/", 1)[1] for s in settings.get("tasks") or [] if "/" in s]
    if not tasks:
        tasks = [t["name"] for t in meta.get("tasks", [])
                 if t.get("task_type") in ("binary_classification", "regression")]

    for task_name in tasks:
        rules = load_metasieve_rules(rules_spec, db_name, task_name)
        task = rules["task"]
        frames, splits = {}, []
        for split in ("train", "val", "test"):
            df = pd.read_parquet(raw_dir / "tasks" / task_name / f"{split}.parquet")
            rows = _task_rows(df, task, info, split)
            if task["target_col"] in df:
                rows["y"] = df[task["target_col"]].to_numpy()
            frames[split] = rows
            splits.append(rows[["key", "ts", "node_idx"]])
        seeds = _seed_frame(splits)
        con = duckdb.connect(resolve_duckdb(str(prep), db_name), read_only=True)
        try:
            contexts = metasieve_context_nodes(con, info, rules, seeds, width).with_seeds()
        finally:
            con.close()
        tick = time.perf_counter()
        recency = historical_seed_orders(seeds)
        t_recency = time.perf_counter() - tick
        tick = time.perf_counter()
        paths = path_seed_orders(contexts, seeds, info)
        t_paths = time.perf_counter() - tick
        label_of = {int(n): float(y) for f in frames.values() if "y" in f
                    for n, y in zip(f["node_idx"], f["y"]) if pd.notna(y)}
        metric = "AUROC" if task["task_type"] == "clf" else "Spearman"
        print(f"{db_name}/{task_name}: {len(seeds):,} task rows; top-{args.k} candidate-label {metric} "
              f"(ranking time: recency {t_recency:.1f}s, paths {t_paths:.1f}s)")
        for split in ("val", "test"):
            f = frames[split]
            if "y" not in f:
                print(f"  {split:4}: no labels in the raw task table")
                continue
            y = f["y"].to_numpy(dtype="float64")
            q = f["node_idx"].to_numpy()
            r_s, r_n = score(task["task_type"], y, knn_scores(recency, q, label_of, args.k))
            p_s, p_n = score(task["task_type"], y, knn_scores(paths, q, label_of, args.k))
            print(f"  {split:4}: recency {r_s:.3f} (n={r_n:,})   paths {p_s:.3f} (n={p_n:,})")


if __name__ == "__main__":
    main()
