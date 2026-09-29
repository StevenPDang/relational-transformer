#!/usr/bin/env python
"""Evaluate an RT checkpoint on the RelBench benchmark tasks.

Loads a checkpoint -- a local dir/file or a Hub model repo -- and evaluates every task of its kind (clf/reg) in
the preprocessed RelBench data (``--pre-dir``, local or Hub), and reports metrics
through RelBench's own leaderboard evaluator: regression predictions are
denormalized to the original target scale, classification logits are sigmoided
to probabilities, and each prediction is keyed back to its relbench
``(entity_col, time_col)``. ``--out-dir`` becomes a valid RelBench *submission
directory* -- one ``<dataset>__<task>.csv`` prediction table per task -- that is
scored with ``relbench.leaderboard.evaluate_task`` (AUROC for clf, NMAE for reg)
and can be re-validated with ``python -m relbench.leaderboard <out-dir>``.

Single-process (one GPU). Example:

    pixi run --environment cuda124 eval --checkpoint checkpoints/rt-j/classification \\
        --pre-dir stanford-star/relbench-preprocessed --out-dir eval_out
"""

from __future__ import annotations

import argparse

import torch

from rt.checkpoints import load_rt_model
from rt.recipes import get_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local path or Hub model repo")
    ap.add_argument("--pre-dir", required=True, help="preprocessed RelBench (local or Hub)")
    ap.add_argument("--mode", default="simple", choices=["simple", "ensemble"],
                    help="simple: one context config on the test split; "
                         "ensemble: tune context config per task on val, then average "
                         "predictions over --ensemble-size context seeds on test")
    ap.add_argument("--recipe", default="relbench_eval_test",
                    help="(simple mode) relbench_eval_test | relbench_eval_val | relbench_eval")
    ap.add_argument("--tasks", nargs="+", default=None, metavar="SEL",
                    help="restrict to these tasks; each SEL is a 'db' (all its tasks) "
                         "or 'db/task-table' (one task), e.g. rel-f1/driver-top3")
    ap.add_argument("--grid", nargs="+", default=["256,32", "512,64"],
                    help="(ensemble mode) candidate 'local_ctx_size,bfs_width' configs")
    ap.add_argument("--ensemble-size", type=int, default=4,
                    help="(ensemble mode) number of context seeds to average on test")
    ap.add_argument("--out-dir", default="eval_out")
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--local-ctx-size", type=int, default=256)
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--sql-context-db", default=None,
                    help="opt-in rel-f1/driver-top3 SQL neighborhoods from this DuckDB file; "
                         "--bfs-width bounds each DB history and historical label rows per seed")
    ap.add_argument("--num-walks", type=int, default=10_000)
    ap.add_argument("--walk-length", type=int, default=20)
    ap.add_argument("--prefer-latest", action=argparse.BooleanOptionalAction, default=True,
                    help="rank same-table neighbors by recency (latest first) vs by frequency")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="seed for val/test subset selection + item shuffle; fix it to keep an "
                         "--items-per-task subsample the same rows across configs")
    ap.add_argument("--tokens-per-gpu", type=int, default=2**18)  # 2**19 overflows RT-J eval kernel @ctx=8192
    ap.add_argument("--items-per-task", type=int, default=10_000_000,
                    help="cap on items per task (default covers the full split)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--reg-metric", default="mae", choices=["mae", "r2"])
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, config = load_rt_model(args.checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)
    task_type = config.get("task_type")
    print(f"loaded {config.get('name', args.checkpoint)} "
          f"(task_type={task_type}, embed={config['embedding_model']}) on {device}")

    def of_kind(tasks):
        return [t for t in tasks if t.task_type == task_type] if task_type in ("clf", "reg") else tasks

    sel = set(args.tasks) if args.tasks else None

    def selected(tasks):
        if sel is None:
            return tasks
        return [t for t in tasks if t.db_name in sel or f"{t.db_name}/{t.table_name}" in sel]

    eval_kwargs = dict(
        embedding_model=config["embedding_model"], d_text=config["d_text"], device=device,
        num_walks=args.num_walks, walk_length=args.walk_length,
        tokens_per_gpu=args.tokens_per_gpu, items_per_task=args.items_per_task,
        num_workers=args.num_workers,
        prefer_latest=args.prefer_latest, shuffle_seed=args.shuffle_seed,
        sql_context_db=args.sql_context_db,
    )

    if args.mode == "ensemble":
        from rt.eval_utils import run_ensemble

        grid = [tuple(int(x) for x in g.split(",")) for g in args.grid]
        val_tasks = selected(of_kind(get_tasks("relbench_eval_val", args.pre_dir)))
        test_tasks = selected(of_kind(get_tasks("relbench_eval_test", args.pre_dir)))
        if not test_tasks:
            raise SystemExit(f"no {task_type} tasks found in {args.pre_dir}")
        run_ensemble(net, args.pre_dir, val_tasks, test_tasks, grid=grid,
                     ensemble_size=args.ensemble_size, ctx_size=args.ctx_size,
                     reg_metric=args.reg_metric, out_dir=args.out_dir, no_csv=args.no_csv,
                     **eval_kwargs)
        return

    from rt.eval_utils import build_evaluator, run_and_report

    tasks = selected(of_kind(get_tasks(args.recipe, args.pre_dir)))
    if not tasks:
        raise SystemExit(f"no {task_type} tasks found in {args.pre_dir}")
    ev = build_evaluator(tasks, args.pre_dir, ctx_size=args.ctx_size,
                         local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
                         **eval_kwargs)
    run_and_report(net, tasks, args.pre_dir, ctx_size=args.ctx_size,
                   reg_metric=args.reg_metric, out_dir=args.out_dir, no_csv=args.no_csv,
                   evaluator=ev, embedding_model=config["embedding_model"])


if __name__ == "__main__":
    main()
