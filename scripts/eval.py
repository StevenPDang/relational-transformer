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

``--raw-dataset`` evaluates a raw RelBench dataset's tasks (all of the
checkpoint's kind unless ``--tasks`` narrows them) and includes cell
preparation, text embeddings, and graph construction (``--sampler bfs``) or raw
SQL import (``--sampler sql``) in the reported runtime, using a fresh
``--prepare-dir``.

Without ``--out-dir``, each run gets its own timestamped folder grouped by task
and run type, e.g. ``runs/eval/rel-f1-driver-top3/20261018-183012_raw-sql-metasieve/``
(raw runs prepare into its ``prepare/`` subfolder).

Single-process (one GPU). Example:

    pixi run --environment cuda124 eval --checkpoint checkpoints/rt-j/classification \\
        --pre-dir stanford-star/relbench-preprocessed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rt.checkpoints import load_rt_model
from rt.recipes import get_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local path or Hub model repo")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pre-dir", help="preprocessed RelBench (local or Hub)")
    inputs.add_argument("--raw-dataset",
                        help="raw RelBench directory or Hub spec (e.g. stanford-star/relbench/rel-trial); "
                             "includes preparation in runtime")
    ap.add_argument("--prepare-dir",
                    help="fresh directory for raw-data preparation (must not exist); "
                         "default: prepare/ inside the run folder")
    ap.add_argument("--sampler", choices=["bfs", "sql"], default="bfs",
                    help="raw-data sampler: bfs builds a graph; sql builds MetaSieve neighborhoods "
                         "(needs --sql-context-rules) and ranks seeds by shared neighborhood rows instead of random walks")
    ap.add_argument("--embedding-batch-size", type=int, default=1024,
                    help="text-embedding batch size for raw-data preparation")
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
    ap.add_argument("--out-dir", default=None,
                    help="submission + runtime.json folder; default: a new timestamped folder "
                         "under --runs-dir/eval/<tasks>/")
    ap.add_argument("--runs-dir", default="runs", help="root for default run folders")
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--local-ctx-size", type=int, default=256)
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--sql-context-db", default=None,
                    help="opt-in SQL neighborhoods from this DuckDB file or a folder of <db>.duckdb "
                         "(needs --sql-context-rules); --bfs-width caps labels and reverse steps per seed")
    ap.add_argument("--sql-context-rules", default=None,
                    help="MetaSieve rules from scripts/metasieve.py: one task's JSON, or a folder "
                         "(e.g. runs/metasieve) holding <db>-<task>/latest.json per task")
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
    if args.raw_dataset is not None and not args.raw_dataset.strip():
        ap.error("--raw-dataset is empty; check that the path variable (e.g. $RAW) is set")
    if args.raw_dataset:
        if args.sql_context_db:
            ap.error("raw SQL builds its own DuckDB database; omit --sql-context-db")
        if args.mode != "simple" or args.recipe != "relbench_eval_test":
            ap.error("raw evaluation currently supports simple mode on the test split")
        from rt.raw_eval import raw_dataset_name

        raw_db = raw_dataset_name(args.raw_dataset)
        if args.tasks is None:
            args.tasks = [raw_db]
        elif any(sel.split("/")[0] != raw_db for sel in args.tasks):
            ap.error(f"--tasks must select tasks of the raw dataset {raw_db}")
        if args.sampler == "sql" and not args.prefer_latest:
            ap.error("raw SQL currently uses recency-ordered seeds; omit --no-prefer-latest")
    elif args.prepare_dir or args.sampler != "bfs":
        ap.error("--prepare-dir and --sampler sql require --raw-dataset; use --sql-context-db with prepared data")
    sql_sampling = bool(args.sql_context_db) or (bool(args.raw_dataset) and args.sampler == "sql")
    if args.sql_context_rules and not sql_sampling:
        ap.error("--sql-context-rules requires --sql-context-db or --raw-dataset with --sampler sql")
    if sql_sampling and not args.sql_context_rules:
        ap.error("SQL sampling needs MetaSieve rules: run `python -m scripts.metasieve --source <dataset>` "
                 "and pass --sql-context-rules runs/metasieve")
    if args.out_dir is None:
        from rt.run_dirs import eval_kind, eval_run_dir

        kind = eval_kind(raw=bool(args.raw_dataset), sampler=args.sampler,
                         sql_context_db=args.sql_context_db,
                         sql_context_rules=args.sql_context_rules, mode=args.mode)
        run_dir = eval_run_dir(args.runs_dir, args.tasks, kind)
        run_dir.mkdir(parents=True)  # claim the name before the long run starts
        args.out_dir = str(run_dir)
    if args.raw_dataset and not args.prepare_dir:
        args.prepare_dir = str(Path(args.out_dir).expanduser() / "prepare")
    print(f"run folder: {args.out_dir}", flush=True)

    started_at = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timings = {}
    preparation_timings = {}
    setup_timings = {}

    def report_runtime(label: str, since: float) -> float:
        # CUDA work is asynchronous; finish it before measuring elapsed time.
        if device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        timings[label] = now - since
        print(f"runtime | {label}: {timings[label]:.2f}s", flush=True)
        return now

    def finish():
        report_runtime("total", started_at)
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "runtime.json").write_text(json.dumps({
            "sampler": (args.sampler if args.raw_dataset else ("sql-hybrid" if args.sql_context_db else "bfs"))
                       + ("+metasieve" if args.sql_context_rules else ""),
            "device": device, "seconds": timings,
            "preparation_seconds": preparation_timings,
            "setup_seconds": setup_timings, "settings": vars(args),
        }, indent=2) + "\n")

    net, config = load_rt_model(args.checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)
    task_type = config.get("task_type")
    print(f"loaded {config.get('name', args.checkpoint)} "
          f"(task_type={task_type}, embed={config['embedding_model']}) on {device}")
    setup_started_at = report_runtime("model loading", started_at)

    if args.raw_dataset:
        from rt.raw_eval import prepare_raw_eval

        args.pre_dir, args.sql_context_db, preparation_timings = prepare_raw_eval(
            args.raw_dataset, args.prepare_dir, sampler=args.sampler,
            embedding_model=config["embedding_model"], d_text=config["d_text"], device=device,
            batch_size=args.embedding_batch_size,
        )
        for label, seconds in preparation_timings.items():
            print(f"runtime | preparation / {label}: {seconds:.2f}s", flush=True)
        setup_started_at = report_runtime("raw data preparation", setup_started_at)
        if args.sampler == "sql":
            print("raw SQL uses SQL historical seeds; --num-walks and --walk-length do not apply", flush=True)

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
        sql_context_rules=args.sql_context_rules,
    )

    if args.mode == "ensemble":
        from rt.eval_utils import run_ensemble

        grid = [tuple(int(x) for x in g.split(",")) for g in args.grid]
        val_tasks = selected(of_kind(get_tasks("relbench_eval_val", args.pre_dir)))
        test_tasks = selected(of_kind(get_tasks("relbench_eval_test", args.pre_dir)))
        if not test_tasks:
            raise SystemExit(f"no {task_type} tasks found in {args.pre_dir}")
        evaluation_started_at = report_runtime("task setup", setup_started_at)
        run_ensemble(net, args.pre_dir, val_tasks, test_tasks, grid=grid,
                     ensemble_size=args.ensemble_size, ctx_size=args.ctx_size,
                     reg_metric=args.reg_metric, out_dir=args.out_dir, no_csv=args.no_csv,
                     **eval_kwargs)
        # Ensemble evaluator construction happens inside run_ensemble.
        report_runtime("ensemble setup, tuning, evaluation and scoring", evaluation_started_at)
        finish()
        return

    from rt.eval_utils import build_evaluator, run_and_report

    if args.raw_dataset:
        from rt.tasks import tasks_from_preprocessed

        # The prepared folder holds one dataset: evaluate its own forecast tasks,
        # not the fixed benchmark list.
        tasks = selected(of_kind(tasks_from_preprocessed(args.pre_dir, splits=("test",))))
    else:
        tasks = selected(of_kind(get_tasks(args.recipe, args.pre_dir)))
    if not tasks:
        raise SystemExit(f"no {task_type} tasks found in {args.pre_dir}")
    ev = build_evaluator(tasks, args.pre_dir, ctx_size=args.ctx_size,
                         local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
                         setup_timings=setup_timings, **eval_kwargs)
    evaluation_started_at = report_runtime("task and evaluator setup", setup_started_at)
    if setup_timings:
        setup_timings["other evaluator setup"] = max(
            0.0, timings["task and evaluator setup"] - sum(setup_timings.values())
        )
        for label, seconds in setup_timings.items():
            print(f"runtime | setup / {label}: {seconds:.2f}s", flush=True)
    run_and_report(net, tasks, args.pre_dir, ctx_size=args.ctx_size,
                   reg_metric=args.reg_metric, out_dir=args.out_dir, no_csv=args.no_csv,
                   evaluator=ev, embedding_model=config["embedding_model"])
    report_runtime("evaluation and scoring", evaluation_started_at)
    finish()


if __name__ == "__main__":
    main()
