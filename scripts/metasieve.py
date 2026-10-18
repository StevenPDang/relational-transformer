"""Select metapaths with MetaSieve from training labels and write the rule sets.

One rule set per task: every enumerated metapath with its scores, GMM cluster
and keep/drop decision, plus the task's entity, key and time columns. Pass a
rule set (or the folder holding them) to ``eval --sql-context-rules`` to build
SQL neighborhoods from the kept paths.

Without ``--out`` each is written to ``runs/metasieve/<db>-<task>/<timestamp>.json``
and ``latest.json`` in that folder is pointed at it. With no ``--task``, every
node-level classification and regression task of the dataset is selected.

    pixi run --environment mac python -m scripts.metasieve \\
      --source stanford-star/relbench/rel-f1 --task driver-top3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

NODE_TASK_TYPES = ("binary_classification", "regression")


def node_tasks(source):
    """Names of the dataset's node-level clf/reg tasks, from its task manifests."""
    import yaml
    from relbench.hf import download_dataset_dir

    root = Path(source).expanduser()
    if not (root / "manifest.yaml").is_file():
        root = Path(download_dataset_dir(source))
    names = []
    for manifest in sorted((root / "tasks").glob("*/manifest.yaml")):
        if yaml.safe_load(manifest.read_text()).get("task_type") in NODE_TASK_TYPES:
            names.append(manifest.parent.name)
    return names


def cap_seeds(train, target_col, task_type, max_seeds, seed=0):
    """At most ``max_seeds`` training rows; classification keeps every label's share.

    Capping per label keeps rare labels from being swamped by the majority
    when scoring (paper section 7.1). Row order is preserved.
    """
    if max_seeds is None or len(train) <= max_seeds:
        return train
    if task_type == "reg":
        return train.sample(n=max_seeds, random_state=seed).sort_index()
    groups = train.groupby(target_col)
    per_label = max(1, max_seeds // groups.ngroups)
    keep = [i for _, g in groups for i in g.sample(n=min(len(g), per_label), random_state=seed).index]
    return train.loc[sorted(keep)]


def select_one(args, source, db_name, db, schema, task_name):
    import duckdb
    import pandas as pd
    import relbench

    from rt import metasieve
    from rt.run_dirs import link_latest, rules_path
    from rt.sql_context import resolve_duckdb

    tick = time.perf_counter()
    task = relbench.load_task(source, task_name)
    info = metasieve.relbench_task_info(schema, db_name, task_name, task)
    train = task.get_table("train", mask_input_cols=False).df.reset_index(drop=True)
    scored = cap_seeds(train, task.target_col, info["task_type"], args.max_seeds)
    seeds = pd.DataFrame({
        "seed": scored.index.to_numpy(),
        "key": scored[task.entity_col].to_numpy(),
        "ts": pd.to_datetime(scored[task.time_col]).to_numpy(),
    })
    con = duckdb.connect(resolve_duckdb(args.duckdb, db_name), read_only=True)
    try:
        rules = metasieve.select_metapaths(
            con, schema, info, seeds, scored[task.target_col].to_numpy(), hops=args.hops,
            num_batches=args.batches, delta=args.delta, strict=args.strict,
        )
    finally:
        con.close()
    rules.update(source=source, train_rows=len(train), seconds=round(time.perf_counter() - tick, 2))
    out = Path(args.out).expanduser() if args.out else rules_path(args.runs_dir, source, task_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, indent=2, default=str) + "\n")
    if not args.out:
        print(f"latest: {link_latest(out)}")

    kept = [row for row in rules["paths"] if row["keep"]]
    print(f"{db_name}/{task_name}: {len(kept)}/{len(rules['paths'])} metapaths kept, "
          f"{len(seeds)}/{len(train)} train rows scored, {rules['seconds']}s -> {out}")
    for row in rules["paths"]:
        print(f"  {'keep' if row['keep'] else 'drop'}  {row['cluster']:<5}  {row['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="stanford-star/relbench/rel-f1",
                        help="RelBench dataset (Hub spec or local path)")
    parser.add_argument("--task", nargs="+", default=None,
                        help="task name(s); default: every node-level clf/reg task of the dataset")
    parser.add_argument("--duckdb", default="data/duckdb",
                        help="DuckDB file, or a folder holding <db>.duckdb (pixi run import-duckdb)")
    parser.add_argument("--out", default=None,
                        help="output rule-set JSON (single task only); default: a timestamped file "
                             "under --runs-dir/metasieve/<db>-<task>/ plus latest.json")
    parser.add_argument("--runs-dir", default="runs", help="root for the default output")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.3,
                        help="LCB significance level in (0, 0.5); smaller is more conservative")
    parser.add_argument("--max-seeds", type=int, default=50_000,
                        help="score on at most this many training rows (label-aware for clf); 0 = all")
    parser.add_argument("--strict", action="store_true",
                        help="score with timestamps < seed time (default <=, like the samplers)")
    args = parser.parse_args()
    args.max_seeds = args.max_seeds or None

    import relbench

    from rt import metasieve
    from rt.sql_context import resolve_duckdb

    db_name = Path(args.source.rstrip("/")).name
    tasks = args.task or node_tasks(args.source)
    if not tasks:
        parser.error(f"{args.source} has no node-level classification or regression tasks")
    if args.out and len(tasks) > 1:
        parser.error("--out names one file; omit it to select several tasks")
    if not Path(resolve_duckdb(args.duckdb, db_name)).is_file():
        parser.error(f"no DuckDB database at {resolve_duckdb(args.duckdb, db_name)}; run "
                     f"`pixi run import-duckdb --dataset {args.source}` first")
    db = relbench.load_dataset(args.source).get_db(upto_test_timestamp=False)
    schema = metasieve.relbench_schema(db)
    for task_name in tasks:
        select_one(args, args.source, db_name, db, schema, task_name)


if __name__ == "__main__":
    main()
