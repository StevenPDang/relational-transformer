"""Select metapaths with MetaSieve from training labels and write the rule set.

The JSON lists every enumerated metapath with its scores, GMM cluster and
keep/drop decision. Pass it to ``eval --sql-context-rules`` to build SQL
neighborhoods from the kept paths instead of the hand-written rel-f1 joins.

Without ``--out`` it is written to ``runs/metasieve/<db>-<task>/<timestamp>.json``
and ``latest.json`` in that folder is pointed at it.

    pixi run --environment mac python -m scripts.metasieve --duckdb data/duckdb/rel-f1.duckdb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="stanford-star/relbench/rel-f1",
                        help="RelBench dataset (Hub spec or local path)")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--duckdb", default="data/duckdb/rel-f1.duckdb")
    parser.add_argument("--out", default=None,
                        help="output rule-set JSON; default: a timestamped file under "
                             "--runs-dir/metasieve/<db>-<task>/ plus latest.json")
    parser.add_argument("--runs-dir", default="runs", help="root for the default output")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.3,
                        help="LCB significance level in (0, 0.5); smaller is more conservative")
    parser.add_argument("--strict", action="store_true",
                        help="score with timestamps < seed time (default <=, like the samplers)")
    args = parser.parse_args()

    import duckdb
    import pandas as pd
    import relbench

    from rt import metasieve
    from rt.run_dirs import link_latest, rules_path

    tick = time.perf_counter()
    db = relbench.load_dataset(args.source).get_db(upto_test_timestamp=False)
    task = relbench.load_task(args.source, args.task)
    train = task.get_table("train", mask_input_cols=False).df.reset_index(drop=True)
    task_type = "reg" if "regression" in str(task.task_type).lower() else "clf"
    seeds = pd.DataFrame({
        "seed": range(len(train)),
        "key": train[task.entity_col],
        "ts": pd.to_datetime(train[task.time_col]),
    })
    con = duckdb.connect(str(Path(args.duckdb).expanduser()), read_only=True)
    try:
        rules = metasieve.select_metapaths(
            con, metasieve.relbench_schema(db), task.entity_table, seeds,
            train[task.target_col].to_numpy(), task_type, hops=args.hops,
            num_batches=args.batches, delta=args.delta, strict=args.strict,
        )
    finally:
        con.close()
    rules.update(source=args.source, task=args.task, task_type=task_type,
                 seconds=round(time.perf_counter() - tick, 2))
    out = Path(args.out).expanduser() if args.out else rules_path(args.runs_dir, args.source, args.task)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rules, indent=2, default=str) + "\n")
    if not args.out:
        print(f"latest: {link_latest(out)}")

    kept = [row for row in rules["paths"] if row["keep"]]
    print(f"{len(kept)}/{len(rules['paths'])} metapaths kept in {rules['seconds']}s -> {out}")
    for row in rules["paths"]:
        print(f"  {'keep' if row['keep'] else 'drop'}  {row['cluster']:<5}  {row['name']}")


if __name__ == "__main__":
    main()
