"""Summarize the contexts in a `predict.py --trace` file.

Checks the time rule (rows stamped after the seed they were reached from) and
shows what the example seeds are: how many, how old, and, with --duckdb, how
many share the target's entity.

    python -m examples.sql.trace_stats sql_out/driver-dnf.jsonl \\
        --duckdb data/relbench-duckdb/rel-f1.duckdb --task driver-dnf
"""

import argparse
import json
from collections import Counter

import numpy as np

DAY = 86400


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("trace")
    p.add_argument("--duckdb", help="look up each seed's entity")
    p.add_argument("--task")
    args = p.parse_args()

    traces = [json.loads(line) for line in open(args.trace)]
    entity_of = None
    if args.duckdb:
        import duckdb

        import rt.sql as rs

        con = duckdb.connect(args.duckdb, read_only=True)
        db = rs.load(con)
        enc = rs.Encoder(con, db)
        task = db.tasks[args.task]
        tables = {t.key: t for t in task.splits.values()}
        col = enc.tables[task.splits["test"].key].fk_columns[0]
        entity_of = {}
        for key, t in tables.items():
            for r, e in con.execute(f'SELECT __row, "{col}" FROM {t.sql}').fetchall():
                entity_of[(key, r)] = e

    late, seeds, same, ages, depth, via = [], [], [], [], Counter(), Counter()
    for tr in traces:
        rows = tr["rows"]
        seed_ts = {
            r["seed"]: r["timestamp"] for r in rows if r["via"] in ("target", "example")
        }
        target = rows[0]
        examples = [r for r in rows if r["via"] == "example"]
        late.append(
            sum(
                r["timestamp"] is not None
                and seed_ts.get(r["seed"]) is not None
                and r["timestamp"] > seed_ts[r["seed"]]
                for r in rows
            )
        )
        seeds.append(len(examples))
        if target["timestamp"] is not None:
            ages += [
                (target["timestamp"] - r["timestamp"]) / DAY / 365.25
                for r in examples
                if r["timestamp"] is not None
            ]
        if entity_of is not None and examples:
            te = entity_of[(target["table"], target["row"])]
            same.append(
                np.mean([entity_of[(r["table"], r["row"])] == te for r in examples])
            )
        for r in rows:
            depth[r["depth"]] += r["cells"]
            via[r["via"].split(" of ")[0]] += r["cells"]

    n = len(traces)
    print(f"{n} contexts")
    print(
        f"rows stamped after their seed: {sum(late)} total, in {sum(x > 0 for x in late)} contexts"
        + ("  <- the time-bound fix was not active" if sum(late) else "")
    )
    print(
        f"example seeds per context: median {np.median(seeds):.0f}, "
        f"min {min(seeds)}, max {max(seeds)}"
    )
    if ages:
        q = np.percentile(ages, [10, 50, 90])
        print(
            f"example age before target (years): p10 {q[0]:.2f}, p50 {q[1]:.2f}, p90 {q[2]:.2f}"
        )
    if same:
        print(f"examples sharing the target's entity: {np.mean(same):.0%} on average")
    total = sum(via.values())
    print("cells by how they were reached:")
    for k, v in via.most_common(12):
        print(f"  {v / total:6.1%}  {k}")
    print("cells by depth:", {d: f"{c / total:.0%}" for d, c in sorted(depth.items())})


if __name__ == "__main__":
    main()
