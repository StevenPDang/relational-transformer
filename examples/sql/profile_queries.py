"""Profile rt.sql's queries one shape at a time, isolated from everything else.

Builds a few contexts, groups their queries by shape (numbers replaced by N),
then reruns one representative call of each shape alone: its time, and its
plan from EXPLAIN ANALYZE with rows out, rows scanned and time per operator.
Writes a Markdown report, most expensive shape first.

    python -m examples.sql.profile_queries --duckdb data/relbench-duckdb/rel-f1.duckdb \\
        --task driver-top3 --rows 8 --out driver-top3-profile.md
"""

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict

import duckdb

import rt.sql as rs
from rt.sql.retrieve import Retriever

KEYWORDS = r"\b(FROM|JOIN|WHERE|AND|QUALIFY|ORDER BY|UNION ALL|LIMIT)\b"


def shape(sql: str) -> str:
    return re.sub(r"\b\d+\b", "N", sql)


def pretty(sql: str) -> str:
    return re.sub(rf"\s+{KEYWORDS}", r"\n\1", sql)


def plan_lines(node: dict, depth: int = 0) -> list[str]:
    """One line per operator: name, table and scan type, rows out, rows scanned, time."""
    name = node.get("operator_name") or node.get("operator_type") or "?"
    out = []
    if name not in ("EXPLAIN_ANALYZE", "PROJECTION"):
        extra = node.get("extra_info") or {}
        where = " ".join(
            str(extra[k]) for k in ("Table", "Type") if isinstance(extra, dict) and k in extra
        )
        scanned = node.get("operator_rows_scanned") or 0
        out.append(
            f"{'  ' * depth}{name}{' ' + where if where else ''}: "
            f"{node.get('operator_cardinality', 0):,} rows out"
            + (f", {scanned:,} scanned" if scanned else "")
            + f", {node.get('operator_timing', 0) * 1e3:.2f} ms"
        )
        depth += 1
    for child in node.get("children", []):
        out += plan_lines(child, depth)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duckdb", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rows", type=int, default=8, help="contexts to build")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument("--local-ctx-size", type=int, default=256)
    p.add_argument("--bfs-width", type=int, default=32)
    p.add_argument("--seed-batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=1, help="DuckDB threads")
    p.add_argument("--reps", type=int, default=20, help="isolated reruns per shape")
    p.add_argument("--out", default="query-profile.md")
    args = p.parse_args()

    con = duckdb.connect(args.duckdb, read_only=True)
    con.execute(f"SET threads = {int(args.threads)}")
    db = rs.load(con)
    enc = rs.Encoder(con, db)
    config = rs.RetrieveConfig(
        ctx_size=args.ctx_size,
        local_ctx_size=args.local_ctx_size,
        bfs_width=args.bfs_width,
        seed_batch=args.seed_batch,
    )

    # Record every query with its parameters as the contexts are built.
    calls = []
    run = Retriever._run

    def recording_run(self, ctx, step, sql, params=()):
        tic = time.perf_counter()
        rows = run(self, ctx, step, sql, params)
        calls.append((step, sql, list(params), len(rows), time.perf_counter() - tic))
        return rows

    Retriever._run = recording_run
    retriever = rs.Retriever(con, db, enc, config)
    n_rows = con.execute(
        f"SELECT count(*) FROM {db.tasks[args.task].splits[args.split].sql}"
    ).fetchone()[0]
    rows = random.Random(args.seed).sample(range(n_rows), min(args.rows, n_rows))
    retriever.build(args.task, args.split, rows[0])  # warm up caches
    calls.clear()
    for r in rows:
        retriever.build(args.task, args.split, r)
    Retriever._run = run
    n = len(rows)

    shapes = defaultdict(list)
    for c in calls:
        shapes[shape(c[1])].append(c)
    report = []
    for s, cs in shapes.items():
        # The call with the median row count stands for the shape.
        rep = sorted(cs, key=lambda c: c[3])[len(cs) // 2]
        step, sql, params, n_out, _ = rep
        con.execute(sql, params).fetchall()
        times = []
        for _ in range(args.reps):
            tic = time.perf_counter()
            con.execute(sql, params).fetchall()
            times.append(time.perf_counter() - tic)
        prof = json.loads(
            con.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params).fetchall()[0][1]
        )
        prof = prof[0] if isinstance(prof, list) else prof
        report.append(
            dict(
                step=step,
                calls=len(cs) / n,
                ms_ctx=sum(c[4] for c in cs) / n * 1e3,
                ms_call=statistics.median(times) * 1e3,
                rows=statistics.mean(c[3] for c in cs),
                rep_rows=n_out,
                scanned=prof.get("cumulative_rows_scanned", 0),
                plan=plan_lines(prof),
                sql=pretty(s),
            )
        )
    report.sort(key=lambda r: -r["ms_ctx"])

    total = sum(r["ms_ctx"] for r in report)
    lines = [
        f"# Query profile: {db.name} / {args.task}",
        "",
        f"{n} contexts, ctx {args.ctx_size}, lcs {args.local_ctx_size}, bw {args.bfs_width}, "
        f"seed_batch {args.seed_batch}, DuckDB threads {args.threads}. "
        f"{len(calls) / n:.0f} queries and {total:.0f} ms of SQL per context.",
        "",
        "ms/call is the median of isolated reruns of one representative call. "
        "Scanned/out compares rows read from tables with rows returned.",
        "",
        "| # | Step | ms/ctx | calls/ctx | ms/call | rows/call | scanned/out |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(report, 1):
        ratio = r["scanned"] / max(r["rep_rows"], 1)
        lines.append(
            f"| {i} | {r['step']} | {r['ms_ctx']:.1f} | {r['calls']:.1f} | "
            f"{r['ms_call']:.2f} | {r['rows']:.0f} | {ratio:,.0f}x |"
        )
    for i, r in enumerate(report, 1):
        lines += [
            "",
            f"## {i}. {r['step']}",
            "",
            f"{r['calls']:.1f} calls per context, {r['ms_call']:.2f} ms per call, "
            f"{r['rep_rows']} rows out, {r['scanned']:,} rows scanned.",
            "",
            "```",
            *r["plan"],
            "```",
            "",
            "```sql",
            r["sql"],
            "```",
        ]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"{len(report)} shapes, {total:.0f} ms SQL per context -> {args.out}")


if __name__ == "__main__":
    main()
