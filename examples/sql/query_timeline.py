"""When each query shape runs while rt.sql builds a context.

Records every query of one inference in order, tagged with where retrieval was:
the target lookup, picking seeds, or a BFS level of a seed batch (parents or
children). Writes a Markdown report with:

1. the shapes (numbers replaced by N), each with an id and its SQL;
2. the timeline of the first context, one line per query;
3. calls, rows and time per shape at each BFS depth, over all contexts built.

    python -m examples.sql.query_timeline --duckdb data/relbench-duckdb/rel-f1.duckdb \\
        --task driver-top3 --rows 8 --out driver-top3-timeline.md
"""

import argparse
import random
import time
from collections import defaultdict

import duckdb

import rt.sql as rs
from examples.sql.profile_queries import pretty, shape
from rt.sql.retrieve import Retriever


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duckdb", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rows", type=int, default=4, help="contexts to build")
    p.add_argument("--row", type=int, help="build this split row first (its timeline is shown)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument("--local-ctx-size", type=int, default=256)
    p.add_argument("--bfs-width", type=int, default=32)
    p.add_argument("--seed-batch", type=int, default=1)
    p.add_argument("--threads", type=int, default=1, help="DuckDB threads")
    p.add_argument("--out", default="query-timeline.md")
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

    # Where retrieval is: set by the BFS fetches, "seeds" between them
    # (example queries run lazily, when build() asks for the next seeds).
    where = {"phase": "seeds", "depth": None, "seeds": None, "active": None}
    calls = []  # one list of query records per context
    run, fetch_parents, fetch_children = (
        Retriever._run,
        Retriever._fetch_parents,
        Retriever._fetch_children,
    )

    def recording_run(self, ctx, step, sql, params=()):
        tic = time.perf_counter()
        rows = run(self, ctx, step, sql, params)
        toc = time.perf_counter()
        phase = where["phase"] if step not in ("target", "label") else "target"
        wanted = len(params[0]) if params and isinstance(params[0], list) else len(params)
        calls[-1].append(
            dict(
                start=tic,
                ms=(toc - tic) * 1e3,
                step=step,
                shape=shape(sql),
                phase=phase,
                depth=where["depth"] if phase in ("parents", "children") else None,
                seeds=where["seeds"] if phase in ("parents", "children") else None,
                active=where["active"] if phase in ("parents", "children") else None,
                wanted=wanted,
                rows=len(rows),
            )
        )
        return rows

    def tagged(fetch, phase):
        def wrapper(self, ctx, states, *rest):
            where.update(
                phase=phase,
                depth=states[0].depth,
                seeds=(states[0].seed_i, states[-1].seed_i),
                active=sum(1 for st in states if st.frontier),
            )
            try:
                return fetch(self, ctx, states, *rest)
            finally:
                where.update(phase="seeds")

        return wrapper

    Retriever._run = recording_run
    Retriever._fetch_parents = tagged(fetch_parents, "parents")
    Retriever._fetch_children = tagged(fetch_children, "children")
    try:
        retriever = rs.Retriever(con, db, enc, config)
        n_rows = con.execute(
            f"SELECT count(*) FROM {db.tasks[args.task].splits[args.split].sql}"
        ).fetchone()[0]
        rows = random.Random(args.seed).sample(range(n_rows), min(args.rows, n_rows))
        if args.row is not None:
            rows = [args.row] + [r for r in rows if r != args.row][: args.rows - 1]
        calls.append([])
        retriever.build(args.task, args.split, rows[0])  # warm up caches
        calls.clear()
        t_build = []
        for r in rows:
            calls.append([])
            tic = time.perf_counter()
            retriever.build(args.task, args.split, r)
            t_build.append((time.perf_counter() - tic) * 1e3)
    finally:
        Retriever._run = run
        Retriever._fetch_parents = fetch_parents
        Retriever._fetch_children = fetch_children

    # Shape ids in order of first appearance.
    ids: dict[str, str] = {}
    steps: dict[str, str] = {}
    for c in (c for ctx in calls for c in ctx):
        if c["shape"] not in ids:
            ids[c["shape"]] = f"Q{len(ids) + 1}"
            steps[c["shape"]] = c["step"]
    n = len(rows)
    all_calls = [c for ctx in calls for c in ctx]

    lines = [
        f"# Query timeline: {db.name} / {args.task} / {args.split}",
        "",
        f"{n} contexts (rows {', '.join(map(str, rows))}), ctx {args.ctx_size}, "
        f"lcs {args.local_ctx_size}, bw {args.bfs_width}, seed_batch {args.seed_batch}, "
        f"DuckDB threads {args.threads}. On average {len(all_calls) / n:.0f} queries, "
        f"{sum(c['ms'] for c in all_calls) / n:.0f} ms of SQL and "
        f"{sum(t_build) / n:.0f} ms of build per context.",
        "",
        "## Shapes",
        "",
        "| Id | Step | calls/ctx | ms/call | ms/ctx | phases |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    by_shape = defaultdict(list)
    for c in all_calls:
        by_shape[c["shape"]].append(c)
    for s, i in ids.items():
        cs = by_shape[s]
        phases = ", ".join(dict.fromkeys(c["phase"] for c in cs))
        ms = sum(c["ms"] for c in cs)
        lines.append(
            f"| {i} | {steps[s]} | {len(cs) / n:.1f} | {ms / len(cs):.2f} | "
            f"{ms / n:.1f} | {phases} |"
        )

    # The first context, query by query.
    first = calls[0]
    t0 = first[0]["start"]
    lines += [
        "",
        f"## Timeline of row {rows[0]}",
        "",
        f"{len(first)} queries, {t_build[0]:.0f} ms. `at` is ms since the first query; "
        "the gap between queries is Python (decoding rows, BFS bookkeeping). "
        "`level` is the BFS depth being expanded, `seeds` the batch and how many of "
        "its seeds still have a frontier, `in` the rows or parents passed in.",
        "",
        "| # | at | phase | seeds | level | Id | step | in | rows | ms |",
        "| ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for k, c in enumerate(first, 1):
        seeds = (
            f"{c['seeds'][0]}–{c['seeds'][1]} ({c['active']})" if c["seeds"] else ""
        )
        lines.append(
            f"| {k} | {(c['start'] - t0) * 1e3:.1f} | {c['phase']} | {seeds} | "
            f"{'' if c['depth'] is None else c['depth']} | {ids[c['shape']]} | "
            f"{c['step']} | {c['wanted']} | {c['rows']} | {c['ms']:.2f} |"
        )

    # Per BFS depth, over every context.
    depths = sorted({c["depth"] for c in all_calls if c["depth"] is not None})
    lines += [
        "",
        "## By BFS level",
        "",
        f"Per context, over all {n}: calls / rows out / ms for each shape at each "
        "level. Level 0 expands the seeds themselves.",
        "",
        "| Id | step | " + " | ".join(f"level {d}" for d in depths) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in depths) + " |",
    ]
    for s, i in ids.items():
        cs = [c for c in by_shape[s] if c["depth"] is not None]
        if not cs:
            continue
        cells = []
        for d in depths:
            at = [c for c in cs if c["depth"] == d]
            cells.append(
                f"{len(at) / n:.1f} / {sum(c['rows'] for c in at) / n:.0f} / "
                f"{sum(c['ms'] for c in at) / n:.1f}"
                if at
                else ""
            )
        lines.append(f"| {i} | {steps[s]} | " + " | ".join(cells) + " |")

    lines += ["", "## SQL", ""]
    for s, i in ids.items():
        lines += [f"### {i}. {steps[s]}", "", "```sql", pretty(s), "```", ""]
    open(args.out, "w").write("\n".join(lines))
    print(
        f"{len(ids)} shapes, {len(all_calls) / n:.0f} queries per context -> {args.out}"
    )


if __name__ == "__main__":
    main()
