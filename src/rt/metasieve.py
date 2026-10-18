"""MetaSieve metapath selection and SQL neighborhoods built from its rules.

MetaSieve (Khan & Aboulnaga, 2026) scores each schema-valid metapath extension
from per-seed SQL counts and the task labels, then prunes the uninformative
ones. ``rt.sql_context`` builds each seed's neighborhood from the rows its
kept metapaths reach, so the context budget is spent only on label-relevant
paths. Nothing here is specific to one database: the schema and the task's
entity table come from the rule set.

Two deliberate differences from the paper:
- Every frontier table holds all seeds at once instead of one batch per worker.
  Batches are still used to score, which is what the LCB needs.
- Rules are applied per full metapath (prefix-aware). The paper approximated
  them per terminal table because PyG cannot track prefixes; SQL can.

A schema maps each table to ``{"pkey": col | None, "time": col | None,
"fkeys": {col: parent_table}}``. A step is ``(direction, src, fk_col, dst)``:
"fwd" follows ``src.fk_col -> dst.pkey`` and "rev" follows
``dst.fk_col -> src.pkey``.
"""

from __future__ import annotations

import math


def relbench_schema(db):
    """Build a schema dict from a ``relbench`` Database."""
    return {
        name: {
            "pkey": table.pkey_col,
            "time": table.time_col,
            "fkeys": dict(table.fkey_col_to_pkey_table),
        }
        for name, table in db.table_dict.items()
    }


def task_info(schema, *, db, name, entity_table, entity_col, time_col, target_col, task_type):
    """Task metadata a rule set carries so samplers need no database knowledge."""
    if task_type not in ("clf", "reg"):
        raise ValueError(f"Unknown task type: {task_type}")
    pkey = schema[entity_table]["pkey"]
    if not pkey:
        raise ValueError(f"Entity table {entity_table} has no primary key")
    return {"db": db, "name": name, "entity_table": entity_table, "entity_col": entity_col,
            "entity_pkey": pkey, "time_col": time_col, "target_col": target_col, "task_type": task_type}


def relbench_task_info(schema, db, name, task):
    """:func:`task_info` from a ``relbench`` task object (which does not know its name)."""
    task_type = "reg" if "regression" in str(task.task_type).lower() else "clf"
    return task_info(schema, db=db, name=name, entity_table=task.entity_table,
                     entity_col=task.entity_col, time_col=task.time_col,
                     target_col=task.target_col, task_type=task_type)


def enumerate_metapaths(schema, entity_table, hops):
    """All schema-valid metapaths of length 1..hops, parents before children.

    Immediate backtracking (a reverse step followed by its own forward step)
    is skipped because it returns to the row it came from.
    """
    if hops <= 0:
        raise ValueError("hops must be positive")
    paths, frontier = [], [()]
    for _ in range(hops):
        nxt = []
        for prefix in frontier:
            table = prefix[-1][3] if prefix else entity_table
            for step in _steps_from(schema, table):
                last = prefix[-1] if prefix else None
                if last and last[0] == "rev" and step == ("fwd", last[3], last[2], last[1]):
                    continue
                nxt.append(prefix + (step,))
        paths.extend(nxt)
        frontier = nxt
    return paths


def _steps_from(schema, table):
    steps = []
    for col, parent in schema[table]["fkeys"].items():
        if schema.get(parent, {}).get("pkey"):
            steps.append(("fwd", table, col, parent))
    if schema[table]["pkey"]:
        for child, spec in schema.items():
            for col, parent in spec["fkeys"].items():
                if parent == table:
                    steps.append(("rev", table, col, child))
    return steps


def path_name(path):
    return " ".join(
        f"{'->' if d == 'fwd' else '<-'}{dst}.{col}" if d == "rev" else f"->{dst}[{col}]"
        for d, _, col, dst in path
    )


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _step_sql(schema, parent_frontier, step, strict):
    """SELECT (seed, rid, t) rows one step past ``parent_frontier``."""
    direction, src, col, dst = step
    s, d = _quote(src), _quote(dst)
    if direction == "fwd":
        on = f"d.{_quote(schema[dst]['pkey'])} = x.{_quote(col)}"
    else:
        on = f"d.{_quote(col)} = x.{_quote(schema[src]['pkey'])}"
    time_col = schema[dst]["time"]
    where = ""
    t = "NULL::DOUBLE"
    if time_col:
        col = f"d.{_quote(time_col)}"
        t = f"epoch({col})::DOUBLE"
        op = "<" if strict else "<="
        # Untimestamped rows always pass, matching the Rust sampler's filter.
        where = f"WHERE ({col} IS NULL OR {col} {op} s.ts)"
    return (
        f"SELECT f.seed, d.rowid AS rid, f.root_rank, {t} AS t "
        f"FROM {parent_frontier} f "
        f"JOIN {s} x ON x.rowid = f.rid "
        f"JOIN {d} d ON {on} "
        f"JOIN ms_seeds s ON s.seed = f.seed {where}"
    )


class _Frontiers:
    """Temp frontier tables keyed by metapath; each prefix is built once."""

    def __init__(self, con, schema, entity_table, seeds, strict, width=None):
        self.con, self.schema, self.strict, self.width = con, schema, strict, width
        self.tables = {}
        con.register("ms_seeds_df", seeds)
        con.execute(
            "CREATE OR REPLACE TEMP TABLE ms_seeds AS "
            "SELECT seed::BIGINT AS seed, key, ts::TIMESTAMP AS ts FROM ms_seeds_df"
        )
        pkey = _quote(schema[entity_table]["pkey"])
        self._create((), (
            f"SELECT s.seed, e.rowid AS rid, 1::BIGINT AS root_rank "
            f"FROM ms_seeds s JOIN {_quote(entity_table)} e ON e.{pkey} = s.key"
        ))

    def _create(self, path, sql):
        name = f"ms_f{len(self.tables)}"
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
        self.tables[path] = name
        return name

    def get(self, path):
        if path in self.tables:
            return self.tables[path]
        parent = self.get(path[:-1])
        rows = _step_sql(self.schema, parent, path[-1], self.strict)
        if self.width is None:
            # Scoring: distinct reachable rows per seed, as in the paper.
            return self._create(path, f"SELECT DISTINCT seed, rid, 1::BIGINT AS root_rank FROM ({rows})")
        # Context: keep each row once per seed with its best root rank. Hop-1
        # rows define root ranks by recency; deeper rows inherit their root's,
        # so descendants stay beside the history row they came from.
        if len(path) == 1:
            ranked = (
                f"SELECT seed, rid, ROW_NUMBER() OVER (PARTITION BY seed ORDER BY t DESC NULLS LAST, rid) "
                f"AS root_rank, t FROM ({rows})"
            )
        else:
            ranked = f"SELECT seed, rid, MIN(root_rank) AS root_rank, ANY_VALUE(t) AS t FROM ({rows}) GROUP BY seed, rid"
        if path[-1][0] == "rev":
            # One-to-many steps are capped per seed; forward steps have at most
            # one parent row per child, so they cannot fan out.
            ranked = (
                f"SELECT * FROM ({ranked}) QUALIFY ROW_NUMBER() OVER "
                f"(PARTITION BY seed ORDER BY root_rank, t DESC NULLS LAST, rid) <= {int(self.width)}"
            )
        return self._create(path, ranked)

    def close(self):
        for name in self.tables.values():
            self.con.execute(f"DROP TABLE IF EXISTS {name}")
        self.con.execute("DROP TABLE IF EXISTS ms_seeds")
        self.con.unregister("ms_seeds_df")


def path_counts(con, schema, entity_table, seeds, paths, strict=False, chunk=20_000):
    """Distinct rows reached per seed for each path: DataFrame (seed x path).

    Frontiers are built ``chunk`` seeds at a time; high-fanout paths on large
    databases can otherwise exhaust memory.
    """
    import pandas as pd

    parts = []
    for start in range(0, len(seeds), chunk):
        part = seeds.iloc[start:start + chunk]
        frontiers = _Frontiers(con, schema, entity_table, part, strict)
        try:
            counts = pd.DataFrame(index=pd.Index(part["seed"].astype("int64"), name="seed"))
            for i, path in enumerate(paths):
                table = frontiers.get(path)
                n = con.execute(f"SELECT seed, COUNT(*) AS n FROM {table} GROUP BY seed").df()
                counts[i] = n.set_index("seed")["n"].reindex(counts.index, fill_value=0).astype("float64")
            parts.append(counts)
        finally:
            frontiers.close()
    return pd.concat(parts) if parts else pd.DataFrame(index=pd.Index([], name="seed"))


def _entropy(y, task_type, bins):
    import numpy as np

    if task_type == "reg":
        y = np.digitize(y, bins)
    _, freq = np.unique(y, return_counts=True)
    p = freq / freq.sum()
    return float(-(p * np.log(p)).sum())


def _lcb(values, delta):
    import numpy as np
    from scipy.stats import t as student_t

    values = np.asarray(values, dtype="float64")
    if len(values) < 2:
        return float(values.mean())
    sd = values.std(ddof=1)
    return float(values.mean() - student_t.ppf(1 - delta, len(values) - 1) * sd / math.sqrt(len(values)))


def score_metapaths(counts, paths, labels, task_type, num_batches=8, delta=0.3, seed=0):
    """Score every path and decide which to keep. Returns one dict per path.

    ``counts`` comes from :func:`path_counts` over labeled training seeds and
    ``labels`` is aligned with its index. Hop-1 paths are never pruned: their
    log-rate is undefined, so the paper starts pruning at hop 2.
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.mixture import GaussianMixture

    if task_type not in ("clf", "reg"):
        raise ValueError(f"Unknown task type: {task_type}")
    if not 0 < delta < 0.5:
        raise ValueError("delta must be in (0, 0.5)")
    y = np.asarray(labels)
    if len(y) != len(counts):
        raise ValueError("labels must align with counts")
    mi = mutual_info_classif if task_type == "clf" else mutual_info_regression
    # Fixed quantile bins from all training labels keep entropies comparable.
    bins = np.unique(np.quantile(y, np.linspace(0, 1, 11)[1:-1])) if task_type == "reg" else None
    # Deterministic batches over seeds ordered by identifier (NTILE in the paper).
    order = np.argsort(counts.index.to_numpy(), kind="stable")
    batches = [b for b in np.array_split(order, num_batches) if len(b)]
    index = {path: i for i, path in enumerate(paths)}

    rows = []
    for i, path in enumerate(paths):
        n = counts[i].to_numpy()
        parent = counts[index[path[:-1]]].to_numpy() if len(path) > 1 else None
        stats = {"count": [], "rate": [], "coverage": []}
        for b in batches:
            nb, yb = n[b], y[b]
            stats["coverage"].append(float((nb > 0).mean()))
            h = _entropy(yb, task_type, bins)
            cost = float(nb.mean())
            features = {"count": np.log1p(nb)}
            if parent is not None:
                pb = parent[b]
                features["rate"] = np.log1p(np.divide(nb, pb, out=np.zeros_like(nb), where=pb > 0))
            for name in ("count", "rate"):
                if name not in features or h == 0 or cost == 0 or np.ptp(features[name]) == 0:
                    stats[name].append(0.0)
                    continue
                x = features[name].reshape(-1, 1)
                value = float(mi(x, yb, discrete_features=False, random_state=seed)[0])
                stats[name].append(value / h / cost)
        rows.append({
            "id": i, "path": path, "name": path_name(path), "hop": len(path),
            "lcb_count": _lcb(stats["count"], delta),
            "lcb_rate": _lcb(stats["rate"], delta) if parent is not None else None,
            "lcb_coverage": _lcb(stats["coverage"], delta),
            "max_coverage": max(stats["coverage"]),
        })

    df = pd.DataFrame(rows)
    df["cluster"] = None
    df.loc[df["max_coverage"] == 0, "cluster"] = "empty"
    df.loc[(df["hop"] == 1) & df["cluster"].isna(), "cluster"] = "hop1"
    scored = df["cluster"].isna()
    if scored.any():
        part = df[scored]
        p_count = part.groupby("hop")["lcb_count"].rank(pct=True)
        p_rate = part.groupby("hop")["lcb_rate"].rank(pct=True)
        p_cov = part.groupby("hop")["lcb_coverage"].rank(pct=True)
        q = np.maximum(p_count, p_rate) + p_cov
        df.loc[scored, "q"] = q
        z = ((q - q.mean()) / q.std(ddof=0)).to_numpy() if q.std(ddof=0) > 0 else None
        if z is None or len(z) < 2:
            df.loc[scored, "cluster"] = "good"
        else:
            gmm = GaussianMixture(2, random_state=seed).fit(z.reshape(-1, 1))
            good = int(np.argmax(gmm.means_.ravel()))
            df.loc[scored, "cluster"] = np.where(gmm.predict(z.reshape(-1, 1)) == good, "good", "bad")

    good = {p for p, c in zip(df["path"], df["cluster"]) if c in ("good", "hop1")}
    # Keep a path if it or any descendant is good: it may be a gateway.
    keep = {p[:k] for p in good for k in range(1, len(p) + 1)}
    df["keep"] = [p in keep and c != "empty" for p, c in zip(df["path"], df["cluster"])]
    return df.drop(columns=["max_coverage"]).to_dict("records")


def select_metapaths(con, schema, task, seeds, labels, *, hops=3, num_batches=8, delta=0.3,
                     strict=False):
    """Enumerate, count, and score metapaths; return the JSON-ready rule set.

    ``task`` comes from :func:`task_info`; ``seeds`` (seed, key, ts) are the
    labeled training rows to score on and ``labels`` is aligned with them.
    """
    entity_table = task["entity_table"]
    paths = enumerate_metapaths(schema, entity_table, hops)
    counts = path_counts(con, schema, entity_table, seeds, paths, strict=strict)
    scored = score_metapaths(counts, paths, labels, task["task_type"], num_batches, delta)
    for row in scored:
        row["path"] = [list(step) for step in row["path"]]
        for key in ("lcb_count", "lcb_rate", "lcb_coverage", "q"):
            value = row.get(key)
            row[key] = None if value is None or (isinstance(value, float) and math.isnan(value)) else value
    return {
        "task": task, "entity_table": entity_table, "schema": schema,
        "params": {"hops": hops, "num_batches": num_batches, "delta": delta,
                   "strict": strict, "num_seeds": len(seeds)},
        "paths": scored,
    }


def keep_all_rules(schema, task, hops=3):
    """A rule set that keeps every metapath: the unpruned baseline."""
    entity_table = task["entity_table"]
    paths = enumerate_metapaths(schema, entity_table, hops)
    return {
        "task": task, "entity_table": entity_table, "schema": schema,
        "params": {"hops": hops, "strict": False, "keep_all": True},
        "paths": [{"id": i, "path": [list(step) for step in path], "name": path_name(path),
                   "hop": len(path), "cluster": "all", "keep": True} for i, path in enumerate(paths)],
    }


def kept_paths(rules):
    """Kept metapaths from a rule set, as tuples of steps in enumeration order."""
    return [tuple(tuple(step) for step in row["path"]) for row in rules["paths"] if row["keep"]]


def path_rows(con, schema, entity_table, seeds, paths, width, strict=False):
    """Rows each path reaches per seed, capped for context use.

    Returns a DataFrame (seed, table, rid, depth, root_rank, root, t, order,
    spine) where ``depth`` counts from the task row (entity row = 1), ``root``
    indexes the hop-1 path a row descends from, ``order`` is the path's
    position and ``spine`` marks paths with only forward steps after hop 1.
    """
    import pandas as pd

    if width <= 0:
        raise ValueError("SQL context width must be positive")
    roots = {path[:1] for path in paths}
    frontiers = _Frontiers(con, schema, entity_table, seeds, strict, width=width)
    try:
        parts = []
        root_index = {root: i for i, root in enumerate(sorted(roots, key=paths.index))}
        for order, path in enumerate(paths):
            table = frontiers.get(path)
            rows = con.execute(f"SELECT seed, rid, root_rank, t FROM {table}").df()
            rows["table"] = path[-1][3]
            rows["depth"] = len(path) + 1
            rows["root"] = root_index[path[:1]]
            rows["order"] = order
            rows["spine"] = all(step[0] == "fwd" for step in path[1:])
            parts.append(rows)
        if not parts:
            return pd.DataFrame(columns=["seed", "rid", "root_rank", "t", "table", "depth", "root", "order",
                                         "spine"])
        return pd.concat(parts, ignore_index=True)
    finally:
        frontiers.close()


def context_nodes(rows, table_info):
    """Group :func:`path_rows` into ordered (node index, depth) lists per seed.

    Spine rows come first: hop-1 history rows interleaved by root rank across
    hop-1 paths (most recent first, round-robin over history tables), each
    followed by its forward parents. Rows past a later reverse step fan out
    (e.g. other drivers' results in a race) and follow in the same order, so they only use budget
    the spine leaves over. A row reached by several paths appears once, at its
    first position.
    """
    import numpy as np

    if rows.empty:
        return {}
    rows = rows.copy()
    offsets = {t: int(table_info[f"{t}:Db"]["node_idx_offset"]) for t in rows["table"].unique()}
    rows["node"] = rows["table"].map(offsets).to_numpy(dtype="int64") + rows["rid"].to_numpy(dtype="int64")
    # Most recent first within a path; untimestamped rows last.
    rows["neg_t"] = (-rows["t"].astype("float64")).fillna(np.inf)
    rows["fan_out"] = ~rows["spine"].astype(bool)
    rows = rows.sort_values(["seed", "fan_out", "root_rank", "root", "depth", "order", "neg_t", "rid"],
                            kind="stable")
    rows = rows.drop_duplicates(["seed", "node"], keep="first")
    out = {}
    for seed, group in rows.groupby("seed", sort=False):
        out[int(seed)] = list(zip(group["node"].astype(int).tolist(), group["depth"].astype(int).tolist()))
    return out
