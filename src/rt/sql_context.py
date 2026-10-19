"""SQL neighborhoods from MetaSieve-selected metapaths, for any RelBench task.

Every seed's neighborhood and historical seed order are built in batched
DuckDB queries over all task rows at once; ``rt.metasieve`` decides which
relational paths a neighborhood follows. The task's entity table, key and time
columns come from the rule set (``rules["task"]``), so nothing here is tied to
one database.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

# Candidate historical seeds kept per seed. Full 8192-cell rel-f1 contexts use
# at most 39 of them with SQL neighborhoods (135 with BFS), so 1024 leaves wide
# headroom while keeping seed orders linear in the number of task rows.
SEED_ORDER_LIMIT = 1024

# Seeds per batch of frontier tables when building neighborhoods.
CONTEXT_CHUNK = 20_000

ENTITY_SQL = """
SELECT s.seed, e.rowid FROM ms_ctx_seeds s JOIN {table} e ON e.{pkey} = s.key
"""

# Most recent strictly earlier labels of the same entity (never the seed's own).
LABELS_SQL = """
SELECT s.seed, l.node_idx FROM ms_ctx_seeds s
JOIN sql_task_labels l ON l.key = s.key AND l.ts < s.ts
QUALIFY ROW_NUMBER() OVER (PARTITION BY s.seed ORDER BY l.ts DESC, l.node_idx) <= ?
ORDER BY s.seed, l.ts DESC, l.node_idx
"""


def _quote(name):
    return '"' + str(name).replace('"', '""') + '"'



def _offsets(seeds, row_seeds):
    """CSR offsets of each (sorted) seed's rows in ``row_seeds``, grouped in seed order."""
    import numpy as np

    return np.append(np.searchsorted(row_seeds, seeds), len(row_seeds)).astype("int64")


@dataclass(frozen=True)
class SqlContexts:
    """Every seed's ordered neighborhood as CSR arrays for the Rust sampler.

    Seed ``seeds[i]`` (strictly increasing) owns ``nodes[offsets[i]:offsets[i + 1]]``
    with the matching ``depths``; flat arrays keep millions of task rows compact.
    """

    seeds: "np.ndarray"  # int32
    offsets: "np.ndarray"  # int64, len(seeds) + 1
    nodes: "np.ndarray"  # int32
    depths: "np.ndarray"  # uint8

    @classmethod
    def from_rows(cls, seeds, row_seeds, nodes, depths):
        """From rows already grouped by seed in ``seeds`` order (sorted, unique)."""
        import numpy as np

        depths = np.asarray(depths)
        if len(depths) and (depths.min() < 0 or depths.max() > np.iinfo(np.uint8).max):
            raise ValueError("SQL context depths must fit in uint8")
        return cls(np.asarray(seeds, dtype="int32"), _offsets(seeds, row_seeds),
                   np.asarray(nodes, dtype="int32"), depths.astype("uint8"))

    @classmethod
    def concat(cls, parts):
        """Join contexts over consecutive, increasing seed ranges."""
        import numpy as np

        if not parts:
            return cls.from_rows([], [], [], [])
        starts = np.cumsum([0] + [len(p.nodes) for p in parts])
        offsets = [p.offsets[:-1] + start for p, start in zip(parts, starts)] + [starts[-1:]]
        return cls(np.concatenate([p.seeds for p in parts]), np.concatenate(offsets).astype("int64"),
                   np.concatenate([p.nodes for p in parts]), np.concatenate([p.depths for p in parts]))

    def with_seeds(self):
        """Each neighborhood led by its own seed at depth 0."""
        import numpy as np

        starts = self.offsets[:-1]
        return SqlContexts(self.seeds, self.offsets + np.arange(len(self.offsets)),
                           np.insert(self.nodes, starts, self.seeds),
                           np.insert(self.depths, starts, 0).astype("uint8"))

    def __getitem__(self, seed):
        import numpy as np

        i = int(np.searchsorted(self.seeds, seed))
        if i == len(self.seeds) or self.seeds[i] != seed:
            raise KeyError(seed)
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return list(zip(self.nodes[lo:hi].tolist(), self.depths[lo:hi].tolist()))

    def to_dict(self):
        return {int(seed): self[seed] for seed in self.seeds}


@dataclass(frozen=True)
class SeedOrders:
    """Every seed's ordered historical seed candidates as CSR arrays."""

    seeds: "np.ndarray"  # int32, strictly increasing
    offsets: "np.ndarray"  # int64, len(seeds) + 1
    candidates: "np.ndarray"  # int32

    def __getitem__(self, seed):
        import numpy as np

        i = int(np.searchsorted(self.seeds, seed))
        if i == len(self.seeds) or self.seeds[i] != seed:
            raise KeyError(seed)
        return self.candidates[self.offsets[i]:self.offsets[i + 1]].tolist()

    def to_dict(self):
        return {int(seed): self[seed] for seed in self.seeds}


def resolve_duckdb(spec, db_name):
    """A DuckDB file, or ``<dir>/<db>.duckdb`` when ``spec`` is a directory."""
    path = Path(spec).expanduser()
    return str(path / f"{db_name}.duckdb" if path.is_dir() else path)


def load_metasieve_rules(spec, db_name, task_name):
    """Rules for one task: a JSON file, or ``<dir>/<db>-<task>/latest.json``.

    The rule set must have been selected for this task: its kept paths and its
    entity/time columns are task-specific.
    """
    path = Path(spec).expanduser()
    if path.is_dir():
        path = path / f"{db_name}-{task_name}" / "latest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No MetaSieve rules for {db_name}/{task_name} at {path}; run `python -m scripts.metasieve "
            f"--source stanford-star/relbench/{db_name} --task {task_name}`"
        )
    rules = json.loads(path.read_text())
    task = rules.get("task") or {}
    if (task.get("db"), task.get("name")) != (db_name, task_name):
        found = f"{task.get('db')}/{task.get('name')}" if task else "an older format without task metadata"
        raise ValueError(f"MetaSieve rules {path} are for {found}, not {db_name}/{task_name}; regenerate them")
    return rules


def metasieve_context_nodes(con, table_info, rules, seeds, width, chunk=CONTEXT_CHUNK):
    """Batched neighborhoods from kept MetaSieve metapaths, for every seed.

    ``seeds`` has columns seed (task node index), key (entity key) and ts. Each
    neighborhood holds the entity row, then up to ``width`` most recent earlier
    labels of that entity, then the DB rows the kept metapaths reach (see
    :func:`rt.metasieve.context_nodes`); returns :class:`SqlContexts`. Seeds are
    processed ``chunk`` at a time in node order to bound DuckDB and pandas
    memory, so no per-seed Python lists are built.
    """
    import numpy as np
    import pandas as pd

    from rt import metasieve

    if width <= 0:
        raise ValueError("SQL context width must be positive")
    task = rules["task"]
    entity = task["entity_table"]
    offset = int(table_info[f"{entity}:Db"]["node_idx_offset"])
    seeds = seeds.sort_values("seed", kind="stable").reset_index(drop=True)
    if seeds["seed"].duplicated().any():
        raise ValueError("Duplicate SQL context seeds")
    entity_sql = ENTITY_SQL.format(table=_quote(entity), pkey=_quote(task["entity_pkey"]))
    paths = metasieve.kept_paths(rules)
    strict = rules["params"].get("strict", False)
    parts = []
    # Every task row is a potential earlier label, whichever chunk its seed is in.
    con.register("sql_task_labels", seeds.rename(columns={"seed": "node_idx"}))
    try:
        for start in range(0, len(seeds), chunk):
            part = seeds.iloc[start:start + chunk]
            con.register("ms_ctx_seeds", part)
            try:
                ent = con.execute(entity_sql).df()
                labels = con.execute(LABELS_SQL, [width]).df()
            finally:
                con.unregister("ms_ctx_seeds")
            rows = metasieve.path_rows(con, rules["schema"], entity, part, paths, width, strict=strict)
            reached = metasieve.context_nodes(rows, table_info)
            # Entity row, labels, then path rows per seed; a stable sort by seed
            # keeps each part's own order, and a row appears once, first.
            frame = pd.concat([
                pd.DataFrame({"seed": ent["seed"], "node": offset + ent["rowid"].astype("int64"), "depth": 1}),
                pd.DataFrame({"seed": labels["seed"], "node": labels["node_idx"], "depth": 2}),
                reached,
            ], ignore_index=True).astype("int64")
            frame = frame.iloc[np.argsort(frame["seed"].to_numpy(), kind="stable")]
            frame = frame[~frame.duplicated(["seed", "node"])]
            # Compact per chunk (int32 nodes, uint8 depths) so peak memory is
            # the flat result plus one chunk's frames.
            parts.append(SqlContexts.from_rows(part["seed"].to_numpy(), frame["seed"].to_numpy(),
                                               frame["node"].to_numpy(), frame["depth"].to_numpy()))
    finally:
        con.unregister("sql_task_labels")
    return SqlContexts.concat(parts)


def _ranges(starts, lengths):
    """Concatenated ``arange(start, start + length)`` for each pair."""
    import numpy as np

    ends = np.cumsum(lengths)
    return np.repeat(starts - (ends - lengths), lengths) + np.arange(ends[-1] if len(ends) else 0)


def historical_seed_orders(seeds, limit=SEED_ORDER_LIMIT, chunk=None):
    """Candidate historical seeds for every seed; replaces random walks in raw SQL mode.

    Strictly earlier task rows, same entity first, then most recent (ties by
    node index), at most ``limit`` per seed; returns :class:`SeedOrders`. All
    rows are sorted once by recency; each seed's candidates are its entity's
    rows past its cutoff, then the other entities' rows among the next
    ``limit`` positions, so the cost is linear in seeds x limit rather than
    quadratic in task rows. Seeds are vectorized ``chunk`` at a time.
    """
    import numpy as np
    import pandas as pd

    seeds = seeds.sort_values("seed", kind="stable")
    node = seeds["seed"].to_numpy(dtype="int64")
    if len(node) == 0:
        return SeedOrders(node.astype("int32"), np.zeros(1, dtype="int64"), node.astype("int32"))
    ts = pd.to_datetime(seeds["ts"]).to_numpy("datetime64[ns]").astype("int64")
    codes = pd.factorize(seeds["key"])[0].astype("int64")
    n = len(node)
    order = np.lexsort((node, -ts))  # newest first, then node index
    g_node, g_neg_ts, g_code = node[order].astype("int32"), -ts[order], codes[order]
    # Positions in that order of each entity's rows, still newest first; the
    # (entity, position) keys are sorted, so searches find per-entity cutoffs.
    by_code = np.argsort(g_code, kind="stable")
    keys = g_code[by_code] * n + by_code
    ends = np.searchsorted(g_code[by_code], np.arange(codes.max() + 2))[1:]
    chunk = chunk or max(1, 2**24 // max(limit, 1))
    out_seeds, out_counts, out_cands = [], [], []
    for lo in range(0, n, chunk):
        c_ts, c_code = ts[lo:lo + chunk], codes[lo:lo + chunk]
        start = np.searchsorted(g_neg_ts, -c_ts, side="right")  # first strictly earlier row
        own_lo = np.searchsorted(keys, c_code * n + start)
        n_own = np.minimum(ends[c_code] - own_lo, limit)
        # Others: the first limit - n_own other-entity rows from the cutoff on.
        # The next ``limit`` positions hold at most n_own own rows, so they suffice.
        width = np.minimum(limit, n - start)
        window = _ranges(start, width)
        seg = np.repeat(np.arange(len(start)), width)
        keep = g_code[window] != c_code[seg]
        window, seg = window[keep], seg[keep]
        count = np.bincount(seg, minlength=len(start))
        rank = np.arange(len(seg)) - (np.cumsum(count) - count)[seg]
        keep = rank < (limit - n_own)[seg]
        window, seg = window[keep], seg[keep]
        n_other = np.bincount(seg, minlength=len(start))
        # Interleave per seed: own candidates, then others.
        total = n_own + n_other
        base = np.cumsum(total) - total
        cands = np.empty(int(total.sum()), dtype="int32")
        cands[_ranges(base, n_own)] = g_node[by_code[_ranges(own_lo, n_own)]]
        cands[_ranges(base + n_own, n_other)] = g_node[window]
        out_seeds.append(node[lo:lo + chunk])
        out_counts.append(total)
        out_cands.append(cands)
    counts = np.concatenate(out_counts)
    return SeedOrders(np.concatenate(out_seeds).astype("int32"),
                      np.append(0, np.cumsum(counts)).astype("int64"),
                      np.concatenate(out_cands))


def _seed_frame(splits):
    """All task rows as one (seed, key, ts) frame."""
    import pandas as pd

    rows = pd.concat(splits, ignore_index=True)
    return pd.DataFrame({
        "seed": rows["node_idx"].astype("int64"),
        "key": rows["key"],
        "ts": pd.to_datetime(rows["ts"]),
    })


def _task_rows(df, task, info, split):
    """One split's (key, ts, node_idx) rows, checked against the preprocessed count."""
    import numpy as np

    ti = info[f"{task['name']}:{split.title()}"]
    if len(df) != ti["num_nodes"]:
        raise ValueError(f"Task row count differs for {task['name']} {split}")
    rows = df[[task["entity_col"], task["time_col"]]].reset_index(drop=True)
    rows.columns = ["key", "ts"]
    if rows.isna().any().any():
        raise ValueError(f"Null entity or timestamp in {task['name']} {split}")
    rows["node_idx"] = np.arange(len(rows)) + ti["node_idx_offset"]
    return rows


def _check_db_alignment(con, raw_db, info, schema):
    """Each table's rowid must identify the row preprocessing encoded there.

    Compares row counts and, by position, the primary key (or the foreign keys
    and time column when there is none) of every table a path may reach. No
    assumptions about key values.
    """
    import pandas as pd

    for table in sorted(schema):
        spec = schema[table]
        raw = raw_db.table_dict[table].df.reset_index(drop=True)
        if len(raw) != info[f"{table}:Db"]["num_nodes"]:
            raise ValueError(f"Preprocessed row count differs for {table}")
        cols = [spec["pkey"]] if spec["pkey"] else [*spec["fkeys"], *([spec["time"]] if spec["time"] else [])]
        for col in cols:
            ours = con.execute(f"SELECT {_quote(col)} FROM {_quote(table)} ORDER BY rowid").df()[col]
            try:
                pd.testing.assert_series_equal(ours, raw[col], check_dtype=False, check_names=False,
                                               check_index_type=False)
            except AssertionError as err:
                raise ValueError(f"DuckDB row order differs from the source for {table}.{col}") from err


def load_contexts(pre_dir, database, db_name, task_name, width, rules):
    """Prepared data: validate source alignment, then build every seed's neighborhood.

    All splits need neighborhoods because Rust may pick historical same-table
    seeds beyond the evaluation split; random walks still rank those seeds.
    """
    import duckdb
    import numpy as np
    import pandas as pd
    import relbench

    from rt._rustler import node_timestamps

    task = rules["task"]
    db_dir = Path(pre_dir) / db_name
    info = json.loads((db_dir / "table_info.json").read_text())
    source = json.loads((db_dir / "meta.json").read_text()).get("source")
    if not source:
        raise ValueError("SQL sampling requires a RelBench source in meta.json")
    rb_task = relbench.load_task(source, task_name)
    # Preprocessing stores the full raw DB; the SQL predicates enforce cutoffs.
    raw_db = relbench.load_dataset(source).get_db(upto_test_timestamp=False)
    con = duckdb.connect(resolve_duckdb(database, db_name), read_only=True)
    try:
        _check_db_alignment(con, raw_db, info, rules["schema"])
        splits = []
        for split in ("train", "val", "test"):
            rows = _task_rows(rb_task.get_table(split, mask_input_cols=False).df, task, info, split)
            # Task timestamps check the split's positional alignment with Rust.
            stored = node_timestamps(str(db_dir), rows.node_idx.tolist())
            ns = pd.to_datetime(rows["ts"]).astype("int64").to_numpy()
            seconds = np.where(ns >= 0, ns // 10**9, -((-ns) // 10**9))
            seconds = np.clip(seconds, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            if stored != seconds.tolist():
                raise ValueError(f"Preprocessed task timestamps differ for {task_name} {split}")
            splits.append(rows)
        return metasieve_context_nodes(con, info, rules, _seed_frame(splits), width).with_seeds()
    finally:
        con.close()


def load_raw_contexts(pre_dir, database, db_name, task_name, width, rules, timings=None):
    """Raw data: neighborhoods and SQL seed orders straight from raw Parquet.

    The cell encoder and DuckDB import consumed these exact files in physical
    row order, so there is no hosted graph/source alignment pass. Offsets only
    translate SQL row positions into the encoder's row identifiers.
    """
    import duckdb
    import pandas as pd

    task = rules["task"]
    db_dir = Path(pre_dir) / db_name
    info = json.loads((db_dir / "table_info.json").read_text())
    meta = json.loads((db_dir / "meta.json").read_text())
    if meta.get("sampling_graph") is not False:
        raise ValueError("Raw SQL sampling requires graph-free cell preparation")
    raw_dir = Path(meta["raw_dataset_dir"])
    splits = [
        _task_rows(pd.read_parquet(raw_dir / "tasks" / task_name / f"{split}.parquet"), task, info, split)
        for split in ("train", "val", "test")
    ]
    seeds = _seed_frame(splits)
    con = duckdb.connect(resolve_duckdb(database, db_name), read_only=True)
    try:
        tick = time.perf_counter()
        contexts = metasieve_context_nodes(con, info, rules, seeds, width).with_seeds()
        if timings is not None:
            timings["SQL neighborhood queries and assembly"] = time.perf_counter() - tick
        tick = time.perf_counter()
        orders = historical_seed_orders(seeds)
        if timings is not None:
            timings["SQL historical seed queries"] = time.perf_counter() - tick
        return contexts, orders
    finally:
        con.close()
