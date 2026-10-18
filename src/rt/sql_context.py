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


def _register_seeds(con, seeds):
    con.register("ms_ctx_seeds", seeds)
    con.register("sql_task_labels", seeds.rename(columns={"seed": "node_idx"}))


def _unregister_seeds(con):
    con.unregister("ms_ctx_seeds")
    con.unregister("sql_task_labels")


def metasieve_context_nodes(con, table_info, rules, seeds, width, chunk=CONTEXT_CHUNK):
    """Batched neighborhoods from kept MetaSieve metapaths, for every seed.

    ``seeds`` has columns seed (task node index), key (entity key) and ts. Each
    list holds the entity row, then up to ``width`` most recent earlier labels
    of that entity, then the DB rows the kept metapaths reach (see
    :func:`rt.metasieve.context_nodes`). Frontier tables are built ``chunk``
    seeds at a time to bound DuckDB and pandas memory.
    """
    from rt import metasieve

    if width <= 0:
        raise ValueError("SQL context width must be positive")
    task = rules["task"]
    entity = task["entity_table"]
    offset = int(table_info[f"{entity}:Db"]["node_idx_offset"])
    contexts = {int(seed): [] for seed in seeds["seed"]}
    _register_seeds(con, seeds)
    try:
        sql = ENTITY_SQL.format(table=_quote(entity), pkey=_quote(task["entity_pkey"]))
        for seed, row in con.execute(sql).fetchall():
            contexts[int(seed)].append((offset + int(row), 1))
        for seed, node in con.execute(LABELS_SQL, [width]).fetchall():
            contexts[int(seed)].append((int(node), 2))
    finally:
        _unregister_seeds(con)
    paths = metasieve.kept_paths(rules)
    strict = rules["params"].get("strict", False)
    for start in range(0, len(seeds), chunk):
        part = seeds.iloc[start:start + chunk]
        rows = metasieve.path_rows(con, rules["schema"], entity, part, paths, width, strict=strict)
        for seed, nodes in metasieve.context_nodes(rows, table_info).items():
            seen = {node for node, _ in contexts[seed]}
            contexts[seed].extend((node, depth) for node, depth in nodes if node not in seen)
    return contexts


def historical_seed_orders(seeds, limit=SEED_ORDER_LIMIT):
    """Candidate historical seeds for every seed; replaces random walks in raw SQL mode.

    Strictly earlier task rows, same entity first, then most recent (ties by
    node index), at most ``limit`` per seed. All rows are sorted once by
    recency; each seed's candidates are its entity's rows past its cutoff plus a
    capped slice of the shared tail, so the cost is linear in seeds x limit
    rather than quadratic in task rows.
    """
    import numpy as np
    import pandas as pd

    if seeds.empty:
        return {}
    node = seeds["seed"].to_numpy(dtype="int64")
    ts = pd.to_datetime(seeds["ts"]).to_numpy("datetime64[ns]").astype("int64")
    codes = pd.factorize(seeds["key"])[0]
    order = np.lexsort((node, -ts))  # newest first, then node index
    g_node, g_neg_ts, g_code = node[order], -ts[order], codes[order]
    # Positions in that order of each entity's rows, still newest first.
    by_code = np.argsort(g_code, kind="stable")
    bounds = np.searchsorted(g_code[by_code], np.arange(codes.max() + 2))
    orders = {}
    for seed, seed_ts, code in zip(node, ts, codes):
        start = np.searchsorted(g_neg_ts, -seed_ts, side="right")  # first strictly earlier row
        own = by_code[bounds[code]:bounds[code + 1]]
        own = own[own >= start][:limit]
        window = np.arange(start, min(len(g_node), start + limit + len(own)))
        other = window[g_code[window] != code][:limit - len(own)]
        orders[int(seed)] = g_node[np.concatenate([own, other])].tolist()
    return orders


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
        contexts = metasieve_context_nodes(con, info, rules, _seed_frame(splits), width)
        return {seed: [(seed, 0)] + nodes for seed, nodes in contexts.items()}
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
        batched = metasieve_context_nodes(con, info, rules, seeds, width)
        contexts = {seed: [(seed, 0)] + nodes for seed, nodes in batched.items()}
        if timings is not None:
            timings["SQL neighborhood queries and assembly"] = time.perf_counter() - tick
        tick = time.perf_counter()
        orders = historical_seed_orders(seeds)
        if timings is not None:
            timings["SQL historical seed queries"] = time.perf_counter() - tick
        return contexts, orders
    finally:
        con.close()
