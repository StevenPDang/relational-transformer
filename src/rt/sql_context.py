"""SQL neighborhoods for rel-f1/driver-top3 from MetaSieve-selected metapaths.

Every seed's neighborhood and historical seed order are built in batched
DuckDB queries over all task rows at once; ``rt.metasieve`` decides which
relational paths a neighborhood follows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


BATCH_ENTITY_SQL = """
SELECT s.seed, d.rowid FROM ms_ctx_seeds s JOIN drivers d ON d.driverId = s.key
"""

BATCH_LABELS_SQL = """
SELECT s.seed, l.node_idx FROM ms_ctx_seeds s
JOIN sql_task_labels l ON l.driverId = s.key AND l.date < s.ts
QUALIFY ROW_NUMBER() OVER (PARTITION BY s.seed ORDER BY l.date DESC, l.node_idx) <= ?
ORDER BY s.seed, l.date DESC, l.node_idx
"""

# Every strictly earlier task row, same driver first, then most recent.
BATCH_SEED_ORDER_SQL = """
SELECT s.seed, l.node_idx FROM ms_ctx_seeds s
JOIN sql_task_labels l ON l.date < s.ts
ORDER BY s.seed, (l.driverId = s.key) DESC, l.date DESC, l.node_idx
"""


def load_metasieve_rules(path):
    """Read a rule set written by scripts/metasieve.py for rel-f1/driver-top3."""
    rules = json.loads(Path(path).expanduser().read_text())
    if rules.get("entity_table") != "drivers":
        raise ValueError("MetaSieve rules must be selected from the drivers entity table")
    return rules


def metasieve_context_nodes(con, table_info, rules, seeds, width):
    """Batched neighborhoods from kept MetaSieve metapaths, for every seed.

    ``seeds`` has columns seed (task node index), key (driverId) and ts. Each
    list holds the driver row, then up to ``width`` most recent earlier labels,
    then the DB rows the kept metapaths reach (see :func:`rt.metasieve.context_nodes`).
    """
    from rt import metasieve

    if width <= 0:
        raise ValueError("SQL context width must be positive")
    driver_offset = int(table_info["drivers:Db"]["node_idx_offset"])
    contexts = {int(seed): [] for seed in seeds["seed"]}
    con.register("ms_ctx_seeds", seeds)
    try:
        for seed, row in con.execute(BATCH_ENTITY_SQL).fetchall():
            contexts[int(seed)].append((driver_offset + int(row), 1))
        for seed, node in con.execute(BATCH_LABELS_SQL, [width]).fetchall():
            contexts[int(seed)].append((int(node), 2))
    finally:
        con.unregister("ms_ctx_seeds")
    paths = metasieve.kept_paths(rules)
    rows = metasieve.path_rows(con, rules["schema"], "drivers", seeds, paths, width,
                               strict=rules["params"].get("strict", False))
    for seed, nodes in metasieve.context_nodes(rows, table_info).items():
        seen = {node for node, _ in contexts[seed]}
        contexts[seed].extend((node, depth) for node, depth in nodes if node not in seen)
    return contexts


def historical_seed_orders(con, seeds):
    """Candidate historical seeds for every seed, in one query.

    Strictly earlier task rows, same driver first, then most recent; this
    replaces random-walk ranking in raw SQL mode.
    """
    import numpy as np

    orders = {int(seed): [] for seed in seeds["seed"]}
    con.register("ms_ctx_seeds", seeds)
    try:
        rows = con.execute(BATCH_SEED_ORDER_SQL).fetchnumpy()
    finally:
        con.unregister("ms_ctx_seeds")
    seed_col, node_col = np.asarray(rows["seed"]), np.asarray(rows["node_idx"])
    if len(seed_col):
        bounds = np.flatnonzero(np.diff(seed_col)) + 1
        for chunk_seeds, chunk in zip(np.split(seed_col, bounds), np.split(node_col, bounds)):
            orders[int(chunk_seeds[0])] = chunk.astype(int).tolist()
    return orders


def _seed_frame(splits):
    import pandas as pd

    labels = pd.concat(splits, ignore_index=True)
    return pd.DataFrame({
        "seed": labels["node_idx"].astype("int64"),
        "key": labels["driverId"].astype("int64"),
        "ts": pd.to_datetime(labels["date"]),
    })


def load_rel_f1_contexts(pre_dir, database, width, rules):
    """Validate source alignment, then materialize all possible task seeds.

    All splits need neighborhoods because Rust may pick historical same-table
    seeds beyond the evaluation split. Earlier rows from all task splits supply
    label neighbors, matching the existing sampler's split policy.
    """
    import duckdb
    import numpy as np
    import pandas as pd
    import relbench

    from rt._rustler import node_timestamps

    db_dir = Path(pre_dir) / "rel-f1"
    info = json.loads((db_dir / "table_info.json").read_text())
    source = json.loads((db_dir / "meta.json").read_text()).get("source")
    if not source:
        raise ValueError("SQL sampling requires a RelBench source in meta.json")
    task = relbench.load_task(source, "driver-top3")
    # Preprocessing stores the full raw DB; the SQL predicates enforce cutoffs.
    raw_db = relbench.load_dataset(source).get_db(upto_test_timestamp=False)
    con = duckdb.connect(str(Path(database).expanduser()), read_only=True)
    try:
        # Verify values and order, not just counts: rowid + offset must identify
        # the same raw row used by preprocessing. No assumptions about PK values.
        # Check every table a metapath may reach, not only the history tables.
        for table in sorted(rules["schema"]):
            raw = raw_db.table_dict[table].df.reset_index(drop=True)
            sql = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid').df()
            if len(raw) != info[f"{table}:Db"]["num_nodes"]:
                raise ValueError(f"Preprocessed row count differs for {table}")
            pd.testing.assert_frame_equal(sql, raw, check_dtype=False)

        splits = []
        for split in ("train", "val", "test"):
            rows = task.get_table(split, mask_input_cols=False).df.reset_index(drop=True)
            ti = info[f"driver-top3:{split.title()}"]
            if len(rows) != ti["num_nodes"]:
                raise ValueError(f"Preprocessed task row count differs for {split}")
            rows = rows[[task.entity_col, task.time_col]].copy()
            if rows.isna().any().any():
                raise ValueError(f"Null entity or timestamp in {split}")
            rows["node_idx"] = np.arange(len(rows)) + ti["node_idx_offset"]
            # Task timestamps check the split's positional alignment with Rust.
            stored = node_timestamps(str(db_dir), rows.node_idx.tolist())
            ns = pd.to_datetime(rows[task.time_col]).astype("int64").to_numpy()
            seconds = np.where(ns >= 0, ns // 10**9, -((-ns) // 10**9))
            seconds = np.clip(seconds, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            if stored != seconds.tolist():
                raise ValueError(f"Preprocessed task timestamps differ for {split}")
            splits.append(rows)
        con.register("sql_task_labels", pd.concat(splits, ignore_index=True))
        contexts = metasieve_context_nodes(con, info, rules, _seed_frame(splits), width)
        return {seed: [(seed, 0)] + nodes for seed, nodes in contexts.items()}
    finally:
        con.close()


def load_raw_rel_f1_contexts(pre_dir, database, width, rules, timings=None):
    """Select local rows and historical seeds directly from raw Parquet data.

    The cell encoder and DuckDB import consumed these exact files in physical
    row order, so there is no hosted graph/source alignment pass. Offsets only
    translate SQL row positions into the encoder's row identifiers.
    """
    import duckdb
    import numpy as np
    import pandas as pd

    db_dir = Path(pre_dir) / "rel-f1"
    info = json.loads((db_dir / "table_info.json").read_text())
    meta = json.loads((db_dir / "meta.json").read_text())
    if meta.get("sampling_graph") is not False:
        raise ValueError("Raw SQL sampling requires graph-free cell preparation")
    raw_dir = Path(meta["raw_dataset_dir"])
    splits = []
    for split in ("train", "val", "test"):
        rows = pd.read_parquet(raw_dir / "tasks" / "driver-top3" / f"{split}.parquet")
        ti = info[f"driver-top3:{split.title()}"]
        if len(rows) != ti["num_nodes"]:
            raise ValueError(f"Raw task row count differs for {split}")
        rows = rows[["driverId", "date"]].copy()
        if rows.isna().any().any():
            raise ValueError(f"Null entity or timestamp in {split}")
        rows["node_idx"] = np.arange(len(rows)) + ti["node_idx_offset"]
        splits.append(rows)
    labels = pd.concat(splits, ignore_index=True)
    con = duckdb.connect(str(Path(database).expanduser()), read_only=True)
    try:
        con.register("sql_task_labels", labels)
        seeds = _seed_frame(splits)
        tick = time.perf_counter()
        batched = metasieve_context_nodes(con, info, rules, seeds, width)
        contexts = {seed: [(seed, 0)] + nodes for seed, nodes in batched.items()}
        if timings is not None:
            timings["SQL neighborhood queries and assembly"] = time.perf_counter() - tick
        tick = time.perf_counter()
        orders = historical_seed_orders(con, seeds)
        if timings is not None:
            timings["SQL historical seed queries"] = time.perf_counter() - tick
        return contexts, orders
    finally:
        con.close()
