"""Non-recursive SQL neighborhoods for the rel-f1/driver-top3 experiment."""

from __future__ import annotations

import json
import time
from itertools import zip_longest
from pathlib import Path


# Keep rowid explicit across joins: Rust's node index is table offset + raw row.
RECENT_DRIVER_ROWS_SQL = """
SELECT r.rowid, race.rowid, constructor.rowid, circuit.rowid
FROM (
    SELECT rowid, raceId, {constructor_col} AS constructorId, date
    FROM {table}
    WHERE driverId = ? AND date <= ?
    ORDER BY date DESC, rowid
    LIMIT ?
) r
LEFT JOIN races race ON race.raceId = r.raceId AND race.date <= ?
LEFT JOIN constructors constructor ON constructor.constructorId = r.constructorId
LEFT JOIN circuits circuit ON circuit.circuitId = race.circuitId
ORDER BY r.date DESC, r.rowid
"""

# Only these fixed table/column names are interpolated; values stay parameters.
HISTORY_TABLES = (
    ("qualifying", "constructorId"),
    ("standings", "NULL::BIGINT"),
    ("results", "constructorId"),
)

HISTORICAL_LABELS_SQL = """
SELECT node_idx FROM sql_task_labels
WHERE driverId = ? AND date < ?
ORDER BY date DESC, node_idx
LIMIT ?
"""

HISTORICAL_SEEDS_SQL = """
SELECT node_idx FROM sql_task_labels
WHERE date < ?
ORDER BY (driverId = ?) DESC, date DESC, node_idx
"""


def query_context_nodes(con, table_info, driver_id, timestamp, width):
    """Return ordered (Rust node index, relationship depth) pairs for one seed."""
    if width <= 0:
        raise ValueError("SQL context width must be positive")
    nodes = []
    seen = set()

    def add(table, row, depth):
        if row is None:
            return
        node = int(table_info[f"{table}:Db"]["node_idx_offset"]) + int(row)
        if node not in seen:
            seen.add(node)
            nodes.append((node, depth))

    for row, in con.execute('SELECT rowid FROM drivers WHERE driverId = ?', [driver_id]).fetchall():
        add("drivers", row, 1)
    # Labels are small, high-value rows. Emit them before DB histories so that
    # the local cell budget cannot discard the entire label-neighbor query.
    for node, in con.execute(
        HISTORICAL_LABELS_SQL, [driver_id, timestamp, width]
    ).fetchall():
        if int(node) not in seen:
            seen.add(int(node))
            nodes.append((int(node), 2))

    histories = [
        con.execute(
            RECENT_DRIVER_ROWS_SQL.format(table=table, constructor_col=constructor_col),
            [driver_id, timestamp, width, timestamp],
        ).fetchall()
        for table, constructor_col in HISTORY_TABLES
    ]
    # Round-robin by recency rank: qualifying, standings, results, then their
    # next-most-recent rows. Parents remain beside the row that references them.
    for group in zip_longest(*histories):
        for (table, _), row in zip(HISTORY_TABLES, group):
            if row is None:
                continue
            idx, race, constructor, circuit = row
            add(table, idx, 2)
            add("races", race, 3)
            add("constructors", constructor, 3)
            add("circuits", circuit, 4)
    return nodes


def load_rel_f1_contexts(pre_dir, database, width):
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
        for table in (
            "drivers", "results", "qualifying", "standings", "races", "constructors", "circuits"
        ):
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
        contexts = {}
        for rows in splits:
            for driver_id, timestamp, node_idx in rows.itertuples(index=False, name=None):
                contexts[int(node_idx)] = [(int(node_idx), 0)] + query_context_nodes(
                    con, info, int(driver_id), timestamp.to_pydatetime(), width
                )
        return contexts
    finally:
        con.close()


def load_raw_rel_f1_contexts(pre_dir, database, width, timings=None):
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
        contexts, orders = {}, {}
        for driver_id, timestamp, node_idx in labels.itertuples(index=False, name=None):
            idx, driver = int(node_idx), int(driver_id)
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
            tick = time.perf_counter()
            contexts[idx] = [(idx, 0)] + query_context_nodes(con, info, driver, timestamp, width)
            if timings is not None:
                timings["SQL neighborhood queries and assembly"] = (
                    timings.get("SQL neighborhood queries and assembly", 0.0) + time.perf_counter() - tick
                )
            tick = time.perf_counter()
            orders[idx] = [int(row[0]) for row in con.execute(
                HISTORICAL_SEEDS_SQL, [timestamp, driver]
            ).fetchall()]
            if timings is not None:
                timings["SQL historical seed queries"] = (
                    timings.get("SQL historical seed queries", 0.0) + time.perf_counter() - tick
                )
        return contexts, orders
    finally:
        con.close()
