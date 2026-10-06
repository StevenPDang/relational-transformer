"""On-demand graph-equivalent SQL neighbors and legacy rel-f1 experiments."""

from __future__ import annotations

import json
import os
import threading
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


def _sql_identifier(name):
    """Quote a single identifier, never an expression or qualified SQL path."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _is_temporal(values):
    import datetime

    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return True
    present = values.dropna()
    return len(present) > 0 and all(
        isinstance(v, (datetime.date, datetime.datetime, pd.Timestamp)) for v in present
    )


def _source_seconds(frame, time_col):
    """Match pre.rs: date/datetime -> ns -> truncating seconds -> clipped i32."""
    import pandas as pd

    if time_col is None:
        return [None] * len(frame)
    values = frame[time_col]
    if not _is_temporal(values):
        return [None] * len(frame)
    result = []
    for value in values:
        if pd.isna(value):
            result.append(None)
        else:
            ns = pd.Timestamp(value).value
            seconds = ns // 10**9 if ns >= 0 else -((-ns) // 10**9)
            result.append(max(-(2**31), min(2**31 - 1, seconds)))
    return result


def _comparison_value(value):
    """Normalize SQL/Parquet list containers without changing their values."""
    import numpy as np
    import pandas as pd

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_comparison_value(item) for item in value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def _fk_positions(value):
    """Rust accepts numeric scalars/lists, truncates floats, and skips nulls."""
    import numbers

    import numpy as np
    import pandas as pd

    values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    result = []
    for item in values:
        if isinstance(item, numbers.Real) and not isinstance(item, (bool, np.bool_)):
            if not pd.isna(item):
                result.append(int(item))
    return result


class SqlNeighborProvider:
    """Graph-equivalent, non-recursive Stage 2 SQL adjacency provider.

    ``provider(node_idx, target_node_idx, seed_node_idx, target_timestamp)``
    returns ``(list[int] f2p, list[int] p2f)``. The cutoff is integer epoch
    seconds or None (unbounded); null neighbor timestamps are always eligible.
    Target and seed identify the Rust traversal but do not alter adjacency.
    There is no width limit, deduplication, historical seed order, or context
    cache. Forward order is source column/list order. Reverse order is nullable
    timestamp ascending (null first), then node index. Duplicate paths remain
        in column/list order, matching the aligned Rust collector.

    Construct with :meth:`from_frames` for synthetic data, or use
    :func:`load_stage2_sql_provider` for validated preprocessed graphs. Each
    worker thread lazily opens its own read-only DuckDB connection and creates
    temporary normalized row tables and an edge *view*, not an edge table.
    Pickling excludes connections, thread-local storage, and locks; a PID
    change resets runtime state before accessing any inherited connection/lock.
    ``stats`` is a thread-safe snapshot of adjacency query_count/total_seconds
    (including lazy setup); disable collection with ``collect_stats=False``.
    Counts and elapsed time are process-local and reset on pickle/fork.
    """

    def __init__(self, database, rows, edge_sql, num_nodes, collect_stats=True):
        self.database = str(Path(database).expanduser().resolve())
        self._rows = rows
        self._edge_sql = edge_sql
        self._num_nodes = num_nodes
        self.collect_stats = collect_stats
        self._reset_runtime()

    @classmethod
    def from_frames(cls, database, table_info, frames, schema, *,
                    node_timestamps=None, collect_stats=True):
        """Validate source frames/schema and construct a lazy provider.

        All mappings use ``'<name>:Db|Train|Val|Test'`` keys and must cover
        exactly table_info. Schema entries have ``fkeys`` (column -> parent DB
        table name), optional ``pkey`` and ``time_col``. FK values are *parent
        row positions*, never primary keys. Frames retain physical column and
        row order. Every Db frame is compared to ``SELECT * ORDER BY rowid``
        from database, including columns, values, and count. Task frames need
        not exist in database and become worker-local temp tables.

        Optional node_timestamps is a callable taking a list of node indices
        and returning int seconds/None, or a mapping from table key to that
        list. The production loader always supplies Rust's timestamp reader;
        synthetic fixtures can omit it. Timestamp mismatches fail closed.
        Empty tables are allowed; offsets must partition contiguous i32 nodes.
        """
        import duckdb
        import pandas as pd

        keys = set(table_info)
        if set(frames) != keys or set(schema) != keys:
            raise ValueError("Source frames/schema must cover every graph table exactly")
        ordered = sorted(keys, key=lambda k: (int(table_info[k]["node_idx_offset"]), k))
        rows, refs, cursor = {}, [], 0
        con = duckdb.connect(str(Path(database).expanduser().resolve()), read_only=True)
        try:
            for rank, key in enumerate(ordered):
                name, kind = key.rsplit(":", 1)
                _sql_identifier(name)
                if kind not in {"Db", "Train", "Val", "Test"}:
                    raise ValueError(f"Unknown graph table type: {key}")
                info, spec = table_info[key], schema[key]
                offset, count = int(info["node_idx_offset"]), int(info["num_nodes"])
                frame = frames[key].reset_index(drop=True)
                if offset != cursor or count < 0 or len(frame) != count:
                    raise ValueError(f"Source row count/offset mismatch for {key}")
                cursor += count
                if cursor > 2**31 - 1:
                    raise ValueError("Graph node indices overflow i32")
                if not frame.columns.is_unique:
                    raise ValueError(f"Duplicate source columns for {key}")
                for column in frame.columns:
                    _sql_identifier(column)
                if kind == "Db":
                    sql = con.execute(
                        f"SELECT * FROM {_sql_identifier(name)} ORDER BY rowid"
                    ).df()
                    # Pandas compares datetime backing integers even with
                    # check_dtype=False; SQL/Parquet may use different units.
                    expected = frame.copy()
                    for column in frame.columns:
                        if _is_temporal(frame[column]) and column in sql:
                            expected[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
                            sql[column] = pd.to_datetime(sql[column], utc=True).astype("datetime64[ns, UTC]")
                        elif column in sql and frame[column].dtype == object:
                            expected[column] = frame[column].map(_comparison_value)
                            sql[column] = sql[column].map(_comparison_value)
                    try:
                        pd.testing.assert_frame_equal(
                            sql, expected, check_dtype=False, check_exact=True,
                            check_categorical=False,
                        )
                    except AssertionError as exc:
                        raise ValueError(f"SQL physical rows differ from source for {key}") from exc
                seconds = _source_seconds(frame, spec.get("time_col"))
                nodes = list(range(offset, cursor))
                if node_timestamps is not None:
                    stored = (node_timestamps(nodes) if callable(node_timestamps)
                              else node_timestamps[key])
                    if list(stored) != seconds:
                        raise ValueError(f"Preprocessed node timestamps differ for {key}")
                normalized = pd.DataFrame({
                    "node_idx": pd.Series(nodes, dtype="int32"),
                    "timestamp": pd.Series(seconds, dtype="Int32"),
                })
                alias = f"_rt_rows_{rank}"
                fkeys = spec.get("fkeys", {})
                if set(fkeys) - set(frame.columns):
                    raise ValueError(f"Missing FK source columns for {key}")
                for col_rank, column in enumerate(frame.columns):
                    if column not in fkeys or column == spec.get("pkey"):
                        continue
                    if pd.api.types.is_datetime64_any_dtype(frame[column].dtype):
                        continue  # pre.rs does not emit datetime foreign keys.
                    parent_key = f"{fkeys[column]}:Db"
                    if parent_key not in keys:
                        raise ValueError(f"Missing FK parent {parent_key} for {key}")
                    positions = [_fk_positions(v) for v in frame[column]]
                    parent_count = int(table_info[parent_key]["num_nodes"])
                    if any(p < 0 or p >= parent_count for cell in positions for p in cell):
                        raise ValueError(f"FK row position out of range for {key}.{column}")
                    fk_alias = f"fk_{col_rank}"
                    normalized[fk_alias] = positions
                    refs.append((rank, col_rank, alias, fk_alias, parent_key))
                rows[key] = (alias, normalized)
        finally:
            con.close()
        queries = []
        for rank, col_rank, alias, fk_alias, parent_key in refs:
            parent_alias = rows[parent_key][0]
            parent_offset = int(table_info[parent_key]["node_idx_offset"])
            # UNNEST list positions as well as values so duplicates and list
            # order survive. JOIN by offset + position, not by the parent PK.
            queries.append(f"""
                SELECT c.node_idx AS child, p.node_idx AS parent,
                       c.timestamp AS child_time, p.timestamp AS parent_time,
                       {rank} AS table_rank, {col_rank} AS column_rank,
                       i.list_rank AS list_rank
                FROM {_sql_identifier(alias)} c,
                     UNNEST(range(1, len(c.{_sql_identifier(fk_alias)}) + 1)) i(list_rank)
                JOIN {_sql_identifier(parent_alias)} p
                  ON p.node_idx = {parent_offset} + c.{_sql_identifier(fk_alias)}[i.list_rank]
            """)
        empty = """SELECT NULL::INTEGER child, NULL::INTEGER parent,
            NULL::INTEGER child_time, NULL::INTEGER parent_time,
            0 table_rank, 0 column_rank, 0::BIGINT list_rank WHERE FALSE"""
        return cls(database, rows, " UNION ALL ".join(queries) or empty,
                   cursor, collect_stats)

    def _reset_runtime(self):
        self._pid = os.getpid()
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._query_count = 0
        self._total_seconds = 0.0

    def _check_process(self):
        if self._pid != os.getpid():
            self._reset_runtime()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items()
                if key not in {"_pid", "_local", "_stats_lock", "_query_count", "_total_seconds"}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_runtime()

    @property
    def stats(self):
        self._check_process()
        with self._stats_lock:
            return {"query_count": self._query_count, "total_seconds": self._total_seconds}

    def close(self):
        """Close only this process/thread's connection; other workers own theirs."""
        self._check_process()
        con = getattr(self._local, "connection", None)
        if con is not None:
            con.close()
            del self._local.connection

    def _connection(self):
        self._check_process()
        con = getattr(self._local, "connection", None)
        if con is None:
            import duckdb

            con = duckdb.connect(self.database, read_only=True)
            try:
                for alias, frame in self._rows.values():
                    con.register("_rt_input", frame)
                    con.execute(f"CREATE TEMP TABLE {_sql_identifier(alias)} AS SELECT * FROM _rt_input")
                    con.unregister("_rt_input")
                con.execute(f"CREATE TEMP VIEW _rt_edges AS {self._edge_sql}")
            except BaseException:
                con.close()
                raise
            self._local.connection = con
        return con

    def __call__(self, node_idx: int, target_node_idx: int, seed_node_idx: int,
                 target_timestamp: int | None) -> tuple[list[int], list[int]]:
        import numbers

        for value in (node_idx, target_node_idx, seed_node_idx):
            if not isinstance(value, numbers.Integral) or not 0 <= value < self._num_nodes:
                raise ValueError(f"Node index out of range: {value!r}")
        if target_timestamp is not None and (
            not isinstance(target_timestamp, numbers.Integral)
            or not -(2**31) <= target_timestamp < 2**31
        ):
            raise ValueError("target_timestamp must be i32 epoch seconds or None")
        cutoff = None if target_timestamp is None else int(target_timestamp)
        self._check_process()
        start = time.perf_counter()
        try:
            con = self._connection()
            # One query per adjacency request; both orders are explicit. Null
            # first matches Rust Option ordering, including null reverse edges.
            found = con.execute("""
                SELECT direction, neighbor FROM (
                    SELECT 0 direction, parent neighbor, NULL::INTEGER edge_time,
                           table_rank, column_rank, child, list_rank
                    FROM _rt_edges WHERE child = ?
                      AND (? IS NULL OR parent_time IS NULL OR parent_time <= ?)
                    UNION ALL
                    SELECT 1 direction, child neighbor, child_time edge_time,
                           table_rank, column_rank, child, list_rank
                    FROM _rt_edges WHERE parent = ?
                      AND (? IS NULL OR child_time IS NULL OR child_time <= ?)
                ) ORDER BY direction, edge_time ASC NULLS FIRST,
                           child, table_rank, column_rank, list_rank
            """, [int(node_idx), cutoff, cutoff, int(node_idx), cutoff, cutoff]).fetchall()
            forward, reverse = [], []
            for direction, neighbor in found:
                (forward if direction == 0 else reverse).append(int(neighbor))
            return forward, reverse
        finally:
            if self.collect_stats:
                elapsed = time.perf_counter() - start
                with self._stats_lock:
                    self._query_count += 1
                    self._total_seconds += elapsed


def load_stage2_sql_provider(pre_dir, database, db_name="rel-f1") -> SqlNeighborProvider:
    """Load/validate all graph tables, returning a lazy SqlNeighborProvider.

    ``pre_dir/db_name/meta.json`` must specify ``source``: either a local raw
    dataset directory containing manifest.yaml, db/*.parquet and task manifests
    and split parquets, or a RelBench dataset identifier. Hosted sources use
    ``load_dataset(source).get_db(False)`` and unmasked task frames for every
    split present in table_info. Requires a sampling graph, not graph-free raw
    preparation. All physical DB rows and all Rust node timestamps are checked
    before returning. Install via sampler.set_sql_neighbor_provider_py(db_name,
    provider); Stage 1 and historical seed selection remain Rust-owned.
    """
    import pandas as pd

    from rt._rustler import node_timestamps

    _sql_identifier(db_name)
    if Path(db_name).name != db_name or db_name in {".", ".."}:
        raise ValueError("db_name must be a single directory name")
    db_dir = Path(pre_dir).expanduser() / db_name
    info = json.loads((db_dir / "table_info.json").read_text())
    meta = json.loads((db_dir / "meta.json").read_text())
    if meta.get("sampling_graph") is False:
        raise ValueError("Stage 2 SQL provider requires a preprocessed sampling graph")
    source = meta.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("Stage 2 SQL provider requires meta.source")
    raw_dir = Path(source).expanduser()
    frames, schema = {}, {}
    if (raw_dir / "manifest.yaml").is_file():
        import yaml

        manifest = yaml.safe_load((raw_dir / "manifest.yaml").read_text())
        for key in info:
            name, kind = key.rsplit(":", 1)
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"Invalid source table path: {name!r}")
            if kind == "Db":
                frames[key] = pd.read_parquet(raw_dir / "db" / f"{name}.parquet")
                schema[key] = manifest.get("tables", {}).get(name, {})
            else:
                task_dir = raw_dir / "tasks" / name
                tm = yaml.safe_load((task_dir / "manifest.yaml").read_text())
                fkeys = {}
                for prefix in ("", "src_", "dst_"):
                    col, parent = tm.get(f"{prefix}entity_col"), tm.get(f"{prefix}entity_table")
                    if col is not None and parent is not None:
                        fkeys[col] = parent
                frames[key] = pd.read_parquet(task_dir / f"{kind.lower()}.parquet")
                schema[key] = {"fkeys": fkeys, "time_col": tm.get("time_col")}
    else:
        import relbench

        dataset = relbench.load_dataset(source)
        raw_db = dataset.get_db(False)
        tasks = {}
        for key in info:
            name, kind = key.rsplit(":", 1)
            if kind == "Db":
                table = raw_db.table_dict[name]
            else:
                if name not in tasks:
                    tasks[name] = relbench.load_task(source, name)
                table = tasks[name].get_table(kind.lower(), mask_input_cols=False)
            frames[key] = table.df
            schema[key] = {"fkeys": table.fkey_col_to_pkey_table,
                           "pkey": table.pkey_col, "time_col": table.time_col}
    return SqlNeighborProvider.from_frames(
        database, info, frames, schema,
        node_timestamps=lambda nodes: node_timestamps(str(db_dir), nodes),
    )


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
