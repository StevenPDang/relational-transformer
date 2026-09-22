"""Check RelBench test-row, rustler node-index, and DuckDB entity alignment.

Example::

    pixi run python -m scripts.check_task_alignment \
      --pre-dir stanford-star/relbench-preprocessed \
      --duckdb data/duckdb/rel-f1.duckdb --task driver-top3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def check_alignment(pre_dir: Path, database: Path, task: str, db_name: str) -> dict:
    import relbench

    meta = json.loads((pre_dir / db_name / "meta.json").read_text())
    source = meta.get("source")
    if not source:
        raise AssertionError("preprocessed meta.json lacks the RelBench source")
    relbench_task = relbench.load_task(source, task)
    dataset = relbench.load_dataset(source)
    task_rows = relbench_task.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    entity_table = relbench_task.entity_table
    entity_col = relbench_task.entity_col
    time_col = relbench_task.time_col
    entity = dataset.get_db().table_dict[entity_table]
    pkey = entity.pkey_col
    raw_relbench = entity.df.reset_index(drop=True)

    info = json.loads((pre_dir / db_name / "table_info.json").read_text())
    task_info = info[f"{task}:Test"]
    entity_info = info[f"{entity_table}:Db"]
    n = len(task_rows)
    if n != int(task_info["num_nodes"]):
        raise AssertionError(f"test count differs: RelBench={n}, rustler={task_info['num_nodes']}")
    node_offset = int(task_info["node_idx_offset"])
    # eval.py recovers this row number as node_idx - node_idx_offset.
    node_idx = pd.Series(range(node_offset, node_offset + n))
    if not (node_idx.to_numpy() - node_offset == task_rows.index.to_numpy()).all():
        raise AssertionError("node indices do not map to RelBench test row positions")
    from rt._rustler import node_timestamps

    stored = node_timestamps(str(pre_dir / db_name), node_idx.tolist())
    if any(value is None for value in stored):
        raise AssertionError("preprocessed test nodes have missing timestamps")
    # rustler normalizes datetimes to nanoseconds, then stores seconds as i32.
    expected_ns = pd.to_datetime(task_rows[time_col], utc=True).astype("int64").to_numpy()
    expected_seconds = np.where(
        expected_ns >= 0,
        expected_ns // 1_000_000_000,
        -((-expected_ns) // 1_000_000_000),
    )
    expected_seconds = np.clip(expected_seconds, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    mismatches = np.flatnonzero(np.asarray(stored, dtype=np.int64) != expected_seconds)
    if mismatches.size:
        row = int(mismatches[0])
        raise AssertionError(
            f"timestamp mismatch at test row {row}, node {int(node_idx[row])}: "
            f"RelBench={int(expected_seconds[row])}, preprocessed={stored[row]}"
        )

    con = duckdb.connect(str(database), read_only=True)
    try:
        quoted = '"' + entity_table.replace('"', '""') + '"'
        raw_duckdb = con.execute(f"SELECT * FROM {quoted}").df()
    finally:
        con.close()
    if len(raw_duckdb) != len(raw_relbench) or len(raw_duckdb) != int(entity_info["num_nodes"]):
        raise AssertionError("entity row counts differ across DuckDB, RelBench, and rustler")
    pd.testing.assert_series_equal(raw_duckdb[pkey], raw_relbench[pkey], check_dtype=False)
    if raw_duckdb[pkey].isna().any() or raw_duckdb[pkey].duplicated().any():
        raise AssertionError("entity primary keys must be unique and non-null")
    if task_rows[entity_col].isna().any() or task_rows[time_col].isna().any():
        raise AssertionError("test rows have null entity IDs or timestamps")
    positions = pd.Series(range(len(raw_duckdb)), index=raw_duckdb[pkey]).reindex(task_rows[entity_col])
    if positions.isna().any():
        raise AssertionError(f"{int(positions.isna().sum())} test entities absent from DuckDB")
    # rustler interprets foreign-key values as zero-based parent row positions.
    if not (positions.to_numpy() == task_rows[entity_col].to_numpy()).all():
        raise AssertionError("task entity IDs are not zero-based DuckDB entity row positions")
    entity_node_idx = int(entity_info["node_idx_offset"]) + positions.astype("int64").to_numpy()
    return {
        "task": f"{db_name}/{task}", "source": source, "test_rows": n,
        "task_node_idx_first": node_offset, "task_node_idx_last": node_offset + n - 1,
        "timestamps_checked": n,
        "entity_table": entity_table, "entity_key": pkey,
        "entity_node_idx_first_example": int(entity_node_idx[0]) if n else None,
        "unique_test_entity_times": int(task_rows[[entity_col, time_col]].drop_duplicates().shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", default="stanford-star/relbench-preprocessed",
                        help="same preprocessed root or Hub repo used by eval.py")
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/rel-f1.duckdb"))
    parser.add_argument("--db", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    args = parser.parse_args()
    from rt.pre import resolve_pre_dir

    pre_dir = Path(resolve_pre_dir(args.pre_dir, [args.db], "all-MiniLM-L12-v2"))
    print(json.dumps(check_alignment(pre_dir, args.duckdb, args.task, args.db), indent=2))


if __name__ == "__main__":
    main()
