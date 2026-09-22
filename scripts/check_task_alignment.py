"""Check RelBench test-row, rustler node-index, and DuckDB entity alignment.

Example::

    pixi run python -m scripts.check_task_alignment \
      --dataset /path/to/rel-f1 --pre-dir /path/to/pre \
      --duckdb data/duckdb/rel-f1.duckdb --task driver-top3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def check_alignment(dataset: Path, pre_dir: Path, database: Path, task: str) -> dict:
    manifest = yaml.safe_load((dataset / "manifest.yaml").read_text())
    task_manifest = yaml.safe_load((dataset / "tasks" / task / "manifest.yaml").read_text())
    db_name = manifest["name"]
    entity_table = task_manifest["entity_table"]
    entity_col = task_manifest["entity_col"]
    time_col = task_manifest["time_col"]
    pkey = manifest["tables"][entity_table]["pkey"]
    info = json.loads((pre_dir / db_name / "table_info.json").read_text())
    task_info = info[f"{task}:Test"]
    entity_info = info[f"{entity_table}:Db"]

    # RelBench uses the task Parquet order; rustler reads the same file in that order.
    task_rows = pd.read_parquet(dataset / "tasks" / task / "test.parquet")
    from relbench import load_task

    source = json.loads((pre_dir / db_name / "meta.json").read_text()).get("source")
    if not source:
        raise AssertionError("preprocessed meta.json lacks the RelBench source")
    relbench_task = load_task(source, task)
    relbench_rows = relbench_task.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    expected = task_rows[[entity_col, time_col]].reset_index(drop=True)
    actual = relbench_rows[[entity_col, time_col]]
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    n = len(task_rows)
    if n != int(task_info["num_nodes"]):
        raise AssertionError(f"test count differs: parquet={n}, rustler={task_info['num_nodes']}")
    node_offset = int(task_info["node_idx_offset"])
    node_idx = pd.Series(range(node_offset, node_offset + n))
    if not (node_idx.to_numpy() - node_offset == task_rows.index.to_numpy()).all():
        raise AssertionError("node indices do not map to task Parquet row positions")

    con = duckdb.connect(str(database), read_only=True)
    try:
        quoted = '"' + entity_table.replace('"', '""') + '"'
        raw = con.execute(f"SELECT * FROM {quoted}").df()
    finally:
        con.close()
    parquet_raw = pd.read_parquet(dataset / "db" / f"{entity_table}.parquet")
    if len(raw) != len(parquet_raw) or len(raw) != int(entity_info["num_nodes"]):
        raise AssertionError("entity table row counts differ across DuckDB, Parquet, and rustler")
    if not raw[pkey].equals(parquet_raw[pkey]):
        raise AssertionError("DuckDB entity primary keys differ from source Parquet row order")
    if raw[pkey].isna().any() or raw[pkey].duplicated().any():
        raise AssertionError("raw entity primary keys must be unique and non-null")
    if task_rows[entity_col].isna().any():
        raise AssertionError("task has null entity IDs")
    if task_rows[time_col].isna().any():
        raise AssertionError("task has null timestamps")
    positions = pd.Series(range(len(raw)), index=raw[pkey]).reindex(task_rows[entity_col])
    if positions.isna().any():
        raise AssertionError(f"{int(positions.isna().sum())} test entities absent from DuckDB")
    # rustler interprets foreign-key values as zero-based parent row positions.
    # Matching a primary key alone is insufficient if its value differs from
    # the row position; that would silently sample the wrong entity node.
    if not (positions.to_numpy() == task_rows[entity_col].to_numpy()).all():
        raise AssertionError("task entity IDs are not zero-based DuckDB entity row positions")
    entity_node_idx = int(entity_info["node_idx_offset"]) + positions.astype("int64").to_numpy()
    return {
        "task": f"{db_name}/{task}", "test_rows": n,
        "task_node_idx_first": node_offset, "task_node_idx_last": node_offset + n - 1,
        "entity_table": entity_table, "entity_key": pkey,
        "entity_node_idx_first_example": int(entity_node_idx[0]) if n else None,
        "unique_test_entity_times": int(task_rows[[entity_col, time_col]].drop_duplicates().shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="local RelBench dataset directory")
    parser.add_argument("--pre-dir", required=True, type=Path, help="preprocessed root containing <dataset>/")
    parser.add_argument("--duckdb", required=True, type=Path)
    parser.add_argument("--task", default="driver-top3")
    args = parser.parse_args()
    result = check_alignment(args.dataset, args.pre_dir, args.duckdb, args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
