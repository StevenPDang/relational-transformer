"""What a converted database holds, read back from the database itself.

`examples/preprocess/to_duckdb.py` writes each RelBench table as a DuckDB
table with a leading `__row` (its parquet row position, which every key value
refers to) and stores the manifests in `_rt_manifest`. This module turns that
into the structure the rest of `rt.sql` works from.
"""

import json
from dataclasses import dataclass, field

import duckdb

ROW = "__row"

SPLITS = ("train", "val", "test")


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class Table:
    """One SQL table: a database table, or one split of a task."""

    name: str  # database table name, or task name for a split
    split: str | None  # None for a database table
    sql: str  # quoted reference, e.g. "transactions" or "customer-churn"."train"
    columns: tuple[tuple[str, str], ...]  # (name, DuckDB type), parquet order
    pkey: str | None
    time_col: str | None
    fkeys: dict[str, str] = field(hash=False)  # FK column -> parent table

    @property
    def key(self) -> str:
        """Unique name for traces: `customers`, `customer-churn/test`."""
        return self.name if self.split is None else f"{self.name}/{self.split}"

    @property
    def is_task(self) -> bool:
        return self.split is not None


@dataclass(frozen=True)
class Task:
    name: str
    target_col: str
    task_type: str  # "binary_classification" | "regression"
    time_col: str | None
    splits: dict[str, Table] = field(hash=False)


@dataclass
class Database:
    name: str
    val_timestamp: str | None
    tables: dict[str, Table]  # database tables by name
    tasks: dict[str, Task]

    def all_tables(self) -> list[Table]:
        out = list(self.tables.values())
        for task in self.tasks.values():
            out.extend(task.splits.values())
        return out

    def children_of(self, parent: str) -> list[tuple[Table, str]]:
        """Every (table, FK column) whose FK points at database table `parent`."""
        return [
            (t, col)
            for t in self.all_tables()
            for col, p in t.fkeys.items()
            if p == parent
        ]


def _columns(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[tuple[str, str], ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple((r[0], r[1]) for r in rows if r[0] != ROW)


def load(con: duckdb.DuckDBPyConnection) -> Database:
    manifests = con.execute("SELECT scope, name, manifest FROM _rt_manifest").fetchall()
    db_manifest = next(json.loads(m) for scope, _, m in manifests if scope == "db")
    specs = db_manifest.get("tables") or {}

    present = {
        (schema, name)
        for schema, name in con.execute(
            "SELECT schema_name, table_name FROM duckdb_tables()"
        ).fetchall()
    }

    tables = {}
    for schema, name in sorted(present):
        if schema != "main" or name == "_rt_manifest":
            continue
        spec = specs.get(name) or {}
        sql = ident(name)
        tables[name] = Table(
            name=name,
            split=None,
            sql=sql,
            columns=_columns(con, sql),
            pkey=spec.get("pkey"),
            time_col=spec.get("time_col"),
            fkeys=dict(spec.get("fkeys") or {}),
        )

    tasks = {}
    for scope, name, m in manifests:
        if scope != "task":
            continue
        tm = json.loads(m)
        fkeys = {
            tm[col]: tm[tbl]
            for col, tbl in (
                ("entity_col", "entity_table"),
                ("src_entity_col", "src_entity_table"),
                ("dst_entity_col", "dst_entity_table"),
            )
            if tm.get(col) and tm.get(tbl)
        }
        splits = {}
        for split in SPLITS:
            if (name, split) not in present:
                continue
            sql = f"{ident(name)}.{ident(split)}"
            splits[split] = Table(
                name=name,
                split=split,
                sql=sql,
                columns=_columns(con, sql),
                pkey=None,
                time_col=tm.get("time_col"),
                fkeys=fkeys,
            )
        tasks[name] = Task(
            name=name,
            target_col=tm.get("target_col"),
            task_type=tm.get("task_type"),
            time_col=tm.get("time_col"),
            splits=splits,
        )

    return Database(
        name=db_manifest["name"],
        val_timestamp=db_manifest.get("val_timestamp"),
        tables=tables,
        tasks=tasks,
    )
