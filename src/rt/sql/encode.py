"""Turn database rows into RT cells, the way `rustler/src/pre.rs` does.

RT-J was trained on cells encoded by the preprocessor, so the rules here mirror
it rule for rule:

- a key column (the pkey, or an FK) is never a cell; null and NaN values are
  not cells either;
- numbers and booleans are `number` cells, normalized by the column's mean and
  sample std (std 0 counts as 1);
- dates and timestamps are `datetime` cells, normalized by one mean and
  population std pooled over every datetime column of the database;
- strings are `text` cells, carried as their sentence embedding;
- a database table drops its constant columns, and a table whose rows cannot
  be told apart gets a synthetic `identifier` column.

Statistics come from SQL aggregates over the same rows the preprocessor uses:
database tables up to the manifest's `val_timestamp`, and the train split of
a task for all of that task's splits.

Known gaps against the preprocessor, each small and listed so a mismatch can
be traced to it:
- the synthetic `identifier` column draws different N(0,1) values (the
  preprocessor seeds Rust's RNG, which Python cannot reproduce) and is not
  re-normalized;
- a list value becomes `str(list)`, where the preprocessor prints polars'
  own formatting;
- INTERVAL columns are not supported.
"""

import math
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
import ml_dtypes
import numpy as np

from rt.sql.meta import ROW, Database, Table, ident

NUMBER, TEXT, DATETIME = 0, 1, 2  # sem_types the model reads

IDENTIFIER_COL = "identifier"
IDENTIFIABILITY_MIN = 0.5
IDENTIFIABILITY_SAMPLE = 1_000_000

_INTS = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
}
_FLOATS = {"FLOAT", "DOUBLE"}


def kind_of(sql_type: str) -> str:
    """number | float | bool | datetime | text | unsupported"""
    if sql_type in _INTS:
        return "number"
    if sql_type in _FLOATS:
        return "float"
    if sql_type == "BOOLEAN":
        return "bool"
    if sql_type == "DATE" or sql_type.startswith("TIMESTAMP"):
        return "datetime"
    if sql_type == "INTERVAL" or sql_type.startswith("DECIMAL"):
        return "unsupported"
    return "text"  # VARCHAR, ENUM, BLOB, lists, structs


@dataclass(frozen=True)
class Cell:
    column: str  # "<col> of <table name>", the preprocessor's column name
    sem_type: int
    number: float = 0.0
    datetime: float = 0.0
    text: str | None = None  # text cells carry their string; embedded later


@dataclass
class TableEncoding:
    """Which columns of a table become cells, with their statistics."""

    table: Table
    cell_columns: list[tuple[str, str]]  # (name, kind) in parquet order
    stats: dict[str, tuple[float, float]]  # column -> (mean, std)
    identifier: bool
    fk_columns: list[str] = field(default_factory=list)  # FK cols, parquet order


def _identifier_policy() -> str:
    policy = os.environ.get("RT_IDENTIFIER_POLICY", "threshold")
    assert policy in ("none", "empty", "empty_or_time", "threshold"), policy
    return policy


def _cutoff_ns(val_timestamp: str | None) -> int | None:
    if not val_timestamp:
        return None
    d = datetime.strptime(val_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp()) * 1_000_000_000


class Encoder:
    """Encodes rows of one converted database. Statistics are computed once."""

    def __init__(self, con: duckdb.DuckDBPyConnection, db: Database):
        self.con = con
        self.db = db
        self.cutoff_ns = _cutoff_ns(db.val_timestamp)
        self.tables: dict[str, TableEncoding] = {}

        # The preprocessor drops constant columns (database tables only) before
        # computing any statistic, so emittability comes first.
        kept = {t.key: self._kept_columns(t) for t in db.all_tables()}

        dt_parts = []
        for t in db.all_tables():
            if t.split in ("val", "test"):
                continue
            where = self._stats_filter(t)
            for col, sql_type in kept[t.key]:
                if kind_of(sql_type) == "datetime":
                    dt_parts.append(
                        f"SELECT epoch_ns({ident(col)})::DOUBLE AS x FROM {t.sql} "
                        f"WHERE {ident(col)} IS NOT NULL{where.replace('WHERE', 'AND')}"
                    )
        if dt_parts:
            n, mean, var = con.execute(
                f"SELECT count(*), avg(x), var_pop(x) FROM ({' UNION ALL '.join(dt_parts)})"
            ).fetchone()
        else:
            n, mean, var = 0, None, None
        self.dt_mean = mean if mean is not None else 0.0
        self.dt_std = math.sqrt(var) if n and n > 1 and var is not None else 1.0

        train_stats: dict[str, dict[str, tuple[float, float]]] = {}
        for t in db.all_tables():
            if t.split in ("val", "test"):
                continue
            stats = self._column_stats(t, kept[t.key])
            if t.split == "train":
                train_stats[t.name] = stats
            self.tables[t.key] = self._table_encoding(t, kept[t.key], stats)
        for task in db.tasks.values():
            for split in ("val", "test"):
                if split in task.splits:
                    t = task.splits[split]
                    assert task.name in train_stats, (
                        f"task {task.name} has a {split} split but no train split "
                        f"to take its column statistics from"
                    )
                    self.tables[t.key] = self._table_encoding(
                        t, kept[t.key], train_stats[task.name]
                    )

    def _stats_filter(self, t: Table) -> str:
        if t.is_task or self.cutoff_ns is None or not t.time_col:
            return ""
        return f" WHERE epoch_ns({ident(t.time_col)}) <= {self.cutoff_ns}"

    def _kept_columns(self, t: Table) -> list[tuple[str, str]]:
        if t.is_task:
            return list(t.columns)
        structural = {t.pkey, t.time_col, *t.fkeys}
        candidates = [c for c, _ in t.columns if c not in structural]
        constant = set()
        if candidates:
            # polars' n_unique counts null as a value of its own.
            exprs = ", ".join(
                f"count(DISTINCT {ident(c)}) + (count(*) > count({ident(c)}))::INT"
                for c in candidates
            )
            counts = self.con.execute(f"SELECT {exprs} FROM {t.sql}").fetchone()
            constant = {c for c, n in zip(candidates, counts) if n <= 1}
        return [(c, ty) for c, ty in t.columns if c not in constant]

    def _column_stats(
        self, t: Table, columns: list[tuple[str, str]]
    ) -> dict[str, tuple[float, float]]:
        exprs, names = [], []
        for col, sql_type in columns:
            kind = kind_of(sql_type)
            c = ident(col)
            if kind == "bool":
                x = f"{c}::DOUBLE"
            elif kind == "number":
                x = f"{c}::DOUBLE"
            elif kind == "float":
                x = f"CASE WHEN isnan({c}) THEN NULL ELSE {c}::DOUBLE END"
            else:
                continue
            exprs += [f"avg({x})", f"stddev_samp({x})"]
            names.append(col)
        if not exprs:
            return {}
        row = self.con.execute(
            f"SELECT {', '.join(exprs)} FROM {t.sql}{self._stats_filter(t)}"
        ).fetchone()
        stats = {}
        for i, col in enumerate(names):
            mean, std = row[2 * i], row[2 * i + 1]
            mean = 0.0 if mean is None else mean
            std = 1.0 if std is None or std == 0.0 else std
            stats[col] = (mean, std)
        return stats

    def _needs_identifier(self, t: Table, columns: list[tuple[str, str]]) -> bool:
        if t.is_task:
            return False
        emitted = [c for c, _ in columns if c != t.pkey and c not in t.fkeys]
        policy = _identifier_policy()
        if policy == "none":
            return False
        if policy == "empty":
            return not emitted
        if policy == "empty_or_time":
            return not emitted or emitted == [t.time_col]
        if not emitted:
            return True
        cols = ", ".join(ident(c) for c in emitted)
        n, distinct = self.con.execute(
            f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT {cols} FROM "
            f"(SELECT {cols} FROM {t.sql} ORDER BY {ROW} LIMIT {IDENTIFIABILITY_SAMPLE})))"
            f" FROM (SELECT 1 FROM {t.sql} ORDER BY {ROW} LIMIT {IDENTIFIABILITY_SAMPLE})"
        ).fetchone()
        identifiability = 1.0 if n == 0 else distinct / n
        return identifiability < IDENTIFIABILITY_MIN

    def _table_encoding(self, t, columns, stats) -> TableEncoding:
        cell_columns = []
        for col, sql_type in columns:
            if col == t.pkey or col in t.fkeys:
                continue
            kind = kind_of(sql_type)
            if kind == "unsupported":
                raise NotImplementedError(
                    f"column {col} of {t.key} has type {sql_type}, which rt.sql "
                    f"does not encode yet"
                )
            cell_columns.append((col, kind))
        fk_columns = [
            c for c, ty in t.columns if c in t.fkeys and kind_of(ty) != "datetime"
        ]
        return TableEncoding(
            table=t,
            cell_columns=cell_columns,
            stats=stats,
            identifier=self._needs_identifier(t, columns),
            fk_columns=fk_columns,
        )

    def select_list(self, t: Table) -> str:
        """The SELECT list `encode_row` expects: `__row`, FK values, cell values."""
        enc = self.tables[t.key]
        parts = [ROW]
        for c in enc.fk_columns:
            parts.append(ident(c))
        for c, kind in enc.cell_columns:
            parts.append(f"epoch_ns({ident(c)})" if kind == "datetime" else ident(c))
        types = dict(t.columns)
        if t.time_col and kind_of(types.get(t.time_col, "")) == "datetime":
            parts.append(f"epoch_ns({ident(t.time_col)}) // 1000000000")
        else:
            parts.append("NULL::BIGINT")
        return ", ".join(parts)

    def decode_row(
        self, t: Table, values: tuple
    ) -> tuple[int, list, list[Cell], int | None]:
        """A fetched row -> (__row, FK values, cells, timestamp in epoch seconds)."""
        enc = self.tables[t.key]
        row = values[0]
        n_fk = len(enc.fk_columns)
        fks = list(values[1 : 1 + n_fk])
        cell_values = values[1 + n_fk : -1]
        ts = values[-1]
        if ts is not None:
            ts = max(min(ts, 2**31 - 1), -(2**31))

        cells = []
        for (col, kind), v in zip(enc.cell_columns, cell_values):
            name = f"{col} of {t.name}"
            if v is None:
                continue
            if kind in ("number", "float", "bool"):
                x = float(v)
                if math.isnan(x):
                    continue
                mean, std = enc.stats.get(col, (0.0, 1.0))
                cells.append(Cell(name, NUMBER, number=(x - mean) / std))
            elif kind == "datetime":
                cells.append(
                    Cell(name, DATETIME, datetime=(v - self.dt_mean) / self.dt_std)
                )
            else:
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="replace")
                cells.append(Cell(name, TEXT, text=str(v)))
        if enc.identifier:
            cells.append(
                Cell(
                    f"{IDENTIFIER_COL} of {t.name}",
                    NUMBER,
                    number=_identifier(t.name, row),
                )
            )
        return row, fks, cells, ts


def _identifier(table: str, row: int) -> float:
    seed = zlib.crc32(table.encode()) ^ (row * 0x9E3779B1 & 0xFFFFFFFF)
    return float(np.random.default_rng(seed).standard_normal())


class TextEmbedder:
    """Sentence embeddings for text cells and column names, cached by string.

    The preprocessor embeds every string once, ahead of time; here only the
    strings a context needs are embedded, when it needs them.
    """

    def __init__(self, embedder: str, device: str = "cpu"):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            f"sentence-transformers/{embedder}",
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32
            },
        )
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            emb = self.model.encode(
                missing, convert_to_numpy=True, show_progress_bar=False
            )
            for t, e in zip(missing, emb):
                self.cache[t] = e.astype(np.float32).astype(ml_dtypes.bfloat16)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 0))
