"""Convert RelBench-format databases to DuckDB files.

    python -m examples.preprocess.to_duckdb data/relbench-v1 --out-dir data/relbench-duckdb

The input is one database (a directory holding manifest.yaml) or a collection
of them (a directory of such directories). Each database becomes one
`<out-dir>/<name>.duckdb`:

- every `db/<table>.parquet` is the table `<table>`;
- every `tasks/<task>/<split>.parquet` is the table `"<task>".<split>`, one
  schema per task;
- every table gains a leading `__row` column, the row's position in its
  parquet file. Keys in RelBench format are row positions (a foreign key value
  is the parent's row number, see notebooks/byod.ipynb), so `__row` is what a
  key joins to, and it is the node order `preprocess` assigns;
- `__row` and every foreign key column get an index, for point lookups;
- `_rt_manifest` holds the database and task manifests as JSON, so the file
  stands without the source directory;
- with `--fk-sorted`, every table with foreign keys also gets one copy per FK,
  `"__by_fk"."<table>:<fk>"` (`<table>` is `results` or `<task>/<split>`),
  sorted by that FK and then newest first. DuckDB keeps min/max statistics
  per block of rows, so a lookup of a few FK values in a sorted copy skips
  the blocks that cannot hold them; `rt.sql` reads a copy for children
  lookups whenever it exists. The cost is one more copy of each child table
  per FK.

No PRIMARY KEY or FOREIGN KEY constraints are declared: DuckDB enforces them,
and RelBench data has dangling keys that preprocessing tolerates.
"""

import argparse
import json
from pathlib import Path

import duckdb
import yaml


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def literal(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def key_columns(con, table: str) -> set[str]:
    """Columns a key can be read from: not lists (a recommendation task's targets)."""
    return {
        name
        for name, type_, *_ in con.execute(f"DESCRIBE {table}").fetchall()
        if not type_.endswith("]")
    }


def load_parquet(con, table: str, path: Path, fkeys: list[str], index: bool) -> int:
    con.execute(
        f"CREATE TABLE {table} AS "
        f"SELECT file_row_number AS __row, * EXCLUDE (file_row_number) "
        f"FROM read_parquet({literal(str(path))}, file_row_number = true) "
        f"ORDER BY __row"
    )
    if index:
        flat = table.replace('"', "").replace(".", "_")
        con.execute(f"CREATE INDEX {ident(f'idx_{flat}___row')} ON {table} (__row)")
        columns = key_columns(con, table)
        for col in fkeys:
            if col in columns:
                con.execute(
                    f"CREATE INDEX {ident(f'idx_{flat}_{col}')} ON {table} ({ident(col)})"
                )
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


FK_SCHEMA = '"__by_fk"'


def fk_sorted_copy(con, table: str, key: str, col: str, time_col: str | None) -> None:
    """A copy of `table` sorted by FK `col`, newest first within each value."""
    columns = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    if col not in key_columns(con, table):
        return
    order = [ident(col)]
    if time_col and time_col in columns:
        order.append(f"{ident(time_col)} DESC NULLS LAST")
    order.append("__row DESC")
    con.execute(
        f"CREATE TABLE {FK_SCHEMA}.{ident(f'{key}:{col}')} AS "
        f"SELECT * FROM {table} ORDER BY {', '.join(order)}"
    )


def convert(
    db_dir: Path, out_dir: Path, *, index: bool, fk_sorted: bool, overwrite: bool
) -> Path:
    manifest = yaml.safe_load((db_dir / "manifest.yaml").read_text())
    name = manifest["name"]
    out = out_dir / f"{name}.duckdb"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    # Build beside the target and rename at the end, so an interrupted run never
    # leaves a half-written file under the final name.
    tmp = out.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)

    con = duckdb.connect(str(tmp))
    con.execute(
        "CREATE TABLE _rt_manifest (scope VARCHAR, name VARCHAR, manifest JSON)"
    )
    con.execute(
        "INSERT INTO _rt_manifest VALUES ('db', ?, ?)", [name, json.dumps(manifest)]
    )

    # (table SQL, key, FK columns, time column) of every table loaded
    loaded: list[tuple[str, str, list[str], str | None]] = []
    specs = manifest.get("tables") or {}
    for path in sorted((db_dir / "db").glob("*.parquet")):
        spec = specs.get(path.stem) or {}
        fkeys = list(spec.get("fkeys") or {})
        n = load_parquet(con, ident(path.stem), path, fkeys, index)
        loaded.append((ident(path.stem), path.stem, fkeys, spec.get("time_col")))
        print(f"  {name}: {path.stem} ({n:,} rows)", flush=True)

    for task_dir in sorted((db_dir / "tasks").glob("*")):
        tm_path = task_dir / "manifest.yaml"
        if not tm_path.is_file():
            continue
        tm = yaml.safe_load(tm_path.read_text())
        con.execute(
            "INSERT INTO _rt_manifest VALUES ('task', ?, ?)",
            [task_dir.name, json.dumps(tm)],
        )
        fkeys = [
            tm[col]
            for col in ("entity_col", "src_entity_col", "dst_entity_col")
            if tm.get(col)
        ]
        con.execute(f"CREATE SCHEMA {ident(task_dir.name)}")
        for path in sorted(task_dir.glob("*.parquet")):
            table = f"{ident(task_dir.name)}.{ident(path.stem)}"
            n = load_parquet(con, table, path, fkeys, index)
            loaded.append(
                (table, f"{task_dir.name}/{path.stem}", fkeys, tm.get("time_col"))
            )
            print(f"  {name}: {task_dir.name}/{path.stem} ({n:,} rows)", flush=True)

    if fk_sorted:
        con.execute(f"CREATE SCHEMA {FK_SCHEMA}")
        for table, key, fkeys, time_col in loaded:
            for col in fkeys:
                fk_sorted_copy(con, table, key, col, time_col)
        n = con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE schema_name = '__by_fk'"
        ).fetchone()[0]
        print(f"  {name}: {n} FK-sorted copies", flush=True)

    con.close()
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "src", type=Path, help="a database directory, or a directory of them"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--no-index", action="store_true", help="skip the __row and FK indexes"
    )
    parser.add_argument(
        "--fk-sorted",
        action="store_true",
        help="also write a copy of each table per FK, sorted by it, for children lookups",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser()
    if (src / "manifest.yaml").is_file():
        db_dirs = [src]
    else:
        db_dirs = sorted(p.parent for p in src.glob("*/manifest.yaml"))
    assert db_dirs, f"{src} holds no manifest.yaml and no <database>/manifest.yaml"

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for db_dir in db_dirs:
        out = convert(
            db_dir,
            out_dir,
            index=not args.no_index,
            fk_sorted=args.fk_sorted,
            overwrite=args.overwrite,
        )
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
