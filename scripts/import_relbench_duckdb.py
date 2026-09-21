"""Copy raw RelBench Parquet tables into a persistent DuckDB database.

Run from the repository root::

    pixi run import-duckdb \
      --dataset stanford-star/relbench/rel-amazon

``--dataset`` can also be a local RelBench-format directory containing
``manifest.yaml`` and ``db/*.parquet``. By default, databases are organized at
``data/duckdb/<dataset>.duckdb``; ``--output`` still accepts an exact path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/duckdb")


def organized_output_path(dataset_name: str, output_dir: Path) -> Path:
    """Return the conventional database path for a manifest dataset name."""
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError(f"Unsafe dataset name in manifest: {dataset_name!r}")
    return output_dir / f"{dataset_name}.duckdb"


def load_dataset(dataset_dir: Path, output: Path) -> list[tuple[str, int]]:
    """Copy database tables into ``output`` and return their row counts.

    The import is transactional: an existing table causes an error without
    changing an existing database.
    """
    import duckdb

    if not (dataset_dir / "manifest.yaml").is_file():
        raise FileNotFoundError(f"Missing RelBench manifest: {dataset_dir / 'manifest.yaml'}")
    files = sorted((dataset_dir / "db").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No database Parquet files in {dataset_dir / 'db'}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output))
    try:
        con.execute("BEGIN TRANSACTION")
        counts = []
        for parquet in files:
            name = '"' + parquet.stem.replace('"', '""') + '"'
            con.execute(
                f"CREATE TABLE {name} AS SELECT * FROM read_parquet(?)",
                [str(parquet.resolve())],
            )
            count = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            counts.append((parquet.stem, count))
        con.execute("COMMIT")
        return counts
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy raw RelBench Parquet tables into a persistent DuckDB file."
    )
    parser.add_argument("--dataset", required=True, help="Local RelBench directory or Hub spec")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="Exact persistent .duckdb path (backward-compatible override)",
    )
    destination.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Organized database directory (default: data/duckdb)",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision for Hub datasets")
    args = parser.parse_args()

    from scripts.preprocess import dataset_name, resolve_dataset_dir

    dataset_dir = resolve_dataset_dir(args.dataset, revision=args.revision)
    output = args.output or organized_output_path(dataset_name(dataset_dir), args.output_dir)
    counts = load_dataset(dataset_dir, output)
    print(f"Imported {len(counts)} tables into {output}:")
    for name, count in counts:
        print(f"  {name}: {count} rows")


if __name__ == "__main__":
    main()
