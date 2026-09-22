# Load a RelBench database into DuckDB

Raw RelBench-format datasets contain `manifest.yaml` and one Parquet file per
database table under `db/`. From the repository root, run the Pixi task to copy
those tables into a persistent DuckDB file:

```bash
pixi run import-duckdb \
  --dataset stanford-star/relbench/rel-amazon
```

The default output is `data/duckdb/rel-amazon.duckdb`, keeping persistent
databases together and named from each dataset's manifest. To use another
collection directory while retaining automatic filenames:

```bash
pixi run import-duckdb \
  --dataset stanford-star/relbench/rel-f1 \
  --output-dir ~/scratch/relbench-duckdb
```

Use `--output path/to/name.duckdb` when you need an exact filename. `--output`
and `--output-dir` are mutually exclusive.

`--dataset` also accepts a local directory containing `manifest.yaml` and
`db/*.parquet`. For a Hub dataset, the script uses the same dataset resolver as
[preprocessing](preprocess.md); add `--revision <commit-or-tag>` to pin a version.
The output path can be anywhere writable and its parent directory is created if
needed.

Open the file later with DuckDB:

```python
import duckdb

con = duckdb.connect("data/duckdb/rel-f1.duckdb", read_only=True)
print(con.execute("SHOW TABLES").fetchall())
con.close()
```

The import copies only the raw `db/` tables, preserving their Parquet column
types. Task label splits under `tasks/` are not imported. The manifest remains
the source of primary-key, foreign-key, and time-column metadata; the script does
not create SQL constraints. Importing into an existing file is allowed if the
new table names do not conflict. A conflict stops the import and rolls back all
tables from that attempt.

## Check one task before comparing SQL contexts

For `rel-f1/driver-top3`, run the row-alignment check against the same
preprocessed data used by evaluation. It reads `meta.json` to find the RelBench
source, then loads the task and raw entity table through the RelBench API:

```bash
pixi run python -m scripts.check_task_alignment \
  --pre-dir stanford-star/relbench-preprocessed \
  --task driver-top3
```

The check reads every test row's entity and timestamp from the RelBench
loader, verifies its `node_idx` range against `table_info.json`, and confirms
that its entity ID resolves to the same zero-based raw entity row in DuckDB.
It fails if the DuckDB import has a different entity row order. The evaluator
uses `node_idx - node_idx_offset` to recover the test row position.

Rust-sampler baseline command for the later score comparison:

```bash
pixi run eval --checkpoint stanford-star/rt-j/classification \
  --pre-dir stanford-star/relbench-preprocessed \
  --tasks rel-f1/driver-top3 --out-dir eval_out/driver-top3-rust \
  --ctx-size 8192 --local-ctx-size 256 --bfs-width 32 \
  --num-walks 10000 --walk-length 20 --prefer-latest --shuffle-seed 0
```

Record the AUROC printed by that run alongside the checkpoint revision, data
revision, and `n` before comparing a SQL neighborhood. No baseline score is
recorded here until the command has actually run.
