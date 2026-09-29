"""Prepare rel-f1 evaluation from raw tables, with or without a traversal graph."""

from __future__ import annotations

import json
import time
from pathlib import Path


def prepare_raw_eval(dataset, out_dir, *, sampler, embedding_model, d_text, device, batch_size=1024):
    """Always build fresh artifacts; include input, encoding and embedding costs.

    SQL imports raw tables into DuckDB and omits traversal edges. BFS builds the
    usual graph. Both encode the same cells and embed their text from scratch.
    """
    import yaml

    from rt._rustler import preprocess

    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        raise ValueError(f"Preparation directory already exists: {out_dir}; use a fresh path")
    if sampler not in ("sql", "bfs"):
        raise ValueError(f"Unknown raw sampler: {sampler}")

    timings = {}
    tick = time.perf_counter()
    raw_dir = Path(dataset).expanduser()
    if not (raw_dir / "manifest.yaml").is_file():
        from relbench.hf import download_dataset_dir

        raw_dir = download_dataset_dir(dataset)
    raw_dir = raw_dir.resolve()
    manifest = yaml.safe_load((raw_dir / "manifest.yaml").read_text())
    if manifest["name"] != "rel-f1":
        raise ValueError("Raw evaluation currently supports only rel-f1/driver-top3")
    if not (raw_dir / "tasks" / "driver-top3" / "manifest.yaml").is_file():
        raise ValueError("Raw dataset needs the driver-top3 task tables")
    timings["raw input resolution"] = time.perf_counter() - tick

    out_dir.mkdir(parents=True)
    tick = time.perf_counter()
    preprocess(str(raw_dir), str(out_dir), source=str(raw_dir), skip_graph=sampler == "sql")
    timings["cell encoding" if sampler == "sql" else "cell encoding and graph construction"] = time.perf_counter() - tick
    db_dir = out_dir / "rel-f1"
    meta_path = db_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["raw_dataset_dir"] = str(raw_dir)

    tick = time.perf_counter()
    from rt.embed import TextEmbedder

    texts = json.loads((db_dir / "text.json").read_text())
    embedder = TextEmbedder(batch_size, embedding_model, device)
    embeddings = embedder(texts, device=device)
    if embeddings.shape != (len(texts), d_text):
        raise ValueError(f"Embedding shape {embeddings.shape} differs from checkpoint dimension {d_text}")
    embeddings.tofile(db_dir / f"text_emb_{embedding_model}.bin")
    # Release the embedder before inference to avoid retaining its GPU memory.
    del embedder, embeddings
    timings["text embeddings"] = time.perf_counter() - tick
    meta["text_embeddings"] = {embedding_model: {
        "file": f"text_emb_{embedding_model}.bin", "d_text": d_text,
    }}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    database = None
    if sampler == "sql":
        import duckdb

        tick = time.perf_counter()
        database = out_dir / "rel-f1.duckdb"
        con = duckdb.connect(str(database))
        try:
            for parquet in sorted((raw_dir / "db").glob("*.parquet")):
                table = '"' + parquet.stem.replace('"', '""') + '"'
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [str(parquet)])
        finally:
            con.close()
        timings["raw DuckDB import"] = time.perf_counter() - tick
    return str(out_dir), str(database) if database else None, timings
