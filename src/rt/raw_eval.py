"""Prepare evaluation of a raw RelBench dataset, with or without a traversal graph."""

from __future__ import annotations

import json
import time
from pathlib import Path

NODE_TASK_TYPES = {"binary_classification": "clf", "regression": "reg"}


def raw_dataset_name(dataset):
    """The dataset's manifest name: read locally when possible, else the Hub spec's last part."""
    import yaml

    manifest = Path(dataset).expanduser() / "manifest.yaml"
    if manifest.is_file():
        return yaml.safe_load(manifest.read_text())["name"]
    return Path(str(dataset).rstrip("/")).name


def raw_task_names(raw_dir, db_name, task_type=None, selectors=None):
    """Node-level tasks of a raw dataset, of ``task_type`` (clf/reg) and matching
    ``selectors`` ('db' or 'db/task'), as eval selects them."""
    import yaml

    names = []
    for manifest in sorted(Path(raw_dir, "tasks").glob("*/manifest.yaml")):
        kind = NODE_TASK_TYPES.get(yaml.safe_load(manifest.read_text()).get("task_type"))
        name = manifest.parent.name
        if kind is None or (task_type in ("clf", "reg") and kind != task_type):
            continue
        if selectors and db_name not in selectors and f"{db_name}/{name}" not in selectors:
            continue
        names.append(name)
    return names


def select_raw_rules(raw_dir, database, db_name, task_name, out, max_seeds=50_000):
    """MetaSieve rules for one task from raw manifests and train labels.

    Part of a timed one-shot run, so it reads only the task's train table and
    the DuckDB import, never the whole database into pandas.
    """
    import duckdb
    import pandas as pd
    import yaml

    from rt import metasieve

    tick = time.perf_counter()
    raw_dir = Path(raw_dir)
    schema = metasieve.manifest_schema(yaml.safe_load((raw_dir / "manifest.yaml").read_text()))
    task_manifest = yaml.safe_load((raw_dir / "tasks" / task_name / "manifest.yaml").read_text())
    task_manifest.setdefault("name", task_name)
    info = metasieve.manifest_task_info(schema, db_name, task_manifest)
    train = pd.read_parquet(raw_dir / "tasks" / task_name / "train.parquet")
    scored = metasieve.cap_seeds(train, info["target_col"], info["task_type"], max_seeds)
    seeds = pd.DataFrame({"seed": scored.index.to_numpy(), "key": scored[info["entity_col"]].to_numpy(),
                          "ts": pd.to_datetime(scored[info["time_col"]]).to_numpy()})
    con = duckdb.connect(str(database), read_only=True)
    try:
        rules = metasieve.select_metapaths(con, schema, info, seeds, scored[info["target_col"]].to_numpy())
    finally:
        con.close()
    rules.update(source=str(raw_dir), train_rows=len(train), seconds=round(time.perf_counter() - tick, 2))
    path = Path(out) / f"{db_name}-{task_name}" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2, default=str) + "\n")
    return rules


def plan_sql_contexts(out_dir, raw_dir, database, db_name, tasks, *, width, rules_spec, timings,
                      max_seeds=50_000):
    """Rules, neighborhoods and seed orders for ``tasks``, before any embedding.

    Saves each task's contexts for the sampler and returns the text-embedding
    rows the test split can look up (:func:`rt.sql_context.reachable_nodes`).
    Without ``rules_spec`` the rules are selected here, under ``out_dir/metasieve``.
    """
    import duckdb
    import numpy as np

    from rt._rustler import node_text_ids
    from rt.sql_context import (load_metasieve_rules, metasieve_context_nodes, path_seed_orders,
                                raw_task_seeds, reachable_nodes, save_contexts)

    db_dir = Path(out_dir) / db_name
    info = json.loads((db_dir / "table_info.json").read_text())
    needed = [np.zeros(0, dtype="int64")]
    for key in ("MetaSieve selection", "SQL contexts and seed orders", "context text ids"):
        timings.setdefault(key, 0.0)
    for task_name in tasks:
        tick = time.perf_counter()
        if rules_spec:
            rules = load_metasieve_rules(rules_spec, db_name, task_name)
        else:
            rules = select_raw_rules(raw_dir, database, db_name, task_name, Path(out_dir) / "metasieve",
                                     max_seeds=max_seeds)
        timings["MetaSieve selection"] += time.perf_counter() - tick

        tick = time.perf_counter()
        seeds = raw_task_seeds(out_dir, db_name, task_name, rules["task"])
        con = duckdb.connect(str(database), read_only=True)
        try:
            contexts = metasieve_context_nodes(con, info, rules, seeds, width).with_seeds()
        finally:
            con.close()
        orders = path_seed_orders(contexts, seeds, info)
        save_contexts(db_dir, task_name, rules, width, contexts, orders)
        timings["SQL contexts and seed orders"] += time.perf_counter() - tick

        tick = time.perf_counter()
        test = info[f"{task_name}:Test"]
        test_seeds = np.arange(test["node_idx_offset"], test["node_idx_offset"] + test["num_nodes"])
        values, names = node_text_ids(str(db_dir), reachable_nodes(contexts, orders, test_seeds).tolist())
        needed += [np.asarray(values, dtype="int64"), np.asarray(names, dtype="int64")]
        timings["context text ids"] += time.perf_counter() - tick
    return np.unique(np.concatenate(needed))


def write_embeddings(path, texts, ids, embedder, d_text, device):
    """Embed ``texts[ids]`` (all texts if ``ids`` is None) into a full-size table.

    Rows outside ``ids`` stay zero in a sparse file: the sampler only looks up
    rows its contexts reach, so their values never enter the model.
    """
    import numpy as np

    subset = texts if ids is None else [texts[i] for i in ids]
    emb = embedder(subset, device=device) if subset else np.zeros((0, d_text))
    if emb.shape != (len(subset), d_text):
        raise ValueError(f"Embedding shape {emb.shape} differs from checkpoint dimension {d_text}")
    if ids is None:
        emb.tofile(path)
        return
    with open(path, "wb") as f:
        f.truncate(len(texts) * d_text * 2)  # bf16; unwritten rows read as zeros
    table = np.memmap(path, dtype=np.uint16, mode="r+", shape=(len(texts), d_text))
    table[np.asarray(ids, dtype="int64")] = np.asarray(emb).view(np.uint16)
    table.flush()
    del table


def prepare_raw_eval(dataset, out_dir, *, sampler, embedding_model, d_text, device, batch_size=1024,
                     sql=None):
    """Always build fresh artifacts; include input, encoding and embedding costs.

    BFS encodes every cell, builds the usual graph and embeds every text. SQL
    encodes every cell and imports raw tables into DuckDB without traversal
    edges. Given ``sql`` (task_type, selectors, width, rules, max_seeds), it
    then selects rules (unless given), builds each task's neighborhoods and seed
    orders, and embeds only the text those can reach for the test split.
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
    name = manifest["name"]
    if not any((raw_dir / "tasks").glob("*/manifest.yaml")):
        raise ValueError(f"Raw dataset {name} has no task tables to evaluate")
    timings["raw input resolution"] = time.perf_counter() - tick

    out_dir.mkdir(parents=True)
    tick = time.perf_counter()
    preprocess(str(raw_dir), str(out_dir), source=str(raw_dir), skip_graph=sampler == "sql")
    timings["cell encoding" if sampler == "sql" else "cell encoding and graph construction"] = time.perf_counter() - tick
    db_dir = out_dir / name
    meta_path = db_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["raw_dataset_dir"] = str(raw_dir)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    database = None
    if sampler == "sql":
        import duckdb

        tick = time.perf_counter()
        database = out_dir / f"{name}.duckdb"
        con = duckdb.connect(str(database))
        try:
            for parquet in sorted((raw_dir / "db").glob("*.parquet")):
                table = '"' + parquet.stem.replace('"', '""') + '"'
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [str(parquet)])
        finally:
            con.close()
        timings["raw DuckDB import"] = time.perf_counter() - tick

    ids = None
    if sampler == "sql" and sql is not None:
        tasks = raw_task_names(raw_dir, name, sql.get("task_type"), sql.get("selectors"))
        if not tasks:
            raise ValueError(f"No {sql.get('task_type') or 'node'} tasks of {name} match {sql.get('selectors')}")
        ids = plan_sql_contexts(out_dir, raw_dir, database, name, tasks, width=sql["width"],
                                rules_spec=sql.get("rules"), timings=timings,
                                max_seeds=sql.get("max_seeds", 50_000))

    tick = time.perf_counter()
    from rt.embed import TextEmbedder

    texts = json.loads((db_dir / "text.json").read_text())
    embedder = TextEmbedder(batch_size, embedding_model, device)
    write_embeddings(db_dir / f"text_emb_{embedding_model}.bin", texts, ids, embedder, d_text, device)
    # Release the embedder before inference to avoid retaining its GPU memory.
    del embedder
    timings["text embeddings"] = time.perf_counter() - tick
    meta["text_embeddings"] = {embedding_model: {
        "file": f"text_emb_{embedding_model}.bin", "d_text": d_text,
        "embedded_texts": len(texts) if ids is None else int(len(ids)), "texts": len(texts),
    }}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return str(out_dir), str(database) if database else None, timings
