import json
import sys
import types

import duckdb  # noqa: F401 -- initialize before torch on macOS
import numpy as np
import pandas as pd
import pytest
import yaml

from rt.raw_eval import prepare_raw_eval


@pytest.fixture
def raw_f1(tmp_path):
    root = tmp_path / "raw"
    (root / "db").mkdir(parents=True)
    tables = {
        "drivers": pd.DataFrame({"driverId": [0, 1], "name": ["A", "B"]}),
        "constructors": pd.DataFrame({"constructorId": [0], "name": ["Team"]}),
        "circuits": pd.DataFrame({"circuitId": [0], "name": ["Track"]}),
        "races": pd.DataFrame({"raceId": [0, 1, 2], "circuitId": [0]*3,
                               "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-07-01"])}),
    }
    history = pd.DataFrame({"driverId": [0, 1]*3, "raceId": [0, 0, 1, 1, 2, 2],
                            "constructorId": [0]*6, "position": [1, 5, 5, 1, 1, 5],
                            "date": pd.to_datetime(["2020-01-01"]*2 + ["2021-01-01"]*2 + ["2022-07-01"]*2)})
    tables["results"] = history.assign(resultId=range(6))
    tables["qualifying"] = history.assign(qualifyId=range(6))
    tables["standings"] = history.drop(columns="constructorId").assign(driverStandingsId=range(6))
    specs = {}
    keys = {"drivers": "driverId", "constructors": "constructorId", "circuits": "circuitId",
            "races": "raceId", "results": "resultId", "qualifying": "qualifyId", "standings": "driverStandingsId"}
    for name, df in tables.items():
        df.to_parquet(root / "db" / f"{name}.parquet", index=False)
        specs[name] = {"pkey": keys[name], "time_col": "date" if "date" in df else None,
                       "fkeys": {col: parent for col, parent in [("driverId", "drivers"), ("raceId", "races"),
                                  ("constructorId", "constructors"), ("circuitId", "circuits")]
                                  if col in df and col != keys[name]}}
    (root / "manifest.yaml").write_text(yaml.safe_dump({"name": "rel-f1", "tables": specs,
        "val_timestamp": "2021-01-01", "test_timestamp": "2022-01-01"}))
    task_dir = root / "tasks" / "driver-top3"
    task_dir.mkdir(parents=True)
    (task_dir / "manifest.yaml").write_text(yaml.safe_dump({"name": "driver-top3", "kind": "forecast",
        "task_type": "binary_classification", "entity_table": "drivers", "entity_col": "driverId",
        "target_col": "qualifying", "time_col": "date", "timedelta": "30 days", "num_eval_timestamps": 1,
        "sql": "SELECT 1"}))
    for split, year in [("train", 2020), ("val", 2021), ("test", 2022)]:
        pd.DataFrame({"driverId": [0, 1], "date": pd.to_datetime([f"{year}-06-01"]*2),
                      "qualifying": [1, 0]}).to_parquet(task_dir / f"{split}.parquet", index=False)
    return root


@pytest.fixture
def zero_embedder(monkeypatch):
    class Embedder:
        def __init__(self, batch_size, embedding_model, device):
            pass

        def __call__(self, texts, device):
            return np.zeros((len(texts), 8), dtype=np.uint16)

    monkeypatch.setitem(sys.modules, "rt.embed", types.SimpleNamespace(TextEmbedder=Embedder))


def dataset_kwargs(pre_dir, database=None):
    from rt.tasks import Task

    return dict(tasks=[Task("rel-f1", "driver-top3", "qualifying", "clf", split="test")],
        pre_dir=pre_dir, global_rank=0, local_rank=0, world_size=1, local_ctx_sizes=[256],
        bfs_widths=[32], num_walks=10000, walk_length=20, prefer_latest=[True], mask_prob_max=0,
        embedding_model="test", d_text=8, shuffle_seed=0, context_seed=0, items_per_task=-1,
        quiet=True, bool_as_num=True, ignore_data_errors=False, skip_text_cols=False,
        mmap_populate=False, balance_labels=[False], timeout_per_item=5,
        ablate_schema_semantics=False, vector_db_path=None, train_only_fallback=False, sql_context_db=database)


def test_raw_sql_needs_no_traversal_graph_and_preserves_model_encoding(raw_f1, zero_embedder, tmp_path):
    import torch
    from rt.data import RustlerDataset
    from rt.sql_context import load_raw_rel_f1_contexts

    sql_dir, database, sql_times = prepare_raw_eval(str(raw_f1), tmp_path / "sql", sampler="sql",
        embedding_model="test", d_text=8, device="cpu")
    bfs_dir, no_database, bfs_times = prepare_raw_eval(str(raw_f1), tmp_path / "bfs", sampler="bfs",
        embedding_model="test", d_text=8, device="cpu")
    sql_path, bfs_path = tmp_path / "sql" / "rel-f1", tmp_path / "bfs" / "rel-f1"
    assert not (sql_path / "p2f_adj.rkyv").exists()
    assert (bfs_path / "p2f_adj.rkyv").exists()
    assert (sql_path / "nodes.rkyv").stat().st_size < (bfs_path / "nodes.rkyv").stat().st_size
    assert no_database is None
    assert "raw DuckDB import" in sql_times
    assert "cell encoding and graph construction" in bfs_times
    assert json.loads((sql_path / "text.json").read_text()) == json.loads((bfs_path / "text.json").read_text())
    with pytest.raises(ValueError, match="fresh path"):
        prepare_raw_eval(str(raw_f1), tmp_path / "sql", sampler="sql", embedding_model="test", d_text=8, device="cpu")
    with pytest.raises(ValueError, match="BFS needs graph"):
        RustlerDataset(**dataset_kwargs(sql_dir))

    contexts, orders = load_raw_rel_f1_contexts(sql_dir, database, 32)
    info = json.loads((sql_path / "table_info.json").read_text())
    test_start = info["driver-top3:Test"]["node_idx_offset"]
    # Earlier same-driver examples lead; no same-time test labels are candidates.
    assert all(idx < test_start or idx >= test_start + 2 for idx in orders[test_start])
    sql_ds = RustlerDataset(**dataset_kwargs(sql_dir, database))
    bfs_ds = RustlerDataset(**dataset_kwargs(bfs_dir))
    bfs_ds.sampler.set_sql_contexts_py("rel-f1", contexts)
    bfs_ds.sampler.set_sql_seed_order_py("rel-f1", orders)
    # Identical SQL selection through graph-backed vs graph-free encoders must
    # give identical model tensors, including foreign-key attention metadata.
    sql_batch = sql_ds._process_batch(sql_ds.sampler.batch_py(0, 2, 256))
    bfs_batch = bfs_ds._process_batch(bfs_ds.sampler.batch_py(0, 2, 256))
    for key, value in sql_batch.items():
        torch.testing.assert_close(value, bfs_batch[key], rtol=0, atol=0, equal_nan=True)
    assert (sql_batch["f2p_nbr_idxs"] >= 0).any()
    assert (sql_batch["is_targets"].sum(dim=1) == 1).all()
    # Graph-free SQL must be invariant to walk settings, even nonzero ones.
    random_kwargs = dataset_kwargs(sql_dir, database)
    random_kwargs.update(num_walks=0, walk_length=0)
    no_walk_ds = RustlerDataset(**random_kwargs)
    no_walk_batch = no_walk_ds._process_batch(no_walk_ds.sampler.batch_py(0, 2, 256))
    for key, value in sql_batch.items():
        torch.testing.assert_close(value, no_walk_batch[key], rtol=0, atol=0, equal_nan=True)
    invalid_orders = dict(orders)
    invalid_orders[test_start] = [test_start + 1]
    with pytest.raises(ValueError, match="earlier same-table"):
        sql_ds.sampler.set_sql_seed_order_py("rel-f1", invalid_orders)


@pytest.mark.parametrize("sampler", ["sql", "bfs"])
def test_raw_eval_cli_reports_full_split_and_preparation_runtime(raw_f1, zero_embedder, tmp_path, monkeypatch, sampler):
    import torch
    import scripts.eval as cli

    class MockModel:
        def to(self, dtype):
            return self

        def eval(self):
            return self

        def predict(self, batch, ctx_sizes, device, task, bool_as_num):
            # Synthetic scores exercise reporting, not actual model accuracy.
            return {ctx: (batch["number_values"][:, :ctx, 0].float()
                          * batch["is_targets"][:, :ctx]).sum(1) for ctx in ctx_sizes}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli, "load_rt_model", lambda *a, **kw: (MockModel(), {
        "task_type": "clf", "embedding_model": "test", "d_text": 8}))
    out = tmp_path / f"report-{sampler}"
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", "mock", "--raw-dataset", str(raw_f1),
        "--prepare-dir", str(tmp_path / f"prep-{sampler}"), "--sampler", sampler,
        "--out-dir", str(out), "--ctx-size", "256", "--local-ctx-size", "128",
        "--tokens-per-gpu", "512", "--num-workers", "0"])
    cli.main()
    predictions = pd.read_csv(out / "rel-f1__driver-top3.csv")
    assert len(predictions) == 2
    assert set(predictions.driverId) == {0, 1}
    runtime = json.loads((out / "runtime.json").read_text())
    assert runtime["sampler"] == sampler
    assert "raw data preparation" in runtime["seconds"]
    assert runtime["seconds"]["total"] >= sum(
        value for name, value in runtime["seconds"].items() if name != "total")
    assert "text embeddings" in runtime["preparation_seconds"]
