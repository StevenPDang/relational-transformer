import duckdb
import pandas as pd
import pytest

from rt.sql_context import historical_seed_orders

# The per-seed query this batched version replaced; kept as the reference order.
PER_SEED_ORDER_SQL = """
SELECT node_idx FROM labels
WHERE ts < ?
ORDER BY (key = ?) DESC, ts DESC, node_idx
"""


def test_batched_seed_order_matches_per_seed_queries():
    con = duckdb.connect()
    seeds = pd.DataFrame({
        "seed": [501, 502, 503, 504, 505, 506],
        "key": [42, 42, 7, 7, 42, 9],
        "ts": pd.to_datetime(["2019-01-01", "2020-01-02", "2019-06-01",
                              "2020-01-02", "2020-01-03", "2018-01-01"]),
    })
    try:
        orders = historical_seed_orders(seeds, limit=len(seeds))
        assert historical_seed_orders(seeds, limit=2)[505] == orders[505][:2]
        assert historical_seed_orders(seeds, limit=1) == {s: o[:1] for s, o in orders.items()}
        con.register("labels", seeds.rename(columns={"seed": "node_idx"}))
        for seed, key, ts in seeds.itertuples(index=False, name=None):
            expected = [n for n, in con.execute(PER_SEED_ORDER_SQL, [ts.to_pydatetime(), key]).fetchall()]
            assert orders[seed] == expected
        # Same driver first, then recency; same-time and later rows never appear.
        assert orders[505] == [502, 501, 504, 503, 506]
        assert orders[502] == [501, 503, 506]
        assert orders[506] == []
    finally:
        con.close()


def test_rust_sql_neighborhood_bypasses_bfs_and_filters_future_nodes(synthetic_dataset, tmp_path):
    import json
    import numpy as np
    from rt._rustler import preprocess
    from rt.data import RustlerDataset
    from rt.tasks import Task

    pre = tmp_path / 'pre'
    preprocess(str(synthetic_dataset), str(pre), skip_tasks=True)
    db = pre / 'synth'
    # Zero vectors suffice to verify sampling without downloading an embedder.
    text = json.loads((db / 'text.json').read_text())
    np.zeros((len(text), 8), dtype=np.uint16).tofile(db / 'text_emb_test.bin')
    info = json.loads((db / 'table_info.json').read_text())
    event = info['events:Db']
    start, n = event['node_idx_offset'], event['num_nodes']
    ds = RustlerDataset(
        tasks=[Task('synth', 'events', 'amount', 'reg')], pre_dir=str(pre),
        global_rank=0, local_rank=0, world_size=1, local_ctx_sizes=[128],
        bfs_widths=[8], num_walks=0, walk_length=0, prefer_latest=[True],
        mask_prob_max=0, embedding_model='test', d_text=8, shuffle_seed=0,
        context_seed=0, items_per_task=-1, quiet=True, bool_as_num=True,
        ignore_data_errors=False, skip_text_cols=False, mmap_populate=False,
        balance_labels=[False], timeout_per_item=3600, ablate_schema_semantics=False,
        vector_db_path=None, train_only_fallback=False,
    )
    sampler = ds.sampler
    baseline = ds._process_batch(sampler.batch_py(0, n, 128))
    with pytest.raises(ValueError, match='missing'):
        sampler.set_sql_contexts_py('synth', {})
    contexts = {i: [(i, 0), (start + n - 1, 1)] for i in range(start, start + n)}
    bad = dict(contexts)
    total = sum(t['num_nodes'] for t in info.values())
    bad[start] = [(total, 0)]
    with pytest.raises(ValueError, match='out of range'):
        sampler.set_sql_contexts_py('synth', bad)
    sampler.set_sql_contexts_py('synth', contexts)
    batch = ds._process_batch(sampler.batch_py(0, n, 128))
    assert batch['is_targets'][:, 0].all()
    assert (batch['is_targets'].sum(dim=1) == 1).all()
    assert (batch['node_idxs'][:, 0] == baseline['node_idxs'][:, 0]).all()
    for row in range(n):
        valid = ~batch['is_padding'][row]
        nodes = batch['node_idxs'][row][valid]
        assert ((nodes >= start) & (nodes < start + n)).all()
        assert (batch['timestamps'][row][valid] <= batch['timestamps'][row, 0]).all()
    # The ordinary BFS reaches user rows, while SQL contexts contain only events.
    baseline_nodes = baseline['node_idxs'][~baseline['is_padding']]
    assert ((baseline_nodes < start) | (baseline_nodes >= start + n)).any()
