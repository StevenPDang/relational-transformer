import duckdb
import pandas as pd
import pytest

from rt.sql_context import SqlContexts, historical_seed_orders

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
        assert historical_seed_orders(seeds, limit=1).to_dict() == {s: o[:1] for s, o in orders.to_dict().items()}
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

    def install(contexts):
        seeds = sorted(contexts)
        rows = np.array([(seed, node, depth) for seed in seeds for node, depth in contexts[seed]],
                        dtype=np.int64).reshape(-1, 3)
        csr = SqlContexts.from_rows(seeds, *rows.T)
        sampler.set_sql_contexts_py('synth', csr.seeds, csr.offsets, csr.nodes, csr.depths)

    with pytest.raises(ValueError, match='missing'):
        install({})
    contexts = {i: [(i, 0), (start + n - 1, 1)] for i in range(start, start + n)}
    bad = dict(contexts)
    total = sum(t['num_nodes'] for t in info.values())
    bad[start] = [(total, 0)]
    with pytest.raises(ValueError, match='out of range'):
        install(bad)
    with pytest.raises(ValueError, match='strictly increasing'):
        sampler.set_sql_contexts_py('synth', np.array([start + 1, start], dtype=np.int32),
                                    np.zeros(3, dtype=np.int64), np.zeros(0, dtype=np.int32),
                                    np.zeros(0, dtype=np.uint8))
    install(contexts)
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


def test_seed_order_chunks_and_limits_match_per_seed_queries():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    seeds = pd.DataFrame({
        "seed": rng.permutation(np.arange(1000, 1000 + n)),
        "key": rng.integers(0, 12, n),
        # Few distinct days, so many rows tie on time.
        "ts": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
    })
    con = duckdb.connect()
    try:
        con.register("labels", seeds.rename(columns={"seed": "node_idx"}))
        for limit, chunk in [(5, 7), (40, 1), (n, None)]:
            orders = historical_seed_orders(seeds, limit=limit, chunk=chunk)
            assert orders.seeds.tolist() == sorted(seeds["seed"])
            for seed, key, ts in seeds.itertuples(index=False, name=None):
                expected = [r for r, in con.execute(PER_SEED_ORDER_SQL, [ts.to_pydatetime(), key]).fetchall()]
                assert orders[seed] == expected[:limit]
    finally:
        con.close()


def test_node_text_ids_cover_only_the_requested_nodes(synthetic_dataset, tmp_path):
    import json

    from rt._rustler import node_text_ids, preprocess

    pre = tmp_path / 'pre'
    preprocess(str(synthetic_dataset), str(pre), skip_tasks=True)
    db = pre / 'synth'
    texts = json.loads((db / 'text.json').read_text())
    total = sum(t['num_nodes'] for t in json.loads((db / 'table_info.json').read_text()).values())
    values, names = node_text_ids(str(db), list(range(total)))
    assert values == sorted(set(values)) and names == sorted(set(names))
    assert set(values) | set(names) <= set(range(len(texts)))
    assert names, 'every cell has a column-name embedding'
    one_values, one_names = node_text_ids(str(db), [0])
    assert set(one_values) <= set(values) and set(one_names) <= set(names)
    with pytest.raises(IndexError, match='out of range'):
        node_text_ids(str(db), [total + 5])


def test_path_seed_orders_rank_shared_rows_first_and_respect_time():
    import numpy as np

    from rt.sql_context import SqlContexts, path_seed_orders

    info = {'rows:Db': {'node_idx_offset': 0, 'num_nodes': 100}, 'task:Train': {'node_idx_offset': 500, 'num_nodes': 6}}
    seeds = pd.DataFrame({
        'seed': [500, 501, 502, 503, 504, 505],
        'key': [1, 2, 3, 4, 5, 6],
        'ts': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06']),
    })
    # 505 shares rows 10 and 11 with 501, row 12 with 503; 504 shares nothing with it.
    # Row 50 is in every neighborhood, a hub; task-row nodes (>= 500) are ignored.
    hoods = {500: [50], 501: [10, 11, 50], 502: [50, 20], 503: [12, 50], 504: [30, 50, 505], 505: [10, 11, 12, 50]}
    rows = [(s, n) for s in sorted(hoods) for n in hoods[s]]
    contexts = SqlContexts.from_rows(sorted(hoods), [s for s, _ in rows], [n for _, n in rows], [2] * len(rows))
    orders = path_seed_orders(contexts, seeds, info, limit=10)
    # Shared rows first (two shared rows beat one), then recency for the rest.
    assert orders[505] == [501, 503, 504, 502, 500]
    assert orders[500] == []  # nothing is earlier
    assert all(c < s for s in hoods for c in orders[s])  # earlier rows only (same order as time here)
    # A hub cap that drops every shared row falls back to recency entirely.
    assert path_seed_orders(contexts, seeds, info, limit=10, max_df=1)[505] == [504, 503, 502, 501, 500]
    assert path_seed_orders(contexts, seeds, info, limit=2)[505] == [501, 503]
    with pytest.raises(ValueError, match='same task rows'):
        path_seed_orders(contexts, seeds.iloc[:5], info)
