from datetime import datetime

import duckdb
import pandas as pd
import pytest

from rt.sql_context import query_context_nodes


def test_sql_context_history_cutoff_order_and_node_mapping():
    con = duckdb.connect()
    con.execute('CREATE TABLE drivers(driverId BIGINT)')
    con.execute('INSERT INTO drivers VALUES (42), (7)')
    con.execute('CREATE TABLE constructors(constructorId BIGINT)')
    con.execute('INSERT INTO constructors VALUES (8)')
    con.execute('CREATE TABLE circuits(circuitId BIGINT)')
    con.execute('INSERT INTO circuits VALUES (9)')
    con.execute('CREATE TABLE races(raceId BIGINT, circuitId BIGINT, date TIMESTAMP)')
    con.execute("INSERT INTO races VALUES (12,9,'2020-01-01'),(13,9,'2020-01-03'),(14,9,'2020-01-02')")
    con.execute('CREATE TABLE results(driverId BIGINT, raceId BIGINT, constructorId BIGINT, date TIMESTAMP)')
    con.execute("INSERT INTO results VALUES (42,12,8,'2020-01-01'),(42,13,8,'2020-01-03'),(7,12,8,'2020-01-01'),(42,14,8,'2020-01-02')")
    con.execute('CREATE TABLE qualifying AS SELECT * FROM results')
    con.execute('CREATE TABLE standings AS SELECT driverId, raceId, date FROM results')
    labels = pd.DataFrame({'node_idx': [501, 502, 503, 504], 'driverId': [42,42,42,7],
                           'date': pd.to_datetime(['2019-01-01','2020-01-02','2020-01-03','2019-01-01'])})
    con.register('sql_task_labels', labels)
    info = {f'{table}:Db': {'node_idx_offset': offset} for table, offset in
            [('drivers',100),('results',200),('races',300),('constructors',400),('circuits',450),
             ('qualifying',600),('standings',700)]}
    try:
        nodes = query_context_nodes(con, info, 42, datetime(2020,1,2), 1)
        # Physical positions map to offsets, even when primary keys aren't positions.
        assert nodes == [(100,1),(501,2),(603,2),(302,3),(400,3),(450,4),(703,2),(203,2)]
        assert query_context_nodes(con, info, 42, datetime(2018,1,1), 1) == [(100,1)]
        # Broad DB rows must not push labels or one history table behind another.
        more = query_context_nodes(con, info, 42, datetime(2020,1,2), 2)
        assert more[:2] == [(100,1),(501,2)]
        history = [idx for idx, _ in more if idx in (600,603,700,703,200,203)]
        assert history == [603,703,203,600,700,200]
        con.execute('DELETE FROM standings WHERE rowid = 0')
        uneven = query_context_nodes(con, info, 42, datetime(2020,1,2), 2)
        history = [idx for idx, _ in uneven if idx in (600,603,700,703,200,203)]
        assert history == [603,703,203,600,200]
        with pytest.raises(ValueError, match='positive'):
            query_context_nodes(con, info, 42, datetime(2020,1,2), 0)
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
