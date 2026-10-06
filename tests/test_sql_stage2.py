"""Stage-2-only SQL parity against the aligned, graph-backed Rust sampler."""
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import duckdb
import pandas as pd
import numpy as np
import pytest

from rt.sql_context import SqlNeighborProvider, load_stage2_sql_provider



@pytest.fixture
def stage2(synthetic_dataset, tmp_path):
    import polars as pl
    from rt._rustler import preprocess
    from rt.data import RustlerDataset
    from rt.tasks import Task

    # Duplicate FK paths, equal timestamps and future rows exercise ordering.
    events_path = synthetic_dataset / 'db/events.parquet'
    events = pl.read_parquet(events_path).with_columns(
        pl.col('user_id').alias('other_user_id'),
        pl.col('timestamp').dt.truncate('2d'),
    )
    events.write_parquet(events_path)
    events.write_parquet(synthetic_dataset / 'db' / 'histories.parquet')
    import yaml
    manifest_path = synthetic_dataset / 'manifest.yaml'
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest['tables']['events']['fkeys']['other_user_id'] = 'users'
    manifest['tables']['histories'] = manifest['tables']['events'].copy()
    manifest_path.write_text(yaml.safe_dump(manifest))
    task_dir = synthetic_dataset / 'tasks' / 'labels'
    task_dir.mkdir(parents=True)
    (task_dir / 'manifest.yaml').write_text(yaml.safe_dump({
        'entity_col': 'user_id', 'entity_table': 'users', 'time_col': 'timestamp',
        'target_col': 'label', 'task_type': 'binary_classification',
    }))
    labels = events.select('user_id', 'timestamp').with_columns(pl.lit(1).alias('label'))
    for split in ('train', 'val', 'test'):
        labels.write_parquet(task_dir / f'{split}.parquet')
    pre = tmp_path / 'pre'
    preprocess(str(synthetic_dataset), str(pre), skip_tasks=False)
    db = pre / 'synth'
    text = json.loads((db / 'text.json').read_text())
    np.zeros((len(text), 8), dtype=np.uint16).tofile(db / 'text_emb_test.bin')
    database = tmp_path / 'source.duckdb'
    con = duckdb.connect(str(database))
    for table in ('events', 'users', 'histories'):
        frame = pd.read_parquet(synthetic_dataset / 'db' / f'{table}.parquet')
        con.register('input_rows', frame)
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM input_rows')
    con.close()
    ds = RustlerDataset(
        tasks=[Task('synth', 'events', 'amount', 'reg')], pre_dir=str(pre),
        global_rank=0, local_rank=0, world_size=1, local_ctx_sizes=[32],
        bfs_widths=[1], num_walks=80, walk_length=6, prefer_latest=[True],
        mask_prob_max=0, embedding_model='test', d_text=8, shuffle_seed=0,
        context_seed=13, items_per_task=-1, quiet=True, bool_as_num=True,
        ignore_data_errors=False, skip_text_cols=False, mmap_populate=False,
        balance_labels=[False], timeout_per_item=3600, ablate_schema_semantics=False,
        vector_db_path=None, train_only_fallback=False,
    )
    provider = load_stage2_sql_provider(pre, database, 'synth')
    info = json.loads((db / 'table_info.json').read_text())
    return ds, provider, info, database, pre


def assert_trace_equal(a, b):
    for field in ('visits', 'candidate_order', 'fallback_candidates', 'collections', 'cells'):
        assert a[field] == b[field], field
    for name, array in dict(a['sequence']).items():
        other = np.asarray(dict(b['sequence'])[name])
        array = np.asarray(array)
        assert array.dtype == other.dtype
        assert array.shape == other.shape
        assert array.tobytes() == other.tobytes(), name


def test_sql_stage2_trace_batch_temporal_budget_and_mask_parity(stage2):
    ds, provider, info, _, _ = stage2
    sampler = ds.sampler
    start = info['events:Db']['node_idx_offset']
    targets = [start, start + 8, start + 17]
    legacy = sampler.trace_py(0, targets[-1], 64, legacy=True)
    assert legacy['policy'] == 'legacy'
    baseline = [sampler.trace_py(0, target, 64) for target in targets]
    batch = dict(sampler.batch_py(0, 6, 64))
    assert provider.stats['query_count'] == 0
    sampler.set_sql_neighbor_provider_py('synth', provider)
    for target, expected in zip(targets, baseline):
        before = provider.stats['query_count']
        actual = sampler.trace_py(0, target, 64)
        assert provider.stats['query_count'] - before == 1
        assert actual['execution_route'] == 'sql_neighbors_batch_plan_replay'
        assert actual['execution_stats']['provider_calls'] == 1
        assert actual['execution_stats']['stage1_passes'] == 2
        assert_trace_equal(expected, actual)
        assert_trace_equal(actual, sampler.trace_py(0, target, 64))
        assert actual['collector'] == 'sql_neighbors'
        seq = dict(actual['sequence'])
        valid = ~seq['is_padding']
        assert valid.sum() <= 64
        assert seq['is_targets'].sum() == 1
        assert seq['node_idxs'][0] == target
        assert (seq['timestamps'][valid] <= seq['timestamps'][0]).all()
        assert len({(c[0], c[1]) for c in actual['cells']}) == len(actual['cells'])
    sql_batch = dict(sampler.batch_py(0, 6, 64))
    for name in batch:
        assert np.asarray(batch[name]).tobytes() == np.asarray(sql_batch[name]).tobytes(), name
    assert provider.stats['query_count'] > 0
    sampler.set_sql_neighbor_provider_py('synth', None)
    assert_trace_equal(baseline[-1], sampler.trace_py(0, targets[-1], 64))


def test_sql_stage2_thread_and_spawn_worker_determinism(stage2):
    _, provider, info, _, _ = stage2
    node = info['users:Db']['node_idx_offset']
    target = info['events:Db']['node_idx_offset'] + 17
    args = (node, target, target, None)
    expected = provider(*args)
    # Both FK columns must survive; DISTINCT would break width RNG parity.
    assert len(expected[1]) > len(set(expected[1]))
    restored = pickle.loads(pickle.dumps(provider))
    assert restored.stats['query_count'] == 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: restored(*args), range(4))) == [expected] * 4
    # Fresh interpreters exercise the same pickle boundary as spawn workers,
    # without requiring OS named semaphores (unavailable in some sandboxes).
    code = ('import duckdb,pandas; import sys,pickle; provider,args=pickle.loads(sys.stdin.buffer.read()); '
            'sys.stdout.buffer.write(pickle.dumps(provider(*args)))')
    def in_process(_):
        result = subprocess.run([sys.executable, '-c', code],
                                input=pickle.dumps((restored, args)),
                                capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        return pickle.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(in_process, range(2))) == [expected] * 2
    # Batched retrieval follows the same pickle/worker boundary.
    batched = restored.query_many([node, target], target, None)
    assert batched[node] == expected
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: restored.query_many([node, target], target, None),
                             range(4))) == [batched] * 4
    code = ('import duckdb,pandas; import sys,pickle; provider,args=pickle.loads(sys.stdin.buffer.read()); '
            'sys.stdout.buffer.write(pickle.dumps(provider.query_many(*args)))')
    args = ([node, target], target, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(in_process, range(2))) == [batched] * 2
    provider.close()


def test_sql_batch_map_matches_scalar_reference_and_queries_only_requests(stage2):
    ds, provider, info, _, _ = stage2
    scalar = pickle.loads(pickle.dumps(provider))
    scalar.batch_enabled = False
    assert scalar.query_many is None
    target = info['events:Db']['node_idx_offset'] + 17
    nodes = [target, info['users:Db']['node_idx_offset']]
    cutoff = 1704499200
    expected = {node: scalar(node, target, target, cutoff) for node in nodes}
    actual = provider.query_many(nodes + nodes, target, cutoff)
    assert actual == expected
    assert set(actual) == set(nodes)
    assert provider.stats['query_count'] == 1
    assert provider.stats['requested_nodes'] == len(nodes)
    assert provider.stats['setup_count'] == 1
    assert provider.query_many([], target, None) == {}
    assert provider.stats['query_count'] == 1
    with pytest.raises(ValueError, match='out of range'):
        provider.query_many([2**30], target, None)
    ds.sampler.set_sql_neighbor_provider_py('synth', scalar)
    reference = ds.sampler.trace_py(0, target, 64)
    assert reference['execution_route'] == 'sql_neighbors_scalar'
    ds.sampler.set_sql_neighbor_provider_py('synth', provider)
    assert_trace_equal(reference, ds.sampler.trace_py(0, target, 64))
    scalar.close()
    provider.close()


def test_sql_providers_coexist_with_live_default_duckdb_connection(stage2):
    _, provider, info, database, pre = stage2
    target = info['events:Db']['node_idx_offset'] + 17
    live = duckdb.connect(str(database), read_only=True)
    second = None
    try:
        expected = provider.query_many([target], target, None)
        # Validation and worker connections must use compatible DB configuration.
        second = load_stage2_sql_provider(pre, database, 'synth')
        assert second.query_many([target], target, None) == expected
        assert live.execute('SELECT count(*) FROM events').fetchone()[0] == 18
    finally:
        if second is not None:
            second.close()
        provider.close()
        live.close()


def test_comparison_cli_reports_batched_route_and_setup(stage2, tmp_path):
    import yaml
    from pathlib import Path
    from rt._rustler import preprocess

    _, _, _, database, pre = stage2
    source = Path(json.loads((pre / 'synth' / 'meta.json').read_text())['source'])
    manifest_path = source / 'manifest.yaml'
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest['name'] = 'rel-f1'
    manifest_path.write_text(yaml.safe_dump(manifest))
    task_dir = source / 'tasks' / 'driver-top3'
    task_dir.mkdir()
    spec = yaml.safe_load((source / 'tasks' / 'labels' / 'manifest.yaml').read_text())
    spec['target_col'] = 'qualifying'
    (task_dir / 'manifest.yaml').write_text(yaml.safe_dump(spec))
    for split in ('train', 'val', 'test'):
        frame = pd.read_parquet(source / 'tasks' / 'labels' / f'{split}.parquet')
        frame.rename(columns={'label': 'qualifying'}).to_parquet(task_dir / f'{split}.parquet')
    cli_pre = tmp_path / 'cli_pre'
    preprocess(str(source), str(cli_pre), skip_tasks=False)
    db_dir = cli_pre / 'rel-f1'
    text = json.loads((db_dir / 'text.json').read_text())
    np.zeros((len(text), 384), dtype=np.uint16).tofile(db_dir / 'text_emb_all-MiniLM-L12-v2.bin')
    output = tmp_path / 'comparison.json'
    result = subprocess.run([
        sys.executable, '-m', 'scripts.compare_sql_stage2',
        '--pre-dir', str(cli_pre), '--duckdb', str(database),
        '--items', '2', '--repeats', '2', '--ctx-size', '64',
        '--local-ctx-size', '32', '--width', '1', '--num-walks', '80',
        '--walk-length', '6', '--sql-route', 'batched', '--output', str(output),
    ], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    report = json.loads(output.read_text())
    assert report['passed'] and report['failures'] == []
    assert report['query_stats']['query_count'] == 4
    assert report['query_stats']['setup_count'] == 1
    for item in report['items']:
        assert item['aligned_vs_sql']['different_fields'] == []
        for timing in item['timings']['sql']:
            assert timing['query_count'] == 1
            assert timing['execution_stats']['provider_calls'] == 1
            assert timing['execution_stats']['stage1_passes'] == 2


def test_sql_stage2_rejects_source_revision_and_bad_neighbors(stage2):
    ds, provider, info, database, pre = stage2
    start = info['events:Db']['node_idx_offset']
    ds.sampler.set_sql_neighbor_provider_py('synth', lambda *args: ([2**30], []))
    with pytest.raises(ValueError, match='identity/order/target cutoff'):
        ds.sampler.trace_py(0, start + 17, 64)
    # Close before updating the source database; validation is fail-closed.
    provider.close()
    con = duckdb.connect(str(database))
    con.execute('UPDATE users SET name = ? WHERE rowid = 0', ['different revision'])
    con.close()
    with pytest.raises(ValueError, match='physical rows differ'):
        load_stage2_sql_provider(pre, database, 'synth')
