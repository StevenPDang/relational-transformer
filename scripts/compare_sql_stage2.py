#!/usr/bin/env python
"""Compare prepared rel-f1/driver-top3 legacy BFS, aligned BFS, and Stage 2 SQL.

Checkpoint-independent CPU context benchmark, not model inference or AUROC.
Targets are a fixed prefix of Test table order (all 726 rows by default).
Stage 1 remains graph-backed; SQL replaces only Stage 2 adjacency retrieval.
Repeats use step=0 to test reproducibility, not different context seeds.
JSON excludes NumPy sequence arrays; temporary disk snapshots bound RAM use.
For later model evaluation, scripts/eval.py separately accepts --sql-context-db;
graph-backed evaluation uses the same on-demand Stage 2 provider.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time


TRACE_FIELDS = ('visits', 'candidate_order', 'fallback_candidates', 'collections', 'cells')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pre-dir', default='stanford-star/relbench-preprocessed')
    parser.add_argument('--duckdb', '--sql-context-db', dest='duckdb',
                        default='data/duckdb/rel-f1.duckdb',
                        help='Stage 2 DuckDB file; alias matches the opt-in eval flag')
    parser.add_argument('--items', type=int, default=726,
                        help='fixed Test-table prefix, capped at split size (default: all 726)')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--ctx-size', type=int, default=256)
    parser.add_argument('--local-ctx-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--num-walks', type=int, default=10_000)
    parser.add_argument('--walk-length', type=int, default=20)
    parser.add_argument('--context-seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('compare_sql_stage2.json'))
    parser.add_argument('--json-cells', action=argparse.BooleanOptionalAction, default=True,
                        help='include cell identities in JSON (always compared in memory)')
    args = parser.parse_args()
    if min(args.items, args.repeats, args.ctx_size, args.local_ctx_size,
           args.width, args.walk_length) <= 0:
        parser.error('items, repeats, context sizes, width, and walk length must be positive')
    if args.local_ctx_size > args.ctx_size:
        parser.error('local context size must not exceed context size')
    if args.num_walks < 0 or not 0 <= args.context_seed < 2**64:
        parser.error('num-walks must be nonnegative and context-seed must fit u64')
    return args


def sequence_diff(left, right):
    """Bitwise NumPy equality, including bfloat16, NaNs, padding and metadata."""
    import numpy as np

    if left.keys() != right.keys():
        return ['sequence.keys']
    differences = []
    for name in left:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if (a.shape != b.shape or a.dtype != b.dtype
                or not np.array_equal(a.reshape(-1).view(np.uint8),
                                      b.reshape(-1).view(np.uint8))):
            differences.append(f'sequence.{name}')
    return differences


def trace_diff(left, right):
    fields = TRACE_FIELDS + ('target', 'step', 'step_seed', 'context_seed',
                             'num_walks', 'walk_length', 'local_ctx_size',
                             'bfs_width', 'prefer_latest', 'balance_labels', 'mask_prob')
    return [name for name in fields if left[name] != right[name]]


def validate_trace(trace, sequence, args, target):
    import numpy as np

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    valid = ~sequence['is_padding']
    size = int(valid.sum())
    check(int(sequence['seq_len']) == args.ctx_size, 'sequence size mismatch')
    check(0 < size <= args.ctx_size and size == len(trace['cells']), 'global cell budget')
    check(np.array_equal(valid, np.arange(args.ctx_size) < size), 'noncontiguous padding')
    check(bool(sequence['batch_mask'][0]), 'phantom item')
    check(bool(sequence['is_targets'][0]) and int(sequence['is_targets'].sum()) == 1,
          'target mask must mark exactly the first cell')
    check(int(sequence['node_idxs'][0]) == target == trace['target'], 'target identity')
    check(trace['mask_prob'] == 0.0, 'evaluation mask probability')
    check(trace['local_ctx_size'] == args.local_ctx_size and trace['bfs_width'] == args.width,
          'local budget/width configuration')
    cells = trace['cells']
    check(len({(c[0], c[1]) for c in cells}) == len(cells), 'duplicate cells')
    check(len({(c[0], c[2]) for c in cells}) == len(cells), 'duplicate node/column')
    check(all(int(sequence['node_idxs'][i]) == c[0]
              and int(sequence['col_name_idxs'][i]) == c[2]
              and int(sequence['seed_node_idxs'][i]) == c[3]
              and int(sequence['bfs_depths'][i]) == c[4]
              for i, c in enumerate(cells)), 'trace/sequence identity')
    target_time = trace['collections'][0]['target_timestamp']
    if target_time is not None:
        check(bool((sequence['timestamps'][valid] <= target_time).all()), 'future context cells')
    labels = (valid & sequence['is_task_nodes']
              & (sequence['col_name_idxs'] == sequence['col_name_idxs'][0])
              & (sequence['node_idxs'] != target))
    # Algorithm 1 permits equal timestamps. Report same-time labels separately
    # rather than silently changing the baseline to a strict cutoff.
    if target_time is not None:
        check(bool((sequence['timestamps'][labels] <= target_time).all()),
              'future context labels')
    check(trace['candidate_order'] and trace['candidate_order'][0] == target,
          'candidate order must be target-first')
    check(trace['collections'] and trace['collections'][0]['seed'] == target,
          'collection order must be target-first')
    by_seed = Counter(c[3] for c in cells)
    check(all(n <= args.local_ctx_size for n in by_seed.values()), 'local emitted cell budget')
    for collection in trace['collections']:
        check(collection['local_ctx_size'] == args.local_ctx_size
              and collection['bfs_width'] == args.width, 'collection budget/width configuration')
        check(0 < collection['remaining_cells'] <= args.ctx_size, 'remaining cell budget')
        # Rust counts raw cells before emitting a row; every row has >=1 cell.
        check(len(collection['rows']) < args.local_ctx_size, 'local row budget')
        if trace['policy'] != 'legacy':
            check(collection['cutoff'] == collection['target_timestamp'], 'non-target cutoff')
    historical = labels & (sequence['timestamps'] < target_time) if target_time is not None else labels
    return failures, {
        'cells': size,
        'nodes': int(np.unique(sequence['node_idxs'][valid]).size),
        'context_label_cells': int(labels.sum()),
        'historical_label_cells': int(historical.sum()),
        'historical_label_nodes': int(np.unique(sequence['node_idxs'][historical]).size),
        'same_time_label_cells': (int((labels & (sequence['timestamps'] == target_time)).sum())
                                  if target_time is not None else 0),
    }


def overlap(left, right):
    def measure(a, b):
        union = a | b
        return {'intersection': len(a & b), 'union': len(union),
                'jaccard': len(a & b) / len(union) if union else 1.0}

    return {
        'cells': measure({(c[0], c[2]) for c in left['cells']},
                         {(c[0], c[2]) for c in right['cells']}),
        'nodes': measure({c[0] for c in left['cells']}, {c[0] for c in right['cells']}),
    }


def run(args, report):
    # Match sample_sql_context.py's dataframe-before-torch import order on macOS.
    import duckdb  # noqa: F401
    import pandas  # noqa: F401
    import numpy as np
    import rt._rustler as native
    from rt.eval_utils import build_evaluator
    from rt.pre import resolve_pre_dir
    from rt.sql_context import load_stage2_sql_provider
    from rt.tasks import Task

    if not hasattr(native.Sampler, 'trace_py'):
        raise RuntimeError('Installed extension is stale: rebuild with pre and pyo3/extension-module')
    report['extension'] = native.__file__
    tick = time.perf_counter()
    pre_dir = resolve_pre_dir(args.pre_dir, ['rel-f1'], 'all-MiniLM-L12-v2')
    db_dir = Path(pre_dir) / 'rel-f1'
    meta = json.loads((db_dir / 'meta.json').read_text())
    if meta.get('sampling_graph') is False:
        raise ValueError('Stage 2 comparison requires prepared graph-backed data')
    info = json.loads((db_dir / 'table_info.json').read_text())['driver-top3:Test']
    count = min(args.items, info['num_nodes'])
    targets = list(range(info['node_idx_offset'], info['node_idx_offset'] + count))
    if not targets:
        raise ValueError('Empty driver-top3 test split')
    report.update(resolved_pre_dir=str(pre_dir), split_items=info['num_nodes'], targets=targets)
    report['setup_seconds']['resolve_and_metadata'] = time.perf_counter() - tick
    tick = time.perf_counter()
    task = Task('rel-f1', 'driver-top3', 'qualifying', 'clf', 'test', ('qualifying',))
    evaluator = build_evaluator(
        [task], pre_dir, embedding_model='all-MiniLM-L12-v2', d_text=384,
        device='cpu', ctx_size=args.ctx_size, local_ctx_size=args.local_ctx_size,
        bfs_width=args.width, num_walks=args.num_walks, walk_length=args.walk_length,
        context_seed=args.context_seed, tokens_per_gpu=args.ctx_size,
        items_per_task=info['num_nodes'], num_workers=0, mmap_populate=False, sql_context_db=None,
    )
    sampler = evaluator.eval_loaders[task].dataset.rustler_dataset.sampler
    report['setup_seconds']['bfs_evaluator'] = time.perf_counter() - tick
    provider = None
    records = {target: {'target': target, 'traces': {}, 'timings': {}} for target in targets}
    report['items'] = list(records.values())

    def capture(mode, target, repeat):
        before = provider.stats if mode == 'sql' else None
        tick = time.perf_counter()
        trace = sampler.trace_py(0, target, args.ctx_size, step=0, legacy=mode == 'legacy')
        seconds = time.perf_counter() - tick
        sequence = dict(trace.pop('sequence'))
        failures, summary = validate_trace(trace, sequence, args, target)
        expected = 'sql_neighbors' if mode == 'sql' else 'bfs'
        if trace['collector'] != expected:
            failures.append(f'wrong collector: {trace["collector"]}')
        for failure in failures:
            report['failures'].append({'mode': mode, 'target': target, 'repeat': repeat,
                                       'violation': failure})
        timing = {'repeat': repeat, 'seconds': seconds}
        if before is not None:
            after = provider.stats
            timing['query_count'] = after['query_count'] - before['query_count']
            timing['query_seconds'] = after['total_seconds'] - before['total_seconds']
        records[target]['timings'].setdefault(mode, []).append(timing)
        if repeat == 0:
            records[target]['traces'][mode] = trace
            records[target].setdefault('context_summary', {})[mode] = summary
        return trace, sequence

    def require_parity(target, repeat, label, differences):
        if differences:
            report['failures'].append({'target': target, 'repeat': repeat,
                                       'violation': label, 'fields': differences})

    # Never retain the dense embedding arrays for the full split in RAM.
    with tempfile.TemporaryDirectory(prefix='rt-stage2-') as tmp:
        try:
            for mode in ('legacy', 'aligned', 'sql'):
                if mode == 'sql':
                    tick = time.perf_counter()
                    provider = load_stage2_sql_provider(pre_dir, args.duckdb)
                    sampler.set_sql_neighbor_provider_py('rel-f1', provider)
                    report['setup_seconds']['sql_provider_load_validate_install'] = time.perf_counter() - tick
                phase_start = time.perf_counter()
                for repeat in range(args.repeats):
                    for target in targets:
                        trace, sequence = capture(mode, target, repeat)
                        path = Path(tmp) / f'{mode}-{target}.npz'
                        # NPZ loses custom bf16 dtype registration; compare its raw bits.
                        canonical = {k: np.asarray(v).view(np.uint16)
                                     if np.asarray(v).dtype.name == 'bfloat16'
                                     else np.asarray(v) for k, v in sequence.items()}
                        if repeat == 0:
                            if mode == 'aligned':
                                baseline = records[target]['traces']['legacy']
                                with np.load(Path(tmp) / f'legacy-{target}.npz') as old:
                                    differences = trace_diff(baseline, trace) + sequence_diff(dict(old), canonical)
                                records[target]['legacy_vs_aligned'] = {
                                    'different_fields': differences, 'overlap': overlap(baseline, trace)}
                            if mode != 'sql':
                                np.savez(path, **canonical)
                        if mode == 'sql' or repeat > 0:
                            baseline_mode = 'aligned' if mode == 'sql' else mode
                            baseline = records[target]['traces'][baseline_mode]
                            with np.load(Path(tmp) / f'{baseline_mode}-{target}.npz') as old:
                                differences = trace_diff(baseline, trace) + sequence_diff(dict(old), canonical)
                            require_parity(target, repeat, f'{baseline_mode}/{mode} parity', differences)
                            if mode == 'sql' and repeat == 0:
                                records[target]['aligned_vs_sql'] = {
                                    'different_fields': differences, 'overlap': overlap(baseline, trace)}
                report['phase_seconds'][mode] = time.perf_counter() - phase_start
                print(f'{mode}: {count} targets x {args.repeats} repeats, '
                      f'{report["phase_seconds"][mode]:.3f}s (includes validation/disk IO)', flush=True)
        finally:
            for mode in ('legacy', 'aligned', 'sql'):
                times = [t['seconds'] for record in records.values()
                         for t in record['timings'].get(mode, [])]
                if times:
                    report['trace_seconds'][mode] = {
                        'count': len(times), 'total': sum(times), 'mean': float(np.mean(times)),
                        'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)),
                    }
            if provider is not None:
                report['query_stats'] = provider.stats
                sampler.set_sql_neighbor_provider_py('rel-f1', None)
                provider.close()


def main():
    args = parse_args()
    started = time.perf_counter()
    report = {'benchmark': 'prepared-graph-stage2-context-only', 'device': 'cpu',
              'model_inference': False, 'auroc_measured': False,
              'settings': {**vars(args), 'output': str(args.output)},
              'setup_seconds': {}, 'phase_seconds': {}, 'trace_seconds': {}, 'failures': [],
              'timing_note': 'trace_py includes trace/NumPy construction; SQL query time includes lazy connection setup'}
    try:
        run(args, report)
    except Exception as exc:
        report['failures'].append({'violation': 'benchmark error',
                                   'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if not args.json_cells:
            for item in report.get('items', []):
                for trace in item['traces'].values():
                    trace.pop('cells', None)
        report['passed'] = not report['failures']
        report['full_runtime_seconds'] = time.perf_counter() - started
        report['runtime_note'] = 'includes imports/setup/all phases/validation/temp IO/cleanup; excludes final JSON write'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{"PASS" if report["passed"] else "FAIL"}: {args.output}; '
          f'{report["full_runtime_seconds"]:.3f}s; {len(report["failures"])} violations')
    if report['failures']:
        print(json.dumps(report['failures'][:10], indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
