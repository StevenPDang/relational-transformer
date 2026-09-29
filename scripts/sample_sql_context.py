"""Build a small rel-f1 SQL context batch without loading a model/checkpoint."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-dir', default='stanford-star/relbench-preprocessed')
    parser.add_argument('--duckdb', default='data/duckdb/rel-f1.duckdb')
    parser.add_argument('--ctx-size', type=int, default=256)
    parser.add_argument('--local-ctx-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--items', type=int, default=4)
    parser.add_argument('--num-walks', type=int, default=0)
    args = parser.parse_args()
    if min(args.items, args.ctx_size, args.local_ctx_size, args.width) <= 0:
        parser.error('items, context sizes, and width must be positive')
    if args.local_ctx_size > args.ctx_size:
        parser.error('local context size must not exceed context size')

    # Load dataframe libraries first in the mac Pixi environment, avoiding its
    # conflicting OpenMP runtime initialization when torch is imported first.
    import duckdb  # noqa: F401
    import pandas  # noqa: F401

    from rt.eval_utils import build_evaluator
    from rt.tasks import Task

    task = Task('rel-f1', 'driver-top3', 'qualifying', 'clf', 'test', ('qualifying',))
    evaluator = build_evaluator(
        [task], args.pre_dir, embedding_model='all-MiniLM-L12-v2', d_text=384,
        device='cpu', ctx_size=args.ctx_size, local_ctx_size=args.local_ctx_size,
        bfs_width=args.width, num_walks=args.num_walks, walk_length=20,
        tokens_per_gpu=args.items * args.ctx_size, items_per_task=args.items,
        num_workers=0, mmap_populate=False, sql_context_db=args.duckdb,
    )
    batch = next(evaluator.eval_loader_iters[task])
    summaries = []
    for row in batch['batch_mask'].nonzero().flatten().tolist():
        valid = ~batch['is_padding'][row]
        target = int(batch['node_idxs'][row, 0])
        assert bool(batch['is_targets'][row, 0])
        assert int(batch['is_targets'][row].sum()) == 1
        assert bool((batch['timestamps'][row][valid] <= batch['timestamps'][row, 0]).all())
        label_cells = valid & batch['is_task_nodes'][row] & (
            batch['col_name_idxs'][row] == batch['col_name_idxs'][row, 0]
        ) & (batch['node_idxs'][row] != target)
        summaries.append({
            'target_node_idx': target, 'cells': int(valid.sum()),
            'nodes': batch['node_idxs'][row][valid].unique().tolist(),
            'context_labels': int(label_cells.sum()),
        })
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
