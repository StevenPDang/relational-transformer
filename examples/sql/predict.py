"""Predict a task straight from a DuckDB database with RT-J, no preprocessing.

For each row of a task split: rt.sql retrieves a context with SQL, encodes it,
and RT-J predicts the masked label. Prints the metric and where the time went,
and writes one JSON trace per context (its rows, why each is there, and every
query with its timing). With --pre-dir, the rustler sampler also predicts the
same rows from preprocessed data, as the baseline to beat.

    python -m examples.sql.predict --duckdb data/mini-shop.duckdb \\
        --task customer-churn --rows 64 --ctx-size 512 --trace traces.jsonl
"""

import argparse
import json
import random
import time

import duckdb
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score

import rt.sql as rs
from examples.sql.parity import load_model, sampler_batch


def score(task_type: str, labels, preds) -> str:
    labels, preds = np.asarray(labels, float), np.asarray(preds, float)
    if task_type == "binary_classification":
        if len(set(labels)) < 2:
            return "AUROC n/a (one class)"
        return f"AUROC {roc_auc_score(labels > 0.5, preds):.4f}"
    return f"MAE {mean_absolute_error(labels, preds):.4f} (normalized units)"


def predict(model, batch, ctx_size, bs, device):
    out = []
    with torch.no_grad():
        for i in range(0, batch["node_idxs"].shape[0], bs):
            sub = {k: v[i : i + bs] for k, v in batch.items()}
            out.append(model.predict(sub, [ctx_size], device, None)[ctx_size].float())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duckdb", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--local-ctx-size", type=int, default=256)
    p.add_argument("--bfs-width", type=int, default=32)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="seeds the row sample")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embedder", default="all-MiniLM-L12-v2")
    p.add_argument("--d-text", type=int, default=384)
    p.add_argument("--trace", help="write one JSON trace per context here")
    p.add_argument("--pre-dir", help="also run the rustler sampler, as a baseline")
    args = p.parse_args()

    tic = time.perf_counter()
    con = duckdb.connect(args.duckdb, read_only=True)
    db = rs.load(con)
    enc = rs.Encoder(con, db)
    task = db.tasks[args.task]
    t_setup = time.perf_counter() - tic

    n_rows = con.execute(
        f"SELECT count(*) FROM {task.splits[args.split].sql}"
    ).fetchone()[0]
    # A random sample: split files are sorted, so the first rows are often one class.
    rows = sorted(
        random.Random(args.seed).sample(range(n_rows), min(args.rows, n_rows))
    )
    retriever = rs.Retriever(
        con,
        db,
        enc,
        rs.RetrieveConfig(
            ctx_size=args.ctx_size,
            local_ctx_size=args.local_ctx_size,
            bfs_width=args.bfs_width,
        ),
    )

    tic = time.perf_counter()
    contexts = [retriever.build(args.task, args.split, r) for r in rows]
    t_retrieve = time.perf_counter() - tic

    embed = rs.TextEmbedder(args.embedder, args.device)
    tic = time.perf_counter()
    batch = rs.to_batch(contexts, embed, args.ctx_size, args.d_text)
    t_encode = time.perf_counter() - tic

    model = load_model(args.device)
    tic = time.perf_counter()
    preds = predict(model, batch, args.ctx_size, args.bs, args.device)
    t_model = time.perf_counter() - tic

    if task.task_type == "binary_classification":
        labels = [c.label for c in contexts]
    else:
        mean, std = enc.tables[task.splits["train"].key].stats[task.target_col]
        labels = [(c.label - mean) / std for c in contexts]

    n = len(rows)
    q = [len(c.queries) for c in contexts]
    toks = [len(c.tokens) for c in contexts]
    print(
        f"rt.sql on {db.name}/{args.task}/{args.split}, {n} rows, ctx {args.ctx_size}"
    )
    print(f"  {score(task.task_type, labels, preds)}")
    print(
        f"  setup {t_setup * 1e3:.0f} ms (schema + column statistics, once per database)"
    )
    print(
        f"  retrieve {t_retrieve / n * 1e3:.1f} ms/context "
        f"({np.mean(q):.0f} queries, {np.mean(toks):.0f} tokens on average)"
    )
    print(f"  encode   {t_encode / n * 1e3:.1f} ms/context (incl. text embedding)")
    print(f"  model    {t_model / n * 1e3:.1f} ms/context ({args.device})")

    if args.trace:
        with open(args.trace, "w") as f:
            for c, y in zip(contexts, preds):
                f.write(json.dumps({**c.trace(), "prediction": float(y)}) + "\n")
        print(f"  traces -> {args.trace}")

    if args.pre_dir:
        tic = time.perf_counter()
        ref = sampler_batch(
            args.pre_dir,
            db.name,
            args.task,
            args.split,
            rows,
            args.ctx_size,
            args.embedder,
            args.d_text,
        )
        t_sample = time.perf_counter() - tic
        ref_preds = predict(model, ref, args.ctx_size, args.bs, args.device)
        print(f"rustler sampler on the same rows (preprocessed {args.pre_dir})")
        print(f"  {score(task.task_type, labels, ref_preds)}")
        print(f"  sample   {t_sample / n * 1e3:.1f} ms/context (incl. sampler setup)")
        corr = np.corrcoef(preds, ref_preds)[0, 1]
        print(f"  prediction correlation with rt.sql: {corr:.3f}")


if __name__ == "__main__":
    main()
