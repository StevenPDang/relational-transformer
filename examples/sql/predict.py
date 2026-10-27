"""Predict a task straight from a DuckDB database with RT-J, no preprocessing.

For each row of a task split: rt.sql retrieves a context with SQL, encodes it,
and RT-J predicts the masked label. Prints the metric and where the time went,
and writes one JSON trace per context (its rows, why each is there, and every
query with its timing).

    python -m examples.sql.predict --duckdb data/mini-shop.duckdb \\
        --task customer-churn --rows 64 --ctx-size 512 --trace traces.jsonl

With --relbench, predictions are written as a RelBench submission CSV and
scored by RelBench itself (`relbench.submit.evaluate_task`), the same scoring
`rt.eval` uses, so the number is comparable with `examples/eval`:

    python -m examples.sql.predict --duckdb data/relbench-duckdb/rel-f1.duckdb \\
        --task driver-dnf --all-rows --ctx-size 8192 \\
        --relbench data/relbench-v1/rel-f1 --csv-out sql_out/
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import sklearn.metrics as M
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm import tqdm

import rt.sql as rs
from examples.sql.parity import load_model


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


def relbench_score(source, task_name, task_type, rows, preds, csv_out):
    """Score with RelBench, as `rt.eval.relbench` does for the sampler.

    `rows` are `__row` positions in the test split, which are RelBench's row
    order. RT-J outputs a logit (classification) or a value normalized by the
    train split's mean and std (regression); both are mapped back to RelBench's
    units. Only a full test split is a valid submission; a partial one is
    scored with sklearn on those rows and marked with `~`.
    """
    from rt.eval.relbench import _load_relbench_task, _train_stats

    rtask = _load_relbench_task(source, task_name)
    masked = rtask.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    gt = rtask.get_table("test", mask_input_cols=False).df.reset_index(drop=True)
    rows = np.asarray(rows, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.float64)
    gt_vals = gt[rtask.target_col].to_numpy(dtype=np.float64)[rows]
    if task_type == "binary_classification":
        out = 1.0 / (1.0 + np.exp(-preds))
    else:
        mean, std = _train_stats(rtask)
        out = preds * std + mean

    sub = masked.iloc[rows][[rtask.entity_col, rtask.time_col]].copy()
    sub[rtask.target_col] = out
    csv_path = None
    if csv_out:
        Path(csv_out).mkdir(parents=True, exist_ok=True)
        csv_path = Path(csv_out) / f"{Path(str(source)).name}__{task_name}.csv"
        sub.to_csv(csv_path, index=False)

    if len(rows) < len(masked):
        if task_type == "binary_classification":
            return "roc_auc~", M.roc_auc_score(gt_vals > 0, out), csv_path
        return "nmae~", M.mean_absolute_error(gt_vals, out) / std, csv_path

    from relbench.submit import evaluate_task

    if csv_path is None:
        import tempfile

        csv_path = Path(tempfile.mkdtemp()) / "pred.csv"
        sub.to_csv(csv_path, index=False)
    name = f"{Path(str(source)).name}/{task_name}"
    metrics = evaluate_task(name, str(csv_path), dataset=str(source))
    mname, mval = next(iter(metrics.items()))
    return mname, float(mval), csv_path if csv_out else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duckdb", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rows", type=int, default=64, help="random sample of the split")
    p.add_argument("--all-rows", action="store_true", help="every row of the split")
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--local-ctx-size", type=int, default=256)
    p.add_argument("--bfs-width", type=int, default=32)
    p.add_argument(
        "--seed-batch",
        type=int,
        default=1,
        help="seeds expanded together, one query per BFS step for the batch",
    )
    p.add_argument("--bs", type=int, default=8)
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="contexts retrieved at once, each on its own DuckDB cursor",
    )
    p.add_argument(
        "--chunk", type=int, default=16, help="contexts built and predicted at once"
    )
    p.add_argument("--seed", type=int, default=0, help="seeds the row sample")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embedder", default="all-MiniLM-L12-v2")
    p.add_argument("--d-text", type=int, default=384)
    p.add_argument("--trace", help="write one JSON trace per context here")
    p.add_argument(
        "--relbench",
        help="RelBench dataset to score against: a local dir (data/relbench-v1/rel-f1) "
        "or a Hub spec (stanford-star/relbench-v1/rel-f1)",
    )
    p.add_argument("--csv-out", help="directory for the RelBench submission CSV")
    p.add_argument(
        "--query-log",
        help="directory for DuckDB's query log, written as each query runs "
        "(tail -f <dir>/duckdb_log_entries.csv)",
    )
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch.cuda.is_available() is False")
    print(
        f"device {args.device} (torch {torch.__version__}, cuda {torch.version.cuda})"
    )

    tic = time.perf_counter()
    con = duckdb.connect(args.duckdb, read_only=True)
    if args.query_log:
        log_dir = str(Path(args.query_log).resolve()).replace("'", "''")
        con.execute(
            f"CALL enable_logging('QueryLog', storage='file', storage_path='{log_dir}')"
        )
        print(f"query log -> {Path(args.query_log) / 'duckdb_log_entries.csv'}")
    db = rs.load(con)
    enc = rs.Encoder(con, db)
    task = db.tasks[args.task]
    t_setup = time.perf_counter() - tic

    n_rows = con.execute(
        f"SELECT count(*) FROM {task.splits[args.split].sql}"
    ).fetchone()[0]
    if args.all_rows:
        rows = list(range(n_rows))
    else:
        # A random sample: split files are sorted, so the first rows are often one class.
        rows = sorted(
            random.Random(args.seed).sample(range(n_rows), min(args.rows, n_rows))
        )
    config = rs.RetrieveConfig(
        ctx_size=args.ctx_size,
        local_ctx_size=args.local_ctx_size,
        bfs_width=args.bfs_width,
        seed_batch=args.seed_batch,
    )
    # One retriever per worker thread, each on its own cursor: a DuckDB
    # connection runs one query at a time, and contexts are independent.
    local = threading.local()

    def build(r):
        if not hasattr(local, "retriever"):
            local.retriever = rs.Retriever(con.cursor(), db, enc, config)
        tic = time.perf_counter()
        ctx = local.retriever.build(args.task, args.split, r)
        return ctx, time.perf_counter() - tic

    embed = rs.TextEmbedder(args.embedder, args.device)
    model = load_model(args.device)

    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace).write_text("")
    t_retrieve = t_wait = t_encode = t_model = 0.0
    preds, raw_labels, q, toks = [], [], [], []
    bar = tqdm(total=len(rows), unit="row", desc=f"{args.task}")
    # In chunks: a full context batch holds two d_text vectors per token.
    # The next chunk is retrieved in the background while this one runs the model.
    chunks = [rows[i : i + args.chunk] for i in range(0, len(rows), args.chunk)]
    pool = ThreadPoolExecutor(max_workers=args.workers)
    t_start = time.perf_counter()
    pending = [pool.submit(build, r) for r in chunks[0]] if chunks else []
    for k in range(len(chunks)):
        bar.set_postfix_str("retrieve")
        tic = time.perf_counter()
        built = [f.result() for f in pending]
        t_wait += time.perf_counter() - tic
        if k + 1 < len(chunks):
            pending = [pool.submit(build, r) for r in chunks[k + 1]]
        contexts = [c for c, _ in built]
        t_retrieve += sum(dt for _, dt in built)
        bar.set_postfix_str("encode")
        tic = time.perf_counter()
        batch = rs.to_batch(contexts, embed, args.ctx_size, args.d_text)
        t_encode += time.perf_counter() - tic
        bar.set_postfix_str("model")
        tic = time.perf_counter()
        y = predict(model, batch, args.ctx_size, args.bs, args.device)
        t_model += time.perf_counter() - tic

        preds.extend(y.tolist())
        raw_labels.extend(c.label for c in contexts)
        q.extend(len(c.queries) for c in contexts)
        toks.extend(len(c.tokens) for c in contexts)
        if args.trace:
            with Path(args.trace).open("a") as trace:
                trace.writelines(
                    json.dumps({**c.trace(), "prediction": float(yi)}) + "\n"
                    for c, yi in zip(contexts, y)
                )
        bar.update(len(contexts))
    bar.close()
    pool.shutdown()
    t_wall = time.perf_counter() - t_start

    if task.task_type == "binary_classification":
        labels = raw_labels
    else:
        mean, std = enc.tables[task.splits["train"].key].stats[task.target_col]
        labels = [(y - mean) / std for y in raw_labels]

    n = len(rows)
    print(
        f"rt.sql on {db.name}/{args.task}/{args.split}, {n} of {n_rows} rows, "
        f"ctx {args.ctx_size}"
    )
    print(f"  {score(task.task_type, labels, preds)}")
    if args.relbench:
        mname, mval, csv_path = relbench_score(
            args.relbench, args.task, task.task_type, rows, preds, args.csv_out
        )
        print(
            f"  RelBench {mname} {mval:.4f}" + (f" -> {csv_path}" if csv_path else "")
        )
    print(
        f"  setup {t_setup * 1e3:.0f} ms (schema + column statistics, once per database)"
    )
    print(
        f"  retrieve {t_retrieve / n * 1e3:.1f} ms/context per worker "
        f"({np.mean(q):.0f} queries, {np.mean(toks):.0f} tokens on average)"
    )
    print(f"  encode   {t_encode / n * 1e3:.1f} ms/context (incl. text embedding)")
    print(f"  model    {t_model / n * 1e3:.1f} ms/context ({args.device})")
    print(
        f"  wall     {t_wall / n * 1e3:.1f} ms/context with {args.workers} workers "
        f"({t_wait / n * 1e3:.1f} ms/context waiting on retrieval)"
    )
    if args.trace:
        print(f"  traces -> {args.trace}")


if __name__ == "__main__":
    main()
