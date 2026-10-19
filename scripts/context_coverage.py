"""How much of a raw-prepared database do MetaSieve SQL contexts touch?

For one-shot raw inference, preparation (cell encoding, and mostly text
embeddings) dominates runtime, yet a context can only reach the rows its kept
metapaths select. This reports, per task, the rows and distinct text-embedding
rows (``text.json`` entries) that contexts can look up, as a share of the whole
database, and, given that run's ``runtime.json``, the embedding time a
contexts-only preparation would need. Embedding cost grows with string length,
so the estimate scales the run's embedding time by the character share.

Two scopes: ``all seeds`` is what the SQL sampler builds today (every task row,
since any earlier row may be a historical seed); ``test items`` is the minimum
for evaluating the test split: test rows, their candidate historical seeds, and
those rows' neighborhoods. Both are upper bounds: Rust may truncate a context
before it uses every listed row.

Run it on the prepare/ folder of a raw SQL run, after that run:

    pixi run --environment cuda124 python -m scripts.context_coverage \\
      --run runs/eval/rel-trial-study-outcome/<timestamp>_raw-sql-metasieve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def coverage(pre_db_dir, info, nodes, lengths, db_rows):
    """Row and text-embedding counts for a set of node indices."""
    import numpy as np

    from rt._rustler import node_text_ids

    nodes = np.unique(nodes)
    values, names = node_text_ids(str(pre_db_dir), nodes.tolist())
    per_table = {}
    for key, span in info.items():
        if key.endswith(":Db"):
            lo, hi = span["node_idx_offset"], span["node_idx_offset"] + span["num_nodes"]
            n = int(np.searchsorted(nodes, hi) - np.searchsorted(nodes, lo))
            if n:
                per_table[key.split(":")[0]] = n
    used = np.union1d(np.asarray(values, dtype="int64"), np.asarray(names, dtype="int64"))
    total_chars = int(lengths.sum())
    chars = int(lengths[used].sum())
    return {"nodes": int(len(nodes)), "db_rows": int(sum(per_table.values())), "db_rows_total": db_rows,
            "texts": int(len(used)), "texts_total": int(len(lengths)),
            "text_share": len(used) / len(lengths) if len(lengths) else 0.0,
            "char_share": chars / total_chars if total_chars else 0.0, "rows_by_table": per_table}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="raw SQL eval run folder (reads runtime.json and prepare/)")
    parser.add_argument("--prepare-dir", help="raw preparation folder, if not --run/prepare")
    parser.add_argument("--rules", default=None, help="MetaSieve rules (default: the run's --sql-context-rules)")
    parser.add_argument("--tasks", nargs="+", default=None, help="task names (default: the run's tasks)")
    parser.add_argument("--width", type=int, default=None, help="default: the run's --bfs-width, else 32")
    parser.add_argument("--out", default=None, help="also write the report as JSON here")
    args = parser.parse_args()

    import duckdb  # noqa: F401 -- before torch on macOS
    import numpy as np

    from rt.sql_context import load_metasieve_rules, load_raw_contexts

    runtime = {}
    if args.run:
        runtime = json.loads((Path(args.run).expanduser() / "runtime.json").read_text())
    settings = runtime.get("settings", {})
    prep = Path(args.prepare_dir or settings.get("prepare_dir") or Path(args.run) / "prepare").expanduser()
    rules_spec = args.rules or settings.get("sql_context_rules")
    width = args.width or settings.get("bfs_width") or 32
    if not rules_spec:
        parser.error("pass --rules (the run did not record --sql-context-rules)")
    dbs = [d for d in prep.iterdir() if (d / "table_info.json").is_file()]
    if len(dbs) != 1:
        parser.error(f"expected one prepared database in {prep}, found {[d.name for d in dbs]}")
    db_dir = dbs[0]
    db_name = db_dir.name
    info = json.loads((db_dir / "table_info.json").read_text())
    meta = json.loads((db_dir / "meta.json").read_text())
    lengths = np.array([len(t) for t in json.loads((db_dir / "text.json").read_text())], dtype="int64")
    text_count = len(lengths)
    db_rows = sum(v["num_nodes"] for k, v in info.items() if k.endswith(":Db"))
    tasks = args.tasks or [s.split("/", 1)[1] for s in settings.get("tasks") or [] if "/" in s]
    if not tasks:
        tasks = [t["name"] for t in meta.get("tasks", [])
                 if t.get("task_type") in ("binary_classification", "regression")]
    embed_seconds = runtime.get("preparation_seconds", {}).get("text embeddings")

    report = {"db": db_name, "prepare_dir": str(prep), "width": width, "texts_total": text_count,
              "db_rows_total": db_rows, "embedding_seconds": embed_seconds, "tasks": {}}
    print(f"{db_name}: {db_rows:,} DB rows, {text_count:,} distinct texts"
          + (f", {embed_seconds:.0f}s to embed them all" if embed_seconds else ""))
    for task in tasks:
        try:
            rules = load_metasieve_rules(rules_spec, db_name, task)
        except (FileNotFoundError, ValueError) as err:
            print(f"  {task}: skipped ({err})")
            continue
        contexts, orders = load_raw_contexts(str(prep), str(prep), db_name, task, width, rules)
        test = info[f"{task}:Test"]
        test_seeds = np.arange(test["node_idx_offset"], test["node_idx_offset"] + test["num_nodes"])
        # Test items, their candidate seeds, and every listed row of those seeds' contexts.
        pos = np.searchsorted(orders.seeds, test_seeds)
        cand = np.concatenate([orders.candidates[orders.offsets[i]:orders.offsets[i + 1]] for i in pos]
                              or [np.array([], dtype="int32")])
        needed_seeds = np.union1d(test_seeds, cand)
        cpos = np.searchsorted(contexts.seeds, needed_seeds)
        test_nodes = np.concatenate([contexts.nodes[contexts.offsets[i]:contexts.offsets[i + 1]] for i in cpos])
        scopes = {"all seeds": coverage(db_dir, info, contexts.nodes, lengths, db_rows),
                  "test items": coverage(db_dir, info, test_nodes, lengths, db_rows)}
        scopes["test items"]["seeds"] = int(len(needed_seeds))
        report["tasks"][task] = scopes
        print(f"  {task}:")
        for name, c in scopes.items():
            est = f", ~{c['char_share'] * embed_seconds:.0f}s to embed" if embed_seconds else ""
            print(f"    {name:10}  rows {c['db_rows']:>10,} / {db_rows:,} ({c['db_rows'] / max(db_rows, 1):6.1%})"
                  f"  texts {c['texts']:>10,} / {text_count:,} ({c['text_share']:6.1%}, "
                  f"{c['char_share']:6.1%} of chars){est}")
        top = sorted(scopes["test items"]["rows_by_table"].items(), key=lambda kv: -kv[1])[:8]
        print("    test-item rows by table: " + ", ".join(f"{t}={n:,}" for t, n in top))
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
