"""Check that rt.sql encodes rows exactly as the preprocessor + sampler do.

The rustler sampler builds contexts for some rows of a task. Each context is
then rebuilt token for token from the database by rt.sql (`replay`), and every
model input is compared by meaning: which row and column a token is, its
semantic type, its values, its embeddings, and its parents. With --model, both
versions also go through RT-J and the predictions are compared.

    python -m examples.sql.parity --duckdb data/mini-shop.duckdb \\
        --pre-dir data/pre --task customer-churn --split test --rows 32 --model

The pre dir must come from this checkout's `preprocess` of the same data.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import torch

import rt.sql as rs
from rt.data import get_tasks
from rt.data.datasets import RustlerDataset
from rt.sql.context import Context, Node, to_batch
from rt.sql.meta import ROW

_SPLIT = {"Db": None, "Train": "train", "Val": "val", "Test": "test"}


def sampler_batch(pre_dir, db_name, task, split, rows, ctx_size, embedder, d_text):
    tasks = get_tasks(pre_dir, [(db_name, task)], (split,))
    ds = RustlerDataset(
        tasks,
        pre_dir,
        0,
        0,
        1,
        local_ctx_size_list=[min(256, ctx_size)],
        bfs_width_list=[32],
        num_walks=1000,
        walk_length=20,
        prefer_latest_list=[True],
        mask_prob_max=0.0,
        embedder=embedder,
        d_text=d_text,
        shuffle_seed=0,
        context_seed=0,
        items_per_task=10**9,
        quiet=True,
        ignore_data_errors=False,
        mmap_populate=False,
        legacy_boolean=False,
        timeout_per_item=60.0,
        vector_db_path=None,
        db_cutoff=None,
    )
    info = json.loads((Path(pre_dir) / db_name / "table_info.json").read_text())
    offset = info[f"{task}:{split.capitalize()}"]["node_idx_offset"]
    raw = ds.sampler.batch_for_nodes_py([offset + r for r in rows], 0, ctx_size)
    batch = ds._process_batch(raw)
    batch.pop("batch_mask")
    return batch


class Locator:
    """Global preprocessor ids -> (table key, __row) and column names."""

    def __init__(self, pre_dir, db_name, db: rs.Database):
        base = Path(pre_dir) / db_name
        self.text = json.loads((base / "text.json").read_text())
        spans = []
        for key, v in json.loads((base / "table_info.json").read_text()).items():
            name, ty = key.rsplit(":", 1)
            split = _SPLIT[ty]
            t = db.tables[name] if split is None else db.tasks[name].splits[split]
            spans.append((v["node_idx_offset"], v["num_nodes"], t))
        self.spans = sorted(spans, key=lambda s: s[0])

    def node(self, idx: int):
        for off, n, t in self.spans:
            if off <= idx < off + n:
                return t, idx - off
        raise KeyError(idx)


def replay(con, db, enc, loc: Locator, batch, b: int, target_column: str) -> Context:
    """Rebuild context `b` of a sampler batch from the database, token for token."""
    pad = batch["is_padding"][b]
    node_idxs = batch["node_idxs"][b][~pad].tolist()
    col_idxs = batch["col_name_idxs"][b][~pad].tolist()
    targets = batch["is_targets"][b][~pad].tolist()

    order: dict[int, int] = {}
    for n in node_idxs:
        order.setdefault(n, len(order))
    fetched = {}
    by_table: dict[str, tuple] = {}
    for n in order:
        t, r = loc.node(n)
        by_table.setdefault(t.key, (t, []))[1].append(r)
    for t, rows in by_table.values():
        for values in con.execute(
            f"SELECT {enc.select_list(t)} FROM {t.sql} "
            f"WHERE {ROW} IN (SELECT unnest(?::BIGINT[]))",
            [rows],
        ).fetchall():
            fetched[(t.key, values[0])] = (t, values)

    nodes = []
    for n in order:
        t, r = loc.node(n)
        tt, values = fetched[(t.key, r)]
        row, fks, cells, ts = enc.decode_row(tt, values)
        enc_t = enc.tables[t.key]
        parents = [
            (t.fkeys[col], int(v))
            for col, v in zip(enc_t.fk_columns, fks)
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        nodes.append(Node(t, row, parents, cells, ts, 0, 0, "replay"))

    target_i = order[node_idxs[targets.index(True)]]
    ctx = Context(target=nodes[target_i], target_column=target_column, label=None)
    ctx.nodes = nodes
    for n, c, is_t in zip(node_idxs, col_idxs, targets):
        node = nodes[order[n]]
        if is_t:
            ctx.tokens.append((order[n], -1))
            continue
        name = loc.text[c]
        ci = next((i for i, cell in enumerate(node.cells) if cell.column == name), None)
        if ci is None:
            raise AssertionError(f"rt.sql emits no cell {name!r} for {node.key}")
        ctx.tokens.append((order[n], ci))
    return ctx


FIELDS = (
    "node",
    "column",
    "is_padding",
    "sem_types",
    "is_targets",
    "number_values",
    "datetime_values",
    "text_values",
    "col_name_values",
)


def compare(a, b, nodes, cols_a, cols_b) -> Counter:
    """Field-by-field mismatches between two batches, by meaning.

    Synthetic identifier cells are left out of `number_values`: their values
    differ by design (see rt.sql.encode).
    """
    bad = Counter({k: 0 for k in FIELDS})
    pa, pb = ~a["is_padding"], ~b["is_padding"]
    bad["is_padding"] += int((pa != pb).sum())
    m = pa & pb
    for k in ("sem_types", "is_targets"):
        bad[k] += int((a[k][m] != b[k][m]).sum())
    ident = torch.tensor([c.startswith("identifier of ") for c in cols_b])
    not_target = m & ~a["is_targets"]
    not_target[0, : len(ident)] &= ~ident
    for k in ("number_values", "datetime_values"):
        x = a[k][..., 0].view(torch.int16)
        y = b[k][..., 0].view(torch.int16)
        bad[k] += int((x[not_target] != y[not_target]).sum())
    is_text = m & (a["sem_types"] == rs.encode.TEXT)
    for k, mask in (("text_values", is_text), ("col_name_values", m)):
        diff = (a[k].float() - b[k].float()).abs().amax(-1)
        bad[k] += int((diff[mask] > 0).sum())
    for name, ka, kb in (("node", *nodes), ("column", cols_a, cols_b)):
        bad[name] += sum(x != y for x, y in zip(ka, kb))
    return bad


def load_model(device: str):
    """RT-J for `device`. Off CUDA, flex attention runs uncompiled: rt compiles
    it at import, and torch.compile cannot lower it for CPU on macOS."""
    from rt import RelationalTransformer
    from rt.model import net

    if not device.startswith("cuda"):
        from torch.nn.attention.flex_attention import flex_attention

        net.flex_attention = flex_attention
    model = RelationalTransformer.from_pretrained("stanford-star/rt-j", device=device)
    return model.to(torch.bfloat16).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duckdb", required=True)
    p.add_argument("--pre-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--ctx-size", type=int, default=1024)
    p.add_argument("--embedder", default="all-MiniLM-L12-v2")
    p.add_argument("--d-text", type=int, default=384)
    p.add_argument("--model", action="store_true", help="also compare RT-J predictions")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    con = duckdb.connect(args.duckdb, read_only=True)
    db = rs.load(con)
    enc = rs.Encoder(con, db)
    loc = Locator(args.pre_dir, db.name, db)
    task = db.tasks[args.task]
    target_column = f"{task.target_col} of {task.name}"
    rows = list(range(args.rows))

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
    contexts = [
        replay(con, db, enc, loc, ref, b, target_column) for b in range(len(rows))
    ]
    embed = rs.TextEmbedder(args.embedder, args.device)
    ours = to_batch(contexts, embed, args.ctx_size, args.d_text)

    # Rows and columns, by meaning: the sampler's global ids against ours.
    def ref_keys(b):
        pad = ref["is_padding"][b]
        nodes = [loc.node(n) for n in ref["node_idxs"][b][~pad].tolist()]
        cols = [loc.text[c] for c in ref["col_name_idxs"][b][~pad].tolist()]
        return [(t.key, r) for t, r in nodes], cols

    def our_keys(ctx):
        nodes = [ctx.nodes[ni].key for ni, _ in ctx.tokens]
        cols = [
            ctx.target_column if ci == -1 else ctx.nodes[ni].cells[ci].column
            for ni, ci in ctx.tokens
        ]
        return nodes, cols

    bad = Counter({k: 0 for k in FIELDS})
    parent_bad = 0
    for b, ctx in enumerate(contexts):
        (nr, cr), (no, co) = ref_keys(b), our_keys(ctx)
        sub = {k: v[b : b + 1] for k, v in ref.items()}
        mine = {k: v[b : b + 1] for k, v in ours.items()}
        bad.update(compare(sub, mine, (nr, no), cr, co))
        # Parents, by meaning.
        pad = ref["is_padding"][b]
        for s, f in enumerate(ref["f2p_nbr_idxs"][b][~pad].tolist()):
            want = [loc.node(x) for x in f if x >= 0]
            want = [(t.name, r) for t, r in want]
            got = ctx.nodes[ctx.tokens[s][0]].parents[:5]
            parent_bad += want != got
    n_tokens = int((~ref["is_padding"]).sum())
    print(f"{len(rows)} contexts, {n_tokens} tokens compared")
    for k in FIELDS:
        print(f"  {k:16s} {bad[k]:>6d} mismatched")
    print(f"  {'f2p parents':16s} {parent_bad:>6d} mismatched")

    ident = sum(
        1
        for ctx in contexts
        for ni, ci in ctx.tokens
        if ci >= 0 and ctx.nodes[ni].cells[ci].column.startswith("identifier of ")
    )
    print(
        f"  ({ident} tokens are synthetic identifier cells; their values differ by design)"
    )

    if args.model:
        model = load_model(args.device)
        with torch.no_grad():
            ya = model.predict(ref, [args.ctx_size], args.device, None)[
                args.ctx_size
            ].float()
            yb = model.predict(ours, [args.ctx_size], args.device, None)[
                args.ctx_size
            ].float()
        d = (ya - yb).abs()
        print(f"predictions: max |diff| {d.max():.4f}, mean |diff| {d.mean():.4f}")
        print(f"  sampler: {np.round(ya[:6].numpy(), 3)}")
        print(f"  rt.sql:  {np.round(yb[:6].numpy(), 3)}")


if __name__ == "__main__":
    main()
