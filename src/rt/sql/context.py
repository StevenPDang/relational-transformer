"""A context: the rows a prediction reads, why each is there, and its tensors.

Every row records how it was reached (the target, a labelled example, or a
parent/child of another row through a named FK) and every query that ran is
logged with its timing, so a prediction can be traced back to SQL.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from rt.sql.encode import NUMBER, Cell, TextEmbedder
from rt.sql.meta import Table

MAX_F2P_NBRS = 5


@dataclass
class Node:
    table: Table
    row: int
    parents: list[tuple[str, int]]  # (parent table name, parent __row) per FK value
    cells: list[Cell]
    timestamp: int | None  # epoch seconds
    seed: int  # index of the seed whose expansion reached this row
    depth: int
    via: str  # "target", "example", "parent via <fk> of <row>", "child via <fk> of <row>"

    @property
    def key(self) -> tuple[str, int]:
        return (self.table.key, self.row)


@dataclass
class Query:
    step: str
    sql: str
    rows: int
    ms: float


@dataclass
class Context:
    target: Node
    target_column: str  # "<col> of <task>"
    label: float | None  # raw label, when the target row has one
    nodes: list[Node] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    # (node index, cell index) in context order; cell index -1 is the target cell
    tokens: list[tuple[int, int]] = field(default_factory=list)

    def trace(self) -> dict:
        """The context as plain data: rows, provenance, queries, timings."""
        return {
            "target": {"table": self.target.table.key, "row": self.target.row},
            "target_column": self.target_column,
            "label": self.label,
            "num_tokens": len(self.tokens),
            "rows": [
                {
                    "table": n.table.key,
                    "row": n.row,
                    "timestamp": n.timestamp,
                    "seed": n.seed,
                    "depth": n.depth,
                    "via": n.via,
                    "cells": len(n.cells),
                }
                for n in self.nodes
            ],
            "queries": [
                {"step": q.step, "rows": q.rows, "ms": round(q.ms, 3), "sql": q.sql}
                for q in self.queries
            ],
            "query_ms": round(sum(q.ms for q in self.queries), 3),
        }


def to_batch(contexts: list[Context], embed: TextEmbedder, ctx_size: int, d_text: int):
    """Contexts -> the tensor dict `RelationalTransformer.predict` reads.

    Row, column and table identifiers are numbered per batch. The model only
    compares them for equality, so the preprocessor's global numbering is not
    needed.
    """
    B, S = len(contexts), ctx_size
    node_idxs = np.full((B, S), -1, np.int32)
    f2p = np.full((B, S, MAX_F2P_NBRS), -1, np.int32)
    col_idxs = np.zeros((B, S), np.int32)
    table_idxs = np.zeros((B, S), np.int32)
    sem_types = np.zeros((B, S), np.int32)
    number = np.zeros((B, S, 1), np.float32)
    dt = np.zeros((B, S, 1), np.float32)
    is_targets = np.zeros((B, S), bool)
    is_task_nodes = np.zeros((B, S), bool)
    is_padding = np.ones((B, S), bool)

    ids: dict = {}

    def nid(key) -> int:
        return ids.setdefault(("node", key), len(ids))

    col_ids: dict[str, int] = {}
    table_ids: dict[str, int] = {}
    texts: list[tuple[int, int, str]] = []  # (b, s, text) for text cells
    col_names: list[str] = [""] * (B * S)

    for b, ctx in enumerate(contexts):
        for s, (ni, ci) in enumerate(ctx.tokens[:S]):
            node = ctx.nodes[ni]
            if ci == -1:
                cell = Cell(ctx.target_column, NUMBER)
                is_targets[b, s] = True
            else:
                cell = node.cells[ci]
            node_idxs[b, s] = nid(node.key)
            for j, (ptable, prow) in enumerate(node.parents[:MAX_F2P_NBRS]):
                f2p[b, s, j] = nid((ptable, prow))
            col_idxs[b, s] = col_ids.setdefault(cell.column, len(col_ids))
            table_idxs[b, s] = table_ids.setdefault(node.table.name, len(table_ids))
            sem_types[b, s] = cell.sem_type
            number[b, s, 0] = cell.number
            dt[b, s, 0] = cell.datetime
            is_task_nodes[b, s] = node.table.is_task or ci == -1
            is_padding[b, s] = False
            col_names[b * S + s] = cell.column
            if cell.text is not None:
                texts.append((b, s, cell.text))

    text_values = np.zeros((B, S, d_text), np.float32)
    if texts:
        emb = embed([t for _, _, t in texts]).astype(np.float32)
        for (b, s, _), e in zip(texts, emb):
            text_values[b, s] = e
    col_name_values = np.zeros((B, S, d_text), np.float32)
    real = [i for i, c in enumerate(col_names) if c]
    if real:
        emb = embed([col_names[i] for i in real]).astype(np.float32)
        col_name_values.reshape(B * S, d_text)[real] = emb

    def bf16(a):
        return torch.from_numpy(a).to(torch.bfloat16)

    return {
        "node_idxs": torch.from_numpy(node_idxs),
        "f2p_nbr_idxs": torch.from_numpy(f2p),
        "col_name_idxs": torch.from_numpy(col_idxs),
        "table_name_idxs": torch.from_numpy(table_idxs),
        "sem_types": torch.from_numpy(sem_types),
        "number_values": bf16(number),
        "datetime_values": bf16(dt),
        "text_values": bf16(text_values),
        "col_name_values": bf16(col_name_values),
        "is_targets": torch.from_numpy(is_targets),
        "is_task_nodes": torch.from_numpy(is_task_nodes),
        "is_padding": torch.from_numpy(is_padding),
    }
