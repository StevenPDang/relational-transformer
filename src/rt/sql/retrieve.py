"""Choose a context's rows with SQL.

v0 follows the shape of the rustler sampler, with each of its random choices
made deterministic so a context can be replayed and traced:

1. Seeds. The target row first, then labelled rows of the same task, earlier
   than the target: rows of the same entity first, then the most recent.
   (The sampler ranks seeds by random-walk visits; that is the first thing to
   revisit.)
2. Neighbourhood. Each seed is expanded breadth first until it has
   `local_ctx_size` cells: every parent through every FK, and up to
   `bfs_width` children per FK, most recent first. (The sampler draws the
   children at random.)
3. Time. Each seed's neighbourhood holds nothing stamped after that seed
   (for the target, after the target), and no other task row stamped at the
   target's own time (it would be a label from the same horizon). With an
   untimestamped seed, only untimestamped children are read, as in the
   sampler.

Each step is one query per table or FK over the whole BFS level, never one
query per row.
"""

import math
import time
from dataclasses import dataclass

import duckdb

from rt.sql.context import Context, Node, Query
from rt.sql.encode import Encoder, kind_of
from rt.sql.meta import ROW, Database, Table, ident


@dataclass
class RetrieveConfig:
    ctx_size: int = 1024
    local_ctx_size: int = 128
    bfs_width: int = 16
    example_batch: int = 64  # labelled seeds fetched per query


def _ts_expr(t: Table, alias: str = "") -> str | None:
    types = dict(t.columns)
    if t.time_col and kind_of(types.get(t.time_col, "")) == "datetime":
        return f"(epoch_ns({alias}{ident(t.time_col)}) // 1000000000)"
    return None


class Retriever:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        db: Database,
        encoder: Encoder,
        config: RetrieveConfig | None = None,
    ):
        self.con = con
        self.db = db
        self.enc = encoder
        self.cfg = config or RetrieveConfig()

    # -- queries ----------------------------------------------------------

    def _run(self, ctx: Context, step: str, sql: str, params=()) -> list[tuple]:
        tic = time.perf_counter()
        rows = self.con.execute(sql, list(params)).fetchall()
        ctx.queries.append(
            Query(step, sql, len(rows), (time.perf_counter() - tic) * 1e3)
        )
        return rows

    def _node(self, t: Table, values: tuple, seed: int, depth: int, via: str) -> Node:
        row, fks, cells, ts = self.enc.decode_row(t, values)
        enc = self.enc.tables[t.key]
        parents = [
            (t.fkeys[col], int(v))
            for col, v in zip(enc.fk_columns, fks)
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        return Node(t, row, parents, cells, ts, seed, depth, via)

    def _visible(self, ts_sql: str | None, t: Table, bound, target_ts) -> str:
        """SQL predicate: row is not after `bound`, and is not a same-horizon task row."""
        if ts_sql is None:
            return "TRUE"
        if bound is None:
            return f"{ts_sql} IS NULL"
        pred = f"({ts_sql} IS NULL OR {ts_sql} <= {int(bound)})"
        if t.is_task and target_ts is not None:
            pred += f" AND ({ts_sql} IS NULL OR {ts_sql} <> {int(target_ts)})"
        return pred

    # -- the context --------------------------------------------------------

    def build(self, task_name: str, split: str, row: int) -> Context:
        task = self.db.tasks[task_name]
        target_table = task.splits[split]
        target_column = f"{task.target_col} of {task.name}"

        ctx = Context(target=None, target_column=target_column, label=None)  # type: ignore[arg-type]
        (values,) = self._run(
            ctx,
            "target",
            f"SELECT {self.enc.select_list(target_table)} FROM {target_table.sql} "
            f"WHERE {ROW} = ?",
            [row],
        )
        target = self._node(target_table, values, seed=0, depth=0, via="target")
        # The label stays out of the context: drop the target column's cell.
        target.cells = [c for c in target.cells if c.column != target_column]
        ctx.target = target
        ctx.label = self._raw_label(ctx, target_table, task.target_col, row)

        visited: set[tuple[str, int]] = set()
        expanded: set[tuple[str, int]] = set()
        ctx.nodes.append(target)
        ctx.tokens.append((0, -1))  # the target cell

        seeds = [(target_table, target)]
        seeds_iter = self._examples(ctx, task, target, target_table)
        seed_i = 0
        while len(ctx.tokens) < self.cfg.ctx_size:
            if seed_i >= len(seeds):
                more = next(seeds_iter, None)
                if more is None:
                    break
                seeds.append(more)
            t, seed_node = seeds[seed_i]
            self._expand(ctx, seed_i, t, seed_node, target, visited, expanded)
            seed_i += 1
        ctx.tokens = ctx.tokens[: self.cfg.ctx_size]
        return ctx

    def _raw_label(self, ctx: Context, t: Table, target_col: str, row: int):
        if target_col not in dict(t.columns):
            return None
        (v,) = self._run(
            ctx,
            "label",
            f"SELECT {ident(target_col)}::DOUBLE FROM {t.sql} WHERE {ROW} = ?",
            [row],
        )[0]
        return v

    def _examples(self, ctx: Context, task, target: Node, target_table: Table):
        """Labelled rows of the task, earlier than the target, best first."""
        target_ts = target.timestamp
        fk_columns = self.enc.tables[target_table.key].fk_columns
        entity_col = fk_columns[0] if fk_columns else None
        target_entity = (
            target.parents[0][1]
            if entity_col
            and target.parents
            and target.parents[0][0] == target_table.fkeys[entity_col]
            else None
        )
        parts = []
        for split, t in task.splits.items():
            ts = _ts_expr(t) or "NULL::BIGINT"
            label = ident(task.target_col)
            types = dict(t.columns)
            if task.target_col not in types:
                continue
            not_nan = (
                f" AND NOT isnan({label})"
                if kind_of(types[task.target_col]) == "float"
                else ""
            )
            when = (
                f" AND {ts} < {int(target_ts)}"
                if target_ts is not None and _ts_expr(t)
                else ""
            )
            same = (
                f"({ident(entity_col)} = {int(target_entity)})"
                if entity_col and target_entity is not None
                else "FALSE"
            )
            not_target = (
                f" AND {ROW} <> {int(target.row)}" if t.key == target_table.key else ""
            )
            parts.append(
                f"SELECT '{split}' AS split, {ROW} AS r, {ts} AS ts, {same} AS same "
                f"FROM {t.sql} WHERE {label} IS NOT NULL{not_nan}{when}{not_target}"
            )
        if not parts:
            return
        order = " UNION ALL ".join(parts)
        offset = 0
        while True:
            picked = self._run(
                ctx,
                "examples",
                f"SELECT split, r FROM ({order}) "
                f"ORDER BY same DESC, ts DESC NULLS LAST, split, r "
                f"LIMIT {self.cfg.example_batch} OFFSET {offset}",
            )
            if not picked:
                return
            offset += len(picked)
            by_split: dict[str, list[int]] = {}
            for split, r in picked:
                by_split.setdefault(split, []).append(r)
            fetched = {}
            for split, rows in by_split.items():
                t = task.splits[split]
                for values in self._run(
                    ctx,
                    f"fetch {t.key}",
                    f"SELECT {self.enc.select_list(t)} FROM {t.sql} "
                    f"WHERE {ROW} IN (SELECT unnest(?::BIGINT[]))",
                    [rows],
                ):
                    fetched[(split, values[0])] = values
            for split, r in picked:
                t = task.splits[split]
                yield t, self._node(t, fetched[(split, r)], 0, 0, "example")

    def _expand(self, ctx, seed_i, seed_table, seed_node, target, visited, expanded):
        """Breadth-first expansion of one seed, appending its rows' cells."""
        # As in the sampler: a seed's neighbourhood is read as of the seed's own
        # time, so an example's rows match what was known when its label was set.
        bound = seed_node.timestamp
        target_ts = target.timestamp
        seed_node.seed = seed_i
        level = [seed_node]
        depth = 0
        cells = 0
        while level:
            nxt: list[tuple[Table, int, str]] = []  # (table, row, via) to fetch
            frontier: list[Node] = []
            for node in level:
                if node.key in expanded:
                    continue
                cells += len(node.cells) + (1 if node is target else 0)
                if cells >= self.cfg.local_ctx_size and node is not seed_node:
                    return
                expanded.add(node.key)
                frontier.append(node)
                self._add(ctx, node, visited)
                if len(ctx.tokens) >= self.cfg.ctx_size:
                    return
                for (ptable, prow), col in zip(
                    node.parents, self.enc.tables[node.table.key].fk_columns
                ):
                    nxt.append(
                        (
                            self.db.tables[ptable],
                            prow,
                            f"parent via {col} of {node.table.key}/{node.row}",
                        )
                    )

            next_nodes = self._fetch_parents(ctx, nxt, seed_i, depth + 1, bound)
            next_nodes += self._fetch_children(
                ctx, frontier, seed_table, seed_i, depth + 1, bound, target_ts
            )
            level = next_nodes
            depth += 1

    def _add(self, ctx: Context, node: Node, visited) -> None:
        if node.key in visited:
            return
        visited.add(node.key)
        if node is not ctx.target:
            ctx.nodes.append(node)
        i = ctx.nodes.index(node) if node is ctx.target else len(ctx.nodes) - 1
        for ci in range(len(node.cells)):
            ctx.tokens.append((i, ci))

    def _fetch_parents(self, ctx, wanted, seed_i, depth, bound) -> list[Node]:
        by_table: dict[str, list[tuple[int, str]]] = {}
        for t, r, via in wanted:
            by_table.setdefault(t.name, []).append((r, via))
        out = []
        for name, items in by_table.items():
            t = self.db.tables[name]
            ts = _ts_expr(t)
            visible = (
                "TRUE"
                if bound is None or ts is None
                else f"({ts} IS NULL OR {ts} <= {int(bound)})"
            )
            rows = sorted({r for r, _ in items})
            fetched = {
                v[0]: v
                for v in self._run(
                    ctx,
                    f"parents {t.key}",
                    f"SELECT {self.enc.select_list(t)} FROM {t.sql} "
                    f"WHERE {ROW} IN (SELECT unnest(?::BIGINT[])) AND {visible}",
                    [rows],
                )
            }
            for r, via in items:
                if r in fetched:
                    out.append(self._node(t, fetched[r], seed_i, depth, via))
        return out

    def _fetch_children(
        self, ctx, frontier, seed_table, seed_i, depth, bound, target_ts
    ):
        by_table: dict[str, list[Node]] = {}
        for node in frontier:
            if not node.table.is_task:
                by_table.setdefault(node.table.name, []).append(node)
        out = []
        for parent_name, parents in by_table.items():
            rows = sorted({n.row for n in parents})
            for child, col in self.db.children_of(parent_name):
                # As in the sampler: a child is a database row or a row of
                # the seed's own table.
                if child.is_task and child.name != seed_table.name:
                    continue
                ts = _ts_expr(child)
                visible = self._visible(ts, child, bound, target_ts)
                order = f"{ts} DESC NULLS LAST, {ROW} DESC" if ts else f"{ROW} DESC"
                fetched = self._run(
                    ctx,
                    f"children {child.key} via {col}",
                    f"SELECT {self.enc.select_list(child)}, {ident(col)} AS __parent "
                    f"FROM {child.sql} "
                    f"WHERE {ident(col)} IN (SELECT unnest(?::BIGINT[])) AND {visible} "
                    f"QUALIFY row_number() OVER (PARTITION BY {ident(col)} ORDER BY {order}) "
                    f"<= {self.cfg.bfs_width}",
                    [rows],
                )
                for values in fetched:
                    parent_row = values[-1]
                    out.append(
                        self._node(
                            child,
                            values[:-1],
                            seed_i,
                            depth,
                            f"child via {col} of {parent_name}/{parent_row}",
                        )
                    )
        return out
