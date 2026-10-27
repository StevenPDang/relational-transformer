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

Each step is one query per table or FK over the whole BFS level of a batch
of `seed_batch` seeds, never one query per row or per seed. With
`seed_batch=1`, seeds expand one after another and each skips what earlier
seeds reached; in a batch, seeds expand independently and a row two of them
reach is added once, by the first.
"""

import math
import time
from dataclasses import dataclass, field

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
    seed_batch: int = 1  # seeds expanded together, one query per BFS step


# Each wanted row with its seed's time bound and index, one parameter array each.
_WANTED = (
    "(SELECT unnest(?::BIGINT[]) AS __pid, unnest(?::BIGINT[]) AS __bound, "
    "unnest(?::BIGINT[]) AS __seed)"
)


@dataclass
class _SeedBfs:
    """One seed's breadth-first expansion, advanced a level at a time."""

    seed_i: int
    table: Table
    node: Node
    level: list[Node] = field(default_factory=list)
    depth: int = 0
    cells: int = 0
    seen: set[tuple[str, int]] = field(default_factory=set)
    collected: list[Node] = field(default_factory=list)  # in BFS order
    frontier: list[Node] = field(default_factory=list)  # claimed on this level
    wanted: list[tuple[Table, int, str]] = field(default_factory=list)

    def __post_init__(self):
        self.level = [self.node]


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
            while len(seeds) < seed_i + self.cfg.seed_batch:
                more = next(seeds_iter, None)
                if more is None:
                    break
                seeds.append(more)
            batch = [
                (i, t, node)
                for i, (t, node) in enumerate(
                    seeds[seed_i : seed_i + self.cfg.seed_batch], seed_i
                )
            ]
            if not batch:
                break
            self._expand_batch(ctx, batch, target, visited, expanded)
            seed_i += len(batch)
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

    def _expand_batch(self, ctx, seeds, target, visited, expanded):
        """Breadth-first expansion of several seeds at once, appending their cells.

        Each BFS level is one query per table or FK for every seed in the batch,
        with each row carrying its seed's time bound. Seeds in a batch expand
        independently and are appended in seed order; a row another seed in the
        batch already added is skipped then, as across batches.
        """
        target_ts = target.timestamp
        states = [_SeedBfs(seed_i, t, node) for seed_i, t, node in seeds]
        for st in states:
            st.node.seed = st.seed_i
        while any(st.level for st in states):
            for st in states:
                self._take_level(st, target, expanded)
            parents = self._fetch_parents(ctx, states)
            children = self._fetch_children(ctx, states, target_ts)
            for st in states:
                st.level = parents.get(st.seed_i, []) + children.get(st.seed_i, [])
                st.depth += 1

        for st in states:
            for node in st.collected:
                expanded.add(node.key)
                self._add(ctx, node, visited)
                if len(ctx.tokens) >= self.cfg.ctx_size:
                    return

    def _take_level(self, st: "_SeedBfs", target: Node, expanded) -> None:
        """Claim this seed's rows on its current level, until its local budget is spent."""
        st.frontier, st.wanted = [], []
        for node in st.level:
            if node.key in expanded or node.key in st.seen:
                continue
            st.cells += len(node.cells) + (1 if node is target else 0)
            if st.cells >= self.cfg.local_ctx_size and node is not st.node:
                st.frontier, st.wanted, st.level = [], [], []
                return
            st.seen.add(node.key)
            st.collected.append(node)
            st.frontier.append(node)
            for (ptable, prow), col in zip(
                node.parents, self.enc.tables[node.table.key].fk_columns
            ):
                # Already claimed: the next level would skip it, so don't fetch it.
                if (ptable, prow) in expanded or (ptable, prow) in st.seen:
                    continue
                st.wanted.append(
                    (
                        self.db.tables[ptable],
                        prow,
                        f"parent via {col} of {node.table.key}/{node.row}",
                    )
                )

    def _add(self, ctx: Context, node: Node, visited) -> None:
        if node.key in visited:
            return
        visited.add(node.key)
        if node is not ctx.target:
            ctx.nodes.append(node)
        i = ctx.nodes.index(node) if node is ctx.target else len(ctx.nodes) - 1
        for ci in range(len(node.cells)):
            ctx.tokens.append((i, ci))

    def _fetch_parents(self, ctx, states) -> dict[int, list[Node]]:
        """Each seed's wanted parents, one query per parent table for the batch."""
        tables: dict[str, Table] = {}
        for st in states:
            for t, _, _ in st.wanted:
                tables.setdefault(t.name, t)
        fetched: dict[tuple[str, int, int], tuple] = {}
        for name, t in tables.items():
            pids, bounds, seed_is = [], [], []
            for st in states:
                for r in sorted({r for pt, r, _ in st.wanted if pt.name == name}):
                    pids.append(r)
                    bounds.append(st.node.timestamp)
                    seed_is.append(st.seed_i)
            ts = _ts_expr(t)
            visible = (
                "TRUE"
                if ts is None
                else f"(w.__bound IS NULL OR {ts} IS NULL OR {ts} <= w.__bound)"
            )
            for v in self._run(
                ctx,
                f"parents {t.key}",
                f"SELECT {self.enc.select_list(t)}, w.__seed FROM {t.sql} "
                f"JOIN {_WANTED} w ON {ROW} = w.__pid WHERE {visible}",
                [pids, bounds, seed_is],
            ):
                fetched[(name, v[0], v[-1])] = v[:-1]
        out: dict[int, list[Node]] = {}
        for st in states:
            # Grouped by table in this seed's own order, as one seed's BFS was.
            by_table: dict[str, list[tuple[int, str]]] = {}
            for t, r, via in st.wanted:
                by_table.setdefault(t.name, []).append((r, via))
            nodes = out.setdefault(st.seed_i, [])
            for name, items in by_table.items():
                for r, via in items:
                    v = fetched.get((name, r, st.seed_i))
                    if v is not None:
                        nodes.append(
                            self._node(tables[name], v, st.seed_i, st.depth + 1, via)
                        )
        return out

    def _fetch_children(self, ctx, states, target_ts) -> dict[int, list[Node]]:
        """Up to `bfs_width` children per FK of each seed's frontier, one query per FK."""
        fetched: dict[tuple[str, str, int], list[tuple]] = {}
        for parent_name in dict.fromkeys(
            n.table.name for st in states for n in st.frontier if not n.table.is_task
        ):
            pids, bounds, seed_is = [], [], []
            for st in states:
                for r in sorted(
                    {n.row for n in st.frontier if n.table.name == parent_name}
                ):
                    pids.append(r)
                    bounds.append(st.node.timestamp)
                    seed_is.append(st.seed_i)
            if not pids:
                continue
            for child, col in self.db.children_of(parent_name):
                # As in the sampler: a child is a database row or a row of
                # the seed's own table. Every seed is the target's task.
                if child.is_task and child.name != states[0].table.name:
                    continue
                ts = _ts_expr(child)
                if ts is None:
                    visible = "TRUE"
                else:
                    visible = (
                        f"(CASE WHEN w.__bound IS NULL THEN {ts} IS NULL "
                        f"ELSE ({ts} IS NULL OR {ts} <= w.__bound) END)"
                    )
                    if child.is_task and target_ts is not None:
                        visible += f" AND ({ts} IS NULL OR {ts} <> {int(target_ts)})"
                order = f"{ts} DESC NULLS LAST, {ROW} DESC" if ts else f"{ROW} DESC"
                for v in self._run(
                    ctx,
                    f"children {child.key} via {col}",
                    f"SELECT {self.enc.select_list(child)}, {ident(col)} AS __parent, "
                    f"w.__seed FROM {child.sql} "
                    f"JOIN {_WANTED} w ON {ident(col)} = w.__pid WHERE {visible} "
                    f"QUALIFY row_number() OVER (PARTITION BY w.__seed, {ident(col)} "
                    f"ORDER BY {order}) <= {self.cfg.bfs_width} "
                    # Without it, DuckDB's parallel scan returns rows in any order.
                    f"ORDER BY w.__seed, {ident(col)}, {order}",
                    [pids, bounds, seed_is],
                ):
                    fetched.setdefault((child.key, col, v[-1]), []).append(v[:-1])
        out: dict[int, list[Node]] = {}
        for st in states:
            # Parent tables in this seed's own frontier order, as one seed's BFS was.
            nodes = out.setdefault(st.seed_i, [])
            for parent_name in dict.fromkeys(
                n.table.name for n in st.frontier if not n.table.is_task
            ):
                for child, col in self.db.children_of(parent_name):
                    for v in fetched.get((child.key, col, st.seed_i), []):
                        nodes.append(
                            self._node(
                                child,
                                v[:-1],
                                st.seed_i,
                                st.depth + 1,
                                f"child via {col} of {parent_name}/{v[-1]}",
                            )
                        )
        return out
