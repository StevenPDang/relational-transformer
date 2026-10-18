"""Timestamped output folders for eval runs and MetaSieve rule sets.

Runs of the same kind share a folder and sort by start time:

    runs/eval/rel-f1-driver-top3/20261018-183012_raw-sql-metasieve/
    runs/metasieve/rel-f1-driver-top3/20261018-182950.json  (+ latest.json)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def stamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-") or "all"


def eval_kind(*, raw: bool, sampler: str, sql_context_db, sql_context_rules, mode: str) -> str:
    """Short run type, e.g. raw-bfs, prepared-sql-metasieve, prepared-bfs-ensemble."""
    kind = sampler if raw else ("sql" if sql_context_db else "bfs")
    if sql_context_rules:
        kind += "-metasieve"
    kind = f"{'raw' if raw else 'prepared'}-{kind}"
    return kind if mode == "simple" else f"{kind}-{mode}"


def task_group(tasks) -> str:
    return slug("+".join(tasks)) if tasks else "all-tasks"


def fresh_dir(parent: Path, name: str) -> Path:
    """``parent/name``, suffixed -2, -3, ... if a run with that name exists."""
    path, n = parent / name, 2
    while path.exists():
        path, n = parent / f"{name}-{n}", n + 1
    return path


def eval_run_dir(runs_dir, tasks, kind, now=None) -> Path:
    parent = Path(runs_dir).expanduser() / "eval" / task_group(tasks)
    return fresh_dir(parent, f"{stamp(now)}_{kind}")


def rules_path(runs_dir, source, task, now=None) -> Path:
    db = Path(str(source).rstrip("/")).name
    parent = Path(runs_dir).expanduser() / "metasieve" / slug(f"{db}-{task}")
    path = fresh_dir(parent, stamp(now))
    return path.with_name(path.name + ".json")


def link_latest(path: Path) -> Path:
    """Point ``latest.json`` beside ``path`` at it, so commands can stay fixed."""
    latest = path.with_name("latest.json")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)
    return latest
