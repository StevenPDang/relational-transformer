from datetime import datetime

from rt.run_dirs import eval_kind, eval_run_dir, link_latest, rules_path

NOW = datetime(2026, 10, 18, 18, 30, 12)


def test_eval_kind_names_mode_and_sampler():
    kind = lambda **kw: eval_kind(**{"raw": False, "sampler": "bfs", "sql_context_db": None,
                                    "sql_context_rules": None, "mode": "simple", **kw})
    assert kind() == "prepared-bfs"
    assert kind(sql_context_db="f1.duckdb") == "prepared-sql-metasieve"
    assert kind(sql_context_db="f1.duckdb", sql_context_rules="r.json") == "prepared-sql-metasieve"
    assert kind(raw=True) == "raw-bfs"
    assert kind(raw=True, sampler="sql", sql_context_rules="r.json") == "raw-sql-metasieve"
    assert kind(mode="ensemble") == "prepared-bfs-ensemble"


def test_eval_run_dirs_group_by_task_and_never_collide(tmp_path):
    first = eval_run_dir(tmp_path, ["rel-f1/driver-top3"], "raw-bfs", now=NOW)
    assert first == tmp_path / "eval" / "rel-f1-driver-top3" / "20261018-183012_raw-bfs"
    first.mkdir(parents=True)
    second = eval_run_dir(tmp_path, ["rel-f1/driver-top3"], "raw-bfs", now=NOW)
    assert second.name == "20261018-183012_raw-bfs-2"
    assert eval_run_dir(tmp_path, None, "prepared-bfs", now=NOW).parent.name == "all-tasks"
    assert eval_run_dir(tmp_path, ["rel-f1", "rel-trial"], "prepared-bfs", now=NOW).parent.name == \
        "rel-f1+rel-trial"


def test_rules_path_and_latest_link(tmp_path):
    path = rules_path(tmp_path, "stanford-star/relbench/rel-f1", "driver-top3", now=NOW)
    assert path == tmp_path / "metasieve" / "rel-f1-driver-top3" / "20261018-183012.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    latest = link_latest(path)
    assert latest.resolve() == path.resolve()
    newer = path.with_name("20261018-190000.json")
    newer.write_text("[]")
    assert link_latest(newer).read_text() == "[]"
