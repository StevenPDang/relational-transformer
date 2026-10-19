import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from rt import metasieve
from rt.sql_context import historical_seed_orders, metasieve_context_nodes

# drivers <- results -> races -> circuits; results -> constructors
SCHEMA = {
    "drivers": {"pkey": "driverId", "time": None, "fkeys": {}},
    "constructors": {"pkey": "constructorId", "time": None, "fkeys": {}},
    "circuits": {"pkey": "circuitId", "time": None, "fkeys": {}},
    "races": {"pkey": "raceId", "time": "date", "fkeys": {"circuitId": "circuits"}},
    "results": {"pkey": "resultId", "time": "date",
                "fkeys": {"driverId": "drivers", "raceId": "races", "constructorId": "constructors"}},
}
OFFSETS = {"drivers": 100, "results": 200, "races": 300, "constructors": 400, "circuits": 450}
INFO = {f"{t}:Db": {"node_idx_offset": o} for t, o in OFFSETS.items()}
TASK = metasieve.task_info(SCHEMA, db="rel-f1", name="driver-top3", entity_table="drivers",
                           entity_col="driverId", time_col="date", target_col="qualifying", task_type="clf")

RES = ("<-results.driverId",)


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE drivers(driverId BIGINT)")
    con.execute("INSERT INTO drivers VALUES (42), (7)")
    con.execute("CREATE TABLE constructors(constructorId BIGINT)")
    con.execute("INSERT INTO constructors VALUES (8), (9)")
    con.execute("CREATE TABLE circuits(circuitId BIGINT)")
    con.execute("INSERT INTO circuits VALUES (5)")
    con.execute("CREATE TABLE races(raceId BIGINT, circuitId BIGINT, date TIMESTAMP)")
    con.execute("INSERT INTO races VALUES (12,5,'2020-01-01'),(13,5,'2020-01-03'),(14,5,'2020-01-02')")
    con.execute("CREATE TABLE results(resultId BIGINT, driverId BIGINT, raceId BIGINT, "
                "constructorId BIGINT, date TIMESTAMP)")
    # rowids 0..4: driver 42 races 12,13,14 (two constructors), driver 7 race 12 and 14
    con.execute("INSERT INTO results VALUES (1,42,12,8,'2020-01-01'),(2,42,13,8,'2020-01-03'),"
                "(3,7,12,9,'2020-01-01'),(4,42,14,9,'2020-01-02'),(5,7,14,9,'2020-01-02')")
    yield con
    con.close()


def seeds(*rows):
    return pd.DataFrame(rows, columns=["seed", "key", "ts"]).astype({"ts": "datetime64[ns]"})


def by_seed(frame):
    """:func:`metasieve.context_nodes` output as {seed: [(node, depth), ...]}."""
    return {seed: list(zip(g["node"].tolist(), g["depth"].tolist())) for seed, g in frame.groupby("seed")}


def by_name(paths):
    return {metasieve.path_name(p): p for p in paths}


def test_enumeration_skips_immediate_backtracking():
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 3))
    assert "<-results.driverId" in paths
    assert "<-results.driverId ->races[raceId]" in paths
    assert "<-results.driverId ->races[raceId] <-results.raceId" in paths  # siblings are new rows
    assert "<-results.driverId ->drivers[driverId]" not in paths  # straight back to the seed
    assert all(len(p) <= 3 for p in paths.values())
    with pytest.raises(ValueError, match="positive"):
        metasieve.enumerate_metapaths(SCHEMA, "drivers", 0)


def test_counts_are_distinct_and_respect_the_time_cutoff(con):
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 3))
    names = ["<-results.driverId", "<-results.driverId ->constructors[constructorId]",
             "<-results.driverId ->races[raceId] ->circuits[circuitId]",
             "<-results.driverId ->races[raceId] <-results.raceId"]
    s = seeds((0, 42, "2020-01-02"), (1, 7, "2020-01-02"), (2, 42, "2019-01-01"))
    counts = metasieve.path_counts(con, SCHEMA, "drivers", s, [paths[n] for n in names])
    assert counts[0].tolist() == [2, 2, 0]  # race 13 is after the cutoff
    assert counts[1].tolist() == [2, 1, 0]  # constructors 8 and 9, counted once each
    assert counts[2].tolist() == [1, 1, 0]  # two races, one shared circuit
    assert counts[3].tolist() == [4, 4, 0]  # every result in races 12 and 14
    strict = metasieve.path_counts(con, SCHEMA, "drivers", s, [paths[names[0]]], strict=True)
    assert strict[0].tolist() == [1, 1, 0]


def test_scoring_keeps_informative_paths_hop1_and_gateways():
    rng = np.random.default_rng(0)
    n = 800
    y = rng.integers(0, 2, n)
    hop1 = ("rev", "drivers", "driverId", "results")
    paths = [(hop1,)]
    counts = {0: rng.poisson(20, n).astype(float)}
    # Hop 2: three paths whose counts track the label, five pure-noise paths.
    for i in range(8):
        paths.append((hop1, ("fwd", "results", f"c{i}", f"t{i}")))
        signal = y * 6 if i < 3 else 0
        counts[len(paths) - 1] = rng.poisson(3, n) + signal
    # A hop-3 informative child below a noise parent keeps the parent as a gateway.
    paths.append(paths[8] + (("rev", "t7", "k", "u"),))
    counts[len(paths) - 1] = rng.poisson(2, n) + y * 6
    # A path no seed ever reaches is removed outright.
    paths.append(paths[4] + (("rev", "t3", "k", "empty"),))
    counts[len(paths) - 1] = np.zeros(n)
    frame = pd.DataFrame(counts, index=pd.RangeIndex(n, name="seed")).astype(float)

    rows = metasieve.score_metapaths(frame, paths, y, "clf", num_batches=8, delta=0.3)
    keep = {i: r["keep"] for i, r in enumerate(rows)}
    assert rows[0]["cluster"] == "hop1" and keep[0]
    assert all(keep[i] for i in (1, 2, 3)), "informative hop-2 paths are kept"
    assert not any(keep[i] for i in (4, 5, 6, 7)), "noise hop-2 paths are dropped"
    assert rows[9]["cluster"] == "good" and keep[9]
    assert rows[8]["cluster"] == "bad" and keep[8], "noise parent kept as a gateway"
    assert rows[10]["cluster"] == "empty" and not keep[10]
    with pytest.raises(ValueError, match="delta"):
        metasieve.score_metapaths(frame, paths, y, "clf", delta=0.5)


def test_regression_labels_score_without_errors():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.normal(size=n)
    hop1 = ("rev", "drivers", "driverId", "results")
    paths = [(hop1,), (hop1, ("fwd", "results", "a", "a")), (hop1, ("fwd", "results", "b", "b"))]
    frame = pd.DataFrame({0: rng.poisson(5, n), 1: np.round(np.exp(y) * 3), 2: rng.poisson(5, n)},
                         index=pd.RangeIndex(n, name="seed")).astype(float)
    rows = metasieve.score_metapaths(frame, paths, y, "reg", num_batches=4)
    assert rows[1]["lcb_count"] > rows[2]["lcb_count"]


def test_context_rows_cap_rev_steps_and_keep_children_beside_parents(con):
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 3))
    kept = [paths["<-results.driverId"], paths["<-results.driverId ->races[raceId]"],
            paths["<-results.driverId ->races[raceId] ->circuits[circuitId]"]]
    s = seeds((900, 42, "2020-01-03"))
    rows = metasieve.path_rows(con, SCHEMA, "drivers", s, kept, width=2)
    nodes = by_seed(metasieve.context_nodes(rows, INFO))[900]
    # Two most recent results (rowids 1: race 13, 3: race 14), each followed by its race;
    # the shared circuit appears once, beside the first race.
    assert nodes == [(201, 2), (301, 3), (450, 4), (203, 2), (302, 3)]
    with pytest.raises(ValueError, match="positive"):
        metasieve.path_rows(con, SCHEMA, "drivers", s, kept, width=0)


def test_fan_out_rows_follow_the_whole_spine(con):
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 3))
    kept = [paths["<-results.driverId"], paths["<-results.driverId ->races[raceId]"],
            paths["<-results.driverId ->races[raceId] <-results.raceId"]]
    s = seeds((900, 42, "2020-01-03"))
    nodes = by_seed(metasieve.context_nodes(metasieve.path_rows(con, SCHEMA, "drivers", s, kept, width=3), INFO))[900]
    # Every history row and its race first; then other results in those races,
    # capped at 3 (two are the driver's own rows), so only driver 7's result 204 is new.
    assert nodes == [(201, 2), (301, 3), (203, 2), (302, 3), (200, 2), (300, 3), (204, 4)]


def test_contexts_put_entity_and_labels_before_metapath_rows(con):
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 2))
    rules = {"task": TASK, "schema": SCHEMA, "params": {"strict": False}, "paths": [
        {"path": [list(st) for st in paths[name]], "keep": keep} for name, keep in [
            ("<-results.driverId", True), ("<-results.driverId ->races[raceId]", False),
            ("<-results.driverId ->constructors[constructorId]", True)]]}
    # Task rows double as the label table: 501 is an earlier label of driver 42.
    s = seeds((501, 42, "2019-06-01"), (502, 42, "2020-01-02"), (503, 7, "2019-06-01"))
    ctx = metasieve_context_nodes(con, INFO, rules, s, width=4)
    # Driver, earlier label 501 (not its own 502), then results with constructors; no races.
    assert ctx[502] == [(100, 1), (501, 2), (203, 2), (401, 3), (200, 2), (400, 3)]
    assert ctx[501] == [(100, 1)]  # driver 42 had no results before 2019-06-01
    assert ctx[503] == [(101, 1)]
    # Chunking frontiers changes nothing.
    assert metasieve_context_nodes(con, INFO, rules, s, width=4, chunk=1).to_dict() == ctx.to_dict()


def test_any_schema_entity_key_names_and_string_keys():
    """Nothing assumes rel-f1: entity key named differently in the task, string keys,
    untimestamped parents, and a task table that is not keyed like the entity."""
    con = duckdb.connect()
    con.execute("CREATE TABLE users(user_id VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO users VALUES ('u1','fr'), ('u2','de')")
    con.execute("CREATE TABLE items(item_id VARCHAR, brand VARCHAR)")
    con.execute("INSERT INTO items VALUES ('i1','x'), ('i2','y')")
    con.execute("CREATE TABLE reviews(review_id BIGINT, uid VARCHAR, item VARCHAR, ts TIMESTAMP)")
    con.execute("INSERT INTO reviews VALUES (1,'u1','i1','2020-01-01'),(2,'u1','i2','2020-02-01'),"
                "(3,'u2','i1','2020-03-01')")
    schema = {
        "users": {"pkey": "user_id", "time": None, "fkeys": {}},
        "items": {"pkey": "item_id", "time": None, "fkeys": {}},
        "reviews": {"pkey": "review_id", "time": "ts", "fkeys": {"uid": "users", "item": "items"}},
    }
    info = {f"{t}:Db": {"node_idx_offset": o} for t, o in [("users", 10), ("items", 20), ("reviews", 30)]}
    task = metasieve.task_info(schema, db="shop", name="user-churn", entity_table="users",
                               entity_col="customer", time_col="when", target_col="churn", task_type="clf")
    assert task["entity_pkey"] == "user_id"
    rules = metasieve.keep_all_rules(schema, task, hops=2)
    s = pd.DataFrame({"seed": [900, 901, 902], "key": ["u1", "u1", "u2"],
                      "ts": pd.to_datetime(["2020-01-15", "2020-03-01", "2020-03-01"])})
    try:
        ctx = metasieve_context_nodes(con, info, rules, s, width=4)
        orders = historical_seed_orders(s)
    finally:
        con.close()
    # u1 at 2020-03-01: user row, earlier label 900, both reviews (newest first) with items.
    assert ctx[901][:2] == [(10, 1), (900, 2)]
    assert [n for n, _ in ctx[901] if 30 <= n < 40] == [31, 30]
    assert {21, 20} <= {n for n, _ in ctx[901]}
    assert ctx[902][0] == (11, 1) and 30 + 2 in {n for n, _ in ctx[902]}
    assert orders.to_dict() == {900: [], 901: [900], 902: [900]}


def test_rules_resolve_per_task_and_reject_other_tasks(tmp_path):
    from rt.sql_context import load_metasieve_rules, resolve_duckdb

    rules = metasieve.keep_all_rules(SCHEMA, TASK, hops=1)
    folder = tmp_path / "rel-f1-driver-top3"
    folder.mkdir()
    (folder / "latest.json").write_text(json.dumps(rules))
    assert load_metasieve_rules(tmp_path, "rel-f1", "driver-top3")["task"] == TASK
    assert load_metasieve_rules(folder / "latest.json", "rel-f1", "driver-top3")["task"] == TASK
    with pytest.raises(ValueError, match="not rel-f1/driver-dnf"):
        load_metasieve_rules(folder / "latest.json", "rel-f1", "driver-dnf")
    with pytest.raises(FileNotFoundError, match="scripts.metasieve"):
        load_metasieve_rules(tmp_path, "rel-trial", "study-outcome")
    old = dict(rules)
    del old["task"]
    (tmp_path / "old.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="older format"):
        load_metasieve_rules(tmp_path / "old.json", "rel-f1", "driver-top3")
    assert resolve_duckdb(tmp_path, "rel-trial") == str(tmp_path / "rel-trial.duckdb")
    assert resolve_duckdb(tmp_path / "f1.duckdb", "rel-trial") == str(tmp_path / "f1.duckdb")


def test_path_counts_chunking_is_exact(con):
    paths = metasieve.enumerate_metapaths(SCHEMA, "drivers", 3)
    s = seeds((0, 42, "2020-01-02"), (1, 7, "2020-01-02"), (2, 42, "2020-01-03"))
    whole = metasieve.path_counts(con, SCHEMA, "drivers", s, paths)
    pd.testing.assert_frame_equal(metasieve.path_counts(con, SCHEMA, "drivers", s, paths, chunk=2), whole)


def test_cap_seeds_keeps_each_label_and_order():
    from scripts.metasieve import cap_seeds

    train = pd.DataFrame({"y": [0] * 90 + [1] * 10, "x": range(100)})
    capped = cap_seeds(train, "y", "clf", 20)
    assert capped["y"].value_counts().to_dict() == {0: 10, 1: 10}
    assert capped.index.is_monotonic_increasing
    assert len(cap_seeds(train, "y", "reg", 30)) == 30
    assert cap_seeds(train, "y", "clf", None) is train
