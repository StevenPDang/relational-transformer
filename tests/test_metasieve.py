import duckdb
import numpy as np
import pandas as pd
import pytest

from rt import metasieve
from rt.sql_context import metasieve_context_nodes

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
    nodes = metasieve.context_nodes(rows, INFO)[900]
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
    nodes = metasieve.context_nodes(metasieve.path_rows(con, SCHEMA, "drivers", s, kept, width=3), INFO)[900]
    # Every history row and its race first; then other results in those races,
    # capped at 3 (two are the driver's own rows), so only driver 7's result 204 is new.
    assert nodes == [(201, 2), (301, 3), (203, 2), (302, 3), (200, 2), (300, 3), (204, 4)]


def test_rel_f1_contexts_put_driver_and_labels_before_metapath_rows(con):
    labels = pd.DataFrame({"node_idx": [501, 502, 503], "driverId": [42, 42, 7],
                           "date": pd.to_datetime(["2019-06-01", "2020-01-02", "2019-06-01"])})
    con.register("sql_task_labels", labels)
    paths = by_name(metasieve.enumerate_metapaths(SCHEMA, "drivers", 2))
    rules = {"entity_table": "drivers", "schema": SCHEMA, "params": {"strict": False}, "paths": [
        {"path": [list(st) for st in paths[name]], "keep": keep} for name, keep in [
            ("<-results.driverId", True), ("<-results.driverId ->races[raceId]", False),
            ("<-results.driverId ->constructors[constructorId]", True)]]}
    s = seeds((502, 42, "2020-01-02"), (503, 7, "2019-06-01"))
    ctx = metasieve_context_nodes(con, INFO, rules, s, width=4)
    # Driver, earlier label 501 (not its own 502), then results with constructors; no races.
    assert ctx[502] == [(100, 1), (501, 2), (203, 2), (401, 3), (200, 2), (400, 3)]
    assert ctx[503] == [(101, 1)]  # nothing of driver 7 happened before 2019-06-01
