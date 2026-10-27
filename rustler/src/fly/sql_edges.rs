//! FK edges served by DuckDB queries instead of the preprocessed adjacency.
//!
//! Every edge `pre.rs` writes is a foreign key, so it can be asked for at
//! sampling time instead:
//!
//! - f2p (child -> parent): `SELECT fk.. FROM child WHERE __row = ?`, then
//!   `SELECT __ts FROM parent WHERE __row = ?` for each FK value.
//! - p2f (parent -> child): `SELECT __row, __ts FROM child WHERE fk = ?`, with
//!   the seed-time cut applied to the returned rows in Rust.
//!
//! Each table is loaded from the same parquet files `pre.rs` reads. A table
//! keeps only the columns these queries touch: `__row` (the parquet row
//! position), each FK cast to BIGINT, and `__ts` (the time column in epoch
//! seconds, as `pre.rs` stores it). `__row` and every FK column get an ART
//! index.
//!
//! The queries are shaped around when DuckDB (1.5) uses those indexes: only
//! for a scan whose sole filter is an equality on the indexed column. Add a
//! time predicate (`fk = ? AND __ts <= ?`) and the lookup falls back to a full
//! scan. The same happens to the parent side of a join on the FK. On a
//! 10M-row child that is ~1.7 ms per lookup against ~0.2 ms indexed. So the
//! time cut is not in SQL, the parent is a second point lookup rather than a
//! join, and `__ts` has no index of its own, since nothing could use it.
//!
//! Node indices still come from the preprocessed `table_info.json` and table
//! name indices from `text.json`. So the edges here match `nodes.rkyv` and
//! `p2f_adj.rkyv` and can replace them one call site at a time.

use super::edge_visible;
use crate::common::{Edge, TableInfo, TableType};
use crate::pre::{DatasetManifest, TaskManifest, load_yaml};
use duckdb::Connection;
use serde::de::{Deserializer as _, SeqAccess, Visitor};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::sync::Mutex;

struct SqlTable {
    node_idx_offset: i32,
    num_nodes: i32,
    table_name_idx: i32,
    table_type: TableType,
    // `__ts` of one row, for the timestamp of an edge onto this table.
    ts_query: String,
    // All FK values of one row, in parquet column order, and the parent table
    // of each. None when the table has no FK columns.
    fk_query: Option<String>,
    fk_parents: Vec<usize>,
    // One query per FK column, in any table, that points at this table, with
    // the child table it reads.
    p2f_queries: Vec<(String, usize)>,
}

pub struct SqlEdges {
    // Sorted by `node_idx_offset`, so a node finds its table by binary search.
    tables: Vec<SqlTable>,
    // DuckDB connections are not `Sync`, so every sampler thread checks one
    // out of the pool. Each connection caches its own prepared statements.
    base: Mutex<Connection>,
    pool: Mutex<Vec<Connection>>,
    stmt_cache_capacity: usize,
}

fn ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn literal(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn parse_table_type(s: &str) -> TableType {
    match s {
        "Db" => TableType::Db,
        "Train" => TableType::Train,
        "Val" => TableType::Val,
        "Test" => TableType::Test,
        _ => panic!("unknown table type {:?} in table_info.json", s),
    }
}

fn is_temporal(sql_type: &str) -> bool {
    sql_type == "DATE" || sql_type.starts_with("TIMESTAMP")
}

fn is_integer(sql_type: &str) -> bool {
    matches!(
        sql_type,
        "TINYINT"
            | "SMALLINT"
            | "INTEGER"
            | "BIGINT"
            | "UTINYINT"
            | "USMALLINT"
            | "UINTEGER"
            | "UBIGINT"
    )
}

fn is_float(sql_type: &str) -> bool {
    matches!(sql_type, "FLOAT" | "DOUBLE")
}

// The FK as a parent row position, as `fk_parent_idxs` in pre.rs reads it:
// floats truncate and NaN is no edge. Any other type (strings, decimals) gives
// pre.rs no edge, so the column is not loaded at all.
fn fk_expr(col: &str, sql_type: &str) -> Option<String> {
    let c = ident(col);
    if is_integer(sql_type) {
        Some(format!("CAST({c} AS BIGINT)"))
    } else if is_float(sql_type) {
        Some(format!(
            "CASE WHEN isnan({c}) THEN NULL ELSE CAST(trunc({c}) AS BIGINT) END"
        ))
    } else {
        None
    }
}

// Epoch seconds clamped to i32, as `read_timestamp` in pre.rs computes them.
// A time column that is not a date or timestamp gives every row no timestamp.
// DuckDB's least/greatest skip NULLs, so the clamp is guarded to keep a null
// time null rather than i32::MAX.
fn ts_expr(col: Option<(&str, &str)>) -> String {
    match col {
        Some((name, sql_type)) if is_temporal(sql_type) => format!(
            "CASE WHEN {c} IS NULL THEN NULL ELSE CAST(greatest(least(\
             epoch_ns({c}) // 1000000000, 2147483647), -2147483648) AS INTEGER) END",
            c = ident(name)
        ),
        _ => "CAST(NULL AS INTEGER)".to_string(),
    }
}

// `text.json` holds every string in the dataset and can be gigabytes, so it is
// streamed. Only the positions of the wanted names are kept.
fn text_idxs(text_path: &str, wanted: &HashSet<String>) -> HashMap<String, i32> {
    struct Find<'a>(&'a HashSet<String>);
    impl<'de> Visitor<'de> for Find<'_> {
        type Value = HashMap<String, i32>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON array of strings")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut found = HashMap::new();
            let mut i = 0i32;
            while let Some(s) = seq.next_element::<String>()? {
                if self.0.contains(&s) {
                    found.entry(s).or_insert(i);
                }
                i += 1;
            }
            Ok(found)
        }
    }
    let file =
        fs::File::open(text_path).unwrap_or_else(|e| panic!("failed to open {}: {}", text_path, e));
    serde_json::Deserializer::from_reader(BufReader::new(file))
        .deserialize_seq(Find(wanted))
        .unwrap_or_else(|e| panic!("failed to read {}: {}", text_path, e))
}

impl SqlEdges {
    /// Loads the FK structure of `dataset_dir` (the directory given to
    /// `preprocess`) into an in-memory DuckDB. `pre_path` is that dataset's
    /// preprocessed directory, which supplies node and table name indices.
    pub fn load(dataset_dir: &str, pre_path: &str) -> Self {
        let dataset_dir = Path::new(dataset_dir);
        let manifest: DatasetManifest = load_yaml(&dataset_dir.join("manifest.yaml"));

        let table_info_path = format!("{}/table_info.json", pre_path);
        let table_info_file = fs::File::open(&table_info_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", table_info_path, e));
        let table_info: HashMap<String, TableInfo> =
            serde_json::from_reader(BufReader::new(table_info_file)).unwrap();

        let mut keys: Vec<(String, TableType, TableInfo)> = table_info
            .into_iter()
            .map(|(key, info)| {
                let (name, ty) = key
                    .rsplit_once(':')
                    .unwrap_or_else(|| panic!("malformed table_info key {:?}", key));
                (name.to_string(), parse_table_type(ty), info)
            })
            .collect();
        keys.sort_by_key(|(_, _, info)| info.node_idx_offset);

        let names: HashSet<String> = keys.iter().map(|(n, _, _)| n.clone()).collect();
        let name_idxs = text_idxs(&format!("{}/text.json", pre_path), &names);

        let conn = Connection::open_in_memory().unwrap();
        let mut task_manifests: HashMap<String, TaskManifest> = HashMap::new();
        let mut tables: Vec<SqlTable> = Vec::with_capacity(keys.len());
        let mut sql_names: Vec<String> = Vec::with_capacity(keys.len());
        // (child table, FK column, parent table name)
        let mut links: Vec<(usize, String, String)> = Vec::new();

        for (i, (name, table_type, info)) in keys.into_iter().enumerate() {
            let (path, fkeys, time_col) = match table_type {
                TableType::Db => {
                    let spec = manifest.tables.get(&name).cloned().unwrap_or_default();
                    (
                        dataset_dir.join("db").join(format!("{}.parquet", name)),
                        spec.fkeys,
                        spec.time_col,
                    )
                }
                _ => {
                    let task_dir = dataset_dir.join("tasks").join(&name);
                    let tm = task_manifests
                        .entry(name.clone())
                        .or_insert_with(|| load_yaml(&task_dir.join("manifest.yaml")));
                    let split = format!("{:?}", table_type).to_lowercase();
                    (
                        task_dir.join(format!("{}.parquet", split)),
                        tm.fkeys(),
                        tm.time_col.clone(),
                    )
                }
            };
            let path = path.to_str().unwrap().to_string();

            let columns: Vec<(String, String)> = conn
                .prepare(&format!(
                    "DESCRIBE SELECT * FROM read_parquet({})",
                    literal(&path)
                ))
                .and_then(|mut stmt| {
                    stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                        .collect()
                })
                .unwrap_or_else(|e| panic!("failed to describe {}: {}", path, e));

            // Parquet column order, as pre.rs walks the columns.
            let mut select = vec!["file_row_number AS __row".to_string()];
            let mut fk_cols: Vec<String> = Vec::new();
            for (col, sql_type) in &columns {
                if !fkeys.contains_key(col) {
                    continue;
                }
                assert!(
                    !sql_type.ends_with("[]"),
                    "FK column {} of {} is a list ({}), which SqlEdges does not \
                     support yet; sample this dataset from p2f_adj.rkyv",
                    col,
                    name,
                    sql_type
                );
                if let Some(expr) = fk_expr(col, sql_type) {
                    select.push(format!("{} AS {}", expr, ident(col)));
                    fk_cols.push(col.clone());
                }
            }
            let time_col = time_col.as_deref().and_then(|t| {
                columns
                    .iter()
                    .find(|(c, _)| c == t)
                    .map(|(c, ty)| (c.as_str(), ty.as_str()))
            });
            select.push(format!("{} AS __ts", ts_expr(time_col)));

            let sql_name = ident(&format!("{}:{:?}", name, table_type));
            conn.execute_batch(&format!(
                "CREATE TABLE {sql_name} AS SELECT {} FROM read_parquet({}, file_row_number = true);
                 CREATE INDEX {} ON {sql_name} (__row);",
                select.join(", "),
                literal(&path),
                ident(&format!("idx_{}_row", i)),
            ))
            .unwrap_or_else(|e| panic!("failed to load {}: {}", path, e));
            for (j, col) in fk_cols.iter().enumerate() {
                conn.execute_batch(&format!(
                    "CREATE INDEX {} ON {sql_name} ({});",
                    ident(&format!("idx_{}_fk_{}", i, j)),
                    ident(col)
                ))
                .unwrap();
                links.push((i, col.clone(), fkeys[col].clone()));
            }

            let table_name_idx = *name_idxs
                .get(&name)
                .unwrap_or_else(|| panic!("table {} has no entry in {}/text.json", name, pre_path));
            tables.push(SqlTable {
                node_idx_offset: info.node_idx_offset,
                num_nodes: info.num_nodes,
                table_name_idx,
                table_type,
                ts_query: format!("SELECT __ts FROM {sql_name} WHERE __row = $1"),
                fk_query: (!fk_cols.is_empty()).then(|| {
                    let cols: Vec<String> = fk_cols.iter().map(|c| ident(c)).collect();
                    format!(
                        "SELECT {} FROM {sql_name} WHERE __row = $1",
                        cols.join(", ")
                    )
                }),
                fk_parents: Vec::new(),
                p2f_queries: Vec::new(),
            });
            sql_names.push(sql_name);
        }

        // pre.rs only links an FK to a parent among the database tables.
        let db_table: HashMap<String, usize> = (0..tables.len())
            .filter(|&i| tables[i].table_type == TableType::Db)
            .map(|i| (sql_names[i].clone(), i))
            .collect();
        for (child, col, parent_name) in links {
            let parent = *db_table
                .get(&ident(&format!("{}:Db", parent_name)))
                .unwrap_or_else(|| {
                    panic!(
                        "FK {} of {} points at {}, which is not a database table",
                        col, sql_names[child], parent_name
                    )
                });
            tables[child].fk_parents.push(parent);
            tables[parent].p2f_queries.push((
                format!(
                    "SELECT __row, __ts FROM {} WHERE {} = $1",
                    sql_names[child],
                    ident(&col)
                ),
                child,
            ));
        }

        let stmt_cache_capacity = tables
            .iter()
            .map(|t| 2 + t.p2f_queries.len())
            .sum::<usize>()
            .max(16);
        Self {
            tables,
            base: Mutex::new(conn),
            pool: Mutex::new(Vec::new()),
            stmt_cache_capacity,
        }
    }

    fn table_of(&self, node_idx: i32) -> (usize, i64) {
        let i = self
            .tables
            .partition_point(|t| t.node_idx_offset <= node_idx)
            .checked_sub(1)
            .unwrap_or_else(|| panic!("node {} precedes every table", node_idx));
        let t = &self.tables[i];
        assert!(
            node_idx < t.node_idx_offset + t.num_nodes,
            "node {} is past the end of its table",
            node_idx
        );
        (i, (node_idx - t.node_idx_offset) as i64)
    }

    fn with_conn<R>(&self, f: impl FnOnce(&Connection) -> R) -> R {
        let conn = self.pool.lock().unwrap().pop().unwrap_or_else(|| {
            let conn = self.base.lock().unwrap().try_clone().unwrap();
            conn.set_prepared_statement_cache_capacity(self.stmt_cache_capacity);
            conn
        });
        let out = f(&conn);
        self.pool.lock().unwrap().push(conn);
        out
    }

    fn edge(&self, table: usize, row: i64, timestamp: Option<i32>) -> Edge {
        let t = &self.tables[table];
        Edge {
            node_idx: t.node_idx_offset + row as i32,
            table_name_idx: t.table_name_idx,
            table_type: t.table_type.clone(),
            timestamp,
        }
    }

    /// The parents of `node_idx`, one per FK value, in FK column order.
    /// Parents stamped after `bound` are left out. With no bound, every parent.
    pub fn f2p_edges(&self, node_idx: i32, bound: Option<i32>) -> Vec<Edge> {
        let (i, row) = self.table_of(node_idx);
        let t = &self.tables[i];
        let Some(fk_query) = &t.fk_query else {
            return Vec::new();
        };
        self.with_conn(|conn| {
            let fks: Vec<Option<i64>> = conn
                .prepare_cached(fk_query)
                .and_then(|mut stmt| {
                    stmt.query_row([row], |r| {
                        (0..t.fk_parents.len()).map(|j| r.get(j)).collect()
                    })
                })
                .unwrap_or_else(|e| panic!("f2p query failed: {}\n{}", e, fk_query));

            let mut edges = Vec::with_capacity(fks.len());
            for (fk, &parent) in fks.into_iter().zip(&t.fk_parents) {
                let Some(prow) = fk else { continue };
                let ts_query = &self.tables[parent].ts_query;
                let mut stmt = conn.prepare_cached(ts_query).unwrap();
                // No row means the FK points past the parent table: no edge.
                let Some(timestamp) = stmt
                    .query([prow])
                    .and_then(|mut rows| rows.next()?.map(|r| r.get(0)).transpose())
                    .unwrap_or_else(|e| panic!("f2p query failed: {}\n{}", e, ts_query))
                else {
                    continue;
                };
                let edge = self.edge(parent, prow, timestamp);
                if edge_visible(&edge, bound, None) {
                    edges.push(edge);
                }
            }
            edges
        })
    }

    /// The children of `node_idx` across every FK pointing at its table,
    /// sorted by timestamp with untimestamped children first, as in
    /// `p2f_adj.rkyv`. A child is kept when `edge_visible` keeps it: it is not
    /// stamped after `bound`, and it is not a task row stamped exactly
    /// `target_ts`. With neither, every child.
    pub fn p2f_edges(
        &self,
        node_idx: i32,
        bound: Option<i32>,
        target_ts: Option<i32>,
    ) -> Vec<Edge> {
        let (i, row) = self.table_of(node_idx);
        let mut edges = self.with_conn(|conn| {
            let mut edges = Vec::new();
            for (sql, child) in &self.tables[i].p2f_queries {
                let mut stmt = conn.prepare_cached(sql).unwrap();
                let rows = stmt
                    .query_map([row], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i32>>(1)?))
                    })
                    .unwrap_or_else(|e| panic!("p2f query failed: {}\n{}", e, sql));
                for r in rows {
                    let (crow, timestamp) = r.unwrap();
                    let edge = self.edge(*child, crow, timestamp);
                    if edge_visible(&edge, bound, target_ts) {
                        edges.push(edge);
                    }
                }
            }
            edges
        });
        edges.sort_by_key(|e| (e.timestamp, e.node_idx));
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets};
    use crate::fly::edge_visible;
    use polars::prelude::*;
    use rkyv::rancor::Error;
    use rkyv::util::AlignedVec;
    use std::path::PathBuf;

    type Key = (i32, i32, String, Option<i32>);

    fn key(e: &Edge) -> Key {
        (
            e.node_idx,
            e.table_name_idx,
            format!("{:?}", e.table_type),
            e.timestamp,
        )
    }

    fn archived_key(e: &ArchivedEdge) -> Key {
        (
            e.node_idx.into(),
            e.table_name_idx.into(),
            format!("{:?}", e.table_type),
            e.timestamp.as_ref().map(|t| i32::from(*t)),
        )
    }

    fn aligned(bytes: &[u8]) -> AlignedVec {
        let mut v = AlignedVec::new();
        v.extend_from_slice(bytes);
        v
    }

    fn write(path: PathBuf, mut df: DataFrame) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        ParquetWriter::new(fs::File::create(path).unwrap())
            .finish(&mut df)
            .unwrap();
    }

    fn datetime_ms(name: &str, ms: &[Option<i64>]) -> Column {
        Series::new(name.into(), ms)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap()
            .into()
    }

    const DAY_MS: i64 = 86_400_000;

    // Covers what pre.rs treats specially: a DATE time column, timestamps
    // before 1970, null and NaN float FKs, an untimestamped child table, two
    // FKs on one table, and a task table that shares the seed's timestamp.
    fn write_dataset(dir: &Path) {
        fs::write(
            dir.join("manifest.yaml"),
            "name: toy\n\
             tables:\n  \
               users: {pkey: id, time_col: signup}\n  \
               posts: {pkey: id, time_col: created, fkeys: {user_id: users}}\n  \
               comments: {time_col: at, fkeys: {post_id: posts, user_id: users}}\n  \
               tags: {fkeys: {user_id: users}}\n",
        )
        .unwrap();
        let db = dir.join("db");

        let signup = Series::new("signup".into(), [-400i32, 10, 20, 30])
            .cast(&DataType::Date)
            .unwrap();
        write(
            db.join("users.parquet"),
            DataFrame::new(vec![
                Column::new("id".into(), [0i64, 1, 2, 3]),
                signup.into(),
                Column::new("name".into(), ["a", "b", "c", "d"]),
            ])
            .unwrap(),
        );
        write(
            db.join("posts.parquet"),
            DataFrame::new(vec![
                Column::new("id".into(), [0i64, 1, 2, 3, 4, 5]),
                Column::new(
                    "user_id".into(),
                    [Some(0i64), Some(0), Some(1), None, Some(2), Some(0)],
                ),
                datetime_ms(
                    "created",
                    &[
                        Some(-5 * DAY_MS - 1),
                        Some(40 * DAY_MS),
                        Some(50 * DAY_MS),
                        Some(60 * DAY_MS),
                        None,
                        Some(40 * DAY_MS),
                    ],
                ),
                Column::new("score".into(), [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ])
            .unwrap(),
        );
        write(
            db.join("comments.parquet"),
            DataFrame::new(vec![
                Column::new(
                    "post_id".into(),
                    [
                        Some(0.0f64),
                        Some(f64::NAN),
                        None,
                        Some(2.0),
                        Some(1.0),
                        Some(5.0),
                    ],
                ),
                Column::new("user_id".into(), [1i32, 2, 3, 0, 0, 1]),
                datetime_ms(
                    "at",
                    &[
                        Some(41 * DAY_MS),
                        Some(42 * DAY_MS),
                        Some(43 * DAY_MS),
                        Some(55 * DAY_MS),
                        None,
                        Some(70 * DAY_MS),
                    ],
                ),
                Column::new("body".into(), ["x", "y", "z", "w", "v", "u"]),
            ])
            .unwrap(),
        );
        write(
            db.join("tags.parquet"),
            DataFrame::new(vec![
                Column::new("user_id".into(), [0i64, 0, 3]),
                Column::new("tag".into(), ["p", "q", "r"]),
            ])
            .unwrap(),
        );

        let task = dir.join("tasks").join("churn");
        fs::create_dir_all(&task).unwrap();
        fs::write(
            task.join("manifest.yaml"),
            "entity_table: users\nentity_col: user_id\ntime_col: timestamp\n\
             target_col: churn\ntask_type: binary_classification\n",
        )
        .unwrap();
        for (split, days) in [("train", [40i64, 50, 40]), ("val", [60, 60, 70])] {
            write(
                task.join(format!("{}.parquet", split)),
                DataFrame::new(vec![
                    datetime_ms("timestamp", &days.map(|d| Some(d * DAY_MS))),
                    Column::new("user_id".into(), [0i64, 0, 1]),
                    Column::new("churn".into(), [0i64, 1, 1]),
                ])
                .unwrap(),
            );
        }
    }

    #[test]
    fn matches_preprocessed_edges() {
        let root = std::env::temp_dir().join(format!("rt_sql_edges_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dataset_dir = root.join("toy");
        fs::create_dir_all(&dataset_dir).unwrap();
        write_dataset(&dataset_dir);

        crate::pre::main(crate::pre::Cli {
            dataset_dir: dataset_dir.to_str().unwrap().to_string(),
            out_dir: root.join("pre").to_str().unwrap().to_string(),
            skip_tasks: false,
            skip_db: false,
            source: None,
        });
        let pre_path = root.join("pre").join("toy");
        let sql = SqlEdges::load(dataset_dir.to_str().unwrap(), pre_path.to_str().unwrap());

        let nodes = fs::read(pre_path.join("nodes.rkyv")).unwrap();
        let offsets = aligned(&fs::read(pre_path.join("offsets.rkyv")).unwrap());
        let offsets = rkyv::access::<ArchivedOffsets, Error>(&offsets).unwrap();
        let adj = aligned(&fs::read(pre_path.join("p2f_adj.rkyv")).unwrap());
        let adj = rkyv::access::<ArchivedAdj, Error>(&adj).unwrap();

        let horizons = [None, Some(-1), Some(0), Some(40), Some(55), Some(60)];
        let horizons: Vec<Option<i32>> = horizons
            .iter()
            .map(|d| d.map(|d: i32| d * 86_400))
            .collect();

        let mut num_f2p = 0;
        let mut num_p2f = 0;
        for n in 0..offsets.offsets.len() - 1 {
            let (l, r) = (
                i64::from(offsets.offsets[n]) as usize,
                i64::from(offsets.offsets[n + 1]) as usize,
            );
            let node = aligned(&nodes[l..r]);
            let node = rkyv::access::<ArchivedNode, Error>(&node).unwrap();
            let n = n as i32;

            for &bound in &horizons {
                let want: Vec<Key> = node
                    .f2p_edges
                    .iter()
                    .filter(|e| edge_visible(*e, bound, None))
                    .map(archived_key)
                    .collect();
                let got: Vec<Key> = sql.f2p_edges(n, bound).iter().map(key).collect();
                assert_eq!(got, want, "f2p of node {} at bound {:?}", n, bound);
                num_f2p += got.len();

                for &target_ts in &horizons {
                    let mut want: Vec<Key> = adj.adj[n as usize]
                        .iter()
                        .filter(|e| edge_visible(*e, bound, target_ts))
                        .map(archived_key)
                        .collect();
                    let got: Vec<Key> =
                        sql.p2f_edges(n, bound, target_ts).iter().map(key).collect();
                    // Timestamp order is the contract; ties may come out in
                    // another order than pre.rs pushed them.
                    assert!(
                        got.is_sorted_by_key(|k| k.3),
                        "p2f of node {} out of order",
                        n
                    );
                    let mut got = got;
                    got.sort();
                    want.sort();
                    assert_eq!(
                        got, want,
                        "p2f of node {} at bound {:?}, target {:?}",
                        n, bound, target_ts
                    );
                    num_p2f += got.len();
                }
            }
        }
        // The comparison above would pass vacuously on an empty graph.
        assert!(num_f2p > 0 && num_p2f > 0);

        fs::remove_dir_all(&root).unwrap();
    }
}
