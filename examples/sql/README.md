# sql: RT-J predictions straight from a database

`rt.sql` builds RT-J's context for a prediction with SQL queries against a
DuckDB database. There is no preprocessing step and no graph built ahead of
time.

```
database --(rt.sql.retrieve: SQL)--> rows --(rt.sql.encode)--> cells
         --(rt.sql.context.to_batch)--> tensors --> RT-J
```

The database is a `.duckdb` file written by
`examples/preprocess/to_duckdb.py` from RelBench-format data.

| Module | Job | Replaces |
|---|---|---|
| `meta` | Tables, keys and tasks, read from the database | `manifest.yaml`, `table_info.json` |
| `encode` | Rows into cells, with statistics from SQL aggregates and live text embeddings | `pre.rs` node building, `text_emb_*.bin` |
| `retrieve` | Which rows go in a context, as set-based SQL per BFS level | the rustler sampler, `p2f_adj.rkyv` |
| `context` | The rows, why each is there, every query, and the model tensors | sampler output |

Every context is traceable: `Context.trace()` lists each row with the step
that reached it (`target`, `example`, `parent via <fk> of <row>`,
`child via <fk> of <row>`) and every query with its row count and time.

To watch the queries live, `predict --query-log DIR` turns on DuckDB's own
query log, which writes each statement as it runs (parameters show as `?`):

```
tail -f DIR/duckdb_log_entries.csv
```

## Scripts

```
# encoding parity against the preprocessor + sampler (needs this checkout's preprocess output)
python -m examples.sql.parity --duckdb mini-shop.duckdb --pre-dir pre \
    --task customer-churn --rows 32 [--model]

# predict a task from the database alone, with traces
python -m examples.sql.predict --duckdb mini-shop.duckdb --task customer-churn \
    --rows 64 --ctx-size 512 --trace traces.jsonl

# score a whole test split with RelBench, comparable with examples/eval
python -m examples.sql.predict --duckdb data/relbench-duckdb/rel-f1.duckdb \
    --task driver-dnf --all-rows --ctx-size 8192 \
    --relbench data/relbench-v1/rel-f1 --csv-out sql_out/
```

With `--relbench`, predictions go through `relbench.submit.evaluate_task`, the
scoring `rt.eval` uses, and `--csv-out` keeps the submission CSV
(`<db>__<task>.csv`, the same name and columns as `rt.eval`'s `eval_out/`).
Only a full split is a submission; a sample is scored with sklearn and marked
`~`. The sampler baseline is `examples/eval` on the same task; the two CSVs
join on the entity and time columns.

Off CUDA, both scripts run flex attention uncompiled, because `torch.compile`
cannot lower it for CPU on macOS. That is slow but gives the same numbers.

## What is verified

`parity.py` rebuilds the sampler's own contexts token for token from the
database. On mini-shop (32 contexts, 32,768 tokens), every model input matches
bit for bit: rows, columns, types, numbers, dates, text and column-name
embeddings, target flags and parents. The one exception is the synthetic
`identifier` cells, whose random values cannot be reproduced from Python.
RT-J's predictions on both differ by at most 0.027, and by 0.002 with
`RT_IDENTIFIER_POLICY=none` on both sides, which is bf16 rounding from a
different token order.

## Where to iterate

The retrieval in `retrieve.py` is v0: deterministic and simple. These are the
gaps to the current system, roughly in order of expected payoff:

1. **Seed ranking.** The sampler ranks labelled examples by random-walk
   visits from the target. v0 takes the same entity's rows, then the most
   recent. A SQL version: count 2-hop paths (target -> parent -> other task
   rows) with a GROUP BY.
2. **Children.** The sampler draws up to `bfs_width` children at random. v0
   takes the most recent.
3. **Query count.** By default each seed's BFS levels are their own queries
   (about 340 for an 8192-token context on mini-shop). `--seed-batch N`
   expands N seeds together: on mini-shop, N=4 runs 3.3x fewer queries and
   retrieves 1.7x faster, but seeds in a batch no longer skip each other's
   rows, so contexts change (89% of rows shared with N=1). Larger N fetches
   and decodes many rows it never keeps; decoding only claimed rows is next.
4. **Speed at scale.** Per-database setup (schema and statistics) is cached in
   the `Encoder`. Text embeddings are cached per string. On GPU, the model
   runs compiled.

Not yet supported: `remove_columns` leakage drops, `db_cutoff`, list-typed
keys, INTERVAL and DECIMAL columns, and recommendation tasks.
