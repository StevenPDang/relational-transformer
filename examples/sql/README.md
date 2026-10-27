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

## Scripts

```
# encoding parity against the preprocessor + sampler (needs this checkout's preprocess output)
python -m examples.sql.parity --duckdb mini-shop.duckdb --pre-dir pre \
    --task customer-churn --rows 32 [--model]

# predict a task from the database alone, with traces; --pre-dir adds the sampler baseline
python -m examples.sql.predict --duckdb mini-shop.duckdb --task customer-churn \
    --rows 64 --ctx-size 512 --trace traces.jsonl [--pre-dir pre]
```

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
3. **Query count.** v0 runs one query per table or FK per BFS level per seed
   (about 70 for a 1024-token context on mini-shop). Expanding all seeds'
   levels together would cut that by the number of seeds.
4. **Speed at scale.** Per-database setup (schema and statistics) is cached in
   the `Encoder`. Text embeddings are cached per string. On GPU, the model
   runs compiled.

Not yet supported: `remove_columns` leakage drops, `db_cutoff`, list-typed
keys, INTERVAL and DECIMAL columns, and recommendation tasks.
