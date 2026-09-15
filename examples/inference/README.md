# Run a pretrained RT checkpoint on your own database

Pick a released checkpoint from the [`stanford-star`](https://huggingface.co/stanford-star)
Hugging Face org and run it on your own database (DuckDB, Postgres, or MySQL)
in three steps. (For the same flow as a notebook, see the
[BYOD Colab](../byod/colab.ipynb).)

You edit exactly one file, [`config.py`](config.py), then run three scripts in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `1_data_prep.py` | Convert your database into RelBench format (declare primary keys, foreign keys, time columns). |
| 2 | `2_task_prep.py` | Define the prediction task (entity, timestamp, target) and its labeled train/val/test rows. |
| 3 | `3_predict.py` | Download the pretrained checkpoint, preprocess, run zero-shot inference, and report the metric. |

## Try it first on the demo database

`config.py` ships pre-filled with the small shop database at
[`examples/byod/mini-shop.duckdb`](../byod/mini-shop.duckdb); `0_make_demo_labels.py`
derives its churn labels, so you can run the whole flow before touching your own
data. From the repo root (after `pixi run build-sampler` and
`pixi run build-pre`, see the [README](../../README.md)):

```bash
pixi run python examples/inference/0_make_demo_labels.py   # demo only: derive labels
pixi run python examples/inference/1_data_prep.py
pixi run python examples/inference/2_task_prep.py
pixi run python examples/inference/3_predict.py            # add --device cpu if no GPU
```

Step 3 prints the AUROC on the held-out month (~0.81 with the default RT-J
classifier) and writes per-row predictions to
`examples/inference/out/customer-churn_predictions.parquet`.

## Pick a checkpoint

`config.CHECKPOINT` (or `3_predict.py --checkpoint ...`) takes any RT checkpoint —
a Hub spec or a local path; `rt.checkpoints.load_rt_model` resolves either:

```bash
# RT-J (default): 86M params, trained on the Join at ctx 8192
pixi run python examples/inference/3_predict.py --checkpoint stanford-star/rt-j/classification
```

`config.CHECKPOINT_REVISION` pins Hub checkpoints to an immutable commit for
reproducible downloads. Override it with `--checkpoint-revision`; it is ignored
when `--checkpoint` names a local path.

Classification tasks need a classifier checkpoint (`rt-j/classification`) and
regression tasks a regressor (`rt-j/regression`). Browse
[huggingface.co/stanford-star](https://huggingface.co/stanford-star) for all
released checkpoints.

## Point it at your own database

Open [`config.py`](config.py) and set:

1. **`SQL_URI`** — how to reach your database:
   - DuckDB: a `.duckdb` file path
   - Postgres: `postgresql+psycopg2://user:pw@host:5432/dbname` (install `psycopg2-binary`)
   - MySQL: `mysql+pymysql://user:pw@host:3306/dbname` (install `pymysql`)

2. **`TABLES`** — your schema: for each table, its primary key, time column, and
   foreign keys (`{column: referenced_table}`).

3. **`TASK`** — what to predict: the entity table/column, the timestamp column, the
   target column, the task type (`binary_classification` or `regression`), and the
   paths to your labeled `train` / `val` / `test` rows (parquet or csv, each with the
   entity column, time column, and target column).

Then rerun steps 1-3 (skip step 0 — that only builds the demo's labels). That's it.
