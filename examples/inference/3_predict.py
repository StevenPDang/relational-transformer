"""Step 3 — Predict and score.

Downloads the pretrained checkpoint named in ``config.CHECKPOINT`` from the
Hugging Face Hub (any released RT checkpoint works — an RT-J variant or a local
training run), preprocesses the dataset prepared in steps 1-2
into the model's tensor format (Rust sampler + text embeddings; one-time, cached
afterwards), runs zero-shot inference over the test split, joins the predictions
back to your labeled rows, and reports the metric (AUROC for classification, MAE
for regression).

    pixi run python examples/inference/3_predict.py               # GPU (default)
    pixi run python examples/inference/3_predict.py --device cpu  # slow; small demos only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config

_REPO = Path(__file__).resolve().parents[2]


def ensure_preprocessed(pre_dir: Path) -> None:
    """Run the rustler preprocessor + text embedding unless already cached."""
    if (pre_dir / config.DB_NAME / "table_info.json").exists():
        print(f"   using cached preprocessed data at {pre_dir / config.DB_NAME}")
        return
    print(f"   preprocessing '{config.DB_NAME}' (rustler + text embeddings)...")
    subprocess.run(
        [
            sys.executable, str(_REPO / "scripts" / "preprocess.py"), "one",
            "--dataset", str(Path(config.DATA_DIR) / config.DB_NAME),
            "--out-dir", str(pre_dir),
        ],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=config.CHECKPOINT,
                   help="Hub spec or local path; overrides config.CHECKPOINT")
    p.add_argument(
        "--checkpoint-revision",
        default=config.CHECKPOINT_REVISION,
        help="Hub branch, tag, or commit; defaults to config.CHECKPOINT_REVISION",
    )
    p.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    args = p.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    task_dir = Path(config.DATA_DIR) / config.DB_NAME / "tasks" / config.TASK["name"]
    if not task_dir.exists():
        raise SystemExit("Run 1_data_prep.py and 2_task_prep.py first.")

    print(f"[step 3] loading checkpoint {args.checkpoint}")
    from rt.checkpoints import load_rt_model

    model, cfg = load_rt_model(
        args.checkpoint, device=device, revision=args.checkpoint_revision
    )
    model = model.to(torch.bfloat16)  # the model runs in bf16 (matches the sampled data)
    want = {"binary_classification": "clf", "regression": "reg"}[config.TASK["task_type"]]
    if cfg.get("task_type", want) != want:
        raise SystemExit(
            f"checkpoint predicts '{cfg['task_type']}' but the task is '{want}' — "
            f"pick a matching checkpoint (see config.py)."
        )

    print("[step 3] preprocessing")
    pre_dir = Path(config.DATA_DIR) / "pre"
    ensure_preprocessed(pre_dir)

    print(f"[step 3] running inference on {config.DB_NAME}/{config.TASK['name']} (device={device})")
    from rt.eval_utils import build_evaluator
    from rt.tasks import tasks_from_preprocessed

    tasks = [
        t
        for t in tasks_from_preprocessed(str(pre_dir), splits=("test",), dbs=[config.DB_NAME])
        if t.table_name == config.TASK["name"]
    ]
    assert tasks, f"task {config.TASK['name']!r} not found in {pre_dir}"
    ctx_size = cfg.get("ctx_len", 8192)
    evaluator = build_evaluator(
        tasks, str(pre_dir),
        embedding_model=cfg.get("embedding_model", "all-MiniLM-L12-v2"),
        d_text=cfg.get("d_text", cfg.get("model", {}).get("d_text", 384)),
        device=device, ctx_size=ctx_size, items_per_task=10_000_000,
    )
    ((task, _ctx, _labels, out, _n, node_idxs),) = evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    )

    # --- join predictions back to the labeled test rows and score -------------
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    table_info = json.loads((pre_dir / config.DB_NAME / "table_info.json").read_text())
    offset = table_info[f"{config.TASK['name']}:Test"]["node_idx_offset"]
    rows = np.asarray(node_idxs) - offset  # seed node index -> test-table row
    df = pd.read_parquet(task_dir / "test.parquet").iloc[rows].reset_index(drop=True)
    raw = np.asarray(out[""], dtype=float)
    y = df[config.TASK["target_col"]].astype(float)
    if task.task_type == "clf":
        df["prediction"] = 1 / (1 + np.exp(-raw))  # logit -> probability
        metric, name = roc_auc_score((y > 0).astype(int), df["prediction"]), "AUROC"
    else:
        train = pd.read_parquet(task_dir / "train.parquet")[config.TASK["target_col"]]
        df["prediction"] = raw * train.std(ddof=1) + train.mean()  # denormalize
        metric, name = mean_absolute_error(y, df["prediction"]), "MAE"

    pred_path = Path(config.DATA_DIR) / f"{config.TASK['name']}_predictions.parquet"
    df.to_parquet(pred_path, index=False)
    print(f"\n[result] {config.DB_NAME}/{config.TASK['name']}   {name} = {metric:.4f}   (n={len(df)})")
    print(f"   per-row predictions -> {pred_path}")


if __name__ == "__main__":
    main()
