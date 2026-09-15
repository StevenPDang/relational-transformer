"""Checkpoint resolution behavior that does not require importing torch."""

from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path


def test_resolve_checkpoint_downloads_exact_hub_files(monkeypatch, tmp_path):
    downloads = []
    config_file = "config.json"

    def fake_download(*, repo_id, filename, revision, **kwargs):
        downloads.append((repo_id, filename, revision, kwargs))
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(config_file):
            path.write_text(json.dumps({"checkpoint_file": "weights.safetensors"}))
        else:
            path.write_bytes(b"weights")
        return str(path)

    pre = types.ModuleType("rt.pre")
    pre.resolve_repo = lambda spec: (
        "/".join(spec.split("/")[:2]),
        "/".join(spec.split("/")[2:]),
    )
    monkeypatch.setitem(sys.modules, "rt", types.ModuleType("rt"))
    monkeypatch.setitem(sys.modules, "rt.pre", pre)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )
    module_spec = importlib.util.spec_from_file_location(
        "checkpoints_under_test", Path(__file__).parents[1] / "src/rt/checkpoints.py"
    )
    checkpoints = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(checkpoints)

    config, model_path = checkpoints.resolve_checkpoint(
        "stanford-star/rt-j/regression", revision="model-revision"
    )

    assert config["checkpoint_file"] == "weights.safetensors"
    assert model_path == tmp_path / "regression" / "weights.safetensors"
    assert [(repo, filename, revision) for repo, filename, revision, _ in downloads] == [
        ("stanford-star/rt-j", "regression/config.json", "model-revision"),
        ("stanford-star/rt-j", "regression/weights.safetensors", "model-revision"),
    ]
