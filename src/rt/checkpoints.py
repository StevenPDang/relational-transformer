"""Load RT checkpoints from a local path *or* the HuggingFace Hub — one interface.

A released checkpoint is a directory (a local folder or a Hub model repo
``org/repo[/subdir]``) holding:

* ``model.safetensors`` -- the weights, a flat ``state_dict`` stored with
  :func:`safetensors.torch.save_file` (the format ``scripts/pretrain.py``
  writes); and
* ``config.json``       -- the model dims + the text-embedding model used, so a
  loader needs no out-of-band knowledge.

Model weights are always safetensors. The one ``torch.save`` pickle the codebase
still writes is ``resume.pt``, which is training state (optimizer, scheduler,
SWA, step) rather than a model, is internal to a run, and holds non-tensor
objects safetensors cannot store.

This mirrors :mod:`rt.pre`'s local-or-Hub *data* resolver, so a checkpoint
reference like ``stanford-star/rt-j/classification`` "just works" (downloaded and cached
on demand), while a local training run's checkpoint dir works with the same call.
A local path always wins, so iterating locally never triggers a download.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rt.pre import resolve_repo

try:
    _RT_VERSION = version("relational-transformer")
except PackageNotFoundError:  # running from a source tree without an install
    _RT_VERSION = None

# huggingface_hub user-agent so Hub downloads are attributed to this library.
_HF_UA = {"library_name": "relational-transformer", "library_version": _RT_VERSION}

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    """Save a flat tensor ``state_dict`` to ``path`` as safetensors.

    ``metadata`` (e.g. ``{"step": 1000}``) is coerced to a str→str header, as
    safetensors metadata only holds strings.
    """
    from safetensors.torch import save_file

    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def load_model(path):
    """Load a flat tensor ``state_dict`` from a ``.safetensors`` checkpoint."""
    from safetensors.torch import load_file

    return load_file(str(path))


def resolve_checkpoint(
    spec, *, revision: str | None = None, subfolder: str | None = None
) -> tuple[dict, Path]:
    """Return ``(config, model_path)`` for a local or Hub checkpoint.

    ``spec`` may be: a local weights file (``model.safetensors``; config from a
    sibling ``config.json`` if present), a local directory, or a Hub
    ``org/repo[/subdir]``. ``subfolder`` selects a sub-directory within the
    repo/directory (the HuggingFace-idiomatic way to pick a checkpoint;
    equivalent to appending it to ``spec``). Within a directory, an explicit
    ``config["checkpoint_file"]`` wins, else ``model.safetensors``.
    """
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return config, p
    if p.is_dir():
        d = p / subfolder if subfolder else p
    else:
        from huggingface_hub import hf_hub_download

        repo_id, subdir = resolve_repo(spec)
        subdir = "/".join(part for part in (subdir, subfolder) if part)
        prefix = f"{subdir}/" if subdir else ""
        config_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}{CONFIG_FILE}",
                revision=revision,
                **_HF_UA,
            )
        )
        config = json.loads(config_path.read_text())
        model_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}{config.get('checkpoint_file', MODEL_FILE)}",
                revision=revision,
                **_HF_UA,
            )
        )
        return config, model_path
    config = json.loads((d / CONFIG_FILE).read_text())
    return config, d / config.get("checkpoint_file", MODEL_FILE)


def load_rt_model(
    spec,
    *,
    device: str = "cpu",
    compile: bool = False,
    revision: str | None = None,
    subfolder: str | None = None,
    model_kwargs: dict | None = None,
):
    """Resolve a checkpoint, build the RelationalTransformer, load its weights.

    Returns ``(model, config)``. Thin backward-compatible wrapper around
    :meth:`rt.model.RelationalTransformer.from_pretrained` (the HuggingFace-style
    entry point, which returns just the model with ``config`` attached as
    ``model.config``). ``model_kwargs`` overrides/fills model dims when a
    checkpoint has no ``config.json`` (e.g. a raw internal ckpt during dev).
    """
    from rt.model import RelationalTransformer

    model = RelationalTransformer.from_pretrained(
        spec,
        device=device,
        compile=compile,
        revision=revision,
        subfolder=subfolder,
        **(model_kwargs or {}),
    )
    return model, model.config
