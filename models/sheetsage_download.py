"""Explicit download of the SheetSage2-Refined snapshot tested with runtime 0.5.0."""
from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath


REFINED_REPO = "anime-song/SheetSage2-Refined"
REFINED_REVISION = "e36ce4fc3c355e7a4e740ef1797275107b7ea257"
LICENSE_FILES = ("LICENSE", "sheetsage2/LICENSE", "sheetsage2/NOTICE.md")


def _artifact_path(folder, filename):
    if not isinstance(filename, str) or not filename or "\\" in filename or ":" in filename:
        raise ValueError(f"Invalid Refined artifact path: {filename!r}")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Refined artifact must stay inside its model folder: {filename}")
    path = folder.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"Refined artifact points outside its model folder: {filename}")
    return path


def _runtime_weight_files():
    try:
        from sheetsage2_refined.pipeline import weight_files
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-sheetsage2-refined.txt with ComfyUI's Python before downloading this model. "
            f"Import error: {exc}"
        ) from exc
    return weight_files


def _required_files(folder, weight_files):
    config = json.loads(_artifact_path(folder, "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("base_weights"), str):
        raise ValueError("SheetSage2-Refined config.json must contain base_weights.")
    weights = weight_files(config)
    for name in weights:
        _artifact_path(folder, name)
        if not name.endswith(".safetensors"):
            raise ValueError(f"Unsupported Refined weight file: {name}")
    return ["config.json", *sorted(set(weights)), *LICENSE_FILES]


def missing_refined_files(folder):
    """Check configured weights and licenses locally, without contacting the Hub."""
    folder = Path(folder).resolve()
    config = _artifact_path(folder, "config.json")
    if not config.is_file() or config.stat().st_size == 0:
        return ["config.json"]
    names = _required_files(folder, _runtime_weight_files())
    return [name for name in names if not (path := _artifact_path(folder, name)).is_file() or path.stat().st_size == 0]


def download_refined_model(folder):
    """Download missing artifacts only when the caller explicitly requests this operation."""
    folder = Path(folder).resolve()
    weight_files = _runtime_weight_files()
    config_path = _artifact_path(folder, "config.json")

    def fetch(filename):
        from huggingface_hub import hf_hub_download

        path = _artifact_path(folder, filename)
        folder.mkdir(parents=True, exist_ok=True)
        logging.info("UniversalMMH: downloading SheetSage2-Refined/%s", filename)
        hf_hub_download(repo_id=REFINED_REPO, filename=filename, revision=REFINED_REVISION,
                        local_dir=str(folder), token=False,
                        force_download=path.is_file() and path.stat().st_size == 0)
        path = _artifact_path(folder, filename)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Download did not produce the requested Refined artifact: {filename}")

    if not config_path.is_file() or config_path.stat().st_size == 0:
        fetch("config.json")
    names = _required_files(folder, weight_files)
    paths = [(name, _artifact_path(folder, name)) for name in names]
    for name, path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            fetch(name)
    logging.info("UniversalMMH: SheetSage2-Refined is ready at %s", folder)
    return folder
