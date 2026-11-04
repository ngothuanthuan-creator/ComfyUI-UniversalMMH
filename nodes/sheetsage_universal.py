"""YuE2 pipeline loader for local Native, HF and Refined SheetSage2 models."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
import os
from pathlib import Path

import folder_paths
from safetensors import safe_open, SafetensorError
import torch

from .comfy_native import YuE2NativePipelineLoader, _loader_inputs, _load_native
from ..models.paths import model_dirs


_NATIVE_KEYS = {"encoder.feature_extractor.mel_mean", "decoder.layernorm_embedding.weight", "layer_weight"}
_MERGED_KEYS = {"encoder.frontend.mel_mean", "decoder.token_embedding.weight", "layer_weights"}
_SKIP_DIRS = {".cache", ".git", "__pycache__"}


def _roots():
    return {"audio_encoders": folder_paths.get_folder_paths("audio_encoders"),
            "SheetSage2": model_dirs("sheetsage2")}


def _inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def _walk(root):
    for parent, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in _SKIP_DIRS)
        yield Path(parent), files


def _keys(path):
    try:
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            return set(weights.keys())
    except (OSError, SafetensorError):
        return set()


def _models():
    """Group Refined configurations; recognize encoder files by their weight keys."""
    found = {}
    for category, roots in _roots().items():
        for index, root_name in enumerate(roots):
            root = Path(root_name).resolve()
            prefix = category if index == 0 else f"{category}[{index + 1}]"
            for folder, files in _walk(root):
                if not _inside(folder, root):
                    continue
                label = f"{prefix}/{folder.relative_to(root).as_posix()}".removesuffix("/.")
                if "config.json" in files:
                    try:
                        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        config = {}
                    if isinstance(config, dict):
                        if "base_weights" in config:
                            found[f"Refined | {label}"] = ("refined", folder, root)
                        elif "modeling_sheetsage2.py" in files and any(
                            name in files for name in ("model.safetensors", "model.safetensors.index.json")
                        ):
                            found[f"HF | {label}"] = ("hf", folder, root)
                for name in sorted(files):
                    if not name.endswith(".safetensors"):
                        continue
                    path = folder / name
                    if not _inside(path, root):
                        continue
                    keys = _keys(path)
                    kind = "native" if _NATIVE_KEYS <= keys else "merged" if _MERGED_KEYS <= keys else None
                    if kind:
                        title = "Native" if kind == "native" else "SheetSage2 merged"
                        found[f"{title} | {prefix}/{path.relative_to(root).as_posix()}"] = (kind, path, root)
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def _mert_models():
    found = {}
    for index, root_name in enumerate(model_dirs("mert2")):
        root = Path(root_name).resolve()
        for folder, files in _walk(root):
            if not _inside(folder, root):
                continue
            if {"config.json", "configuration_mert2.py", "modeling_mert2.py"} <= set(files) and any(
                name in files for name in ("model.safetensors", "model.safetensors.index.json")
            ):
                name = folder.relative_to(root).as_posix()
                found[f"MERT2[{index + 1}]/{name}"] = folder
    return found


def _validate_refined(folder, root):
    from sheetsage2_refined.pipeline import weight_files

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    for name in weight_files(config):
        path = (folder / name).resolve()
        if not _inside(path, root):
            raise ValueError(f"Refined weight is outside its model search folder: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"SheetSage2-Refined is missing {path}. No download was started.")
    return config


def _mert_path(folder, selection):
    models = _mert_models()
    if selection != "auto":
        if selection not in models:
            raise ValueError("The selected MERT2 folder is no longer available. Refresh the model list.")
        return str(models[selection])
    nested = folder / "MERT-v2-FullSong"
    if all((nested / name).is_file() for name in (
        "config.json", "model.safetensors", "configuration_mert2.py", "modeling_mert2.py"
    )):
        return str(nested)
    if models:
        return str(next(iter(models.values())))
    raise FileNotFoundError(
        "This HF SheetSage2 adapter needs a local MERT-v2-FullSong snapshot in models/MERT2/. "
        "Alternatively select a Native or merged SheetSage2 model. No download was started."
    )


def _abc_events(events, melody_only):
    result = copy.deepcopy(events)
    reduced = set()
    for event in result:
        values = event["values"]
        if melody_only:
            values.pop("chord", None)
            continue
        chord = values.get("chord", "N")
        if chord in {"N", "X", "?"}:
            continue
        root, descriptor = chord.split(":", 1)
        quality, separator, bass = descriptor.partition("/")
        if quality == "sus4(b7)":
            continue
        base = quality.split("(", 1)[0]
        if base == "5":
            base = "maj"
        replacement = root + ":" + base + (separator + bass if separator else "")
        if replacement != chord:
            reduced.add(f"{chord} -> {replacement}")
            values["chord"] = replacement
    if reduced:
        logging.warning("SheetSage2 ABC chord simplification: %s", "; ".join(sorted(reduced)))
    return result


class LocalSheetSage2Encoder:
    """Expose generate_abc without retaining a second model in RAM or VRAM."""

    def __init__(self, kind, path, root, device="auto", mert2_model="auto"):
        self.kind, self.path, self.root = kind, path, root
        self.device, self.mert2_model = device, mert2_model

    def generate_abc(self, audio, sample_rate, melody_only=True):
        import comfy.model_management as mm

        target = mm.get_torch_device() if self.device == "auto" else torch.device(self.device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("For HF/Refined SheetSage2 on this system, select sheetsage_device=cpu.")
        mm.throw_exception_if_processing_interrupted()
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            # Refined's decoder reads parameter version counters. Loading on a fresh
            # thread avoids creating inference tensors in ComfyUI's execution context.
            with ThreadPoolExecutor(max_workers=1) as worker:
                future = worker.submit(self._transcribe, audio, sample_rate, melody_only, target)
                return future.result()
        finally:
            gc.collect()
            mm.soft_empty_cache()

    def _transcribe(self, audio, sample_rate, melody_only, target):
        import comfy.audio
        import comfy.model_management as mm
        from comfy.audio_encoders.sheetsage2_abc import events_to_abc

        waves = comfy.audio.resample(audio.detach().float().cpu().mean(dim=1), int(sample_rate), 24000)
        if self.kind == "hf":
            from ..compat.sheetsage2 import load_sheetsage2
            from ..models.sheetsage2 import transcribe

            path = _mert_path(self.path, self.mert2_model)
            dtype = torch.bfloat16 if target.type == "cuda" and mm.should_use_bf16(target) else torch.float32
            model = load_sheetsage2(str(self.path), device=str(target), dtype=dtype, base_model_path=path)
            try:
                scores = []
                for wave in waves:
                    mm.throw_exception_if_processing_interrupted()
                    abc, _structure, _result, _midi = transcribe(model, wave, 24000, melody_only=melody_only)
                    scores.append(abc)
                return scores
            finally:
                del model

        from sheetsage2_refined import SheetSage2Refined
        from sheetsage2_refined.model import SheetSage
        from sheetsage2_refined.transcribe import transcribe_many

        if self.kind == "refined":
            _validate_refined(self.path, self.root)
            model = SheetSage2Refined.load(folder=self.path, device=str(target))
        else:
            model = SheetSage.load(self.path, device=str(target))
        try:
            scores = []
            for wave in waves:
                mm.throw_exception_if_processing_interrupted()
                if self.kind == "refined":
                    events = model.transcribe([wave], batch_size=1, encode_batch=1)[0]
                else:
                    events = transcribe_many(model, [wave], batch_size=1, encode_batch=1)[0]
                mm.throw_exception_if_processing_interrupted()
                scores.append(events_to_abc(_abc_events(events, melody_only), wave.numel() / 24000,
                                           melody_only=melody_only))
            return scores
        finally:
            del model


class YuE2UniversalPipelineLoader(YuE2NativePipelineLoader):
    DESCRIPTION = (
        "Replacement for YuE2 Native Pipeline Loader. Select Native single-file, "
        "merged SheetSage2, HF + local MERT2, or a complete SheetSage2-Refined folder. "
        "Refined loads its configured decoder and heads together. Keeps all native YuE2 ports. "
        "Uses local models only. HF/Refined models load for transcription and are then released."
    )
    CATEGORY = "YuE2/Universal SheetSage2"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = _loader_inputs()
        models = _models()
        options = ["none", *models]
        checkpoint_input = inputs["required"]["checkpoint"]
        checkpoint_input[1]["default"] = next(
            (name for name in checkpoint_input[0] if Path(name).name == "yue2_3b_int8_convrot.safetensors"),
            checkpoint_input[1]["default"],
        )
        default = next((name for name, (kind, path, _root) in models.items()
                        if kind == "native" and path.name == "sheetsage2_bf16.safetensors"), "none")
        inputs["required"]["audio_encoder"] = (options, {
            "default": default,
            "tooltip": "Choose a complete SheetSage2 model or Refined folder; decoder/head components are grouped automatically.",
        })
        inputs["optional"]["sheetsage_device"] = (["auto", "cuda", "cpu"], {
            "default": "auto", "tooltip": "Device for HF/Refined transcription. Native models use ComfyUI memory management.",
        })
        inputs["optional"]["mert2_model"] = (["auto", *_mert_models()], {
            "default": "auto", "tooltip": "Local parent encoder for HF adapter models. Native, merged and Refined models include the encoder.",
        })
        return inputs

    def load(self, checkpoint, audio_encoder="none", memory_budget_gib=0, offload_ar=True,
             sheetsage_device="auto", mert2_model="auto"):
        import comfy.audio_encoders.audio_encoders
        import comfy.sd
        import comfy.utils

        encoder = None
        path = None
        if audio_encoder != "none":
            models = _models()
            if audio_encoder not in models:
                raise ValueError("Selected SheetSage2 model is unavailable. Refresh and select a complete model or Refined folder.")
            kind, path, root = models[audio_encoder]
            if kind == "native":
                state = comfy.utils.load_torch_file(str(path), safe_load=True)
                encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(state)
            else:
                if kind in {"refined", "merged"}:
                    try:
                        import sheetsage2_refined
                    except ImportError as exc:
                        raise RuntimeError(
                            "This model requires SheetSage2-Refined. Install "
                            "requirements-sheetsage2-refined.txt with ComfyUI's Python, then restart ComfyUI. "
                            f"Import error: {exc}"
                        ) from exc
                    if kind == "refined":
                        _validate_refined(path, root)
                else:
                    _mert_path(path, mert2_model)
                encoder = LocalSheetSage2Encoder(kind, path, root, sheetsage_device, mert2_model)
        pipe, model, clip, vae, _encoder = _load_native(checkpoint, "none", memory_budget_gib, offload_ar)
        pipe.update(audio_encoder=encoder, audio_encoder_name=audio_encoder,
                    audio_encoder_path=str(path) if path else None)
        return pipe, model, clip, vae, encoder


NODE_CLASS_MAPPINGS = {"YuE2UniversalPipelineLoader": YuE2UniversalPipelineLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"YuE2UniversalPipelineLoader": "YuE2 Universal Pipeline Loader (SheetSage2)"}
