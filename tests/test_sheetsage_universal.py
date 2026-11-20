"""Universal SheetSage2 discovery and pipeline contracts without model weights."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
from unittest import mock

import pytest
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def universal(monkeypatch, tmp_path):
    roots = {name: tmp_path / name for name in ("audio_encoders", "sheetsage2", "mert2")}
    for root in roots.values():
        root.mkdir()
    folders = types.SimpleNamespace(
        get_folder_paths=lambda name: [str(roots.get(name, tmp_path / name))],
        get_filename_list=lambda name: ["YuE2/yue2_3b_int8_convrot.safetensors"] if name == "checkpoints" else [],
        get_full_path_or_raise=lambda name, filename: str(tmp_path / name / filename),
    )
    name = "_universal_sheetsage_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    node_package = types.ModuleType(name + ".nodes")
    node_package.__path__ = [str(ROOT / "nodes")]
    model_package = types.ModuleType(name + ".models")
    model_package.__path__ = [str(ROOT / "models")]
    paths = types.ModuleType(name + ".models.paths")
    paths.model_dirs = lambda kind: [str(roots[kind])]
    for module_name, value in {
        name: package, name + ".nodes": node_package, name + ".models": model_package,
        name + ".models.paths": paths, "folder_paths": folders,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, value)
    for filename in ("comfy_native", "sheetsage_universal"):
        module_name = name + ".nodes." + filename
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "nodes" / f"{filename}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    module.test_roots = roots
    return module


@pytest.fixture
def comfy_modules(monkeypatch):
    names = ("comfy", "comfy.sd", "comfy.utils", "comfy.audio", "comfy.model_management",
             "comfy.audio_encoders", "comfy.audio_encoders.audio_encoders", "comfy.audio_encoders.sheetsage2_abc")
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules["comfy.sd"].load_checkpoint_guess_config = mock.Mock(return_value=(object(), object(), object()))
    modules["comfy.utils"].load_torch_file = mock.Mock(return_value={"weights": True})
    modules["comfy.audio_encoders.audio_encoders"].load_audio_encoder_from_sd = mock.Mock(
        return_value=types.SimpleNamespace(generate_abc=mock.Mock())
    )
    management = modules["comfy.model_management"]
    management.get_torch_device = lambda: torch.device("cpu")
    management.throw_exception_if_processing_interrupted = mock.Mock()
    management.unload_all_models = mock.Mock()
    management.soft_empty_cache = mock.Mock()
    modules["comfy.audio"].resample = mock.Mock(side_effect=lambda audio, *rates: audio)
    modules["comfy.audio_encoders.sheetsage2_abc"].events_to_abc = mock.Mock(return_value="K:C\nV:Ins\nC|")
    return modules


@pytest.fixture
def refined_modules(monkeypatch):
    pipeline = types.ModuleType("sheetsage2_refined.pipeline")
    pipeline.weight_files = lambda config: [config["base_weights"], *([config["decoder"]] if config.get("decoder") else [])]
    refined = types.ModuleType("sheetsage2_refined")
    refined.__path__ = []
    refined.SheetSage2Refined = mock.Mock()
    model = types.ModuleType("sheetsage2_refined.model")
    model.SheetSage = mock.Mock()
    transcribe = types.ModuleType("sheetsage2_refined.transcribe")
    transcribe.transcribe_many = mock.Mock()
    for value in (pipeline, refined, model, transcribe):
        monkeypatch.setitem(sys.modules, value.__name__, value)
    return refined


def write_weights(path, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: torch.zeros(1) for key in keys}, str(path))


def test_discovery_groups_refined_heads_and_skips_invalid_or_cached_files(universal):
    root = universal.test_roots["audio_encoders"]
    refined = root / "Refined"
    write_weights(root / "native.safetensors", universal._NATIVE_KEYS)
    write_weights(refined / "base.safetensors", universal._MERGED_KEYS)
    write_weights(refined / "decoder.safetensors", {"decoder.weight"})
    write_weights(root / ".cache/cached.safetensors", universal._NATIVE_KEYS)
    (refined / "config.json").write_text('{"base_weights":"base.safetensors","decoder":"decoder.safetensors"}')
    (root / "invalid.safetensors").write_bytes(b"invalid")
    choices = universal._models()
    assert {entry[0] for entry in choices.values()} == {"native", "merged", "refined"}
    assert len(choices) == 3
    assert not any("decoder.safetensors" in name or ".cache" in name for name in choices)


def test_schema_retains_native_ports_and_handles_nested_checkpoint_names(universal):
    schema = universal.YuE2UniversalPipelineLoader.INPUT_TYPES()
    assert schema["required"]["checkpoint"][1]["default"] == "YuE2/yue2_3b_int8_convrot.safetensors"
    assert schema["required"]["audio_encoder"][0] == ["none", universal._AUTO_REFINED]
    assert universal.YuE2UniversalPipelineLoader.RETURN_TYPES == universal.YuE2NativePipelineLoader.RETURN_TYPES
    assert "YuE2UniversalPipelineLoader" in universal.NODE_CLASS_MAPPINGS


def test_unknown_selection_is_rejected_before_checkpoint_loading(universal, comfy_modules):
    with pytest.raises(ValueError, match="unavailable"):
        universal.YuE2UniversalPipelineLoader().load("checkpoint", "../../anything")
    comfy_modules["comfy.sd"].load_checkpoint_guess_config.assert_not_called()


def test_native_load_keeps_pipe_and_standard_outputs_without_optional_package(universal, comfy_modules, monkeypatch):
    root = universal.test_roots["audio_encoders"]
    write_weights(root / "native.safetensors", universal._NATIVE_KEYS)
    monkeypatch.setitem(sys.modules, "sheetsage2_refined", None)
    selected = next(iter(universal._models()))
    pipe, model, clip, vae, encoder = universal.YuE2UniversalPipelineLoader().load("checkpoint", selected, 8, False)
    assert (pipe["model"], pipe["clip"], pipe["vae"]) == (model, clip, vae)
    assert pipe["audio_encoder"] is encoder
    assert pipe["memory_budget_gib"] == 8 and pipe["offload_ar"] is False
    comfy_modules["comfy.utils"].load_torch_file.assert_called_once_with(str(root / "native.safetensors"), safe_load=True)


def test_optional_import_failure_explains_installation(universal, comfy_modules, monkeypatch):
    root = universal.test_roots["audio_encoders"]
    write_weights(root / "merged.safetensors", universal._MERGED_KEYS)
    monkeypatch.setitem(sys.modules, "sheetsage2_refined", None)
    with pytest.raises(RuntimeError, match="requirements-sheetsage2-refined.txt"):
        universal.YuE2UniversalPipelineLoader().load("checkpoint", next(iter(universal._models())))
    comfy_modules["comfy.sd"].load_checkpoint_guess_config.assert_not_called()


def test_auto_download_choice_works_without_an_installed_model(universal, comfy_modules, refined_modules, monkeypatch):
    downloader = types.ModuleType("_universal_sheetsage_tests.models.sheetsage_download")
    root = universal.test_roots["audio_encoders"]
    target = root / "SheetSage2-Refined"
    downloader.download_refined_model = mock.Mock(return_value=target)
    monkeypatch.setitem(sys.modules, downloader.__name__, downloader)
    monkeypatch.setattr(universal, "_validate_refined", mock.Mock())
    pipe, _model, _clip, _vae, encoder = universal.YuE2UniversalPipelineLoader().load(
        "checkpoint", universal._AUTO_REFINED
    )
    downloader.download_refined_model.assert_called_once_with(target)
    assert encoder.kind == "refined"
    assert pipe["audio_encoder_name"] == universal._AUTO_REFINED


def test_refined_missing_files_and_path_escape_are_reported(universal, refined_modules):
    root = universal.test_roots["audio_encoders"]
    folder = root / "Refined"
    folder.mkdir()
    config = folder / "config.json"
    config.write_text('{"base_weights":"missing.safetensors"}')
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        universal._validate_refined(folder, root)
    config.write_text('{"base_weights":"../../outside.safetensors"}')
    with pytest.raises(ValueError, match="outside"):
        universal._validate_refined(folder, root)


def test_hf_requires_local_parent_and_revalidates_parent_selection(universal):
    root = universal.test_roots["sheetsage2"]
    with pytest.raises(FileNotFoundError, match="No download was started"):
        universal._mert_path(root, "auto")
    with pytest.raises(ValueError, match="no longer available"):
        universal._mert_path(root, "../../parent")
    parent = universal.test_roots["mert2"] / "MERT-v2-FullSong"
    parent.mkdir()
    for name in ("config.json", "model.safetensors", "configuration_mert2.py", "modeling_mert2.py"):
        (parent / name).touch()
    assert universal._mert_path(root, "auto") == str(parent)


def test_refined_batch_uses_worker_thread_and_matches_audio_to_abc_contract(
    universal, comfy_modules, refined_modules,
):
    root = universal.test_roots["audio_encoders"]
    (root / "config.json").write_text('{"base_weights":"base.safetensors"}')
    (root / "base.safetensors").touch()
    events = [{"time": 0.0, "values": {"chord": "C:maj", "melody": []}}]
    model = mock.Mock()
    model.transcribe.return_value = [events]
    threads = []

    def load(**kwargs):
        threads.append(threading.current_thread())
        return model

    refined_modules.SheetSage2Refined.load.side_effect = load
    audio = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
    encoder = universal.LocalSheetSage2Encoder("refined", root, root, device="cpu")
    scores = encoder.generate_abc(audio, 48000, melody_only=True)
    assert scores == ["K:C\nV:Ins\nC|", "K:C\nV:Ins\nC|"]
    assert len(threads) == 1 and threads[0] is not threading.current_thread()
    resample = comfy_modules["comfy.audio"].resample
    assert resample.call_args.args[1:] == (48000, 24000)
    assert torch.equal(resample.call_args.args[0], audio.mean(dim=1))
    assert model.transcribe.call_count == 2
    assert "chord" in events[0]["values"]
    comfy_modules["comfy.model_management"].unload_all_models.assert_called_once()
    assert comfy_modules["comfy.model_management"].soft_empty_cache.call_count == 2


def test_transcription_failure_still_cleans_up_comfy_memory(universal, comfy_modules, monkeypatch):
    root = universal.test_roots["audio_encoders"]
    encoder = universal.LocalSheetSage2Encoder("merged", root, root, device="cpu")
    monkeypatch.setattr(encoder, "_transcribe", mock.Mock(side_effect=RuntimeError("transcription failed")))
    with pytest.raises(RuntimeError, match="transcription failed"):
        encoder.generate_abc(torch.zeros(1, 1, 10), 24000)
    assert comfy_modules["comfy.model_management"].soft_empty_cache.call_count == 2


def test_extended_chord_reduction_is_reported_without_mutating_events(universal, caplog):
    events = [{"time": 0.0, "values": {"chord": "C:maj7(9)/3"}}]
    result = universal._abc_events(events, False)
    assert result[0]["values"]["chord"] == "C:maj7/3"
    assert events[0]["values"]["chord"] == "C:maj7(9)/3"
    assert "C:maj7(9)/3 -> C:maj7/3" in caplog.text
