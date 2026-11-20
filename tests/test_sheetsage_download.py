"""Download scope, resuming, and offline behavior for Refined model artifacts."""
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def download_env(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("_refined_download_test", ROOT / "models/sheetsage_download.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = types.ModuleType("sheetsage2_refined.pipeline")
    runtime.weight_files = lambda config: [config["base_weights"], config["decoder"]]
    package = types.ModuleType("sheetsage2_refined")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "sheetsage2_refined", package)
    monkeypatch.setitem(sys.modules, "sheetsage2_refined.pipeline", runtime)
    config = {"base_weights": "sheetsage2/base.safetensors", "decoder": "decoder.safetensors"}
    contents = {"config.json": json.dumps(config).encode(),
                config["base_weights"]: b"base weights", config["decoder"]: b"decoder weights",
                **{name: b"license text" for name in module.LICENSE_FILES}}

    def fetch(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[kwargs["filename"]])
        return str(path)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(side_effect=fetch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    folder = tmp_path / "SheetSage2-Refined"
    return module, hub.hf_hub_download, folder, contents


def test_fresh_download_fetches_only_pinned_model_files_and_licenses(download_env):
    module, fetch, folder, contents = download_env
    assert module.download_refined_model(folder) == folder
    assert {call.kwargs["filename"] for call in fetch.call_args_list} == set(contents)
    for call in fetch.call_args_list:
        assert call.kwargs["repo_id"] == "anime-song/SheetSage2-Refined"
        assert call.kwargs["revision"] == module.REFINED_REVISION
        assert call.kwargs["token"] is False
        assert Path(call.kwargs["local_dir"]) == folder
    assert module.missing_refined_files(folder) == []


def test_ready_local_model_makes_no_hub_requests(download_env):
    module, fetch, folder, contents = download_env
    for name, data in contents.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    module.download_refined_model(folder)
    assert module.missing_refined_files(folder) == []
    fetch.assert_not_called()


def test_missing_or_empty_weight_is_repaired_without_redownloading_other_files(download_env):
    module, fetch, folder, contents = download_env
    module.download_refined_model(folder)
    (folder / "decoder.safetensors").write_bytes(b"")
    fetch.reset_mock()
    module.download_refined_model(folder)
    assert fetch.call_count == 1
    assert fetch.call_args.kwargs["filename"] == "decoder.safetensors"
    assert fetch.call_args.kwargs["force_download"] is True
    assert all((folder / name).read_bytes() == data for name, data in contents.items())


def test_retry_after_network_failure_keeps_completed_artifacts(download_env):
    module, fetch, folder, contents = download_env
    original = fetch.side_effect

    def interrupted(**kwargs):
        if kwargs["filename"] == "sheetsage2/base.safetensors":
            raise OSError("network interrupted")
        return original(**kwargs)

    fetch.side_effect = interrupted
    with pytest.raises(OSError, match="network interrupted"):
        module.download_refined_model(folder)
    assert (folder / "config.json").is_file()
    assert (folder / "decoder.safetensors").is_file()
    fetch.side_effect = original
    fetch.reset_mock()
    module.download_refined_model(folder)
    downloaded = {call.kwargs["filename"] for call in fetch.call_args_list}
    assert "config.json" not in downloaded and "decoder.safetensors" not in downloaded
    assert module.missing_refined_files(folder) == []


@pytest.mark.parametrize("filename", ("../outside.safetensors", "/outside.safetensors",
                                     "C:/outside.safetensors", "heads\\outside.safetensors", "model.py"))
def test_untrusted_config_paths_are_rejected_before_weight_download(download_env, filename):
    module, fetch, folder, _contents = download_env
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"base_weights": filename, "decoder": "decoder.safetensors"}))
    with pytest.raises(ValueError):
        module.download_refined_model(folder)
    fetch.assert_not_called()


def test_checking_an_empty_folder_never_downloads(download_env):
    module, fetch, folder, _contents = download_env
    assert module.missing_refined_files(folder) == ["config.json"]
    assert not folder.exists()
    fetch.assert_not_called()


def test_missing_runtime_is_reported_before_download(download_env, monkeypatch):
    module, fetch, folder, _contents = download_env
    monkeypatch.setitem(sys.modules, "sheetsage2_refined.pipeline", None)
    with pytest.raises(RuntimeError, match="requirements-sheetsage2-refined.txt"):
        module.download_refined_model(folder)
    fetch.assert_not_called()
