"""Saved audio stays playable while embedded tags and workflow data are stripped."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audio_save(monkeypatch, tmp_path):
    api = types.ModuleType("comfy_api.latest")
    api.IO = types.SimpleNamespace(
        ComfyNode=object, FolderType=types.SimpleNamespace(output="output"),
        NodeOutput=lambda *values, **kwargs: types.SimpleNamespace(result=values, ui=kwargs["ui"]),
    )
    helper = types.SimpleNamespace(save_audio=mock.Mock(return_value=[{
        "filename": "song_00001.mp3", "subfolder": "audio", "type": "output",
    }]))
    api.UI = types.SimpleNamespace(AudioSaveHelper=helper, SavedAudios=lambda results: {"audio": results})
    monkeypatch.setitem(sys.modules, "comfy_api.latest", api)
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    spec = importlib.util.spec_from_file_location("_audio_save_test", ROOT / "nodes/audio_save.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_helper = helper
    return module


def make_audio(path, file_format):
    av = pytest.importorskip("av")
    codecs = {"flac": "flac", "mp3": "libmp3lame", "opus": "libopus"}
    samples = np.sin(np.arange(24000) * (2 * np.pi * 440 / 48000)).astype(np.float32)[None]
    with av.open(str(path), "w", format=file_format) as container:
        container.metadata.update({"prompt": "PRIVATE_PROMPT", "workflow": "PRIVATE_WORKFLOW", "artist": "PRIVATE_ARTIST"})
        stream = container.add_stream(codecs[file_format], rate=48000, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
        frame.sample_rate = 48000
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def decoded_samples(path):
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        return np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=-1)


@pytest.mark.parametrize("file_format", ("flac", "mp3", "opus"))
def test_metadata_cleanup_preserves_decoded_audio(audio_save, tmp_path, file_format):
    if audio_save.mutagen is None:
        pytest.skip("mutagen is optional outside the ComfyUI test environment")
    path = tmp_path / ("tagged." + file_format)
    make_audio(path, file_format)
    media = audio_save.mutagen.File(str(path))
    assert media.tags
    if file_format == "flac":
        from mutagen.flac import Picture
        picture = Picture()
        picture.mime = "image/png"
        picture.type = 3
        picture.data = b"test artwork"
        media.add_picture(picture)
        media.save()
    before = decoded_samples(path)
    audio_save._strip_metadata(path, file_format)
    cleaned = audio_save.mutagen.File(str(path))
    assert not cleaned.tags
    if file_format == "flac":
        assert cleaned.pictures == []
        assert all(block.code in (0, 1, 3) for block in cleaned.metadata_blocks)
        assert all(block.length == 0 for block in cleaned.metadata_blocks if block.code == 1)
    if file_format == "opus":
        assert cleaned.tags.vendor == ""
    assert np.array_equal(before, decoded_samples(path))


def test_strip_enabled_avoids_embedding_workflow_and_returns_player_ui(audio_save, monkeypatch):
    monkeypatch.setattr(audio_save, "mutagen", object())
    strip = mock.Mock()
    monkeypatch.setattr(audio_save, "_strip_metadata", strip)
    audio = {"waveform": object(), "sample_rate": 48000}
    result = audio_save.UniversalMMHSaveAudio.execute(
        audio, "audio/song", {"format": "mp3", "quality": "320k"}, True
    )
    kwargs = audio_save.test_helper.save_audio.call_args.kwargs
    assert kwargs["cls"] is None
    assert kwargs["format"] == "mp3" and kwargs["quality"] == "320k"
    assert result.result == (audio,)
    assert result.ui["audio"][0]["filename"] == "song_00001.mp3"
    assert strip.call_count == 1


def test_strip_disabled_keeps_comfy_metadata_and_skips_cleanup(audio_save, monkeypatch):
    strip = mock.Mock()
    monkeypatch.setattr(audio_save, "_strip_metadata", strip)
    audio_save.UniversalMMHSaveAudio.execute({"waveform": object()}, remove_metadata=False)
    assert audio_save.test_helper.save_audio.call_args.kwargs["cls"] is audio_save.UniversalMMHSaveAudio
    strip.assert_not_called()


def test_missing_cleanup_dependency_fails_before_saving(audio_save, monkeypatch):
    monkeypatch.setattr(audio_save, "mutagen", None)
    with pytest.raises(RuntimeError, match="requirements-audio.txt"):
        audio_save.UniversalMMHSaveAudio.execute({"waveform": object()}, remove_metadata=True)
    audio_save.test_helper.save_audio.assert_not_called()


def test_cleanup_cannot_modify_files_outside_output_directory(audio_save, monkeypatch):
    monkeypatch.setattr(audio_save, "mutagen", object())
    audio_save.test_helper.save_audio.return_value = [{"filename": "outside.mp3", "subfolder": "..", "type": "output"}]
    strip = mock.Mock()
    monkeypatch.setattr(audio_save, "_strip_metadata", strip)
    with pytest.raises(ValueError, match="outside"):
        audio_save.UniversalMMHSaveAudio.execute({"waveform": object()})
    strip.assert_not_called()
