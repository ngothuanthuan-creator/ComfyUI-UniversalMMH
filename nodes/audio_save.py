"""Save audio with ComfyUI's player and optional removal of embedded tags."""
from pathlib import Path

import folder_paths
from comfy_api.latest import IO, UI

try:
    import mutagen
except ImportError:
    mutagen = None


def _strip_metadata(path, file_format):
    media = mutagen.File(str(path))
    if media is None:
        raise ValueError(f"Cannot read saved audio for metadata cleanup: {path}")
    if file_format == "flac":
        # Keep the stream description and seek index; remove optional metadata,
        # comments, pictures, applications, cuesheets, and leading/trailing ID3.
        media.metadata_blocks[:] = [block for block in media.metadata_blocks if block.code in (0, 3)]
        media.tags = None
        media.cuesheet = None
        media.save(deleteid3=True, padding=lambda _info: 0)
    elif file_format == "opus":
        media.tags.clear()
        media.tags.vendor = ""
        media.save()
    else:
        media.delete()


class UniversalMMHSaveAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="UniversalMMHSaveAudio",
            display_name="UniversalMMH · Save Audio + Player",
            category="UniversalMMH/Audio",
            description="Save FLAC, MP3 or Opus and play the saved file in ComfyUI. Optionally strip embedded tags, workflow and prompt metadata without re-encoding.",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/UniversalMMH"),
                IO.DynamicCombo.Input("format", options=[
                    IO.DynamicCombo.Option("flac", []),
                    IO.DynamicCombo.Option("mp3", [
                        IO.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0"),
                    ]),
                    IO.DynamicCombo.Option("opus", [
                        IO.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k"),
                    ]),
                ]),
                IO.Boolean.Input("remove_metadata", default=True,
                                 tooltip="Remove embedded tags, prompt/workflow and cover art. Audio headers needed for decoding remain."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            outputs=[IO.Audio.Output("audio")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="audio/UniversalMMH", format=None, remove_metadata=True):
        if audio is None:
            raise ValueError("Save Audio + Player needs an AUDIO input.")
        if remove_metadata and mutagen is None:
            raise RuntimeError("Install requirements-audio.txt with ComfyUI's Python, then restart to enable metadata cleanup.")
        file_format = (format or {}).get("format", "flac")
        quality = (format or {}).get("quality", "128k")
        results = UI.AudioSaveHelper.save_audio(
            audio, filename_prefix=filename_prefix, folder_type=IO.FolderType.output,
            cls=None if remove_metadata else cls, format=file_format, quality=quality,
        )
        if remove_metadata:
            root = Path(folder_paths.get_output_directory()).resolve()
            for result in results:
                path = (root / result["subfolder"] / result["filename"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Saved audio path is outside the ComfyUI output directory.")
                _strip_metadata(path, file_format)
        return IO.NodeOutput(audio, ui=UI.SavedAudios(results))


NODE_CLASS_MAPPINGS = {"UniversalMMHSaveAudio": UniversalMMHSaveAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalMMHSaveAudio": "Save Audio + Player"}
