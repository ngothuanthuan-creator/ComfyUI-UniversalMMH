"""ComfyUI-UniversalMMH: YuE2 generation and SheetSage2 transcription.

The YuE2 wheel is loaded from the selected model directory without pip dependency
resolution. SheetSage2 can use a fully local MERT2 parent model.
"""
from __future__ import annotations

import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _try_register(module_name: str, mapping: dict) -> None:
    """Register one node module without breaking unrelated node groups."""
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except Exception as exc:  # noqa: BLE001
        mapping[module_name] = f"{type(exc).__name__}: {exc}"
        print(f"[ComfyUI-UniversalMMH] Skipping {module_name}: {exc}")
        return
    classes = getattr(module, "NODE_CLASS_MAPPINGS", {})
    names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(classes)
    for node_name, node_class in classes.items():
        if not node_class.CATEGORY.startswith("UniversalMMH/"):
            node_class.CATEGORY = "UniversalMMH/" + node_class.CATEGORY
        display_name = names.get(node_name, node_name)
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = (
            display_name if display_name.startswith("UniversalMMH · ") else "UniversalMMH · " + display_name
        )


_failed: dict = {}

_try_register("nodes.comfy_native", _failed)
_try_register("nodes.sheetsage_universal", _failed)
_try_register("nodes.audio_save", _failed)
_try_register("nodes.yue2_lora", _failed)
_try_register("nodes.yue2", _failed)
_try_register("nodes.yue2_advanced", _failed)
_try_register("nodes.sheetsage2", _failed)
_try_register("nodes.abc_tools", _failed)

if not NODE_CLASS_MAPPINGS:
    print("[ComfyUI-UniversalMMH] No nodes were registered; check dependencies and the errors above")
else:
    print(f"[ComfyUI-UniversalMMH] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
    for name, err in _failed.items():
        print(f"[ComfyUI-UniversalMMH]   Failed to load {name}: {err}")

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
