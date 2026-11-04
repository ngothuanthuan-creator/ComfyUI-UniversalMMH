"""Pack branding must preserve the node identifiers used by saved workflows."""
from pathlib import Path
import importlib.util
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("comfy_native", "sheetsage_universal", "yue2_lora", "yue2", "yue2_advanced", "sheetsage2", "abc_tools")


@pytest.fixture
def branded_pack(monkeypatch):
    name = "_universalmmh_branding_test"
    expected = {}
    monkeypatch.setattr(sys, "path", list(sys.path))
    for module_name in MODULES:
        module = types.ModuleType(name + ".nodes." + module_name)
        node_id = "SavedWorkflow_" + module_name
        node_class = type(node_id, (), {"CATEGORY": "YuE2/Test"})
        module.NODE_CLASS_MAPPINGS = {node_id: node_class}
        module.NODE_DISPLAY_NAME_MAPPINGS = {node_id: "YuE2 " + module_name}
        monkeypatch.setitem(sys.modules, module.__name__, module)
        expected[node_id] = node_class
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    pack = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, pack)
    spec.loader.exec_module(pack)
    return pack, expected


def test_rebranding_keeps_workflow_ids_and_classes(branded_pack):
    pack, expected = branded_pack
    assert pack.NODE_CLASS_MAPPINGS == expected
    assert len(pack.NODE_DISPLAY_NAME_MAPPINGS) == len(expected)
    assert all(label.startswith("UniversalMMH · ") for label in pack.NODE_DISPLAY_NAME_MAPPINGS.values())
    assert all(node.CATEGORY == "UniversalMMH/YuE2/Test" for node in pack.NODE_CLASS_MAPPINGS.values())


def test_registering_again_does_not_duplicate_branding(branded_pack):
    pack, expected = branded_pack
    pack._try_register("nodes.comfy_native", {})
    assert pack.NODE_CLASS_MAPPINGS == expected
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["SavedWorkflow_comfy_native"] == "UniversalMMH · YuE2 comfy_native"
    assert expected["SavedWorkflow_comfy_native"].CATEGORY == "UniversalMMH/YuE2/Test"
