import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "ComfyUI-UniversalMMH.AudioPreview",
  beforeRegisterNodeDef(_nodeType, nodeData) {
    if (nodeData.name !== "UniversalMMHSaveAudio") return;
    // Reuse ComfyUI's audio widget, saved-output URLs and playback handling.
    nodeData.input ??= {};
    nodeData.input.required ??= {};
    nodeData.input.required.audioUI = ["AUDIO_UI", {}];
  },
});
