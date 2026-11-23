import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "BExplains.RefPacks.Preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["BExplainsRefPacksPreviewText", "BExplainsSaveRefPack", "BExplainsSaveRefPacksReferences", "BExplainsInspectRefPacks", "BExplainsInspectReferences"].includes(nodeData.name)) return;
    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      const text = Array.isArray(message.text) ? message.text.join("\n") : String(message.text ?? "");
      let widget = this.widgets?.find(w => w.name === "refpacks_result");
      if (!widget) {
        widget = ComfyWidgets.STRING(this, "refpacks_result", ["STRING", { multiline: true }], app).widget;
        widget.options.serialize = false;
        if (widget.inputEl) widget.inputEl.readOnly = true;
      }
      widget.value = text;
      this.setSize([Math.max(this.size[0], 480), Math.max(this.size[1], 180)]);
      this.setDirtyCanvas(true, true);
    };
  }
});
