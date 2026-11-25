import { app } from "../../scripts/app.js";

const MAX_BUNDLES = 64;

function updateSlots(node) {
  const inputs = node.inputs ?? [];
  let lastConnected = -1;
  for (let i = 0; i < inputs.length; i++) {
    if (inputs[i].link != null) lastConnected = i;
  }
  const count = Math.min(MAX_BUNDLES, Math.max(2, lastConnected + 2));
  while ((node.inputs?.length ?? 0) > count) node.removeInput(node.inputs.length - 1);
  while ((node.inputs?.length ?? 0) < count) {
    node.addInput(`bundle_${node.inputs.length + 1}`, "BEXPLAINS_REFERENCES");
  }
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "BExplains.RefPacks.MergeBundles",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BExplainsMergeRefPackBundles") return;
    for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
      const original = nodeType.prototype[hook];
      nodeType.prototype[hook] = function (...args) {
        const result = original?.apply(this, args);
        if (!this._bundleSlotsPending) {
          this._bundleSlotsPending = true;
          queueMicrotask(() => {
            updateSlots(this);
            this._bundleSlotsPending = false;
          });
        }
        return result;
      };
    }
  },
});
