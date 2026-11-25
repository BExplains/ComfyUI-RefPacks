const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

let extension;
const scheduled = [];
const code = fs.readFileSync(path.join(__dirname, '../web/merge_bundles.js'), 'utf8');
vm.runInNewContext(code.replace(/^import .*;\r?\n/, ''), {
  app: { registerExtension(value) { extension = value; } },
  queueMicrotask(callback) { scheduled.push(callback); },
});
class Node {
  constructor() {
    this.inputs = Array.from({ length: 64 }, (_, i) => ({ name: `bundle_${i + 1}`, link: null }));
    this.size = [300, 1300];
  }
  addInput(name, type) { this.inputs.push({ name, type, link: null }); }
  removeInput(index) {
    assert.equal(this.inputs[index].link, null, 'Never remove a connected slot');
    this.inputs.splice(index, 1);
    this.onConnectionsChange();
  }
  computeSize() { return [200, 50 + this.inputs.length * 20]; }
  setSize(size) { this.size = size; }
  setDirtyCanvas() {}
}
function flush() { while (scheduled.length) scheduled.shift()(); }
extension.beforeRegisterNodeDef(Node, { name: 'BExplainsMergeRefPackBundles' });
const node = new Node();
node.onNodeCreated(); flush();
assert.equal(node.inputs.length, 2);
assert.equal(node.size[0], 300);
node.inputs[0].link = 10;
node.inputs[1].link = 11;
node.onConnectionsChange(); flush();
assert.equal(node.inputs.length, 3);
node.inputs[2].link = 12;
node.onConnectionsChange(); flush();
assert.equal(node.inputs.length, 4);
node.inputs[1].link = null;
node.onConnectionsChange(); flush();
assert.equal(node.inputs[2].link, 12, 'Preserve slot indices across middle disconnects');
const saved = JSON.parse(JSON.stringify(node.inputs));
const restored = new Node();
restored.onNodeCreated();
restored.inputs = saved;
restored.onConfigure(); flush();
assert.equal(restored.inputs.length, 4);
assert.equal(restored.inputs[2].link, 12);
restored.inputs[2].link = null;
restored.onConnectionsChange(); flush();
assert.equal(restored.inputs.length, 2);
const full = new Node();
full.inputs.forEach((input, i) => { input.link = i; });
full.onConfigure(); flush();
assert.equal(full.inputs.length, 64);
console.log('Merge bundle slots: growth, gaps, disconnects, workflow restore and 64-slot limit passed.');
