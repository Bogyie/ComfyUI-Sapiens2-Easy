import { app } from "../../scripts/app.js";

const SEG_PARTS = [
  "00: Background",
  "01: Apparel",
  "02: Eyeglass",
  "03: Face_Neck",
  "04: Hair",
  "05: Left_Foot",
  "06: Left_Hand",
  "07: Left_Lower_Arm",
  "08: Left_Lower_Leg",
  "09: Left_Shoe",
  "10: Left_Sock",
  "11: Left_Upper_Arm",
  "12: Left_Upper_Leg",
  "13: Lower_Clothing",
  "14: Right_Foot",
  "15: Right_Hand",
  "16: Right_Lower_Arm",
  "17: Right_Lower_Leg",
  "18: Right_Shoe",
  "19: Right_Sock",
  "20: Right_Upper_Arm",
  "21: Right_Upper_Leg",
  "22: Torso",
  "23: Upper_Clothing",
  "24: Lower_Lip",
  "25: Upper_Lip",
  "26: Lower_Teeth",
  "27: Upper_Teeth",
  "28: Tongue",
];

function parseRows(value) {
  try {
    const rows = JSON.parse(value || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function hideWidget(widget) {
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function syncParts(node) {
  const rows = (node.sapiens2PartRows || []).map((row) => ({
    enabled: Boolean(row.enabled.value),
    part: row.part.value,
  }));
  const partsWidget = node.widgets?.find((widget) => widget.name === "parts");
  if (partsWidget) {
    partsWidget.value = JSON.stringify(rows);
  }
}

function addPartRow(node, row = {}) {
  node.sapiens2PartRows ||= [];
  const index = node.sapiens2PartRows.length + 1;
  const enabled = node.addWidget("toggle", `part_${index}_on`, row.enabled ?? true, () => syncParts(node));
  const part = node.addWidget(
    "combo",
    `part_${index}`,
    row.part || "04: Hair",
    () => syncParts(node),
    { values: SEG_PARTS }
  );
  const remove = node.addWidget("button", `remove_${index}`, "remove", () => {
    enabled.value = false;
    enabled.disabled = true;
    part.disabled = true;
    remove.disabled = true;
    syncParts(node);
    node.setDirtyCanvas(true, true);
  });
  node.sapiens2PartRows.push({ enabled, part, remove });
  syncParts(node);
  node.setDirtyCanvas(true, true);
}

function setupSegmentationNode(node) {
  if (node.sapiens2PartsReady) {
    return;
  }
  node.sapiens2PartsReady = true;
  const partsWidget = node.widgets?.find((widget) => widget.name === "parts");
  if (!partsWidget) {
    return;
  }
  const savedRows = parseRows(partsWidget.value);
  hideWidget(partsWidget);
  node.addWidget("button", "+ add part", "add", () => addPartRow(node));
  for (const row of savedRows) {
    addPartRow(node, row);
  }

  const originalOnSerialize = node.onSerialize;
  node.onSerialize = function (data) {
    syncParts(this);
    originalOnSerialize?.apply(this, arguments);
  };
}

function restorePartRows(node) {
  if (!node.sapiens2PartsReady || node.sapiens2PartRows?.length) {
    return;
  }
  const partsWidget = node.widgets?.find((widget) => widget.name === "parts");
  for (const row of parseRows(partsWidget?.value)) {
    addPartRow(node, row);
  }
}

app.registerExtension({
  name: "ComfyUI.Sapiens2Easy",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Sapiens2Segmentation") {
      return;
    }
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      setupSegmentationNode(this);
    };
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      originalOnConfigure?.apply(this, arguments);
      setupSegmentationNode(this);
      restorePartRows(this);
    };
  },
});
