import { app } from "../../scripts/app.js";

const SEG_PARTS = {
  Background: ["all"],
  Apparel: ["all"],
  Eyeglass: ["all"],
  "Face Neck": ["all"],
  Hair: ["all"],
  Foot: ["all", "left", "right"],
  Hand: ["all", "left", "right"],
  Arm: ["all", "left", "right"],
  "Lower Arm": ["all", "left", "right"],
  "Upper Arm": ["all", "left", "right"],
  Leg: ["all", "left", "right"],
  "Lower Leg": ["all", "left", "right"],
  "Upper Leg": ["all", "left", "right"],
  Shoe: ["all", "left", "right"],
  Sock: ["all", "left", "right"],
  Clothing: ["all", "upper", "lower"],
  Torso: ["all"],
  Lip: ["all", "upper", "lower"],
  Teeth: ["all", "upper", "lower"],
  Tongue: ["all"],
};

const SEG_PART_NAMES = Object.keys(SEG_PARTS);

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
  widget.hidden = true;
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function syncParts(node) {
  const rows = (node.sapiens2Parts || []).map((row) => ({
    enabled: Boolean(row.enabled.value),
    name: row.name.value,
    detail: row.detail.value,
  }));
  const partsWidget = node.widgets?.find((widget) => widget.name === "parts");
  if (partsWidget) {
    partsWidget.value = JSON.stringify(rows);
  }
}

function partRows(node) {
  return node.sapiens2Parts || [];
}

function legacyName(row = {}) {
  const value = row.name || row.part || "Hair";
  const normalized = String(value).replace(/^\d+\s*:\s*/, "").replaceAll("_", " ");
  if (normalized.startsWith("Left ") || normalized.startsWith("Right ")) {
    return normalized.replace(/^Left |^Right /, "");
  }
  if (normalized.endsWith(" Clothing")) {
    return "Clothing";
  }
  if (normalized.endsWith(" Lip")) {
    return "Lip";
  }
  if (normalized.endsWith(" Teeth")) {
    return "Teeth";
  }
  if (normalized === "Face Neck") {
    return "Face Neck";
  }
  return SEG_PARTS[normalized] ? normalized : "Hair";
}

function legacyDetail(row = {}) {
  const value = String(row.detail || row.part || "all").toLowerCase();
  if (value.includes("left")) {
    return "left";
  }
  if (value.includes("right")) {
    return "right";
  }
  if (value.includes("upper")) {
    return "upper";
  }
  if (value.includes("lower")) {
    return "lower";
  }
  return "all";
}

function setDetailOptions(nameWidget, detailWidget) {
  const values = SEG_PARTS[nameWidget.value] || ["all"];
  detailWidget.options.values = values;
  if (!values.includes(detailWidget.value)) {
    detailWidget.value = "all";
  }
}

function rowState(row = {}) {
  return {
    enabled: row.enabled ?? true,
    name: legacyName(row),
    detail: legacyDetail(row),
  };
}

function removePartWidgets(node) {
  node.widgets = (node.widgets || []).filter((widget) => !widget.name?.startsWith("sapiens2_part_"));
  node.sapiens2Parts = [];
}

function rebuildPartRows(node, rows) {
  removePartWidgets(node);
  for (const row of rows) {
    addPartRow(node, rowState(row), false);
  }
  syncParts(node);
  node.setDirtyCanvas(true, true);
}

function addPartRow(node, row = {}, dirty = true) {
  node.sapiens2Parts ||= [];
  const index = node.sapiens2Parts.length + 1;
  const enabled = node.addWidget("toggle", `sapiens2_part_${index}_on`, row.enabled ?? true, () => syncParts(node));
  const name = node.addWidget(
    "combo",
    `sapiens2_part_${index}_name`,
    legacyName(row),
    () => {
      setDetailOptions(name, detail);
      syncParts(node);
    },
    { values: SEG_PART_NAMES }
  );
  const detail = node.addWidget(
    "combo",
    `sapiens2_part_${index}_detail`,
    legacyDetail(row),
    () => syncParts(node),
    { values: SEG_PARTS[legacyName(row)] || ["all"] }
  );
  setDetailOptions(name, detail);
  const remove = node.addWidget("button", `sapiens2_part_${index}_remove`, "remove", () => {
    const rows = partRows(node)
      .filter((item) => item.remove !== remove)
      .map((item) => ({
        enabled: item.enabled.value,
        name: item.name.value,
        detail: item.detail.value,
      }));
    rebuildPartRows(node, rows);
  });
  node.sapiens2Parts.push({ enabled, name, detail, remove });
  syncParts(node);
  if (dirty) {
    node.setDirtyCanvas(true, true);
  }
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
    addPartRow(node, row, false);
  }
  syncParts(node);

  const originalOnSerialize = node.onSerialize;
  node.onSerialize = function (data) {
    syncParts(this);
    originalOnSerialize?.apply(this, arguments);
  };
}

function restorePartRows(node) {
  if (!node.sapiens2PartsReady || node.sapiens2Parts?.length) {
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
