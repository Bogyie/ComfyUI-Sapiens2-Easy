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
const ROW_HEIGHT = 28;
const ROW_GAP = 6;

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
  widget.label = "";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
  widget.options ||= {};
  widget.options.hidden = true;

  for (const element of [widget.element, widget.inputEl, widget.textElement]) {
    if (!element) {
      continue;
    }
    element.hidden = true;
    element.style.display = "none";
  }
  const parent = widget.inputEl?.parentElement || widget.element?.parentElement;
  if (parent) {
    parent.hidden = true;
    parent.style.display = "none";
  }
}

function syncParts(node) {
  const rows = (node.sapiens2Parts || []).map((widget) => rowState(widget.value));
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

function rowState(row = {}) {
  const name = legacyName(row);
  const detail = legacyDetail(row);
  const details = SEG_PARTS[name] || ["all"];
  return {
    enabled: row.enabled ?? true,
    name,
    detail: details.includes(detail) ? detail : "all",
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

function drawBox(ctx, x, y, width, height, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, width, height, 4);
  } else {
    ctx.rect(x, y, width, height);
  }
  ctx.fill();
}

function rowLayout(width) {
  const padding = 8;
  const toggleWidth = 26;
  const removeWidth = 24;
  const available = Math.max(180, width - padding * 2 - toggleWidth - removeWidth - ROW_GAP * 3);
  const detailWidth = Math.min(78, Math.max(58, Math.floor(available * 0.32)));
  const nameWidth = Math.max(90, available - detailWidth);
  let x = padding;
  const toggle = { x, width: toggleWidth };
  x += toggleWidth + ROW_GAP;
  const name = { x, width: nameWidth };
  x += nameWidth + ROW_GAP;
  const detail = { x, width: detailWidth };
  x += detailWidth + ROW_GAP;
  const remove = { x, width: removeWidth };
  return { toggle, name, detail, remove };
}

function drawText(ctx, text, x, y, width, height, align = "center") {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();
  ctx.fillStyle = globalThis.LiteGraph?.WIDGET_TEXT_COLOR || "#ddd";
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  const textX = align === "left" ? x + 8 : x + width / 2;
  ctx.fillText(text, textX, y + height / 2);
  ctx.restore();
}

function menuChoice(value) {
  return typeof value === "string" ? value : value?.content ?? value?.value;
}

function pickFromMenu(event, values, currentValue, callback) {
  const LiteGraph = globalThis.LiteGraph;
  if (LiteGraph?.ContextMenu) {
    event?.preventDefault?.();
    new LiteGraph.ContextMenu(values, {
      event,
      callback: (value) => callback(menuChoice(value)),
    });
    return true;
  }

  const index = values.indexOf(currentValue);
  callback(values[(index + 1) % values.length]);
  return true;
}

function markDirty(node) {
  syncParts(node);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
  app.canvas?.draw?.(true, true);
}

function makePartRowWidget(node, row = {}) {
  const value = rowState(row);
  node.sapiens2PartRowIndex = (node.sapiens2PartRowIndex || 0) + 1;
  return {
    type: "sapiens2_part_row",
    name: `sapiens2_part_row_${node.sapiens2PartRowIndex}`,
    value,
    serialize: false,
    computeSize(width) {
      return [width || 300, ROW_HEIGHT];
    },
    draw(ctx, _node, width, y, height) {
      this.lastWidth = width;
      const layout = rowLayout(width);
      const rowY = y + 3;
      const rowHeight = Math.min(height - 6, 24);
      const bg = globalThis.LiteGraph?.WIDGET_BGCOLOR || "#333";
      const muted = "#666";

      ctx.font = "12px Arial";

      drawBox(ctx, layout.toggle.x, rowY, layout.toggle.width, rowHeight, value.enabled ? "#5a8" : muted);
      drawText(ctx, value.enabled ? "on" : "off", layout.toggle.x, rowY, layout.toggle.width, rowHeight);

      drawBox(ctx, layout.name.x, rowY, layout.name.width, rowHeight, bg);
      drawText(ctx, value.name, layout.name.x, rowY, layout.name.width, rowHeight, "left");

      drawBox(ctx, layout.detail.x, rowY, layout.detail.width, rowHeight, bg);
      drawText(ctx, value.detail, layout.detail.x, rowY, layout.detail.width, rowHeight);

      drawBox(ctx, layout.remove.x, rowY, layout.remove.width, rowHeight, "#734");
      drawText(ctx, "x", layout.remove.x, rowY, layout.remove.width, rowHeight);
    },
    mouse(event, pos, node) {
      if (!["pointerdown", "mousedown"].includes(event.type)) {
        return false;
      }
      const layout = rowLayout(this.lastWidth || node.size?.[0] || 300);
      const x = pos[0];

      if (x >= layout.toggle.x && x <= layout.toggle.x + layout.toggle.width) {
        value.enabled = !value.enabled;
        markDirty(node);
        return true;
      }
      if (x >= layout.name.x && x <= layout.name.x + layout.name.width) {
        return pickFromMenu(event, SEG_PART_NAMES, value.name, (name) => {
          if (!SEG_PARTS[name]) {
            return;
          }
          value.name = name;
          if (!SEG_PARTS[name].includes(value.detail)) {
            value.detail = "all";
          }
          markDirty(node);
        });
      }
      if (x >= layout.detail.x && x <= layout.detail.x + layout.detail.width) {
        return pickFromMenu(event, SEG_PARTS[value.name] || ["all"], value.detail, (detail) => {
          if (!(SEG_PARTS[value.name] || ["all"]).includes(detail)) {
            return;
          }
          value.detail = detail;
          markDirty(node);
        });
      }
      if (x >= layout.remove.x && x <= layout.remove.x + layout.remove.width) {
        const rows = partRows(node)
          .filter((widget) => widget !== this)
          .map((widget) => rowState(widget.value));
        rebuildPartRows(node, rows);
        return true;
      }
      return false;
    },
  };
}

function addPartRow(node, row = {}, dirty = true) {
  node.sapiens2Parts ||= [];
  const widget = makePartRowWidget(node, row);
  if (node.addCustomWidget) {
    node.addCustomWidget(widget);
  } else {
    node.widgets ||= [];
    node.widgets.push(widget);
  }
  node.sapiens2Parts.push(widget);
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
