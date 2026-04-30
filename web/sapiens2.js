import { app } from "../../scripts/app.js";

const SEG_PARTS = {
  Background: ["all"],
  Apparel: ["all"],
  Eyeglass: ["all"],
  Face: ["all", "skin", "with eyeglass", "with mouth"],
  Hair: ["all"],
  Foot: ["all", "left", "right"],
  Hand: ["all", "left", "right"],
  Arm: ["all", "left", "right", "upper", "lower", "left upper", "left lower", "right upper", "right lower"],
  Leg: ["all", "left", "right", "upper", "lower", "left upper", "left lower", "right upper", "right lower"],
  Shoe: ["all", "left", "right"],
  Sock: ["all", "left", "right"],
  Clothing: ["all", "upper", "lower"],
  Torso: ["all"],
  Lip: ["all", "upper", "lower"],
  Teeth: ["all", "upper", "lower"],
  Tongue: ["all"],
  Mouth: ["all", "lip", "teeth", "tongue"],
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
  const withoutSide = normalized.replace(/^Left |^Right /, "");
  if (normalized === "Face Neck") {
    return "Face";
  }
  if (withoutSide === "Lower Arm" || withoutSide === "Upper Arm") {
    return "Arm";
  }
  if (withoutSide === "Lower Leg" || withoutSide === "Upper Leg") {
    return "Leg";
  }
  if (normalized.startsWith("Left ") || normalized.startsWith("Right ")) {
    return withoutSide;
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
  return SEG_PARTS[normalized] ? normalized : "Hair";
}

function legacyDetail(row = {}) {
  const value = String(row.detail || row.part || "all").toLowerCase().replaceAll("_", " ");
  const name = String(row.name || row.part || "").replace(/^\d+\s*:\s*/, "").replaceAll("_", " ");
  const hasLeft = value.includes("left");
  const hasRight = value.includes("right");
  const hasUpper = value.includes("upper");
  const hasLower = value.includes("lower");
  if (name === "Face Neck") {
    return "skin";
  }
  if (value === "full") {
    return "all";
  }
  if (hasLeft && hasUpper) {
    return "left upper";
  }
  if (hasLeft && hasLower) {
    return "left lower";
  }
  if (hasRight && hasUpper) {
    return "right upper";
  }
  if (hasRight && hasLower) {
    return "right lower";
  }
  if (hasLeft) {
    return "left";
  }
  if (hasRight) {
    return "right";
  }
  if (hasUpper) {
    return "upper";
  }
  if (hasLower) {
    return "lower";
  }
  return value || "all";
}

function defaultDetail(name) {
  return (SEG_PARTS[name] || ["all"])[0];
}

function rowState(row = {}) {
  const name = legacyName(row);
  const detail = legacyDetail(row);
  const details = SEG_PARTS[name] || ["all"];
  return {
    enabled: row.enabled ?? true,
    name,
    detail: details.includes(detail) ? detail : defaultDetail(name),
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
  refreshCanvas(node);
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

function refreshCanvas(node) {
  syncParts(node);
  const canvases = [app.canvas, globalThis.LGraphCanvas?.active_canvas].filter(Boolean);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  for (const canvas of canvases) {
    canvas.dirty_canvas = true;
    canvas.dirty_bgcanvas = true;
    canvas.setDirty?.(true, true);
    canvas.draw?.(true, true);
  }
}

function updatePartRow(node, targetWidget, value) {
  const rows = partRows(node).map((widget) => rowState(widget === targetWidget ? value : widget.value));
  rebuildPartRows(node, rows);
}

function makePartRowWidget(node, row = {}) {
  node.sapiens2PartRowIndex = (node.sapiens2PartRowIndex || 0) + 1;
  return {
    type: "sapiens2_part_row",
    name: `sapiens2_part_row_${node.sapiens2PartRowIndex}`,
    value: rowState(row),
    serialize: false,
    computeSize(width) {
      return [width || 300, ROW_HEIGHT];
    },
    draw(ctx, _node, width, y, height) {
      const value = this.value;
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
        updatePartRow(node, this, { ...this.value, enabled: !this.value.enabled });
        return true;
      }
      if (x >= layout.name.x && x <= layout.name.x + layout.name.width) {
        return pickFromMenu(event, SEG_PART_NAMES, this.value.name, (name) => {
          if (!SEG_PARTS[name]) {
            return;
          }
          const detail = SEG_PARTS[name].includes(this.value.detail) ? this.value.detail : defaultDetail(name);
          updatePartRow(node, this, { ...this.value, name, detail });
        });
      }
      if (x >= layout.detail.x && x <= layout.detail.x + layout.detail.width) {
        const details = SEG_PARTS[this.value.name] || ["all"];
        return pickFromMenu(event, details, this.value.detail, (detail) => {
          if (!details.includes(detail)) {
            return;
          }
          updatePartRow(node, this, { ...this.value, detail });
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
