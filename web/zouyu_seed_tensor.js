/**
 * ZouyuSeedTensor 前端扩展
 *
 * 功能：
 * 6.  @ 引用 / 加载时自动显示种子文件下拉菜单（@ 自动补全 + 文件列表刷新）
 * 7.  批量处理进度（后端 ProgressBar 驱动，前端无需额外逻辑）
 * 10. 前端动态 LoRA 槽管理（+/- 增删）与动态参考槽管理（+/- 增删）
 * 5.  目录刷新按钮
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CATEGORY = "ZouyuAI/SeedTensor";
const MAX_REF_IMAGE = 9;
const MAX_REF_VIDEO = 3;
const MAX_REF_AUDIO = 3;
const MAX_LORA = 8;

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

async function listSeedFiles() {
  try {
    const d = await fetchJson("/zouyu_seed_tensor/files");
    return d.files || d || [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// 文件下拉刷新（功能 6）
// ---------------------------------------------------------------------------
async function refreshFileComboWidget(widget) {
  const files = await listSeedFiles();
  const full = (files || []).filter((f) => String(f).endsWith(".pt"));
  const values = full.length ? full : ["(暂无文件)"];
  if (widget.options) {
    widget.options.values = values;
  }
  if (!values.includes(widget.value)) {
    widget.value = values[0];
  }
}

async function refreshAllFileCombos() {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    const w = n.widgets?.find((x) => x.name === "file_name" && x.type === "combo");
    if (w) await refreshFileComboWidget(w);
  }
  app.graph?.setDirtyCanvas(true, false);
}

function addRefreshButton(node) {
  const w = node.widgets?.find((x) => x.name === "file_name");
  if (!w) return;
  const btn = node.addWidget("button", "🔄 刷新目录", null, async () => {
    await refreshFileComboWidget(w);
    app.graph?.setDirtyCanvas(true, false);
  });
  btn.serialize = false;
}

// ---------------------------------------------------------------------------
// @ 提及自动补全（功能 6）
// ---------------------------------------------------------------------------
const MENTION_STYLE = `
.zouyu-mention-menu{position:fixed;z-index:99999;min-width:220px;max-width:340px;max-height:260px;
  overflow:auto;background:#2a2a2a;border:1px solid #4a4a4a;border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.5);padding:4px 0;font-size:12px}
.zouyu-mention-menu.hidden{display:none}
.zouyu-mention-item{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;color:#ddd}
.zouyu-mention-item:hover,.zouyu-mention-item.active{background:#3a3a3a;color:#fff}
.zouyu-mention-label{font-weight:600;color:#4fff8f}
.zouyu-mention-meta{color:#999;font-size:11px}
.zouyu-mention-empty{padding:10px 12px;color:#999;text-align:center}
`;

let mentionStylesInjected = false;

function injectMentionStyles() {
  if (mentionStylesInjected) return;
  mentionStylesInjected = true;
  const el = document.createElement("style");
  el.textContent = MENTION_STYLE;
  document.head.appendChild(el);
}

function setupMentionAutocomplete(node) {
  const promptWidget = node.widgets?.find((w) => w.name === "prompt");
  if (!promptWidget) return;

  let menu = null;
  let filtered = [];
  let activeIndex = 0;
  let mentionStart = -1;

  const ensureMenu = () => {
    if (menu) return menu;
    menu = document.createElement("div");
    menu.className = "zouyu-mention-menu hidden";
    document.body.appendChild(menu);
    menu.addEventListener("mousedown", (e) => e.preventDefault());
    return menu;
  };

  const closeMenu = () => {
    mentionStart = -1;
    filtered = [];
    activeIndex = 0;
    menu?.classList.add("hidden");
  };

  const positionMenu = (ta) => {
    const rect = ta.getBoundingClientRect();
    menu.classList.remove("hidden");
    const mh = menu.offsetHeight || 180;
    const mw = menu.offsetWidth || 220;
    let top = rect.bottom + 4;
    if (top + mh > window.innerHeight - 8) top = Math.max(8, rect.top - mh - 4);
    let left = Math.min(rect.left, window.innerWidth - mw - 8);
    left = Math.max(8, left);
    menu.style.top = `${Math.round(top)}px`;
    menu.style.left = `${Math.round(left)}px`;
  };

  const renderMenu = async (query, ta) => {
    const files = await listSeedFiles();
    const names = files
      .map((f) => String(f).replace(/\.pt$/i, ""))
      .filter(Boolean)
      .sort();
    const q = (query || "").toLowerCase();
    filtered = names.filter((n) => !q || n.toLowerCase().includes(q));

    const m = ensureMenu();
    m.innerHTML = "";
    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "zouyu-mention-empty";
      empty.textContent = names.length ? "无匹配的种子文件" : "暂无种子文件，请先保存";
      m.appendChild(empty);
    } else {
      filtered.forEach((name, i) => {
        const row = document.createElement("div");
        row.className = `zouyu-mention-item${i === activeIndex ? " active" : ""}`;
        row.innerHTML = `<span class="zouyu-mention-label">@${name}</span>`;
        row.onmousedown = (e) => {
          e.preventDefault();
          insertMention(name, ta);
        };
        m.appendChild(row);
      });
    }
    positionMenu(ta);
  };

  const insertMention = (name, ta) => {
    const value = ta.value || "";
    const caret = ta.selectionStart ?? value.length;
    const start = mentionStart >= 0 ? mentionStart : caret;
    const next = value.slice(0, start) + `@${name} ` + value.slice(caret);
    ta.value = next;
    // 同步到 widget + 触发输入事件
    promptWidget.value = next;
    promptWidget.callback?.(next);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    const pos = start + name.length + 2;
    ta.setSelectionRange(pos, pos);
    closeMenu();
    ta.focus();
  };

  const openIfMention = (ta) => {
    const value = ta.value || "";
    const caret = ta.selectionStart ?? value.length;
    const before = value.slice(0, caret);
    const match = before.match(/@([^\s@<]*)$/);
    if (!match) {
      closeMenu();
      return;
    }
    mentionStart = caret - match[0].length;
    activeIndex = 0;
    renderMenu(match[1], ta);
  };

  const moveActive = (delta) => {
    if (!filtered.length) return;
    const prev = activeIndex;
    activeIndex = (activeIndex + delta + filtered.length) % filtered.length;
    menu?.querySelectorAll(".zouyu-mention-item").forEach((row, i) => {
      row.classList.toggle("active", i === activeIndex);
    });
    const activeRow = menu?.querySelectorAll(".zouyu-mention-item")[activeIndex];
    activeRow?.scrollIntoView({ block: "nearest" });
  };

  const wire = (ta) => {
    if (!ta || ta.__zouyuMentionWired) return;
    ta.__zouyuMentionWired = true;
    ta.addEventListener("input", () => openIfMention(ta));
    ta.addEventListener("click", () => openIfMention(ta));
    ta.addEventListener("keyup", (e) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) openIfMention(ta);
    });
    ta.addEventListener("keydown", (e) => {
      if (menu && !menu.classList.contains("hidden") && filtered.length) {
        if (e.key === "ArrowDown") { e.preventDefault(); moveActive(1); return; }
        if (e.key === "ArrowUp") { e.preventDefault(); moveActive(-1); return; }
        if (e.key === "Enter" || e.key === "Tab") {
          e.preventDefault();
          insertMention(filtered[activeIndex], ta);
          return;
        }
        if (e.key === "Escape") { e.preventDefault(); closeMenu(); return; }
      }
    });
    ta.addEventListener("blur", () => {
      setTimeout(closeMenu, 150);
    });
  };

  injectMentionStyles();

  // 等待 textarea 元素被渲染后挂载
  const tryAttach = () => {
    let ta = promptWidget.element;
    if (ta && ta.tagName !== "TEXTAREA" && ta.tagName !== "INPUT") {
      ta = ta.querySelector?.("textarea, input") || ta;
    }
    if (!ta) {
      setTimeout(tryAttach, 120);
      return;
    }
    wire(ta);
  };
  setTimeout(tryAttach, 0);
}

// ---------------------------------------------------------------------------
// 动态 LoRA 槽（功能 10）
// ---------------------------------------------------------------------------
function setupLoraSlots(node) {
  const countWidget = node.widgets?.find((w) => w.name === "lora_count");

  const applyCount = () => {
    const count = Math.max(0, Math.min(MAX_LORA, Number(countWidget?.value ?? 1)));
    // 移除超出 count 的槽位输入
    for (let i = node.inputs.length - 1; i >= 0; i--) {
      const name = node.inputs[i].name;
      const m = name.match(/^lora_(name|strength)_(\d+)$/);
      if (m && Number(m[2]) > count) {
        node.removeInput(i);
      }
    }
    // 补齐缺失的槽位输入（从 1 到 count）
    for (let i = 1; i <= count; i++) {
      if (!node.inputs.some((inp) => inp.name === `lora_name_${i}`)) {
        node.addInput(`lora_name_${i}`, "STRING");
      }
      if (!node.inputs.some((inp) => inp.name === `lora_strength_${i}`)) {
        node.addInput(`lora_strength_${i}`, "FLOAT");
      }
    }
    // 重新排序：把 lora 槽位输入放到 node.inputs 末尾（+/- 按钮之后）
  };

  const addBtn = node.addWidget("button", "＋ 添加 LoRA 槽", null, () => {
    const cur = Number(countWidget?.value ?? 0);
    if (countWidget) countWidget.value = Math.min(MAX_LORA, cur + 1);
    applyCount();
    app.graph?.setDirtyCanvas(true, false);
  });
  addBtn.serialize = false;

  const delBtn = node.addWidget("button", "－ 移除 LoRA 槽", null, () => {
    const cur = Number(countWidget?.value ?? 0);
    if (countWidget) countWidget.value = Math.max(0, cur - 1);
    applyCount();
    app.graph?.setDirtyCanvas(true, false);
  });
  delBtn.serialize = false;

  applyCount();
}

// ---------------------------------------------------------------------------
// 动态参考槽（功能 10 / 功能 4）：参考图 / 视频 / 音频
// ---------------------------------------------------------------------------
function setupRefSlots(node) {
  const groups = [
    { prefix: "reference_image_", type: "IMAGE", max: MAX_REF_IMAGE, label: "参考图", def: 1 },
    { prefix: "ref_video_audio_", type: "AUDIO", max: MAX_REF_AUDIO, label: "视频配乐", def: 0 },
    { prefix: "ref_video_", type: "IMAGE", max: MAX_REF_VIDEO, label: "参考视频", def: 0 },
    { prefix: "ref_audio_", type: "AUDIO", max: MAX_REF_AUDIO, label: "参考音频", def: 0 },
  ];

  const isSlot = (name, prefix) => new RegExp("^" + prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\d+$").test(name);

  const visibleCount = (prefix) => {
    let c = 0;
    for (let i = 0; i < 12; i++) {
      if (node.inputs.some((inp) => inp.name === prefix + i)) c++;
      else break;
    }
    return c;
  };

  // 初始：隐藏超出默认数量的槽位
  for (const g of groups) {
    let count = 0;
    for (let i = g.max - 1; i >= 0; i--) {
      const name = g.prefix + i;
      const idx = node.inputs.findIndex((inp) => inp.name === name);
      if (idx < 0) continue;
      if (i >= g.def) {
        node.removeInput(idx);
      } else {
        count++;
      }
    }
  }

  const makeButtons = (g) => {
    const addBtn = node.addWidget("button", `＋ ${g.label}`, null, () => {
      const count = visibleCount(g.prefix);
      if (count < g.max) {
        node.addInput(g.prefix + count, g.type);
        app.graph?.setDirtyCanvas(true, false);
      }
    });
    addBtn.serialize = false;

    const delBtn = node.addWidget("button", `－ ${g.label}`, null, () => {
      const count = visibleCount(g.prefix);
      if (count > 0) {
        const name = g.prefix + (count - 1);
        const idx = node.inputs.findIndex((inp) => inp.name === name);
        if (idx >= 0) node.removeInput(idx);
        app.graph?.setDirtyCanvas(true, false);
      }
    });
    delBtn.serialize = false;
  };

  for (const g of groups) makeButtons(g);
}

// ---------------------------------------------------------------------------
// 扩展注册
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "ZouyuSeedTensor.ui",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.category !== CATEGORY) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      try {
        if (nodeData.name === "ZouyuSeedBlender") {
          setupMentionAutocomplete(node);
        } else if (nodeData.name === "ZouyuLoraStack") {
          setupLoraSlots(node);
        } else if (nodeData.name === "ZouyuSaveSeedConditioning") {
          setupRefSlots(node);
        } else if (nodeData.name === "ZouyuLoadSeedConditioning" || nodeData.name === "ZouyuExtractSeedMedia") {
          addRefreshButton(node);
        }
      } catch (e) {
        console.error("[ZouyuSeedTensor] UI setup error:", e);
      }

      return r;
    };
  },
});

// 后端保存/刷新后自动更新所有文件下拉
api.addEventListener("Zouyu-seed-files-refresh", () => {
  refreshAllFileCombos();
});
