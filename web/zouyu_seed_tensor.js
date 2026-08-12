/**
 * ZouyuSeedTensor 前端扩展
 *
 * 功能：
 * 1. 参考槽位自动增减：端口被链接时自动显示下一个端口（无 +/- 按钮）
 * 2. UI 全中文化 + 中英文切换联动（language 开关切换时更新整个节点界面）
 * 3. 节点颜色区分（UI 优化）
 * 4. @ 引用自动补全下拉
 * 5. 文件下拉刷新 + 清空临时存储按钮
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CATEGORY = "ZouyuAI/SeedTensor";

// 槽位组（Python 端静态定义所有接口，JS 端动态增减）
const SLOT_GROUPS = [
  { prefix: "reference_image_", type: "IMAGE", max: 9 },
  { prefix: "ref_video_audio_", type: "AUDIO", max: 3 },
  { prefix: "ref_video_", type: "IMAGE", max: 3 },
  { prefix: "ref_audio_", type: "AUDIO", max: 3 },
];

// ---------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------
const LANG_KEY = "zouyu_seed_tensor_lang";

const LABELS = {
  conditioning: { zh: "条件张量", en: "Conditioning" },
  seed: { zh: "种子", en: "Seed" },
  filename: { zh: "文件名", en: "Filename" },
  storage: { zh: "存储位置", en: "Storage" },
  language: { zh: "界面语言", en: "Language" },
  canvas_mode: { zh: "画布模式", en: "Canvas Mode" },
  width: { zh: "宽度", en: "Width" },
  height: { zh: "高度", en: "Height" },
  ref_image_size: { zh: "参考图缩放", en: "Ref Image Size" },
  crop_mode: { zh: "裁剪方式", en: "Crop Mode" },
  prompt_text: { zh: "提示词", en: "Prompt" },
  duration: { zh: "时长(秒)", en: "Duration (s)" },
  ref_image_format: { zh: "参考图格式", en: "Ref Format" },
  file_name: { zh: "种子文件", en: "Seed File" },
  prompt: { zh: "提示词", en: "Prompt" },
  weights: { zh: "权重", en: "Weights" },
  rebuild: { zh: "重建目录", en: "Rebuild" },
  trigger: { zh: "触发", en: "Trigger" },
};

const SLOT_LABELS = {
  reference_image_: { zh: "参考图", en: "Ref Image" },
  ref_video_audio_: { zh: "视频配乐", en: "Video Audio" },
  ref_video_: { zh: "参考视频", en: "Ref Video" },
  ref_audio_: { zh: "参考音频", en: "Ref Audio" },
};

const TITLES = {
  ZouyuSaveSeedConditioning: { zh: "保存种子张量", en: "Save Seed Tensor" },
  ZouyuLoadSeedConditioning: { zh: "加载种子张量", en: "Load Seed Tensor" },
  ZouyuSeedBlender: { zh: "多种子混合器", en: "Seed Blender" },
  ZouyuExtractSeedMedia: { zh: "提取参考媒体", en: "Extract Seed Media" },
  ZouyuSeedCatalog: { zh: "种子目录", en: "Seed Catalog" },
  ZouyuSeedPreview: { zh: "种子预览", en: "Seed Preview" },
  ZouyuClearTemp: { zh: "清空临时存储", en: "Clear Temp Storage" },
};

const NODE_COLORS = {
  ZouyuSaveSeedConditioning: { color: "#2e7d4f", bgcolor: "#16321f" },
  ZouyuLoadSeedConditioning: { color: "#2f6b8f", bgcolor: "#162a38" },
  ZouyuSeedBlender: { color: "#7a4fa0", bgcolor: "#2c1a3a" },
  ZouyuExtractSeedMedia: { color: "#1f8a8a", bgcolor: "#123232" },
  ZouyuSeedCatalog: { color: "#6b6b6b", bgcolor: "#262626" },
  ZouyuSeedPreview: { color: "#b0722a", bgcolor: "#3a2812" },
  ZouyuClearTemp: { color: "#a03838", bgcolor: "#3a1616" },
};

function getLang() {
  return localStorage.getItem(LANG_KEY) || "中文";
}

function setLang(lang) {
  localStorage.setItem(LANG_KEY, lang);
}

function isZh(lang) {
  return lang !== "English";
}

function slotLabel(name, lang) {
  for (const prefix in SLOT_LABELS) {
    const m = name.match(new RegExp("^" + prefix + "(\\d+)$"));
    if (m) {
      const num = Number(m[1]) + 1;
      const base = isZh(lang) ? SLOT_LABELS[prefix].zh : SLOT_LABELS[prefix].en;
      return `${base} ${num}`;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// 网络工具
// ---------------------------------------------------------------------------
async function fetchJson(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

async function listSeedFiles() {
  try {
    const d = await fetchJson("/zouyu_seed_tensor/files");
    return d.all || [];
  } catch {
    return [];
  }
}

async function refreshFileComboWidget(widget) {
  const files = await listSeedFiles();
  const full = (files || []).filter((f) => String(f).endsWith(".pt"));
  const values = full.length ? full : ["(暂无文件)"];
  if (widget.options) widget.options.values = values;
  if (!values.includes(widget.value)) widget.value = values[0];
}

async function refreshAllFileCombos() {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    const w = n.widgets?.find((x) => x.name === "file_name" && x.type === "combo");
    if (w) await refreshFileComboWidget(w);
  }
  app.graph?.setDirtyCanvas(true, false);
}

// ---------------------------------------------------------------------------
// i18n 应用
// ---------------------------------------------------------------------------
function applyLanguage(node, lang) {
  const zh = isZh(lang);

  // 标题
  const t = TITLES[node.comfyClass || node.type];
  if (t) node.title = zh ? t.zh : t.en;

  // widget label
  for (const w of node.widgets || []) {
    if (w.__zouyuButtonKey) {
      const key = w.__zouyuButtonKey;
      w.label = zh ? key.zh : key.en;
      w.name = w.label;
      continue;
    }
    const entry = LABELS[w.name];
    if (entry) w.label = zh ? entry.zh : entry.en;
  }

  // 输入槽 label（参考图/视频/音频）
  for (const inp of node.inputs || []) {
    const sl = slotLabel(inp.name, lang);
    if (sl) {
      inp.label = sl;
      continue;
    }
    const entry = LABELS[inp.name];
    if (entry) inp.label = zh ? entry.zh : entry.en;
  }

  app.graph?.setDirtyCanvas(true, false);
}

function applyLanguageToAll(lang) {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if (n.comfyClass?.startsWith?.("Zouyu") || CATEGORY === n.constructor?.category) {
      applyLanguage(n, lang);
    }
  }
}

// ---------------------------------------------------------------------------
// 参考槽位自动增减（功能 1）
// ---------------------------------------------------------------------------
function groupOf(name) {
  for (const g of SLOT_GROUPS) {
    if (new RegExp("^" + g.prefix + "\\d+$").test(name)) return g;
  }
  return null;
}

function refreshGroup(node, g) {
  let count = 0;
  while (count < g.max && node.inputs.some((inp) => inp.name === g.prefix + count)) count++;

  // 最后一个有链接的槽
  let lastLinked = -1;
  for (let i = 0; i < count; i++) {
    const inp = node.inputs.find((x) => x.name === g.prefix + i);
    if (inp && inp.link != null) lastLinked = i;
  }

  // 目标槽数：至少 1；最后一个已链接则 +1 扩展
  let target = Math.max(1, lastLinked + 1);
  if (lastLinked === count - 1 && count < g.max) target = count + 1;

  // 收缩
  while (count > target) {
    const idx = node.inputs.findIndex((x) => x.name === g.prefix + (count - 1));
    if (idx >= 0) node.removeInput(idx);
    count--;
  }
  // 扩展
  while (count < target) {
    node.addInput(g.prefix + count, g.type);
    const inp = node.inputs.find((x) => x.name === g.prefix + count);
    if (inp) {
      const sl = slotLabel(inp.name, getLang());
      if (sl) inp.label = sl;
    }
    count++;
  }
}

function refreshAllGroups(node) {
  for (const g of SLOT_GROUPS) refreshGroup(node, g);
  // 强制重新计算节点尺寸并重绘画布
  node.size = node.computeSize();
  app.graph?.setDirtyCanvas(true, false);
}

function initSlots(node) {
  for (const g of SLOT_GROUPS) {
    let lastLinked = -1;
    for (let i = 0; i < g.max; i++) {
      const inp = node.inputs.find((x) => x.name === g.prefix + i);
      if (inp && inp.link != null) lastLinked = i;
    }
    const keep = Math.max(1, lastLinked + 1);
    for (let i = g.max - 1; i >= keep; i--) {
      const idx = node.inputs.findIndex((x) => x.name === g.prefix + i);
      if (idx >= 0) node.removeInput(idx);
    }
  }
}

// ---------------------------------------------------------------------------
// @ 提及自动补全
// ---------------------------------------------------------------------------
const MENTION_STYLE = `
.zouyu-mention-menu{position:fixed;z-index:99999;min-width:220px;max-width:340px;max-height:260px;
  overflow:auto;background:#2a2a2a;border:1px solid #4a4a4a;border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.5);padding:4px 0;font-size:12px}
.zouyu-mention-menu.hidden{display:none}
.zouyu-mention-item{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;color:#ddd}
.zouyu-mention-item:hover,.zouyu-mention-item.active{background:#3a3a3a;color:#fff}
.zouyu-mention-label{font-weight:600;color:#4fff8f}
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
    const names = files.map((f) => String(f).replace(/\.pt$/i, "")).filter(Boolean).sort();
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
    activeIndex = (activeIndex + delta + filtered.length) % filtered.length;
    menu?.querySelectorAll(".zouyu-mention-item").forEach((row, i) => {
      row.classList.toggle("active", i === activeIndex);
    });
    menu?.querySelectorAll(".zouyu-mention-item")[activeIndex]?.scrollIntoView({ block: "nearest" });
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
    ta.addEventListener("blur", () => setTimeout(closeMenu, 150));
  };

  injectMentionStyles();
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
// 按钮
// ---------------------------------------------------------------------------
function addButton(node, keyZh, keyEn, onClick) {
  const lang = getLang();
  const label = isZh(lang) ? keyZh : keyEn;
  const btn = node.addWidget("button", label, null, onClick);
  btn.serialize = false;
  btn.__zouyuButtonKey = { zh: keyZh, en: keyEn };
  return btn;
}

// ---------------------------------------------------------------------------
// 扩展注册
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "ZouyuSeedTensor.ui",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.category !== CATEGORY) return;

    const comfyClass = nodeData.name;

    // 拦截连接变化：自动增减参考槽位
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = origOnConnectionsChange?.apply(this, arguments);
      try {
        if (this.__zouyuRefreshSlots && !this.__zouyuRefreshing) {
          this.__zouyuRefreshing = true;
          try {
            this.__zouyuRefreshSlots();
          } finally {
            this.__zouyuRefreshing = false;
          }
        }
      } catch (e) {
        console.error("[ZouyuSeedTensor] slot refresh error:", e);
      }
      return r;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      node.comfyClass = comfyClass;

      try {
        // 节点颜色（UI 优化）
        const color = NODE_COLORS[comfyClass];
        if (color) {
          node.color = color.color;
          node.bgcolor = color.bgcolor;
        }

        // 应用当前语言
        applyLanguage(node, getLang());

        // 保存节点：参考槽自动增减
        if (comfyClass === "ZouyuSaveSeedConditioning") {
          initSlots(node);
          node.__zouyuRefreshSlots = () => refreshAllGroups(node);
          node.size = node.computeSize();
        }

        // 混合器：@ 自动补全
        if (comfyClass === "ZouyuSeedBlender") {
          setupMentionAutocomplete(node);
        }

        // 语言切换联动
        const langWidget = node.widgets?.find((w) => w.name === "language");
        if (langWidget) {
          const origCallback = langWidget.callback;
          langWidget.callback = function (value) {
            const r = origCallback?.call(this, value);
            setLang(value);
            applyLanguage(node, value);
            return r;
          };
        }

        // 文件下拉 + 刷新按钮
        if (comfyClass === "ZouyuLoadSeedConditioning" || comfyClass === "ZouyuExtractSeedMedia" || comfyClass === "ZouyuSeedPreview") {
          const fw = node.widgets?.find((w) => w.name === "file_name");
          if (fw) {
            addButton(node, "🔄 刷新", "🔄 Refresh", async () => {
              await refreshFileComboWidget(fw);
              app.graph?.setDirtyCanvas(true, false);
            });
          }
        }

        // 目录节点：清空临时 + 刷新
        if (comfyClass === "ZouyuSeedCatalog") {
          addButton(node, "🗑 清空临时存储", "🗑 Clear Temp", async () => {
            try {
              const d = await fetchJson("/zouyu_seed_tensor/clear_temp", { method: "POST" });
              alert(`已清空临时存储（移除 ${d.removed} 项）`);
            } catch (e) {
              alert("清空失败: " + e);
            }
          });
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
