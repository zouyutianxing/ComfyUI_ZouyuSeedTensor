/**
 * ZouyuSeedTensor 前端扩展
 *
 * 功能：
 * 1. 参考槽位自动增减（参考图上限 50），同类端口连续排列，断开自动收缩 + 重排补齐缺口
 * 2. 中英文切换：仅改 UI（getOptionLabel 值/显示分离），语言开关只作用于单节点（节点级语言状态）
 * 3. 节点颜色区分
 * 4. @ 引用自动补全下拉（种子文件 + 已连接的参考端口）
 * 5. 文件下拉刷新 + 清空临时存储按钮
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CATEGORY = "ZouyuAI/SeedTensor";

// 槽位组（按节点类型区分：Save 与 Loader 端口类型不同）
// 顺序须与后端 INPUT_TYPES 的可选端口声明顺序一致
const SLOT_GROUPS_BY_TYPE = {
  ZouyuSaveSeedConditioning: [
    { prefix: "reference_image_", type: "IMAGE", max: 50 },
    { prefix: "ref_video_", type: "IMAGE", max: 3 },
    { prefix: "ref_video_audio_", type: "AUDIO", max: 3 },
    { prefix: "ref_audio_", type: "AUDIO", max: 3 },
  ],
  ZouyuSeedLoader: [
    { prefix: "reference_image_", type: "IMAGE", max: 50 },
    { prefix: "ref_video_", type: "VIDEO", max: 3 },
    { prefix: "ref_audio_", type: "AUDIO", max: 3 },
  ],
};

// 可被提示词 @ 提及的参考端口前缀（不含配乐子端口）
const MENTION_PREFIXES = ["reference_image_", "ref_video_", "ref_audio_"];

// ---------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------
const LANG_KEY = "zouyu_seed_tensor_lang";

function getLang() {
  return localStorage.getItem(LANG_KEY) || "中文";
}

function setLang(lang) {
  localStorage.setItem(LANG_KEY, lang);
}

function isZh(lang) {
  return lang !== "English";
}

// widget / 端口 label（key 为内部名，值始终英文键）
const LABELS = {
  conditioning: { zh: "条件", en: "Conditioning" },
  guider: { zh: "引导器", en: "Guider" },
  latent: { zh: "Latent 图像", en: "Latent" },
  logs: { zh: "日志", en: "Logs" },
  clip: { zh: "文本编码器", en: "CLIP" },
  vae: { zh: "视频 VAE", en: "VAE" },
  audio_vae: { zh: "音频 VAE", en: "Audio VAE" },
  model: { zh: "模型", en: "Model" },
  seed: { zh: "种子", en: "Seed" },
  filename: { zh: "文件名", en: "Filename" },
  storage: { zh: "存储位置", en: "Storage" },
  language: { zh: "界面语言", en: "Language" },
  canvas_mode: { zh: "画布模式", en: "Canvas Mode" },
  width: { zh: "宽度", en: "Width" },
  height: { zh: "高度", en: "Height" },
  duration: { zh: "时长(秒)", en: "Duration (s)" },
  fps: { zh: "帧率", en: "FPS" },
  ref_image_size: { zh: "参考图缩放", en: "Ref Image Size" },
  crop_mode: { zh: "裁剪方式", en: "Crop Mode" },
  prompt_text: { zh: "提示词", en: "Prompt" },
  ref_image_format: { zh: "参考图格式", en: "Ref Format" },
  file_name: { zh: "种子文件", en: "Seed File" },
  prompt: { zh: "提示词", en: "Prompt" },
  weights: { zh: "权重", en: "Weights" },
  rebuild: { zh: "重建目录", en: "Rebuild" },
  trigger: { zh: "触发", en: "Trigger" },
  backup: { zh: "备份位置", en: "Backup" },
  samples: { zh: "Latent", en: "Latent" },
  vae_image: { zh: "视频 VAE", en: "Video VAE" },
  vae_audio: { zh: "音频 VAE", en: "Audio VAE" },
  images: { zh: "图像", en: "Images" },
  audio: { zh: "音频", en: "Audio" },
  // 输出端口
  saved_path: { zh: "保存路径", en: "Saved Path" },
  metadata: { zh: "元数据", en: "Metadata" },
  source_names: { zh: "来源名称", en: "Source Names" },
  cleaned_prompt: { zh: "清理后提示词", en: "Cleaned Prompt" },
  ref_images: { zh: "参考图", en: "Ref Images" },
  ref_videos: { zh: "参考视频", en: "Ref Videos" },
  ref_audio: { zh: "参考音频", en: "Ref Audio" },
  catalog_json: { zh: "目录 JSON", en: "Catalog JSON" },
  count: { zh: "数量", en: "Count" },
  cleared_info: { zh: "清理信息", en: "Cleared Info" },
};

// combo：keys 为后端英文键（稳定不变），zh/en 为前端显示文字
const COMBOS = {
  storage: { keys: ["permanent", "temp"], zh: ["永久存储", "临时存储"], en: ["Permanent", "Temporary"] },
  backup: { keys: ["permanent", "temp", "none"], zh: ["永久备份", "临时备份", "不备份"], en: ["Permanent", "Temporary", "None"] },
  canvas_mode: { keys: ["auto", "max", "custom"], zh: ["自动", "最大", "自定义"], en: ["Auto", "Max", "Custom"] },
  ref_image_size: { keys: ["match", "max"], zh: ["匹配画布", "短边2048"], en: ["Match", "Max(2048)"] },
  crop_mode: { keys: ["disabled", "center", "contain"], zh: ["不裁剪", "居中裁剪", "等比填充"], en: ["Disabled", "Center", "Contain"] },
};

const BOOL_LABELS = {
  rebuild: { zh: ["重建目录", "读取目录"], en: ["Rebuild", "Read"] },
};

const SLOT_LABELS = {
  reference_image_: { zh: "参考图", en: "Ref Image" },
  ref_video_audio_: { zh: "视频配乐", en: "Video Audio" },
  ref_video_: { zh: "参考视频", en: "Ref Video" },
  ref_audio_: { zh: "参考音频", en: "Ref Audio" },
};

const TITLES = {
  ZouyuSaveSeedConditioning: { zh: "保存种子张量", en: "Save Seed Tensor" },
  ZouyuSeedLoader: { zh: "融合加载器", en: "Seed Loader" },
  ZouyuVAEDecodeAV: { zh: "音视频联合解码", en: "AV VAE Decode" },
  ZouyuExtractSeedMedia: { zh: "提取参考媒体", en: "Extract Seed Media" },
  ZouyuSeedCatalog: { zh: "种子目录", en: "Seed Catalog" },
  ZouyuSeedPreview: { zh: "种子预览", en: "Seed Preview" },
  ZouyuClearTemp: { zh: "清空临时存储", en: "Clear Temp Storage" },
};

const NODE_COLORS = {
  ZouyuSaveSeedConditioning: { color: "#2e7d4f", bgcolor: "#16321f" },
  ZouyuSeedLoader: { color: "#7a4fa0", bgcolor: "#2c1a3a" },
  ZouyuVAEDecodeAV: { color: "#2f6b8f", bgcolor: "#162a38" },
  ZouyuExtractSeedMedia: { color: "#1f8a8a", bgcolor: "#123232" },
  ZouyuSeedCatalog: { color: "#6b6b6b", bgcolor: "#262626" },
  ZouyuSeedPreview: { color: "#b0722a", bgcolor: "#3a2812" },
  ZouyuClearTemp: { color: "#a03838", bgcolor: "#3a1616" },
};

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

function numOf(name) {
  const m = (name || "").match(/(\d+)$/);
  return m ? parseInt(m[1], 10) : 0;
}

function buildComboOptions(widgetName, node) {
  const cb = COMBOS[widgetName];
  return {
    values: cb.keys.slice(),
    getOptionLabel: (value) => {
      const idx = cb.keys.indexOf(value);
      if (idx < 0) return value;
      const lang = node?.__zouyuLang || getLang();
      return isZh(lang) ? cb.zh[idx] : cb.en[idx];
    },
  };
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
// i18n 应用（仅作用于单个节点）
// ---------------------------------------------------------------------------
function applyLanguage(node, lang) {
  const zh = isZh(lang);
  node.__zouyuLang = lang;

  // 标题
  const t = TITLES[node.comfyClass || node.type];
  if (t) node.title = zh ? t.zh : t.en;

  // widget：按钮 / 语言选择器 / boolean / combo / label
  for (const w of node.widgets || []) {
    if (w.__zouyuButtonKey) {
      const key = w.__zouyuButtonKey;
      w.label = zh ? key.zh : key.en;
      w.name = w.label;
      continue;
    }
    if (w.name === "language") {
      w.value = lang;
      const entry = LABELS[w.name];
      if (entry) w.label = zh ? entry.zh : entry.en;
      continue;
    }
    const bl = BOOL_LABELS[w.name];
    if (bl && w.options) {
      w.options.label_on = zh ? bl.zh[0] : bl.en[0];
      w.options.label_off = zh ? bl.zh[1] : bl.en[1];
    }
    const cb = COMBOS[w.name];
    if (cb && w.type === "combo") {
      w.options = buildComboOptions(w.name, node);
    }
    const entry = LABELS[w.name];
    if (entry) w.label = zh ? entry.zh : entry.en;
  }

  // 输入端口 label
  for (const inp of node.inputs || []) {
    const sl = slotLabel(inp.name, lang);
    if (sl) {
      inp.label = sl;
      continue;
    }
    const entry = LABELS[inp.name];
    if (entry) inp.label = zh ? entry.zh : entry.en;
  }

  // 输出端口 label（融合加载器的 conditioning 输出显示为「张量输出」）
  for (const out of node.outputs || []) {
    let label = null;
    if (node.comfyClass === "ZouyuSeedLoader" && out.name === "conditioning") {
      label = zh ? "张量输出" : "Tensor Output";
    } else {
      const entry = LABELS[out.name];
      label = entry ? (zh ? entry.zh : entry.en) : null;
    }
    if (label) out.label = label;
  }

  app.graph?.setDirtyCanvas(true, false);
}

// ---------------------------------------------------------------------------
// 参考槽位自动增减 + 重排（同类连续、缺口自动补齐）
// ---------------------------------------------------------------------------
function getLink(graph, linkId) {
  if (!graph || linkId == null) return null;
  const links = graph.links;
  if (!links) return null;
  if (typeof links.get === "function") return links.get(linkId);
  return links[linkId];
}

// 修正单个组的端口数量与连续性（缺口自动补齐，链接保留），返回是否有改动
function refreshGroup(node, g) {
  const re = new RegExp("^" + g.prefix + "(\\d+)$");
  const slots = [];
  for (let i = 0; i < node.inputs.length; i++) {
    const m = (node.inputs[i].name || "").match(re);
    if (m) slots.push({ idx: i, num: parseInt(m[1], 10), inp: node.inputs[i] });
  }
  slots.sort((a, b) => a.num - b.num);

  const linked = slots.filter((s) => s.inp.link != null);
  const target = Math.min(g.max, Math.max(1, linked.length + 1));

  // 已连续且数量正确 → 无需改动
  const contiguous =
    slots.length === target &&
    slots.every((s, i) => s.num === i) &&
    linked.every((s, i) => s.num === i);
  if (contiguous) return false;

  // 捕获链接来源（按数值顺序）
  const origins = [];
  for (const s of linked) {
    const l = getLink(app.graph, s.inp.link);
    if (l) origins.push({ origin_id: l.origin_id, origin_slot: l.origin_slot });
  }

  // 移除本组所有端口（removeInput 会自动修正 target_slot）
  const desc = [...slots].sort((a, b) => b.idx - a.idx);
  for (const s of desc) node.removeInput(s.idx);

  // 重新添加 target 个端口（0 起点）
  const added = [];
  for (let j = 0; j < target; j++) {
    const inp = node.addInput(g.prefix + j, g.type);
    const sl = slotLabel(g.prefix + j, node.__zouyuLang || getLang());
    if (sl) inp.label = sl;
    added.push(inp);
  }

  // 重新连接（缺口已补齐，链接连续）
  for (let j = 0; j < origins.length; j++) {
    const o = origins[j];
    const originNode =
      app.graph?.getNodeById?.(o.origin_id) ||
      app.graph?._nodes?.find((n) => n.id === o.origin_id);
    if (!originNode) continue;
    const targetSlot = node.inputs.indexOf(added[j]);
    try {
      originNode.connect(o.origin_slot, node, targetSlot);
    } catch (e) {
      console.warn("[ZouyuSeedTensor] reconnect failed:", e);
    }
  }

  return true;
}

// 重排各参考组块的顺序（同类连续、按组顺序），修正链接 target_slot，返回是否有改动
function reorderGroupInputs(node) {
  const groups = SLOT_GROUPS_BY_TYPE[node.comfyClass] || [];
  if (!groups.length) return false;

  const prefixOrder = {};
  groups.forEach((g, i) => (prefixOrder[g.prefix] = i));

  function groupPrefixOf(name) {
    if (!name) return null;
    for (const g of groups) {
      if (name.startsWith(g.prefix) && /^\d+$/.test(name.slice(g.prefix.length))) return g.prefix;
    }
    return null;
  }

  // 非参考组输入（required）始终在前，参考组输入在后
  let tailStart = node.inputs.length;
  for (let i = 0; i < node.inputs.length; i++) {
    if (groupPrefixOf(node.inputs[i].name)) {
      tailStart = i;
      break;
    }
  }

  const head = node.inputs.slice(0, tailStart);
  const tailObjs = node.inputs.slice(tailStart);

  // 期望的尾部队列：按组顺序 + 组内数值顺序
  const desiredTail = [];
  for (const g of groups) {
    const re = new RegExp("^" + g.prefix + "(\\d+)$");
    const grp = tailObjs.filter((inp) => re.test(inp.name || ""));
    grp.sort((a, b) => numOf(a.name) - numOf(b.name));
    desiredTail.push(...grp);
  }

  // 判断顺序是否已正确
  let changed = false;
  if (desiredTail.length !== tailObjs.length) {
    changed = true;
  } else {
    for (let i = 0; i < desiredTail.length; i++) {
      if (desiredTail[i] !== tailObjs[i]) {
        changed = true;
        break;
      }
    }
  }
  if (!changed) return false;

  // 对象身份映射（旧 index -> 新 index）
  const oldInputs = node.inputs.slice();
  const newIndexByObj = new Map();
  head.forEach((inp, i) => newIndexByObj.set(inp, i));
  desiredTail.forEach((inp, i) => newIndexByObj.set(inp, tailStart + i));

  // 用 splice 原地重排尾部
  node.inputs.splice(tailStart, tailObjs.length, ...desiredTail);

  // 修正指向本节点的链接 target_slot
  const links = app.graph?.links;
  if (links) {
    let entries;
    if (typeof links.get === "function") {
      entries = [];
      for (const [k, v] of links.entries()) entries.push([k, v]);
    } else {
      entries = Object.entries(links);
    }
    for (const [, l] of entries) {
      if (l && l.target_id === node.id) {
        const oldInp = oldInputs[l.target_slot];
        if (oldInp && newIndexByObj.has(oldInp)) {
          l.target_slot = newIndexByObj.get(oldInp);
        }
      }
    }
  }

  return true;
}

function refreshAllGroups(node) {
  if (node.__zouyuRefreshing) return;
  node.__zouyuRefreshing = true;
  try {
    const groups = SLOT_GROUPS_BY_TYPE[node.comfyClass] || [];
    let changed = false;
    for (const g of groups) {
      if (refreshGroup(node, g)) changed = true;
    }
    if (reorderGroupInputs(node)) changed = true;
    if (changed) {
      node.size = node.computeSize();
      app.graph?.setDirtyCanvas(true, false);
    }
  } finally {
    node.__zouyuRefreshing = false;
  }
}

// ---------------------------------------------------------------------------
// @ 提及自动补全（种子文件 + 已连接的参考端口）
// ---------------------------------------------------------------------------
const MENTION_STYLE = `
.zouyu-mention-menu{position:fixed;z-index:99999;min-width:220px;max-width:340px;max-height:260px;
  overflow:auto;background:#2a2a2a;border:1px solid #4a4a4a;border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.5);padding:4px 0;font-size:12px}
.zouyu-mention-menu.hidden{display:none}
.zouyu-mention-item{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;color:#ddd}
.zouyu-mention-item:hover,.zouyu-mention-item.active{background:#3a3a3a;color:#fff}
.zouyu-mention-label{font-weight:600;color:#4fff8f}
.zouyu-mention-port{font-weight:600;color:#8fd0ff}
.zouyu-mention-tag{margin-left:auto;font-size:10px;color:#888}
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

// 收集已连接的参考端口显示名（如 "参考图1"）
function getConnectedRefPorts(node) {
  const entries = [];
  for (const g of SLOT_GROUPS_BY_TYPE[node.comfyClass] || []) {
    if (!MENTION_PREFIXES.includes(g.prefix)) continue;
    const base = SLOT_LABELS[g.prefix]?.zh;
    if (!base) continue;
    const re = new RegExp("^" + g.prefix + "(\\d+)$");
    for (const inp of node.inputs || []) {
      const m = (inp.name || "").match(re);
      if (m && inp.link != null) entries.push(`${base}${parseInt(m[1], 10) + 1}`);
    }
  }
  return entries;
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
    const fileNames = files.map((f) => String(f).replace(/\.pt$/i, "")).filter(Boolean).sort();
    const ports = getConnectedRefPorts(node);

    // 条目：{ name, isPort }
    const items = [];
    for (const n of fileNames) items.push({ name: n, isPort: false });
    for (const p of ports) items.push({ name: p, isPort: true });

    const q = (query || "").toLowerCase();
    filtered = items.filter((it) => !q || it.name.toLowerCase().includes(q));

    const m = ensureMenu();
    m.innerHTML = "";
    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "zouyu-mention-empty";
      empty.textContent = items.length ? "无匹配的种子文件或参考端口" : "暂无种子文件，请先保存";
      m.appendChild(empty);
    } else {
      filtered.forEach((it, i) => {
        const row = document.createElement("div");
        row.className = `zouyu-mention-item${i === activeIndex ? " active" : ""}`;
        const labelCls = it.isPort ? "zouyu-mention-port" : "zouyu-mention-label";
        const tag = it.isPort ? '<span class="zouyu-mention-tag">参考端口</span>' : "";
        row.innerHTML = `<span class="${labelCls}">@${it.name}</span>${tag}`;
        row.onmousedown = (e) => {
          e.preventDefault();
          insertMention(it.name, ta);
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
          insertMention(filtered[activeIndex].name, ta);
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
  const label = isZh(getLang()) ? keyZh : keyEn;
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
    const hasSlots = comfyClass === "ZouyuSaveSeedConditioning" || comfyClass === "ZouyuSeedLoader";

    // 拦截连接变化：自动增减/重排参考槽位
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = origOnConnectionsChange?.apply(this, arguments);
      try {
        if (this.__zouyuRefreshSlots) this.__zouyuRefreshSlots();
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
        const color = NODE_COLORS[comfyClass];
        if (color) {
          node.color = color.color;
          node.bgcolor = color.bgcolor;
        }

        // 应用当前语言（仅本节点）
        applyLanguage(node, getLang());

        if (hasSlots) {
          node.__zouyuRefreshSlots = () => refreshAllGroups(node);
          // 初始收敛到正确状态（默认每个参考组仅保留 0 号端口）
          refreshAllGroups(node);
          node.size = node.computeSize();
        }

        if (comfyClass === "ZouyuSeedLoader") {
          setupMentionAutocomplete(node);
        }

        // 语言切换（仅本节点）
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

        if (comfyClass === "ZouyuExtractSeedMedia" || comfyClass === "ZouyuSeedPreview") {
          const fw = node.widgets?.find((w) => w.name === "file_name");
          if (fw) {
            addButton(node, "🔄 刷新", "🔄 Refresh", async () => {
              await refreshFileComboWidget(fw);
              app.graph?.setDirtyCanvas(true, false);
            });
          }
        }

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
