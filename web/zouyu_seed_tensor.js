/**
 * ZouyuSeedTensor 前端扩展（V3 API）
 *
 * 功能：
 * 1. 参考端口由官方 io.Autogrow 原生动态扩展（前端无需自定义增减逻辑）
 * 2. 中英文切换：仅改 UI（getOptionLabel 值/显示分离），语言开关只作用于单节点
 * 3. 节点颜色区分
 * 4. @ 引用自动补全下拉（种子文件 + 已连接的参考端口）
 * 5. 文件下拉刷新 + 清空临时存储按钮
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CATEGORY = "ZouyuAI/SeedTensor";

// 可被提示词 @ 提及的参考端口（V3 autogrow 组名 -> 中文前缀）
const MENTION_GROUPS = [
  { re: /^ref_images\.ref_image_(\d+)$/, zh: "参考图" },
  { re: /^ref_videos\.ref_video_(\d+)$/, zh: "参考视频" },
  { re: /^ref_audios\.ref_audio_(\d+)$/, zh: "参考音频" },
];

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
  ref_scale: { zh: "参考值放大", en: "Ref Scale" },
  crop_mode: { zh: "裁剪方式", en: "Crop Mode" },
  prompt_text: { zh: "提示词", en: "Prompt" },
  ref_image_format: { zh: "参考图格式", en: "Ref Format" },
  file_name: { zh: "种子文件", en: "Seed File" },
  prompt: { zh: "提示词", en: "Prompt" },
  rebuild: { zh: "重建目录", en: "Rebuild" },
  trigger: { zh: "触发", en: "Trigger" },
  backup: { zh: "备份位置", en: "Backup" },
  samples: { zh: "Latent", en: "Latent" },
  vae_image: { zh: "视频 VAE", en: "Video VAE" },
  vae_audio: { zh: "音频 VAE", en: "Audio VAE" },
  images: { zh: "图像", en: "Images" },
  audio: { zh: "音频", en: "Audio" },
  saved_path: { zh: "保存路径", en: "Saved Path" },
  metadata: { zh: "元数据", en: "Metadata" },
  catalog_json: { zh: "目录 JSON", en: "Catalog JSON" },
  count: { zh: "数量", en: "Count" },
  cleared_info: { zh: "清理信息", en: "Cleared Info" },
  ref_images: { zh: "参考图", en: "Ref Images" },
  ref_videos: { zh: "参考视频", en: "Ref Videos" },
  ref_audio: { zh: "参考音频", en: "Ref Audio" },
  // ---- 模型加载器 / 模型占用检测 ----
  unet_folder: { zh: "UNET 文件夹", en: "UNET Folder" },
  unet_name: { zh: "UNET 模型", en: "UNET Model" },
  weight_dtype: { zh: "权重精度", en: "Weight Dtype" },
  clip_folder: { zh: "CLIP 文件夹", en: "CLIP Folder" },
  clip_name: { zh: "文本编码器", en: "CLIP Model" },
  clip_type: { zh: "编码器类型", en: "CLIP Type" },
  clip_device: { zh: "编码器设备", en: "CLIP Device" },
  vae_folder: { zh: "视频VAE 文件夹", en: "Video VAE Folder" },
  vae_name: { zh: "视频VAE", en: "Video VAE" },
  audio_vae_folder: { zh: "音频VAE 文件夹", en: "Audio VAE Folder" },
  audio_vae_name: { zh: "音频VAE", en: "Audio VAE" },
  low_vram_mode: { zh: "低显存模式", en: "Low VRAM Mode" },
  trigger: { zh: "触发(执行时机)", en: "Trigger (timing)" },
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
  low_vram_mode: { zh: ["彻底卸载", "CPU缓存"], en: ["Full unload", "CPU cache"] },
};

const TITLES = {
  ZouyuSaveSeedConditioning: { zh: "保存种子张量", en: "Save Seed Tensor" },
  ZouyuSeedLoader: { zh: "融合加载器", en: "Seed Loader" },
  ZouyuVAEDecodeAV: { zh: "音视频联合解码", en: "AV VAE Decode" },
  ZouyuExtractSeedMedia: { zh: "提取参考媒体", en: "Extract Seed Media" },
  ZouyuSeedCatalog: { zh: "种子目录", en: "Seed Catalog" },
  ZouyuSeedPreview: { zh: "种子预览", en: "Seed Preview" },
  ZouyuClearTemp: { zh: "清空临时存储", en: "Clear Temp Storage" },
  ZouyuModelLoader: { zh: "Zouyu 模型加载器", en: "Zouyu Model Loader" },
  ZouyuModelGuard: { zh: "Zouyu 模型占用检测", en: "Zouyu Model Guard" },
};

const NODE_COLORS = {
  ZouyuSaveSeedConditioning: { color: "#2e7d4f", bgcolor: "#16321f" },
  ZouyuSeedLoader: { color: "#7a4fa0", bgcolor: "#2c1a3a" },
  ZouyuVAEDecodeAV: { color: "#2f6b8f", bgcolor: "#162a38" },
  ZouyuExtractSeedMedia: { color: "#1f8a8a", bgcolor: "#123232" },
  ZouyuSeedCatalog: { color: "#6b6b6b", bgcolor: "#262626" },
  ZouyuSeedPreview: { color: "#b0722a", bgcolor: "#3a2812" },
  ZouyuClearTemp: { color: "#a03838", bgcolor: "#3a1616" },
  ZouyuModelLoader: { color: "#3d8b40", bgcolor: "#142b16" },
  ZouyuModelGuard: { color: "#8a6d1f", bgcolor: "#2b2310" },
};

// 参考端口槽位 label（V3 autogrow 嵌套名 ref_images.ref_image_0 -> 参考图 1）
const SLOT_LABEL_RE = [
  { re: /^ref_images\.ref_image_(\d+)$/, zh: "参考图", en: "Ref Image" },
  { re: /^ref_videos\.ref_video_(\d+)$/, zh: "参考视频", en: "Ref Video" },
  { re: /^ref_video_audios\.ref_video_audio_(\d+)$/, zh: "视频配乐", en: "Video Audio" },
  { re: /^ref_audios\.ref_audio_(\d+)$/, zh: "参考音频", en: "Ref Audio" },
];

function slotLabel(name, lang) {
  if (!name) return null;
  for (const m of SLOT_LABEL_RE) {
    const match = name.match(m.re);
    if (match) {
      const num = Number(match[1]) + 1;
      const base = isZh(lang) ? m.zh : m.en;
      return `${base} ${num}`;
    }
  }
  return null;
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

  const t = TITLES[node.comfyClass || node.type];
  if (t) node.title = zh ? t.zh : t.en;

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
    if (bl) {
      const on = zh ? bl.zh[0] : bl.en[0];
      const off = zh ? bl.zh[1] : bl.en[1];
      if (w.options) {
        w.options.label_on = on;
        w.options.label_off = off;
      }
      if (typeof w.label_on === "string") w.label_on = on;
      if (typeof w.label_off === "string") w.label_off = off;
    }
    const cb = COMBOS[w.name];
    if (cb && w.type === "combo") {
      w.options = buildComboOptions(w.name, node);
    }
    const entry = LABELS[w.name];
    if (entry) w.label = zh ? entry.zh : entry.en;
  }

  for (const inp of node.inputs || []) {
    const sl = slotLabel(inp.name, lang);
    if (sl) {
      inp.label = sl;
      continue;
    }
    const entry = LABELS[inp.name];
    if (entry) inp.label = zh ? entry.zh : entry.en;
  }

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

  if (node.__zouyuStatus) refreshStatusDOM(node);

  app.graph?.setDirtyCanvas(true, false);
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
  for (const inp of node.inputs || []) {
    for (const g of MENTION_GROUPS) {
      const m = (inp.name || "").match(g.re);
      if (m && inp.link != null) {
        entries.push(`${g.zh}${Number(m[1]) + 1}`);
      }
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
// 模型加载器 / 模型占用检测：文件夹选择、文件刷新、红/绿/蓝状态灯
// ---------------------------------------------------------------------------

const MAX_MODELS = 8;
const MODEL_TYPE_OPTIONS = ["未使用", "主模型", "文本模型", "视频VAE", "音频VAE", "LoRA", "其他"];
const TYPE_EN = { 未使用: "Unused", 主模型: "Main Model", 文本模型: "Text(CLIP)", 视频VAE: "Video VAE", 音频VAE: "Audio VAE", LoRA: "LoRA", 其他: "Other" };
const TYPE_KEYS = { 主模型: "main", 文本模型: "clip", 视频VAE: "vae", 音频VAE: "avae", LoRA: "lora", 其他: "other" };
const TYPE_CATEGORY = { 主模型: "diffusion_models", 文本模型: "text_encoders", 视频VAE: "vae", 音频VAE: "vae", LoRA: "loras", 其他: "diffusion_models" };
const TYPE_PORT_NAMES = { main: "主模型", clip: "文本模型", vae: "视频VAE", avae: "音频VAE", lora: "lora", other: "其他" };
const TYPE_PORT_NAMES_EN = { main: "Main", clip: "CLIP", vae: "VideoVAE", avae: "AudioVAE", lora: "lora", other: "other" };

const KIND_NAMES = {
  unet: { zh: "UNET 模型", en: "UNET Model" },
  clip: { zh: "文本编码器", en: "CLIP" },
  vae: { zh: "视频VAE", en: "Video VAE" },
  audio_vae: { zh: "音频VAE", en: "Audio VAE" },
};

const STATE_INFO = {
  gpu: { zh: "已加载(GPU)", en: "Loaded (GPU)", color: "#4caf50" },
  cpu: { zh: "CPU缓存", en: "CPU cached", color: "#2196f3" },
  free: { zh: "未加载", en: "Not loaded", color: "#f44336" },
  unknown: { zh: "未知", en: "Unknown", color: "#9e9e9e" },
};

function stateText(state, lang) {
  const e = STATE_INFO[state] || STATE_INFO.unknown;
  return lang !== "English" ? e.zh : e.en;
}

const STATUS_STYLE = `
.zouyu-status-block{padding:2px 4px;min-width:130px;margin-top:8px}
.zouyu-status-block .zouyu-model-line{display:flex;align-items:center;gap:6px;font-size:12px;line-height:1.6}
.zouyu-status-block .dot{width:10px;height:10px;border-radius:50%;flex:none;border:1px solid rgba(0,0,0,.4);box-shadow:0 0 4px rgba(0,0,0,.5)}
.zouyu-status-block .zn{color:#e8e8e8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:190px}
.zouyu-status-block .zt{color:#c9a05f;font-size:11px;flex:none}
.zouyu-status-block .zs{color:#9a9a9a;font-size:11px;flex:none}
.zouyu-gap{height:10px}
.zouyu-slot-dot{display:flex;justify-content:flex-end;padding:2px 6px;min-height:16px}
.zouyu-slot-dot .dot{width:10px;height:10px;border-radius:50%;display:inline-block;border:1px solid rgba(0,0,0,.4);box-shadow:0 0 4px rgba(0,0,0,.5)}
`;

let statusStylesInjected = false;

function ensureStatusStyles() {
  if (statusStylesInjected) return;
  statusStylesInjected = true;
  const el = document.createElement("style");
  el.textContent = STATUS_STYLE;
  document.head.appendChild(el);
}

function addDOMWidgetSafe(node, name, element) {
  if (node.widgets?.find((w) => w.name === name)) return null;
  try {
    return node.addDOMWidget(name, "div", element, { serialize: false });
  } catch (e) {
    console.error("[ZouyuSeedTensor] addDOMWidget failed:", e);
    return null;
  }
}

function makeStatusBlock() {
  const el = document.createElement("div");
  el.className = "zouyu-status-block";
  el.innerHTML = `
    <div class="zouyu-model-line"><span class="dot"></span><span class="zn"></span><span class="zt"></span><span class="zs"></span></div>`;
  return el;
}

function updateStatusBlock(el, info, lang) {
  if (!el) return;
  const zh = lang !== "English";
  const st = STATE_INFO[info.state] || STATE_INFO.unknown;
  el.querySelector(".dot").style.background = st.color;
  el.querySelector(".zn").textContent = info.nameDisplay || "—";
  el.querySelector(".zt").textContent = info.typeText || "";
  el.querySelector(".zs").textContent = zh ? st.zh : st.en;
}

let statusNodes = new Set();
let statusTimer = null;

async function refreshStatusDOM(node) {
  if (!node || !node.__zouyuStatus) return;
  try {
    const r = await fetch("/zouyu_model_loader/status");
    if (!r.ok) return;
    const payload = await r.json();
    const byKind = {};
    for (const m of payload.models || []) byKind[m.kind] = m;
    const lang = node.__zouyuLang || getLang();
    const zh = lang !== "English";
    for (const [kind, info] of Object.entries(node.__zouyuStatus)) {
      const m = byKind[kind];
      const st = STATE_INFO[m?.state || "unknown"];
      if (info.dotOnly) {
        // 行尾三色灯：info.el 即灯元素本身；info.tag 为类型标签（集成模式）
        if (info.el) info.el.style.background = st.color;
        if (info.tag && m?.type) {
          info.tag.textContent = zh ? m.type_zh : m.type_en;
        }
        continue;
      }
      const fname = info.fileWidget ? String(info.fileWidget.value || "").split(/[\\/]/).pop() : "";
      const kindName = KIND_NAMES[kind] ? (zh ? KIND_NAMES[kind].zh : KIND_NAMES[kind].en) : kind;
      updateStatusBlock(info.el, {
        state: m?.state || "unknown",
        typeText: m?.type ? (zh ? m.type_zh : m.type_en) : "",
        nameDisplay: info.fileWidget ? (fname || "—") : kindName,
      }, lang);
    }
  } catch (e) {
    /* 服务端未就绪时忽略 */
  }
}

function startStatusPolling() {
  if (statusTimer) return;
  statusTimer = setInterval(async () => {
    for (const node of [...statusNodes]) await refreshStatusDOM(node);
  }, 2500);
}

// ===========================================================================
// 动态模型加载器（Vue 前端：schema 原生控件 + options.hidden 显隐 + 行内端口/状态灯）
// 说明：
// - 显隐：同时设置 w.hidden 与 w.options.hidden（Vue 网格按 options.hidden 判显隐）
// - 行布局：每个可见槽位一行 [类型][文件夹][文件下拉]，行尾叠加 状态灯+类型标签+📁
// - 端口：原生输出端口点用 CSS transform 平移到对应下拉行（与下拉平行、位于最右）
// - 三色灯：直接挂载在节点 DOM 内，随节点移动/缩放自动定位
// - 拖入文件夹：node.onDragOver/onDragDrop → 上传到 models/<同名文件夹>（同名复用/新建）
// - 节点尺寸随可见行数自动收缩/伸长（DOM 网格驱动，隐藏控件不占行）
// ===========================================================================

const ZOUYU_OVERLAY_STYLE = `
.zouyu-loader-node .lg-slot--output{height:0 !important;margin:0 !important;padding:0 !important;overflow:visible !important}
.zouyu-loader-node .lg-slot--output [data-slot-key]{position:relative;z-index:22;cursor:crosshair}
.zouyu-loader-overlay{position:absolute;inset:0;pointer-events:none;z-index:10}
.zouyu-row-tail{position:absolute;right:20px;display:flex;align-items:center;gap:6px;pointer-events:none;transform:translateY(-50%)}
.zouyu-rt-light{width:12px;height:12px;border-radius:50%;border:1px solid rgba(0,0,0,.45);box-shadow:0 0 4px rgba(0,0,0,.5);display:inline-block;background:#9e9e9e}
.zouyu-rt-type{font-size:10px;line-height:14px;color:#c9a05f;background:rgba(0,0,0,.4);padding:0 4px;border-radius:3px;white-space:nowrap}
.zouyu-rt-folder{font-size:11px;line-height:15px;border:none;background:rgba(0,0,0,.4);border-radius:3px;padding:0 4px;color:#ddd;cursor:pointer;pointer-events:auto}
.zouyu-rt-folder:hover{background:rgba(0,0,0,.6)}
.zouyu-import-bar{position:absolute;left:8px;right:8px;bottom:3px;height:16px;font-size:10px;line-height:16px;text-align:center;color:#9a9a9a;background:rgba(0,0,0,.28);border-radius:3px;cursor:pointer;pointer-events:auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.zouyu-import-bar:hover{background:rgba(0,0,0,.45)}
.zouyu-import-bar.dragover{background:rgba(76,175,80,.4);color:#fff}
`;

let zouyuOverlayStyleInjected = false;

function ensureLoaderOverlayStyle() {
  if (zouyuOverlayStyleInjected) return;
  zouyuOverlayStyleInjected = true;
  const el = document.createElement("style");
  el.textContent = ZOUYU_OVERLAY_STYLE;
  document.head.appendChild(el);
}

function setWidgetHidden(w, hidden) {
  if (!w) return;
  w.hidden = !!hidden;
  if (!w.options) w.options = {};
  w.options.hidden = !!hidden;
  if (hidden) {
    w.computeLayoutSize = () => ({ minHeight: 0, minWidth: 0, maxHeight: 0, maxWidth: 0 });
  } else {
    w.computeLayoutSize = undefined;
  }
}

function slotWidgets(node, i) {
  return {
    type: node.widgets?.find((w) => w.name === `model_${i}_type`),
    folder: node.widgets?.find((w) => w.name === `model_${i}_folder`),
    name: node.widgets?.find((w) => w.name === `model_${i}_name`),
  };
}

/** 可见槽位数：0..最后已填槽位+1（填好后自动弹出下一个）。 */
function visibleSlotCount(st) {
  let lastFilled = -1;
  for (let i = 0; i < MAX_MODELS; i++) {
    const s = st.slots[i];
    if (!s) continue;
    const filled = !!s.name && s.name !== "(未选择)" && s.name !== "(无文件)";
    const typeSet = !!s.type && s.type !== "未使用";
    // 集成模式：只填文件也算已用（类型由后端自动识别）
    if (filled && (typeSet || st.compact)) lastFilled = i;
  }
  return Math.min(Math.max(lastFilled + 2, 1), MAX_MODELS);
}

/** 刷新某槽位文件下拉选项（按类型+文件夹过滤显示；Vue 组合框读 widget.options.values）。 */
async function refreshSlotFileOptions(node, i) {
  const st = node.__zouyuSlotState;
  const s = st && st.slots[i];
  const w = slotWidgets(node, i).name;
  if (!w || !s) return;
  if (!s.type || s.type === "未使用") {
    w.options = w.options || {};
    w.options.values = ["(未选择)"];
    w.value = "(未选择)";
    return;
  }
  const category = TYPE_CATEGORY[s.type] || "diffusion_models";
  const folder = s.folder || category;
  try {
    const d = await fetchJson(
      `/zouyu_model_loader/files?category=${encodeURIComponent(category)}&folder=${encodeURIComponent(folder)}`
    );
    s.files = (d.files || []).length ? d.files : [];
    w.options = w.options || {};
    w.options.values = s.files.length ? s.files : ["(未选择)"];
    if (!s.files.includes(w.value)) w.value = s.files[0] || "(未选择)";
  } catch (e) {
    w.options.values = ["(未选择)"];
  }
  app.graph?.setDirtyCanvas(true, false);
}

function _cssEsc(v) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(String(v));
  return String(v).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c);
}

/** 节点根 DOM 元素（Vue 渲染，[data-node-id]）。 */
function nodeEl(node) {
  if (!node || node.id == null) return null;
  return document.querySelector(`.lg-node[data-node-id="${_cssEsc(node.id)}"]`) || null;
}

/** 按 data-slot-key 后缀找槽位点元素（nodeId 可能是数字或字符串）。 */
function findSlotEl(rootEl, suffix) {
  const all = rootEl.querySelectorAll("[data-slot-key]");
  for (const el of all) {
    const k = el.dataset?.slotKey || "";
    if (k.endsWith(suffix)) return el;
  }
  return null;
}

/** 测量单个可见控件行的高度（Vue 网格行）。 */
function measureRowHeight(node) {
  const el = nodeEl(node);
  const row = el?.querySelector('[data-testid="node-widget"]');
  return row ? row.getBoundingClientRect().height : 0;
}

/** 槽位 i 的名字控件在可见控件网格中的行序号（估算兜底用）。 */
function visibleRowIndex(node, i) {
  const st = node.__zouyuSlotState;
  if (!st) return i;
  const count = visibleSlotCount(st);
  const compact = !!st.compact;
  let idx = 0;
  for (let j = 0; j < i; j++) {
    const s = st.slots[j];
    const vis = j < count;
    if (!vis) continue;
    const used = s && s.type !== "未使用";
    if (compact) {
      idx += 1;
    } else {
      idx += used ? 3 : 1;
    }
  }
  return idx;
}

/** 行尾叠加：状态灯 + 类型标签(集成模式) + 📁(常规模式)。 */
function updateRowTail(node, overlay, i, rowCY, visible, used, compact, zh) {
  let tail = overlay.querySelector(`[data-zouyu-tail="${i}"]`);
  if (!visible || !used) {
    if (tail) tail.remove();
    if (node.__zouyuStatus) delete node.__zouyuStatus[`slot${i}`];
    return;
  }
  if (!tail) {
    tail = document.createElement("div");
    tail.className = "zouyu-row-tail";
    tail.setAttribute("data-zouyu-tail", String(i));
    overlay.appendChild(tail);
    tail.addEventListener("pointerdown", (e) => e.stopPropagation());
  }
  tail.style.top = (rowCY != null ? rowCY : 12) + "px";
  tail.innerHTML = "";
  const s = node.__zouyuSlotState.slots[i];
  let tag = null;
  if (compact && s) {
    // 集成模式：类型只显示在端口旁（未选类型时显示"自动"，执行后刷新为识别结果）
    tag = document.createElement("span");
    tag.className = "zouyu-rt-type";
    tag.textContent = s.type === "未使用"
      ? (zh ? "自动" : "Auto")
      : (zh ? s.type : (TYPE_PORT_NAMES_EN[TYPE_KEYS[s.type]] || s.type));
    tail.appendChild(tag);
  } else {
    const btn = document.createElement("button");
    btn.className = "zouyu-rt-folder";
    btn.textContent = "📁";
    btn.title = zh ? "选择模型文件夹" : "Choose model folder";
    btn.addEventListener("click", (e) => { e.stopPropagation(); pickSlotFolder(node, i); });
    tail.appendChild(btn);
  }
  const light = document.createElement("span");
  light.className = "zouyu-rt-light";
  light.title = zh ? "绿=已加载(GPU) 蓝=CPU缓存 红=未加载" : "Green=GPU Blue=CPU Red=Not loaded";
  tail.appendChild(light);
  if (node.__zouyuStatus) node.__zouyuStatus[`slot${i}`] = { el: light, tag, dotOnly: true };
}

/** 布局一次：端口点平移到下拉行 + 行尾灯/标签/📁 定位（节点 DOM 内，自动跟随节点）。 */
function layoutLoaderOverlay(node) {
  const el = nodeEl(node);
  const st = node.__zouyuSlotState;
  if (!el || !st) return;
  el.classList.add("zouyu-loader-node");
  let overlay = el.querySelector(".zouyu-loader-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "zouyu-loader-overlay";
    overlay.setAttribute("data-zouyu-overlay", "1");
    el.appendChild(overlay);
    const bar = document.createElement("div");
    bar.className = "zouyu-import-bar";
    bar.setAttribute("data-zouyu-import", "1");
    bar.dataset.text = zhText(node, "📥 拖入文件夹导入模型（或点击选择）", "📥 Drop folder to import models (or click)");
    bar.textContent = bar.dataset.text;
    overlay.appendChild(bar);
    bar.addEventListener("click", () => pickAndImportFolder(node));
    bar.addEventListener("dragover", (e) => { e.preventDefault(); bar.classList.add("dragover"); });
    bar.addEventListener("dragleave", () => bar.classList.remove("dragover"));
    bar.addEventListener("drop", (e) => { e.preventDefault(); bar.classList.remove("dragover"); });
  }
  const nodeRect = el.getBoundingClientRect();
  const count = visibleSlotCount(st);
  const compact = !!st.compact;
  const zh = st.lang !== "English";
  const rowH = measureRowHeight(node);
  for (let i = 0; i < MAX_MODELS; i++) {
    const visible = i < count;
    const s = st.slots[i];
    const used = visible && !!s && s.type !== "未使用";
    // 行锚点：该行名字控件的输入端口点
    const anchor = findSlotEl(el, `-in-${3 * i + 2}`);
    let rowCY = null;
    if (anchor) {
      const r = anchor.getBoundingClientRect();
      rowCY = r.top + r.height / 2 - nodeRect.top;
    } else if (rowH) {
      rowCY = (visibleRowIndex(node, i) + 0.5) * rowH;
    }
    // 输出端口点：可见槽位平移到该行（与下拉平行、最右）；隐藏槽位隐藏端口
    const outDot = findSlotEl(el, `-out-${i}`);
    if (outDot) {
      if (!visible) {
        outDot.style.opacity = "0";
        outDot.style.pointerEvents = "none";
      } else {
        outDot.style.opacity = "";
        outDot.style.pointerEvents = "";
        if (rowCY != null) {
          const dr = outDot.getBoundingClientRect();
          const curCY = dr.top + dr.height / 2 - nodeRect.top;
          const dy = rowCY - curCY;
          if (Math.abs(dy) > 0.5) outDot.style.transform = `translateY(${dy.toFixed(1)}px)`;
        }
      }
    }
    updateRowTail(node, overlay, i, rowCY, visible, used, compact, zh);
  }
  if (node.__zouyuLang !== st.lang) {
    const bar = overlay.querySelector(".zouyu-import-bar");
    if (bar) {
      bar.dataset.text = zhText(node, "📥 拖入文件夹导入模型（或点击选择）", "📥 Drop folder to import models (or click)");
      bar.textContent = bar.dataset.text;
    }
    node.__zouyuLang = st.lang;
  }
}

/** 触发 widgets 网格尺寸变化 → ResizeObserver → 槽位布局重测（让端口点新位置生效）。 */
function forceSlotResync(node) {
  const el = nodeEl(node);
  const grid = el?.querySelector('[data-testid="node-widgets"]');
  if (!grid) return;
  grid.style.minHeight = "1px";
  setTimeout(() => { grid.style.minHeight = ""; }, 80);
}

/** 节点 DOM 被 Vue 重挂载时自动补挂叠加层。 */
function watchLoaderOverlay(node) {
  const el = nodeEl(node);
  if (!el || el.__zouyuOverlayObs) return;
  el.__zouyuOverlayObs = new MutationObserver(() => {
    if (!el.querySelector(".zouyu-loader-overlay")) layoutLoaderOverlay(node);
  });
  el.__zouyuOverlayObs.observe(el, { childList: true });
}

function zhText(node, zh, en) {
  return node.__zouyuLang !== "English" ? zh : en;
}

function importBarEl(node) {
  return nodeEl(node)?.querySelector(".zouyu-import-bar") || null;
}

function showImportResult(node, text, isError) {
  const bar = importBarEl(node);
  if (!bar) return;
  bar.textContent = text;
  bar.style.color = isError ? "#ff8a80" : "#8bc34a";
  clearTimeout(bar._zouyuTimer);
  bar._zouyuTimer = setTimeout(() => {
    bar.textContent = bar.dataset.text || "";
    bar.style.color = "";
  }, 6000);
}

/** 递归收集拖入目录的所有文件（webkitGetAsEntry）。 */
function collectDroppedFolder(dt) {
  return new Promise((resolve) => {
    const items = dt?.items ? [...dt.items] : [];
    const entries = items
      .map((it) => (typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null))
      .filter(Boolean);
    const dir = entries.find((en) => en.isDirectory);
    if (!dir) { resolve(null); return; }
    const files = [];
    let pending = 1;
    const maybeDone = () => { pending--; if (pending <= 0) resolve({ name: dir.name || null, files }); };
    const walk = (entry) => {
      const reader = entry.createReader();
      const readBatch = () => {
        reader.readEntries((batch) => {
          if (!batch.length) { maybeDone(); return; }
          for (const en of batch) {
            if (en.isFile) {
              pending++;
              en.file((f) => { files.push(f); maybeDone(); });
            } else if (en.isDirectory) {
              pending++;
              walk(en);
              maybeDone();
            }
          }
          readBatch();
        }, () => maybeDone());
      };
      readBatch();
    };
    walk(dir);
    setTimeout(() => resolve({ name: dir.name || null, files }), 30000);
  });
}

/** 把文件上传到 models/<name>（同名文件夹复用/新建），并刷新槽位。 */
async function importFolderEntries(node, folderName, files) {
  if (!folderName || !files.length) return;
  const bar = importBarEl(node);
  if (bar) bar.textContent = zhText(node, `上传中 ${files.length} 个文件…`, `Uploading ${files.length} files…`);
  try {
    const fd = new FormData();
    fd.append("folder_name", folderName);
    for (const f of files) fd.append("files", f, f.name);
    const r = await fetch("/zouyu_model_loader/import_folder", { method: "POST", body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !d.ok) { showImportResult(node, (d.error || r.statusText || "导入失败"), true); return; }
    showImportResult(node, zhText(node, `已导入 ${d.count} 个文件 → models/${d.target}`, `Imported ${d.count} files → models/${d.target}`), false);
    // 刷新各槽位下拉选项
    const st = node.__zouyuSlotState;
    for (let i = 0; i < MAX_MODELS; i++) {
      const s = st && st.slots[i];
      if (s && s.type && s.type !== "未使用") await refreshSlotFileOptions(node, i);
    }
    // 第一个可见空槽位的文件夹指向导入目标
    for (let i = 0; i < visibleSlotCount(st); i++) {
      const s = st.slots[i];
      if (s && s.type && s.type !== "未使用" && (!s.folder || s.folder === TYPE_CATEGORY[s.type])) {
        s.folder = d.target;
        const fw = slotWidgets(node, i).folder;
        if (fw) fw.value = d.target;
        await refreshSlotFileOptions(node, i);
        break;
      }
    }
  } catch (err) {
    showImportResult(node, "导入失败: " + (err && err.message || err), true);
  }
}

/** 点击导入条：showDirectoryPicker 选择文件夹并导入。 */
async function pickAndImportFolder(node) {
  try {
    if (window.showDirectoryPicker) {
      const handle = await window.showDirectoryPicker();
      const files = [];
      const walk = async (dh) => {
        for await (const [name, h] of dh.entries()) {
          if (h.kind === "file") {
            const f = await h.getFile();
            try { Object.defineProperty(f, "webkitRelativePath", { value: `${dh.name}/${name}` }); } catch { /* noop */ }
            files.push(f);
          } else if (h.kind === "directory") {
            await walk(h);
          }
        }
      };
      await walk(handle);
      if (!files.length) { showImportResult(node, zhText(node, "文件夹为空", "Folder is empty"), true); return; }
      await importFolderEntries(node, handle.name, files);
      return;
    }
    // 兼容回退：隐藏 <input webkitdirectory>
    const picked = await new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.webkitdirectory = true;
      input.style.display = "none";
      document.body.appendChild(input);
      let done = false;
      const finish = (v) => { if (!done) { done = true; input.remove(); resolve(v); } };
      input.onchange = () => {
        const files = [...(input.files || [])];
        const name = files[0]?.webkitRelativePath?.split("/")[0] || null;
        finish(name ? { name, files } : null);
      };
      input.oncancel = () => finish(null);
      input.click();
    });
    if (!picked) return;
    await importFolderEntries(node, picked.name, picked.files);
  } catch (err) {
    if (err && err.name === "AbortError") return;
    showImportResult(node, "选择失败: " + (err && err.message || err), true);
  }
}

/** 节点级拖入文件夹（Vue 前端通过 node.onDragOver/onDragDrop 回调）。 */
function setupLoaderDragDrop(node) {
  node.onDragOver = (e) => {
    try {
      const items = e.dataTransfer?.items ? [...e.dataTransfer.items] : [];
      return items.some((it) => {
        const entry = typeof it.webkitGetAsEntry === "function" && it.webkitGetAsEntry();
        return !!entry && entry.isDirectory;
      });
    } catch { return false; }
  };
  node.onDragDrop = async (e) => {
    try {
      const folder = await collectDroppedFolder(e.dataTransfer);
      if (!folder || !folder.files.length) {
        showImportResult(node, zhText(node, "未识别到文件夹（请使用 Chrome/Edge 拖入）", "No folder detected (use Chrome/Edge)"), true);
        return true;
      }
      await importFolderEntries(node, folder.name, folder.files);
      return true;
    } catch (err) {
      showImportResult(node, "导入失败: " + (err && err.message || err), true);
      return true;
    }
  };
}

/** 按当前状态应用控件显隐 + 行布局 + 端口平移。 */
function applySlotVisibility(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const compact = !!st.compact;
  const count = visibleSlotCount(st);
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    const visible = i < count;
    const s = st.slots[i] || (st.slots[i] = { type: "未使用", folder: "", name: "(未选择)", files: [] });
    const used = visible && s.type !== "未使用";
    // 集成模式：不显示类型/文件夹，只显示文件列（类型只在端口旁显示）
    setWidgetHidden(w.type, !visible || compact);
    setWidgetHidden(w.name, !visible || (!used && !compact));
    setWidgetHidden(w.folder, !visible || !used || compact);
    if (w.type && w.type.value !== s.type) w.type.value = s.type;
    if (w.folder && w.folder.value !== s.folder) w.folder.value = s.folder;
    if (w.name && w.name.value !== s.name) w.name.value = s.name;
  }
  syncLoaderOutputs(node);
  app.graph?.setDirtyCanvas(true, true);
  requestAnimationFrame(() => {
    layoutLoaderOverlay(node);
    forceSlotResync(node);
  });
  setTimeout(() => {
    layoutLoaderOverlay(node);
    forceSlotResync(node);
  }, 60);
}

/** 输出端口与模型一一对应：按类型编号命名（主模型0/主模型1/lora0/...），端口文字隐藏由行尾标签展示。 */
function syncLoaderOutputs(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const zh = st.lang !== "English";
  const ordinals = {};
  for (let i = 0; i < MAX_MODELS; i++) {
    const s = st.slots[i];
    if (!node.outputs[i]) continue;
    const used = s && s.type && s.type !== "未使用" && s.name && s.name !== "(未选择)";
    if (!used) {
      node.outputs[i].name = "";
      node.outputs[i].label = "";
      continue;
    }
    const tkey = TYPE_KEYS[s.type] || "other";
    ordinals[tkey] = (ordinals[tkey] || 0) + 1;
    const base = zh ? (TYPE_PORT_NAMES[tkey] || "其他") : (TYPE_PORT_NAMES_EN[tkey] || "other");
    node.outputs[i].name = base + (ordinals[tkey] - 1);
    node.outputs[i].label = "";
    node.outputs[i].type = "*";
  }
  app.graph?.setDirtyCanvas(true, false);
}

/** 设置各槽位控件的中文/英文标签。 */
function applyLoaderLabels(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const zh = st.lang !== "English";
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    if (w.type) w.label = zh ? `模型 ${i}` : `Model ${i}`;
    if (w.folder) w.label = zh ? "文件夹" : "Folder";
    if (w.name) w.label = zh ? "模型文件" : "Model file";
  }
}

/** 「选择模型文件夹」（行尾 📁）：原生目录选择，取消一律静默。 */
async function pickSlotFolder(node, i) {
  const st = node.__zouyuSlotState;
  const s = st.slots[i];
  if (!s || !s.type || s.type === "未使用") return;
  const category = TYPE_CATEGORY[s.type] || "diffusion_models";
  let folderName = null;
  if (window.showDirectoryPicker) {
    try {
      const handle = await window.showDirectoryPicker();
      folderName = handle?.name || null;
    } catch (e) {
      return;
    }
  }
  if (!folderName) {
    folderName = await new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.webkitdirectory = true;
      input.style.display = "none";
      document.body.appendChild(input);
      let done = false;
      const finish = (v) => { if (!done) { done = true; input.remove(); resolve(v); } };
      input.onchange = () => {
        const f = input.files?.[0];
        finish(f?.webkitRelativePath ? f.webkitRelativePath.split("/")[0] : null);
      };
      input.oncancel = () => finish(null);
      input.click();
    });
    if (!folderName) return;
  }
  try {
    const d = await fetchJson(`/zouyu_model_loader/find_folder?name=${encodeURIComponent(folderName)}`);
    const found = d.found || [];
    if (!found.length) {
      alert(`[Zouyu] models 目录下找不到「${folderName}」，请手动在文件夹输入框填写`);
      return;
    }
    let chosen = found[0];
    for (const rel of found) {
      const fl = await fetchJson(
        `/zouyu_model_loader/files?category=${encodeURIComponent(category)}&folder=${encodeURIComponent(rel)}`
      );
      if (fl.files?.length) {
        chosen = rel;
        break;
      }
    }
    s.folder = chosen;
    const fw = slotWidgets(node, i).folder;
    if (fw) fw.value = chosen;
    await refreshSlotFileOptions(node, i);
  } catch (e) {
    alert("[Zouyu] 定位文件夹失败: " + e);
  }
}

function removeSlot(node, i) {
  const st = node.__zouyuSlotState;
  for (let j = i; j < MAX_MODELS - 1; j++) {
    st.slots[j] = st.slots[j + 1] || { type: "未使用", folder: "", name: "(未选择)", files: [] };
  }
  st.slots[MAX_MODELS - 1] = { type: "未使用", folder: "", name: "(未选择)", files: [] };
  applySlotVisibility(node);
  refreshStatusDOM(node);
}

function setupModelLoaderNode(node) {
  ensureLoaderOverlayStyle();

  // 从 schema 原生控件读取状态
  const st = { compact: false, lowVram: false, lang: getLang(), slots: {} };
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    st.slots[i] = {
      type: w.type?.value || "未使用",
      folder: w.folder?.value || "",
      name: w.name?.value || "(未选择)",
      files: [],
    };
  }
  st.compact = !!node.widgets?.find((w) => w.name === "compact_mode")?.value;
  st.lowVram = !!node.widgets?.find((w) => w.name === "low_vram_mode")?.value;
  st.lang = node.widgets?.find((w) => w.name === "language")?.value || getLang();
  node.__zouyuSlotState = st;
  node.__zouyuLang = st.lang;
  node.__zouyuStatus = {};

  // 挂接回调（包装原有回调，保证值仍然流向后端）
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    if (!w.type) continue;
    const s = st.slots[i];
    const origTypeCb = w.type.callback;
    w.type.callback = (v) => {
      if (origTypeCb) origTypeCb.call(w.type, v);
      s.type = v;
      if (v !== "未使用") {
        s.folder = TYPE_CATEGORY[v] || "";
        if (w.folder) w.folder.value = s.folder;
        s.name = "(未选择)";
        if (w.name) w.name.value = "(未选择)";
        refreshSlotFileOptions(node, i);
      }
      applySlotVisibility(node);
    };
    if (w.name) {
      const origNameCb = w.name.callback;
      w.name.callback = (v) => {
        if (origNameCb) origNameCb.call(w.name, v);
        s.name = v;
        applySlotVisibility(node);
      };
    }
    if (w.folder) {
      const origFolderCb = w.folder.callback;
      w.folder.callback = (v) => {
        if (origFolderCb) origFolderCb.call(w.folder, v);
        s.folder = v;
        refreshSlotFileOptions(node, i);
      };
    }
  }
  const compactW = node.widgets.find((w) => w.name === "compact_mode");
  if (compactW) {
    const orig = compactW.callback;
    compactW.callback = (v) => { if (orig) orig.call(compactW, v); st.compact = !!v; applySlotVisibility(node); };
  }
  const lowW = node.widgets.find((w) => w.name === "low_vram_mode");
  if (lowW) {
    const orig = lowW.callback;
    lowW.callback = (v) => { if (orig) orig.call(lowW, v); st.lowVram = !!v; };
  }
  const langW = node.widgets.find((w) => w.name === "language");
  if (langW) {
    const orig = langW.callback;
    langW.callback = (v) => {
      if (orig) orig.call(langW, v);
      st.lang = v;
      setLang(v);
      node.__zouyuLang = v;
      applyLoaderLabels(node);
      syncLoaderOutputs(node);
      applyLanguage(node, v);
    };
  }

  applyLoaderLabels(node);
  applySlotVisibility(node);
  for (const i of Object.keys(st.slots)) {
    const s = st.slots[i];
    if (s.type && s.type !== "未使用") refreshSlotFileOptions(node, Number(i));
  }
  refreshStatusDOM(node);

  // 拖入文件夹导入
  setupLoaderDragDrop(node);

  // 布局：等 Vue 渲染出节点 DOM 后挂载叠加层 + 平移端口点
  const layoutPass = () => {
    layoutLoaderOverlay(node);
    forceSlotResync(node);
  };
  setTimeout(layoutPass, 0);
  setTimeout(layoutPass, 120);
  setTimeout(layoutPass, 400);
  setTimeout(() => watchLoaderOverlay(node), 600);
  // 延时重同步：防止框架在 onNodeCreated 后重建控件覆盖显隐/回调
  setTimeout(() => {
    try { applySlotVisibility(node); applyLoaderLabels(node); } catch (e) { /* ignore */ }
  }, 150);
}


function setupModelGuardNode(node) {
  ensureStatusStyles();
  node.__zouyuStatus = {};
  statusNodes.add(node);
  startStatusPolling();

  const pairs = [
    ["model", "unet"],
    ["clip", "clip"],
    ["vae", "vae"],
    ["audio_vae", "audio_vae"],
  ];
  for (const [inpName, kind] of pairs) {
    const inp = node.inputs?.find((i) => i.name === inpName);
    if (!inp || inp.link == null) continue; // 只为已连接的模型亮灯
    const el = makeStatusBlock();
    addDOMWidgetSafe(node, "zouyu_guard_" + kind, el);
    node.__zouyuStatus[kind] = { el, folderWidget: null, fileWidget: null };
  }

  refreshStatusDOM(node);
}

// ---------------------------------------------------------------------------
// 扩展注册
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "ZouyuSeedTensor.ui",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.category !== CATEGORY) return;

    const comfyClass = nodeData.name;

    // 连接变化时重新应用语言（确保 autogrow 新增槽位的 label 被翻译）
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = origOnConnectionsChange?.apply(this, arguments);
      try {
        applyLanguage(this, this.__zouyuLang || getLang());
        if (this.comfyClass === "ZouyuModelGuard") setupModelGuardNode(this);
      } catch (e) {
        console.error("[ZouyuSeedTensor] i18n refresh error:", e);
      }
      return r;
    };

    const origOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const r = origOnRemoved?.apply(this, arguments);
      statusNodes.delete(this);
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

        applyLanguage(node, getLang());

        if (comfyClass === "ZouyuSeedLoader") {
          setupMentionAutocomplete(node);
        }

        if (comfyClass === "ZouyuModelLoader") {
          setupModelLoaderNode(node);
        }

        if (comfyClass === "ZouyuModelGuard") {
          setupModelGuardNode(node);
        }

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

// 模型加载器/占用检测执行完成后立即刷新状态灯（兜底：另有 2.5s 轮询）
api.addEventListener("executed", (event) => {
  try {
    const detail = event.detail;
    if (!detail || detail.nodeId == null) return;
    const node = app.graph?._nodes?.find((n) => String(n.id) === String(detail.nodeId));
    if (!node) return;
    if (node.comfyClass === "ZouyuModelLoader" || node.comfyClass === "ZouyuModelGuard") {
      refreshStatusDOM(node);
    }
  } catch (e) {
    /* ignore */
  }
});
