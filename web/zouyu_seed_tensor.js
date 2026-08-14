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
        if (info.el) info.el.querySelector(".dot").style.background = st.color;
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

async function refreshSlotFiles(node, i) {
  const st = node.__zouyuSlotState;
  const s = st.slots[i];
  if (!s || !s.type || s.type === "未使用") return;
  const category = TYPE_CATEGORY[s.type] || "diffusion_models";
  const folder = s.folder || category;
  try {
    const d = await fetchJson(
      `/zouyu_model_loader/files?category=${encodeURIComponent(category)}&folder=${encodeURIComponent(folder)}`
    );
    s.files = (d.files || []).length ? d.files : ["(未选择)"];
    if (!s.files.includes(s.name)) s.name = s.files[0];
  } catch (e) {
    s.files = ["(未选择)"];
  }
  rebuildLoaderWidgets(node);
}

/** 「选择模型文件夹」：直接打开系统原生文件夹对话框（取消一律静默）。 */
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
      return; // 取消 → 静默
    }
  } else {
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
    if (!folderName) return; // 取消或浏览器不支持 → 静默
  }
  try {
    const d = await fetchJson(`/zouyu_model_loader/find_folder?name=${encodeURIComponent(folderName)}`);
    const found = d.found || [];
    if (!found.length) {
      alert(`[Zouyu] models 目录下找不到名为「${folderName}」的文件夹，请手动在文件夹输入框填写`);
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
    await refreshSlotFiles(node, i);
  } catch (e) {
    alert("[Zouyu] 定位文件夹失败: " + e);
  }
}

/** 可见槽位数：0..最后已填槽位+1（保证填好后自动弹出下一个）。 */
function computeVisibleCount(st) {
  let lastFilled = -1;
  for (let i = 0; i < MAX_MODELS; i++) {
    const s = st.slots[i];
    if (s && s.type && s.type !== "未使用" && s.name && s.name !== "(未选择)") lastFilled = i;
  }
  return Math.min(Math.max(lastFilled + 2, 1), MAX_MODELS);
}

function addComboWidget(node, name, value, options, callback) {
  const w = node.addWidget("combo", name, value, callback, {});
  w.options = w.options || {};
  w.options.values = options || [];
  w.serialize = true;
  return w;
}

function addStringWidget(node, name, value, callback) {
  const w = node.addWidget("string", name, value, callback, {});
  w.serialize = true;
  return w;
}

function addToggleWidget(node, name, value, callback, on, off) {
  const w = node.addWidget("toggle", name, value, callback, {});
  w.options = w.options || {};
  w.options.label_on = on;
  w.options.label_off = off;
  w.serialize = true;
  return w;
}

function makeSlotDot(i) {
  const el = document.createElement("div");
  el.className = "zouyu-slot-dot";
  el.innerHTML = '<span class="dot"></span>';
  el.title = "绿=已加载(GPU) 蓝=CPU缓存 红=未加载";
  return el;
}

function onSlotTypeChange(node, i, v) {
  const st = node.__zouyuSlotState;
  const s = st.slots[i] || (st.slots[i] = { type: "未使用", folder: "", name: "(未选择)", files: [] });
  s.type = v;
  if (v !== "未使用") {
    s.folder = TYPE_CATEGORY[v] || "";
    s.name = "(未选择)";
    s.files = [];
    refreshSlotFiles(node, i);
  } else {
    s.folder = "";
    s.name = "(未选择)";
    s.files = [];
    rebuildLoaderWidgets(node);
  }
}

function onSlotNameChange(node, i, v) {
  const st = node.__zouyuSlotState;
  const s = st.slots[i];
  if (!s) return;
  s.name = v;
  if (v && v !== "(未选择)" && i + 1 < MAX_MODELS && !st.slots[i + 1]) {
    // 填好当前模型 → 自动弹出下一个选择界面
    st.slots[i + 1] = { type: "未使用", folder: "", name: "(未选择)", files: [] };
  }
  rebuildLoaderWidgets(node);
}

function removeSlot(node, i) {
  const st = node.__zouyuSlotState;
  for (let j = i; j < MAX_MODELS - 1; j++) {
    st.slots[j] = st.slots[j + 1] || { type: "未使用", folder: "", name: "(未选择)", files: [] };
  }
  st.slots[MAX_MODELS - 1] = { type: "未使用", folder: "", name: "(未选择)", files: [] };
  rebuildLoaderWidgets(node);
}

/** 输出端口与模型一一对应：按类型编号（主模型0/主模型1/lora0/...），数量随模型增减。 */
function syncLoaderOutputs(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const zh = st.lang !== "English";
  const count = computeVisibleCount(st);
  while (node.outputs.length > count) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < count) node.addOutput("model_" + node.outputs.length, "*");
  const ordinals = {};
  for (let i = 0; i < count; i++) {
    const s = st.slots[i];
    const tkey = TYPE_KEYS[s?.type] || "other";
    ordinals[tkey] = (ordinals[tkey] || 0) + 1;
    const base = zh ? (TYPE_PORT_NAMES[tkey] || "其他") : (TYPE_PORT_NAMES_EN[tkey] || "other");
    node.outputs[i].name = base + (ordinals[tkey] - 1);
    node.outputs[i].type = "*";
  }
  app.graph?.setDirtyCanvas?.(true, false);
}

/** 重建节点控件列表：只保留可见槽位的类型/文件夹/文件下拉 + 状态灯 + 端口。 */
function rebuildLoaderWidgets(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const zh = st.lang !== "English";
  const compact = !!st.compact;
  const count = computeVisibleCount(st);
  node.widgets = [];
  node.__zouyuStatus = {};
  for (let i = 0; i < count; i++) {
    const s = st.slots[i] || (st.slots[i] = { type: "未使用", folder: "", name: "(未选择)", files: [] });
    const used = s.type !== "未使用";
    const typeW = addComboWidget(node, `model_${i}_type`, s.type, MODEL_TYPE_OPTIONS, (v) => onSlotTypeChange(node, i, v));
    typeW.options.getOptionLabel = (v) => {
      if (v === "未使用") return zh ? "请选择模型类型" : "Select Model Type";
      return zh ? v : (TYPE_EN[v] || v);
    };
    typeW.label = zh ? `模型 ${i}` : `Model ${i}`;
    if (used) {
      if (!compact) {
        const folderW = addStringWidget(node, `model_${i}_folder`, s.folder, (v) => { s.folder = v; refreshSlotFiles(node, i); });
        folderW.label = zh ? "文件夹" : "Folder";
        const btn = node.addWidget("button", "📁", null, () => pickSlotFolder(node, i));
        btn.serialize = false;
        btn.title = zh ? "选择模型文件夹" : "Choose model folder";
      }
      const nameW = addComboWidget(node, `model_${i}_name`, s.name, s.files.length ? s.files : ["(未选择)"], (v) => onSlotNameChange(node, i, v));
      nameW.label = zh ? "模型文件" : "Model file";
    } else {
      const nameW = addComboWidget(node, `model_${i}_name`, "(未选择)", ["(未选择)"], null);
      nameW.label = zh ? "模型文件" : "Model file";
    }
    // 三色状态灯：放在文件下拉之后（端口旁，位于端口与下拉之间）
    const dotEl = makeSlotDot(i);
    addDOMWidgetSafe(node, `zouyu_slot_dot_${i}`, dotEl);
    node.__zouyuStatus[`slot${i}`] = { el: dotEl, dotOnly: true };
    if (i > 0) {
      const rm = node.addWidget("button", zh ? "− 移除" : "− Remove", null, () => removeSlot(node, i));
      rm.serialize = false;
    }
  }
  // 底部：集成模式 / 低显存模式 / 界面语言
  const compactW = addToggleWidget(node, "compact_mode", !!st.compact, (v) => { st.compact = v; rebuildLoaderWidgets(node); }, zh ? "集成" : "Compact", zh ? "展开" : "Full");
  compactW.label = zh ? "集成模式" : "Compact";
  const lowW = addToggleWidget(node, "low_vram_mode", !!st.lowVram, (v) => { st.lowVram = v; }, zh ? "彻底卸载" : "Full unload", zh ? "CPU缓存" : "CPU cache");
  lowW.label = zh ? "低显存模式" : "Low VRAM";
  const langW = addComboWidget(node, "language", st.lang, ["中文", "English"], (v) => {
    st.lang = v;
    setLang(v);
    node.__zouyuLang = v;
    applyLanguage(node, v);
    rebuildLoaderWidgets(node);
  });
  langW.label = zh ? "界面语言" : "Language";
  syncLoaderOutputs(node);
  refreshStatusDOM(node);
  app.graph?.setDirtyCanvas(true, false);
}

function setupModelLoaderNode(node) {
  ensureStatusStyles();
  statusNodes.add(node);
  startStatusPolling();
  // 从 schema 控件读取已保存状态（首次创建/工作流加载）
  const st = { compact: false, lowVram: false, lang: getLang(), slots: {} };
  for (let i = 0; i < MAX_MODELS; i++) {
    const tw = node.widgets?.find((w) => w.name === `model_${i}_type`);
    const nw = node.widgets?.find((w) => w.name === `model_${i}_name`);
    const fw = node.widgets?.find((w) => w.name === `model_${i}_folder`);
    const t = tw?.value || "未使用";
    if (t !== "未使用" || i === 0) {
      st.slots[i] = { type: t, folder: fw?.value || "", name: nw?.value || "(未选择)", files: [] };
    }
  }
  st.compact = !!node.widgets?.find((w) => w.name === "compact_mode")?.value;
  st.lowVram = !!node.widgets?.find((w) => w.name === "low_vram_mode")?.value;
  st.lang = node.widgets?.find((w) => w.name === "language")?.value || getLang();
  node.__zouyuSlotState = st;
  node.__zouyuLang = st.lang;
  rebuildLoaderWidgets(node);
  // 为已保存的槽位拉取文件选项
  for (const i of Object.keys(st.slots)) {
    const s = st.slots[i];
    if (s.type && s.type !== "未使用") refreshSlotFiles(node, Number(i));
  }
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
