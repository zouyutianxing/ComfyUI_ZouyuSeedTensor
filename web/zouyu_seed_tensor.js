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
  // ---- 模型加载器 / 模型加载开关 ----
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
  signal: { zh: "信号", en: "Signal" },
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
  action: { zh: ["加载", "卸载"], en: ["Load", "Unload"] },
  compact_view: { zh: ["完整", "简洁"], en: ["Full", "Compact"] },
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
  ZouyuModelSwitch: { zh: "模型加载开关", en: "Model Load Switch" },
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
  ZouyuModelSwitch: { color: "#8a6d1f", bgcolor: "#2b2310" },
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

  if (node.comfyClass === "ZouyuModelSwitch") {
    applySwitchSubtitle(node);
    refreshSwitchModelCombo(node);
  }

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
// 模型加载器 / 模型加载开关：文件夹选择、文件刷新、红/绿/蓝状态灯 + 状态文字
// ---------------------------------------------------------------------------

const MAX_MODELS = 8;
const TYPE_KEYS = { 主模型: "main", 文本模型: "clip", 视频VAE: "vae", 音频VAE: "avae", LoRA: "lora", 其他: "other" };
const TYPE_CATEGORY = { 主模型: "diffusion_models", 文本模型: "text_encoders", 视频VAE: "vae", 音频VAE: "vae", LoRA: "loras", 其他: "diffusion_models" };
// 端口名 = 类型 + 序号（如 主模型0 / Main0），避免与行尾状态灯/文字重叠；
// 完整分类（Diffusion/CLIP/VAE 等）与位置提示由灯悬停 tooltip 展示
const TYPE_DISPLAY = {
  main: { zh: "主模型", en: "Main", clsZh: "Diffusion", clsEn: "Diffusion" },
  clip: { zh: "文本模型", en: "Text", clsZh: "CLIP", clsEn: "CLIP" },
  vae: { zh: "视频VAE", en: "VideoVAE", clsZh: "VAE", clsEn: "VAE" },
  avae: { zh: "音频VAE", en: "AudioVAE", clsZh: "VAE", clsEn: "VAE" },
  lora: { zh: "LoRA", en: "LoRA", clsZh: "LoRA", clsEn: "LoRA" },
  other: { zh: "其他", en: "Other", clsZh: "自动识别", clsEn: "Auto" },
};
const DETECT_DISPLAY = {
  unet: { zh: "主模型", en: "Main", clsZh: "Diffusion", clsEn: "Diffusion" },
  checkpoint: { zh: "主模型", en: "Main", clsZh: "Checkpoint", clsEn: "Checkpoint" },
  clip: { zh: "文本模型", en: "Text", clsZh: "CLIP", clsEn: "CLIP" },
  vae: { zh: "视频VAE", en: "VideoVAE", clsZh: "VAE", clsEn: "VAE" },
  avae: { zh: "音频VAE", en: "AudioVAE", clsZh: "VAE", clsEn: "VAE" },
  lora: { zh: "LoRA", en: "LoRA", clsZh: "LoRA", clsEn: "LoRA" },
  other: { zh: "其他", en: "Other", clsZh: "", clsEn: "" },
  unknown: { zh: "未知", en: "Unknown", clsZh: "", clsEn: "" },
};

function makePortName(zh, d, ordinal) {
  return `${zh ? d.zh : d.en}${ordinal}`;
}

// 状态文字保持简短（避免与下拉内文件名文字重叠），位置提示由灯 tooltip 展示
const STATE_INFO = {
  gpu: { zh: "已加载", en: "Loaded", color: "#4caf50" },
  cpu: { zh: "未加载", en: "Not loaded", color: "#2196f3" },
  free: { zh: "已卸载", en: "Unloaded", color: "#f44336" },
  unknown: { zh: "未知", en: "Unknown", color: "#9e9e9e" },
};

function stateText(state, lang) {
  const e = STATE_INFO[state] || STATE_INFO.unknown;
  return lang !== "English" ? e.zh : e.en;
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
      // 行尾/画布：三色灯 + 状态文字（绿=已加载(显存) 蓝=未加载(内存) 红=已卸载(硬盘)）
      info.color = st.color;
      info.state = m?.state || "unknown";
      info.zh = st.zh;
      info.en = st.en;
      if (info.el) info.el.style.background = st.color;
      if (info.textEl) info.textEl.textContent = zh ? st.zh : st.en;
      if (info.tag && m?.type) {
        info.tag.textContent = zh ? m.type_zh : m.type_en;
      }
    }
    if (node.comfyClass === "ZouyuModelSwitch") {
      // 开关节点的标题副文案由 action 决定，无需在此刷新；模型下拉由 loaders 执行事件刷新
    }
    if (node.comfyClass === "ZouyuModelLoader") {
      // 执行后：按后端识别类型刷新端口名（类型 + 序号 + 分类，如 主模型0 (Diffusion)）
      const ord = {};
      for (let i = 0; i < MAX_MODELS; i++) {
        const out = node.outputs && node.outputs[i];
        if (!out || !out.name) continue;
        const m = byKind[`slot${i}`];
        if (!m || !m.type) continue;
        const d = DETECT_DISPLAY[m.type] || DETECT_DISPLAY.other;
        ord[m.type] = (ord[m.type] || 0) + 1;
        out.name = makePortName(zh, d, ord[m.type] - 1);
        out.label = "";
      }
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
.zouyu-row-tail{position:absolute;right:74px;display:flex;align-items:center;gap:6px;pointer-events:none;transform:translateY(-50%)}
.zouyu-rt-light{width:12px;height:12px;border-radius:50%;border:1px solid rgba(0,0,0,.45);box-shadow:0 0 4px rgba(0,0,0,.5);display:inline-block;background:#9e9e9e}
.zouyu-rt-text{font-size:10px;line-height:14px;color:#9a9a9a;white-space:nowrap}
.zouyu-rt-type{font-size:10px;line-height:14px;color:#c9a05f;background:rgba(0,0,0,.4);padding:0 4px;border-radius:3px;white-space:nowrap}
.zouyu-rt-folder{font-size:11px;line-height:15px;border:none;background:rgba(0,0,0,.4);border-radius:3px;padding:0 4px;color:#ddd;cursor:pointer;pointer-events:auto}
.zouyu-rt-folder:hover{background:rgba(0,0,0,.6)}
.zouyu-import-bar{position:absolute;left:8px;right:8px;bottom:3px;height:16px;font-size:10px;line-height:16px;text-align:center;color:#9a9a9a;background:rgba(0,0,0,.28);border-radius:3px;cursor:pointer;pointer-events:auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.zouyu-import-bar:hover{background:rgba(0,0,0,.45)}
.zouyu-import-bar.dragover{background:rgba(76,175,80,.4);color:#fff}
/* Vue 模式：槽位行下的 文件夹/加载·卸载 按钮（旧版 Canvas 用真实按钮控件，见 setupLegacyLoaderLayout） */
.zouyu-row-btns{position:absolute;left:8px;display:flex;gap:5px;align-items:center;z-index:12;transform:translateY(-50%);pointer-events:auto}
.zouyu-btn{font-size:10px;line-height:14px;border:1px solid rgba(255,255,255,.18);background:rgba(0,0,0,.45);color:#ddd;border-radius:4px;padding:1px 6px;cursor:pointer;white-space:nowrap}
.zouyu-btn:hover{background:rgba(255,255,255,.15);color:#fff}
/* 模型文件夹浏览器弹窗（功能九：点击下拉下方的 📁 打开 models 目录树） */
.zouyu-fb-overlay{position:fixed;inset:0;z-index:99998;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center}
.zouyu-fb-overlay.hidden{display:none}
.zouyu-fb-panel{width:480px;max-width:92vw;max-height:70vh;background:#232323;border:1px solid #4a4a4a;border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,.6);display:flex;flex-direction:column;font-size:12px;color:#ddd;overflow:hidden}
.zouyu-fb-head{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid #3a3a3a}
.zouyu-fb-title{font-weight:600;color:#4fff8f;flex:1}
.zouyu-fb-close{border:none;background:none;color:#aaa;font-size:14px;cursor:pointer;padding:2px 6px}
.zouyu-fb-close:hover{color:#fff}
.zouyu-fb-crumb{display:flex;flex-wrap:wrap;gap:4px;padding:8px 14px;border-bottom:1px solid #333;align-items:center}
.zouyu-fb-crumb span{color:#8fd0ff;cursor:pointer;padding:2px 4px;border-radius:3px}
.zouyu-fb-crumb span:hover{background:rgba(0,0,0,.4)}
.zouyu-fb-crumb .sep{color:#666;cursor:default}
.zouyu-fb-crumb .sep:hover{background:none}
.zouyu-fb-list{flex:1;overflow:auto;padding:6px 8px;min-height:120px}
.zouyu-fb-item{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:5px;cursor:pointer;color:#ccc}
.zouyu-fb-item:hover{background:rgba(255,255,255,.08);color:#fff}
.zouyu-fb-item .dir-ico{color:#e8c14a}
.zouyu-fb-empty{padding:18px;text-align:center;color:#888}
.zouyu-fb-foot{display:flex;gap:8px;padding:10px 14px;border-top:1px solid #3a3a3a;justify-content:flex-end;align-items:center}
.zouyu-fb-foot button{border:1px solid #4a4a4a;background:#2e2e2e;color:#ddd;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:12px}
.zouyu-fb-foot button:hover{background:#3a3a3a}
.zouyu-fb-foot .zouyu-fb-ok{background:#2e7d4f;border-color:#3a9a63;color:#fff}
.zouyu-fb-foot .zouyu-fb-ok:hover{background:#3a9a63}
.zouyu-fb-foot .zouyu-fb-reveal{margin-right:auto}
.zouyu-fb-current{padding:6px 14px;font-size:11px;color:#9a9a9a;border-bottom:1px solid #333}
.zouyu-fb-current b{color:#c9a05f;font-weight:600}
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

/** 可见槽位数：0..最后已填槽位+1（选好模型后自动弹出下一个，最多 8 个）。
 *  「已填」= 已选模型文件 且 槽位类型不是「未使用」（旧工作流残留值不算，防止全部显示）。 */
function visibleSlotCount(st) {
  let lastFilled = -1;
  for (let i = 0; i < MAX_MODELS; i++) {
    const s = st.slots[i];
    if (!s) continue;
    const filled = !!s.name && s.name !== "(未选择)" && s.name !== "(无文件)" && s.type !== "未使用";
    if (filled) lastFilled = i;
  }
  return Math.min(Math.max(lastFilled + 2, 1), MAX_MODELS);
}

/** 刷新某槽位文件下拉选项：未选文件夹时显示全部模型（请加载模型），选了文件夹按文件夹过滤。 */
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
  // 未选文件夹 → 全量模型列表；已选文件夹 → 该文件夹下的模型文件
  const category = s.folder ? (TYPE_CATEGORY[s.type] || "diffusion_models") : "all";
  const folder = s.folder || ".";
  try {
    const d = await fetchJson(
      `/zouyu_model_loader/files?category=${encodeURIComponent(category)}&folder=${encodeURIComponent(folder)}`
    );
    s.files = (d.files || []).length ? d.files : [];
    w.options = w.options || {};
    // 选项始终包含「(未选择)」占位符，避免下拉当前值不在选项中而被框架自动改成第一个模型
    w.options.values = ["(未选择)", ...(s.files.length ? s.files : [])];
    // 未选择（占位符）保持不动，避免把下拉自动填成第一个文件；
    // 只有当前值已失效时才回退到第一个可用文件
    if (w.value !== "(未选择)" && w.value !== "(无文件)" && !s.files.includes(w.value)) {
      w.value = s.files[0] || "(未选择)";
    }
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

/** 槽位 i 的名字控件在可见控件网格中的行序号（type/folder 已隐藏，每槽位仅 1 个控件行）。 */
function visibleRowIndex(node, i) {
  const st = node.__zouyuSlotState;
  if (!st) return i;
  const count = visibleSlotCount(st);
  let idx = 0;
  for (let j = 0; j < i; j++) {
    if (j < count) idx += 1;
  }
  return idx;
}

/** 行尾叠加：三色状态灯 + 状态文字（与下拉同一水平线，位于最右端口旁）。
 *  未选模型=灰色「未选择」；已选模型=按状态三色。 */
function updateRowTail(node, overlay, i, rowCY, visible, zh) {
  let tail = overlay.querySelector(`[data-zouyu-tail="${i}"]`);
  if (!visible) {
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
  const st2 = node.__zouyuSlotState;
  const s = st2 && st2.slots[i];
  const filled = !!(s && s.name && s.name !== "(未选择)" && s.name !== "(无文件)" && s.type !== "未使用");
  const info = node.__zouyuStatus && node.__zouyuStatus[`slot${i}`];
  const stInfo = filled ? (STATE_INFO[(info && info.state) || "unknown"] || STATE_INFO.unknown) : null;
  const light = document.createElement("span");
  light.className = "zouyu-rt-light";
  light.title = zh ? "绿=已加载(显存) 蓝=未加载(内存) 红=已卸载(硬盘)" : "Green=VRAM Blue=RAM Red=Disk";
  light.style.background = stInfo ? stInfo.color : "#9e9e9e";
  tail.appendChild(light);
  const textEl = document.createElement("span");
  textEl.className = "zouyu-rt-text";
  textEl.textContent = filled
    ? (zh ? (info && info.zh) || "未知" : (info && info.en) || "Unknown")
    : (zh ? "未选择" : "Not chosen");
  tail.appendChild(textEl);
  if (node.__zouyuStatus) node.__zouyuStatus[`slot${i}`] = { el: light, textEl, tag: null, dotOnly: true };
}

/** Vue 模式：槽位行下的按钮行（📁 模型文件夹 + ⤓加载/⤒卸载）。旧版 Canvas 模式用真实按钮控件。 */
function updateRowButtons(node, overlay, i, rowCY, visible, zh) {
  const st = node.__zouyuSlotState;
  const s = st && st.slots[i];
  const filled = !!(s && s.name && s.name !== "(未选择)" && s.name !== "(无文件)" && s.type !== "未使用");
  // 至少选过一个模型、且精简显示开关打开时，所有可见下拉都显示按钮（与 applySlotVisibility 一致）
  let anyFilled = false;
  for (let j = 0; j < MAX_MODELS; j++) {
    const sj = st && st.slots[j];
    if (sj && sj.name && sj.name !== "(未选择)" && sj.name !== "(无文件)" && sj.type !== "未使用") { anyFilled = true; break; }
  }
  const showBtn = visible && (filled || anyFilled) && !!st.compactView;
  let bar = overlay.querySelector(`[data-zouyu-btns="${i}"]`);
  if (!showBtn) {
    if (bar) bar.remove();
    return;
  }
  if (!bar) {
    bar = document.createElement("div");
    bar.className = "zouyu-row-btns";
    bar.setAttribute("data-zouyu-btns", String(i));
    overlay.appendChild(bar);
    const folderBtn = document.createElement("button");
    folderBtn.type = "button";
    folderBtn.className = "zouyu-btn";
    folderBtn.setAttribute("data-zouyu-folder", String(i));
    folderBtn.addEventListener("click", (e) => { e.stopPropagation(); openFolderBrowser(node, i); });
    bar.appendChild(folderBtn);
  }
  const folderBtn = bar.querySelector(`[data-zouyu-folder="${i}"]`);
  if (folderBtn) folderBtn.textContent = zh ? "📁 模型文件夹" : "📁 Model Folder";
  bar.style.top = (rowCY != null ? rowCY : 12) + "px";
}

/** 布局一次：端口点平移到下拉行 + 行尾灯/文字/标签 定位（节点 DOM 内，自动跟随节点）。 */
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
  const zh = st.lang !== "English";
  const rowH = measureRowHeight(node);
  for (let i = 0; i < MAX_MODELS; i++) {
    const visible = i < count;
    // 行锚点：优先用控件行 DOM（type/folder 隐藏后端口索引不可靠），端口点作兜底
    let rowCY = null;
    const rows = el.querySelectorAll('[data-testid="node-widget"]');
    const row = rows[visibleRowIndex(node, i)];
    if (row) {
      const rr = row.getBoundingClientRect();
      rowCY = rr.top + rr.height / 2 - nodeRect.top;
    }
    if (rowCY == null) {
      const anchor = findSlotEl(el, `-in-${3 * i + 2}`);
      if (anchor) {
        const r = anchor.getBoundingClientRect();
        rowCY = r.top + r.height / 2 - nodeRect.top;
      } else if (rowH) {
        rowCY = (visibleRowIndex(node, i) + 0.5) * rowH;
      }
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
    updateRowTail(node, overlay, i, rowCY, visible, zh);
    updateRowButtons(node, overlay, i, rowCY, visible, zh);
  }
  // 精简显示开关：关闭（简洁）时隐藏底部导入条
  const ib = overlay.querySelector(".zouyu-import-bar");
  if (ib) ib.style.display = st.compactView ? "" : "none";
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

/** 节点级拖入文件夹（两种渲染模式通用：canvas 画布也走 app 级 drop 分发）。 */
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

function isVueMode() {
  return typeof LiteGraph !== "undefined" && !!LiteGraph.vueNodesMode;
}

/** 把当前控件值同步回状态（工作流加载/配置后控件值才恢复，需要重新读取）。 */
function syncStateFromWidgets(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    // 类型：「未使用」=空槽位（即使 name 有残留值也不显示/不加载，兼容旧工作流）；
    // 其它明确类型或「其他」（自动识别）视为已启用；旧工作流中用户明确选过类型则尊重
    const tv = w.type?.value;
    st.slots[i].type = (tv === "未使用") ? "未使用" : ((tv && tv !== "未使用") ? tv : "其他");
    st.slots[i].folder = w.folder?.value || "";
    st.slots[i].name = w.name?.value || "(未选择)";
    // 「未使用」槽位强制清空残留 name（旧工作流曾被自动填充的模型名），
    // 确保这类槽位既不显示也不参与加载
    if (st.slots[i].type === "未使用") {
      st.slots[i].name = "(未选择)";
      st.slots[i].folder = "";
      if (w.name && w.name.value !== "(未选择)") w.name.value = "(未选择)";
    }
  }
  st.lowVram = !!node.widgets?.find((w) => w.name === "low_vram_mode")?.value;
  st.compactView = !!node.widgets?.find((w) => w.name === "compact_view")?.value;
  st.lang = node.widgets?.find((w) => w.name === "language")?.value || getLang();
}

/** 布局分派：Vue 节点模式用 DOM 叠加层；旧版 Canvas 模式（默认）用 output.pos + 画布绘制。 */
function applyLoaderLayout(node) {
  if (isVueMode()) {
    requestAnimationFrame(() => {
      layoutLoaderOverlay(node);
      forceSlotResync(node);
    });
    setTimeout(() => {
      layoutLoaderOverlay(node);
      forceSlotResync(node);
    }, 60);
  } else {
    layoutLegacyLoader(node);
  }
}

// ===========================================================================
// 旧版 Canvas 渲染模式（LiteGraph.vueNodesMode=false，ComfyUI 默认）：
// - 端口：node.outputs[i].pos 硬编码到对应下拉行（画布原生尊重 pos）
// - 灯/📁/类型标签：node.onDrawForeground 画布绘制（自动跟随节点位置与大小）
// - 📁 点击：node.onMouseDown 命中检测
// - 节点尺寸：行数变化时按 computeSize() 收缩/伸长
// ===========================================================================

/** 旧版布局：端口对齐各行 + 行数变化时自动伸缩节点。 */
function layoutLegacyLoader(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  try { node.arrange(); } catch (e) { /* 忽略 */ }
  const count = visibleSlotCount(st);
  const H = LiteGraph.NODE_WIDGET_HEIGHT;
  // 下拉控件宽度收窄，行尾留出空间给 状态文字+三色灯+端口名（避免重叠）
  const namePad = 150;
  for (let i = 0; i < MAX_MODELS; i++) {
    const nameW = slotWidgets(node, i).name;
    if (nameW) nameW.width = Math.max(60, node.size[0] - namePad);
  }
  for (let i = 0; i < MAX_MODELS; i++) {
    const out = node.outputs && node.outputs[i];
    if (!out) continue;
    const s = st.slots[i];
    // 仅已选模型的槽位显示端口（与 syncLoaderOutputs 一致）
    const used = i < count && !!s && s.type !== "未使用" && !!s.name
      && s.name !== "(未选择)" && s.name !== "(无文件)";
    if (used) {
      const nameW = slotWidgets(node, i).name;
      const rowY = slotRowCenterY(nameW, H); // 端口与下拉同一水平线（行中心）
      out.pos = [node.size[0] - 8, rowY];
      if (!node.__zouyuStatus[`slot${i}`]) node.__zouyuStatus[`slot${i}`] = { color: "#9e9e9e" };
    } else {
      out.pos = [-500, -500]; // 未使用的端口移出画布（不再堆在顶部）
    }
  }
  // 节点尺寸随可见行数自动收缩/伸长
  if (count !== node.__zouyuLastCount) {
    node.__zouyuLastCount = count;
    try {
      const h = node.computeSize()[1];
      if (Math.abs(h - node.size[1]) > 2) node.setSize([node.size[0], h]);
    } catch (e) { /* 忽略 */ }
  }
}

/** 槽位下拉行的垂直中心（相对节点顶部）：优先用控件实际高度 computedHeight，
 *  避免 NODE_WIDGET_HEIGHT 与真实行高不一致导致灯/端口偏离下拉中线。 */
function slotRowCenterY(nameW, H) {
  if (!nameW) return H / 2;
  const h = (nameW.computedHeight && nameW.computedHeight > 0) ? nameW.computedHeight : H;
  return (nameW.y != null ? nameW.y : 0) + h / 2;
}

/** 画布绘制：行尾 = 三色状态灯 + 状态文字（与下拉同一水平线）。
 *  布局：下拉 → 状态文字（右对齐 W-80）→ 灯（W-66）→ 端口名（W-56 起，类型+序号）→ 端口点（W-8）。
 *  每个可见下拉行都绘制灯：未选模型=灰色「未选择」，已选模型=按状态三色。 */
function drawLegacyOverlay(node, ctx) {
  const st = node.__zouyuSlotState;
  if (!st || !ctx) return;
  const zh = st.lang !== "English";
  const count = visibleSlotCount(st);
  const H = LiteGraph.NODE_WIDGET_HEIGHT;
  const W = node.size[0];
  ctx.save();
  ctx.textBaseline = "middle";
  for (let i = 0; i < MAX_MODELS; i++) {
    if (i >= count) break;
    const s = st.slots[i];
    if (!s) continue;
    const nameW = slotWidgets(node, i).name;
    const y = slotRowCenterY(nameW, H);
    const filled = !!s.name && s.name !== "(未选择)" && s.name !== "(无文件)" && s.type !== "未使用";
    const info = node.__zouyuStatus && node.__zouyuStatus[`slot${i}`];
    let color, text;
    if (!filled) {
      color = "#9e9e9e";
      text = zh ? "未选择" : "Not chosen";
    } else {
      color = (info && info.color) || "#9e9e9e";
      text = zh ? (info && info.zh) || "未知" : (info && info.en) || "Unknown";
    }
    // 状态文字（右对齐，端口名左侧）
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#c8c8c8";
    ctx.textAlign = "right";
    ctx.fillText(text, W - 86, y);
    ctx.textAlign = "left";
    // 三色灯（状态文字与端口名之间）
    ctx.beginPath();
    ctx.arc(W - 70, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,.5)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();
}

/** 旧版点击处理：📁/加载/卸载 均为真实按钮控件（原生点击），无需画布命中检测。 */
function legacyMouseDown(node, pos) {
  return false;
}

/** 旧版布局初始化：固定控件起始行（与端口位置解耦）+ 画布绘制 + 点击 + 底部导入条按钮。 */
function setupLegacyLoaderLayout(node) {
  // 固定控件起始行：使 widget.y 不受输出端口 pos 影响（否则端口对齐会反作用于布局）
  node.widgets_start_y = 30;
  // 每个槽位的下拉下方：模型文件夹选择开关（真实按钮，选中模型后显示）
  // 说明：模型的加载/卸载由「模型加载控制开关」的信号自动控制，这里不提供手动加载按钮
  for (let i = 0; i < MAX_MODELS; i++) {
    const folderBtn = addButton(node, "📁 模型文件夹", "📁 Model Folder", () => openFolderBrowser(node, i));
    folderBtn.__zouyuSlotBtn = `folder_${i}`;
  }
  // 把文件夹按钮移动到对应 name 下拉之后（数组顺序决定控件行顺序：下拉 → 📁 文件夹）
  for (let i = 0; i < MAX_MODELS; i++) {
    const nameW = slotWidgets(node, i).name;
    const folderBtn = node.widgets?.find((w) => w.__zouyuSlotBtn === `folder_${i}`);
    if (!nameW || !folderBtn) continue;
    const bi = node.widgets.indexOf(folderBtn);
    if (bi >= 0) node.widgets.splice(bi, 1);
    const ni = node.widgets.indexOf(nameW);
    node.widgets.splice(ni + 1, 0, folderBtn);
  }
  node.onDrawForeground = (ctx) => {
    layoutLegacyLoader(node); // 每帧刷新端口 X（跟随节点宽度变化）
  };
  // ComfyUI 绘制顺序：onDrawForeground → drawSlots → drawWidgets（控件）。
  // 控件（combo 全宽背景）会盖住 onDrawForeground 里画的内容，
  // 因此把三色灯/状态文字移到 drawWidgets 之后绘制，保证灯可见于控件之上。
  const origDrawWidgets = node.drawWidgets ? node.drawWidgets.bind(node) : null;
  node.drawWidgets = function (ctx, opts) {
    const r = origDrawWidgets ? origDrawWidgets(ctx, opts) : undefined;
    try {
      drawLegacyOverlay(node, ctx);
    } catch (e) { /* 绘制异常不影响控件 */ }
    return r;
  };
  node.onMouseDown = (e, pos) => legacyMouseDown(node, pos);
  // 底部"拖入文件夹导入"按钮（画布原生可点击，占用一行）
  const bar = addButton(node, "📥 拖入文件夹导入模型", "📥 Drop folder to import", () => pickAndImportFolder(node));
  bar.__zouyuImportBar = true;
  applySlotVisibility(node);
  setTimeout(() => {
    layoutLegacyLoader(node);
    app.graph?.setDirtyCanvas(true, true);
  }, 60);
}

/** 按当前状态应用控件显隐 + 行布局 + 端口对齐（自动分派 Vue/旧版Canvas 两种布局）。 */
function applySlotVisibility(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const count = visibleSlotCount(st);
  // 是否已选过至少一个模型：选过之后，所有可见下拉（含新出现的空下拉）都显示下方的文件夹/加载按钮
  let anyFilled = false;
  for (let j = 0; j < MAX_MODELS; j++) {
    const sj = st.slots[j];
    if (sj && sj.name && sj.name !== "(未选择)" && sj.name !== "(无文件)" && sj.type !== "未使用") { anyFilled = true; break; }
  }
  // 精简显示开关：关闭（简洁）= 隐藏文件夹/加载按钮与导入条，只保留下拉/灯/提示/端口/低显存/语言/本开关
  const showActions = !!st.compactView;
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    const visible = i < count;
    const s = st.slots[i] || (st.slots[i] = { type: "其他", folder: "", name: "(未选择)", files: [] });
    // 「未使用」槽位兜底：强制清空残留 name，避免旧工作流自动填充的模型名被显示/加载
    if (s.type === "未使用" && s.name !== "(未选择)" && s.name !== "(无文件)") {
      s.name = "(未选择)";
      if (w.name) w.name.value = "(未选择)";
    }
    const filled = !!s.name && s.name !== "(未选择)" && s.name !== "(无文件)" && s.type !== "未使用";
    // 简洁模式：类型/文件夹控件隐藏（类型自动识别、文件夹用行下按钮），只显示「请加载模型」文件下拉
    setWidgetHidden(w.type, true);
    setWidgetHidden(w.folder, true);
    setWidgetHidden(w.name, !visible);
    // 行下文件夹按钮：至少选过一个模型、且精简显示开关打开时显示
    const showBtn = visible && (filled || anyFilled) && showActions;
    const folderBtn = node.widgets?.find((x) => x.__zouyuSlotBtn === `folder_${i}`);
    if (folderBtn) setWidgetHidden(folderBtn, !showBtn);
    if (w.type && w.type.value !== s.type) w.type.value = s.type;
    if (w.folder && w.folder.value !== s.folder) w.folder.value = s.folder;
    if (w.name && w.name.value !== s.name) w.name.value = s.name;
  }
  // 集成模式开关不再需要（简洁模式为默认且唯一），隐藏
  const compactW = node.widgets?.find((w) => w.name === "compact_mode");
  if (compactW) setWidgetHidden(compactW, true);
  // 底部拖入导入条：精简模式下隐藏
  const importBar = node.widgets?.find((w) => w.__zouyuImportBar);
  if (importBar) setWidgetHidden(importBar, !showActions);
  syncLoaderOutputs(node);
  applyLoaderLayout(node);
  fitNodeHeight(node); // 显隐变化后节点高度自动收缩（简洁模式）
  app.graph?.setDirtyCanvas(true, true);
}

/** 简洁模式下节点高度自动收缩：框架 _arrangeWidgets 只在内容超高时撑大节点、不会缩小，
 *  这里基于可见控件计算内容高度并 setSize（在 applySlotVisibility 显隐变化后调用）。 */
function fitNodeHeight(node) {
  if (isVueMode() || !node.widgets) return;
  try {
    const H = LiteGraph.NODE_WIDGET_HEIGHT;
    let l = node.widgets_start_y != null ? node.widgets_start_y : 30;
    let any = false;
    for (const w of node.widgets) {
      if (w.hidden) continue;
      any = true;
      l += (w.computedHeight && w.computedHeight > 0 ? w.computedHeight : H) + 4;
    }
    if (!any) return;
    l += 6;
    if (Math.abs(node.size[1] - l) > 2) {
      node.setSize([node.size[0], l]);
    }
  } catch (e) { /* ignore */ }
}

/** 输出端口与模型一一对应：按类型编号命名（主模型0 (Diffusion)/文本模型0 (CLIP)/...），端口文字显示类型+分类。 */
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
    node.outputs[i].name = makePortName(zh, TYPE_DISPLAY[tkey] || TYPE_DISPLAY.other, ordinals[tkey] - 1);
    node.outputs[i].label = "";
    node.outputs[i].type = "*";
  }
  app.graph?.setDirtyCanvas(true, false);
}

/** 设置各槽位控件的标签（简洁模式：只显示「请加载模型」文件下拉）。 */
function applyLoaderLabels(node) {
  const st = node.__zouyuSlotState;
  if (!st) return;
  const zh = st.lang !== "English";
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    if (w.name) w.label = zh ? `请加载模型 ${i + 1}` : `Load model ${i + 1}`;
  }
}

/**
 * 模型文件夹浏览器弹窗（功能九）：
 * 点击下拉下方的「📁 模型文件夹」打开，逐级浏览 models 目录树，
 * 选定文件夹后自动刷新该下拉中的模型文件列表；也可一键在系统资源管理器中打开 models 目录。
 */
let _folderBrowser = null;

function openFolderBrowser(node, slotIndex) {
  const st = node.__zouyuSlotState;
  const s = st && st.slots[slotIndex];
  if (!s) return;
  if (!_folderBrowser) _folderBrowser = createFolderBrowser();
  _folderBrowser.open(node, slotIndex);
}

function createFolderBrowser() {
  const overlay = document.createElement("div");
  overlay.className = "zouyu-fb-overlay hidden";
  overlay.innerHTML = `
    <div class="zouyu-fb-panel">
      <div class="zouyu-fb-head">
        <span class="zouyu-fb-title">📁 模型文件夹</span>
        <button class="zouyu-fb-close">✕</button>
      </div>
      <div class="zouyu-fb-crumb"></div>
      <div class="zouyu-fb-current"></div>
      <div class="zouyu-fb-list"></div>
      <div class="zouyu-fb-foot">
        <button class="zouyu-fb-reveal">📂 打开 models 目录</button>
        <button class="zouyu-fb-cancel">取消</button>
        <button class="zouyu-fb-ok">选择此文件夹</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const titleEl = overlay.querySelector(".zouyu-fb-title");
  const crumbEl = overlay.querySelector(".zouyu-fb-crumb");
  const curEl = overlay.querySelector(".zouyu-fb-current");
  const listEl = overlay.querySelector(".zouyu-fb-list");
  const state = { node: null, slot: -1, path: "" };

  const close = () => overlay.classList.add("hidden");

  const setLang = () => {
    const zh = state.node && state.node.__zouyuLang !== "English";
    titleEl.textContent = zh ? "📁 模型文件夹" : "📁 Model Folder";
    overlay.querySelector(".zouyu-fb-reveal").textContent = zh ? "📂 打开 models 目录" : "📂 Open models folder";
    overlay.querySelector(".zouyu-fb-cancel").textContent = zh ? "取消" : "Cancel";
    overlay.querySelector(".zouyu-fb-ok").textContent = zh ? "选择此文件夹" : "Choose folder";
  };

  const renderCrumb = () => {
    crumbEl.innerHTML = "";
    const root = document.createElement("span");
    root.textContent = "models";
    root.addEventListener("click", () => navigate(""));
    crumbEl.appendChild(root);
    const parts = state.path ? state.path.split("/") : [];
    let acc = "";
    for (const p of parts) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = "/";
      crumbEl.appendChild(sep);
      acc = acc ? acc + "/" + p : p;
      const el = document.createElement("span");
      el.textContent = p;
      el.addEventListener("click", () => navigate(acc));
      crumbEl.appendChild(el);
    }
  };

  const navigate = async (path) => {
    state.path = path || "";
    renderCrumb();
    curEl.innerHTML = "";
    const zh = state.node && state.node.__zouyuLang !== "English";
    const cur = document.createElement("span");
    cur.innerHTML = (zh ? "当前：" : "Current: ")
      + `<b>models/${path || ""}</b>`;
    curEl.appendChild(cur);
    listEl.innerHTML = "";
    try {
      const d = await fetchJson(`/zouyu_model_loader/list_dirs?path=${encodeURIComponent(path || ".")}`);
      const dirs = d.dirs || [];
      if (!dirs.length) {
        const empty = document.createElement("div");
        empty.className = "zouyu-fb-empty";
        empty.textContent = zh ? "（此文件夹下没有子目录）" : "(no subfolders)";
        listEl.appendChild(empty);
      } else {
        for (const rel of dirs) {
          const item = document.createElement("div");
          item.className = "zouyu-fb-item";
          item.innerHTML = `<span class="dir-ico">📁</span><span></span>`;
          item.querySelector("span:last-child").textContent = rel.split("/").pop();
          item.addEventListener("click", () => navigate(rel));
          listEl.appendChild(item);
        }
      }
    } catch (e) {
      const empty = document.createElement("div");
      empty.className = "zouyu-fb-empty";
      empty.textContent = "error: " + e;
      listEl.appendChild(empty);
    }
  };

  overlay.querySelector(".zouyu-fb-close").addEventListener("click", close);
  overlay.querySelector(".zouyu-fb-cancel").addEventListener("click", close);
  overlay.querySelector(".zouyu-fb-ok").addEventListener("click", () => {
    const s = state.node && state.node.__zouyuSlotState && state.node.__zouyuSlotState.slots[state.slot];
    if (!s) return;
    s.folder = state.path || "";
    const fw = slotWidgets(state.node, state.slot).folder;
    if (fw) fw.value = s.folder;
    refreshSlotFileOptions(state.node, state.slot);
    close();
  });
  overlay.querySelector(".zouyu-fb-reveal").addEventListener("click", async () => {
    try {
      // 在系统文件资源管理器中打开 models 目录
      const st = await fetchJson("/zouyu_model_loader/status");
      const root = st.models_root || "";
      await fetchJson(`/zouyu_model_loader/reveal?path=${encodeURIComponent(root)}`);
    } catch (e) { /* ignore */ }
  });
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });

  return {
    open(node, slot) {
      state.node = node;
      state.slot = slot;
      const s = node.__zouyuSlotState && node.__zouyuSlotState.slots[slot];
      state.path = (s && s.folder) || "";
      setLang();
      overlay.classList.remove("hidden");
      navigate(state.path);
    },
  };
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

  // 从 schema 原生控件读取状态（类型默认「其他」= 自动识别）
  const st = { lowVram: false, lang: getLang(), slots: {} };
  for (let i = 0; i < MAX_MODELS; i++) {
    const w = slotWidgets(node, i);
    const tv = w.type?.value;
    st.slots[i] = {
      type: (tv && tv !== "未使用") ? tv : "其他",
      folder: w.folder?.value || "",
      name: w.name?.value || "(未选择)",
      files: [],
    };
  }
  st.lowVram = !!node.widgets?.find((w) => w.name === "low_vram_mode")?.value;
  st.compactView = !!node.widgets?.find((w) => w.name === "compact_view")?.value;
  st.lang = node.widgets?.find((w) => w.name === "language")?.value || getLang();
  node.__zouyuSlotState = st;
  node.__zouyuLang = st.lang;
  node.__zouyuStatus = {};
  // 加入 2.5s 状态轮询：开关节点卸载/加载后，无需加载器重跑即可刷新灯与文字
  statusNodes.add(node);
  startStatusPolling();

  // 挂接回调（包装原有回调，保证值仍然流向后端）
  const configDirty = () => {
    // 配置变化 → 先推送给后端（register_config 完成后），再刷新图中所有开关的下拉（避免竞态）
    Promise.resolve(pushLoaderConfig(node)).then(() => refreshAllSwitchCombos());
  };
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
      configDirty();
    };
    if (w.name) {
      const origNameCb = w.name.callback;
      w.name.callback = (v) => {
        if (origNameCb) origNameCb.call(w.name, v);
        s.name = v;
        applySlotVisibility(node);
        configDirty();
      };
    }
    if (w.folder) {
      const origFolderCb = w.folder.callback;
      w.folder.callback = (v) => {
        if (origFolderCb) origFolderCb.call(w.folder, v);
        s.folder = v;
        refreshSlotFileOptions(node, i);
        configDirty();
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
  // 精简显示开关：关闭（简洁）= 只保留下拉/灯/提示/端口/低显存/语言/本开关；打开（完整）= 显示文件夹/加载按钮与导入条
  const cvW = node.widgets.find((w) => w.name === "compact_view");
  if (cvW) {
    const orig = cvW.callback;
    cvW.callback = (v) => {
      if (orig) orig.call(cvW, v);
      st.compactView = !!v;
      applySlotVisibility(node);
    };
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
  // 只刷新可见槽位的文件下拉（避免多余请求）
  const visCnt = visibleSlotCount(st);
  for (let i = 0; i < visCnt; i++) {
    const s = st.slots[i];
    if (s && s.type && s.type !== "未使用") refreshSlotFileOptions(node, i);
  }
  refreshStatusDOM(node);

  // 拖入文件夹导入（两种渲染模式通用）
  setupLoaderDragDrop(node);

  // 工作流加载：控件值在 configure 之后才恢复，配置完成后重新同步状态并布局
  const origCfg = node.onAfterGraphConfigured;
  node.onAfterGraphConfigured = function (...args) {
    try {
      syncStateFromWidgets(node);
      applySlotVisibility(node);
      applyLoaderLabels(node);
      pushLoaderConfig(node);
      refreshAllSwitchCombos();
    } catch (e) { /* ignore */ }
    const r = origCfg ? origCfg.apply(this, args) : undefined;
    return r;
  };

  // 布局初始化：Vue 节点模式（叠加层）或 旧版 Canvas 模式（output.pos + 画布绘制）
  if (isVueMode()) {
    const layoutPass = () => {
      layoutLoaderOverlay(node);
      forceSlotResync(node);
    };
    setTimeout(layoutPass, 0);
    setTimeout(layoutPass, 120);
    setTimeout(layoutPass, 400);
    setTimeout(() => watchLoaderOverlay(node), 600);
  } else {
    setupLegacyLoaderLayout(node);
  }
  // 延时重同步：防止框架在 onNodeCreated 后重建控件覆盖显隐/回调；并兜底工作流值恢复
  setTimeout(() => {
    try {
      syncStateFromWidgets(node);
      applySlotVisibility(node);
      applyLoaderLabels(node);
      pushLoaderConfig(node);
      refreshAllSwitchCombos();
    } catch (e) { /* ignore */ }
  }, 150);
  setTimeout(() => {
    try {
      syncStateFromWidgets(node);
      applySlotVisibility(node);
      pushLoaderConfig(node);
      refreshAllSwitchCombos();
    } catch (e) { /* ignore */ }
  }, 600);
  // 初始推送一次（界面新建节点即让开关下拉可见）
  pushLoaderConfig(node);
  refreshAllSwitchCombos();
}


function setupModelSwitchNode(node) {
  node.__zouyuLang = getLang();
  statusNodes.add(node);
  startStatusPolling();

  // 『动作』开关 → 副标题实时显示 加载/卸载
  const aw = node.widgets?.find((w) => w.name === "action");
  if (aw) {
    const orig = aw.callback;
    aw.callback = (v) => {
      if (orig) orig.call(aw, v);
      applySwitchSubtitle(node);
    };
  }
  applySwitchSubtitle(node);

  // 模型下拉：只显示加载器中已配置的模型（美化标签），并在加载器执行后自动刷新
  refreshSwitchModelCombo(node);
  setTimeout(() => refreshSwitchModelCombo(node), 400);
  setTimeout(() => refreshSwitchModelCombo(node), 1200);
  // 工作流加载：控件值在 configure 后才恢复 → 延迟重读动作并更新副标题
  setTimeout(() => applySwitchSubtitle(node), 150);
  setTimeout(() => applySwitchSubtitle(node), 600);
}

/** 开关副标题：加载 / 卸载（卸载深度由加载器的低显存模式开关决定）。 */
function applySwitchSubtitle(node) {
  if (!node) return;
  const zh = node.__zouyuLang !== "English";
  const aw = node.widgets?.find((w) => w.name === "action");
  const on = !!(aw && aw.value);
  const base = zh ? "模型加载开关" : "Model Load Switch";
  node.title = base + (on ? (zh ? " · 加载" : " · Load") : (zh ? " · 卸载" : " · Unload"));
  app.graph?.setDirtyCanvas(true, true);
}

/** 开关模型下拉：从 /status 拉取已配置模型（只显示 slot 槽位，选项 = 加载器中已选择的模型文件）。 */
async function refreshSwitchModelCombo(node) {
  if (!node) return;
  const w = node.widgets?.find((x) => x.name === "model");
  if (!w) return;
  try {
    const r = await fetch("/zouyu_model_loader/status");
    if (!r.ok) return;
    const payload = await r.json();
    // 只列出加载器中已配置/已选模型的槽位；显示名 = 模型文件名（与加载器下拉一致）
    const items = (payload.models || [])
      .filter((m) => String(m.kind || "").startsWith("slot"))
      .map((m) => ({ value: m.kind, label: m.name || m.kind }));
    w.options = w.options || {};
    const values = ["(未选择)", ...items.map((i) => i.value)];
    if (!w.options.values || w.options.values.join("|") !== values.join("|")) {
      w.options.values = values;
    }
    w.options.getOptionLabel = (v) => {
      const it = items.find((i) => i.value === v);
      return it ? it.label : v;
    };
    app.graph?.setDirtyCanvas(true, true);
  } catch (e) {
    /* 服务端未就绪时忽略 */
  }
}

/** 刷新图中所有开关节点的模型下拉（加载器执行后调用）。 */
function refreshAllSwitchCombos() {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if ((n.comfyClass || n.type) === "ZouyuModelSwitch") refreshSwitchModelCombo(n);
  }
}

/** 把加载器当前配置推送后端：开关下拉无需运行加载器即可识别（槽位配置即推即见）。 */
async function pushLoaderConfig(node) {
  const st = node && node.__zouyuSlotState;
  if (!st) return;
  const slots = [];
  for (let i = 0; i < MAX_MODELS; i++) {
    const s = st.slots[i];
    if (!s || s.type === "未使用" || !s.name || s.name === "(未选择)" || s.name === "(无文件)") continue;
    slots.push({ slot: i, tkey: TYPE_KEYS[s.type] || "other", folder: s.folder || "", name: s.name });
  }
  try {
    await fetch("/zouyu_model_loader/register_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slots }),
    });
  } catch (e) {
    /* 服务端未就绪时忽略 */
  }
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
        if (this.comfyClass === "ZouyuModelSwitch") applySwitchSubtitle(this);
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

        if (comfyClass === "ZouyuModelSwitch") {
          setupModelSwitchNode(node);
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

// 模型加载器/开关执行完成后立即刷新状态灯与开关下拉（兜底：另有 2.5s 轮询）
api.addEventListener("executed", (event) => {
  try {
    const detail = event.detail;
    if (!detail || detail.nodeId == null) return;
    const node = app.graph?._nodes?.find((n) => String(n.id) === String(detail.nodeId));
    if (!node) return;
    if (node.comfyClass === "ZouyuModelLoader") {
      refreshStatusDOM(node);
      // 加载器执行后：登记了新配置 → 刷新所有开关节点的模型下拉
      refreshAllSwitchCombos();
    } else if (node.comfyClass === "ZouyuModelSwitch") {
      refreshSwitchModelCombo(node);
    }
  } catch (e) {
    /* ignore */
  }
});
