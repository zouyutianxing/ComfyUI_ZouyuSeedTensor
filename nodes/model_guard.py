"""
Zouyu Model Guard — 共享模型状态注册表 + 节点2（模型加载开关 ZouyuModelSwitch）。

共享注册表：模型加载器（节点1）每次执行后登记模型，模型加载开关（节点2）按信号
触发加载/卸载任务，HTTP 路由（前端轮询红/绿/蓝状态灯与状态文字）互通。
"""

import os
import time
import threading

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io
from comfy.patcher_extension import CallbacksMP

# ---------------------------------------------------------------------------
# 共享注册表（同一进程内，节点1 / 节点2 / HTTP 路由互通）
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()

_REGISTRY = {
    "switch": False,      # 节点1 的低显存开关（True=低显存彻底卸载；False=CPU缓存，开关退化为转接点）
    "switch_ts": 0.0,
    "loader_ts": 0.0,     # 节点1 最近一次执行时间
    "models": {},         # kind -> {kind, name, obj, patcher, state, ts}  （加载器执行后登记）
    "configured": {},     # kind -> {kind, tkey, folder, name}               （前端推送的配置，未运行即可用）
    "events": [],         # 日志环形缓冲
}

_MAX_EVENTS = 40

KIND_LABELS = {"unet": "UNET", "clip": "CLIP", "vae": "视频VAE", "audio_vae": "音频VAE"}

KIND_LABELS_EN = {"unet": "UNET", "clip": "CLIP", "vae": "Video VAE", "audio_vae": "Audio VAE"}

# 槽位类型标签（含后端识别键 unet/checkpoint：加载器执行后按识别结果显示）
SLOT_TYPE_LABELS = {
    "main":      {"zh": "主模型",  "en": "Main"},
    "unet":      {"zh": "主模型",  "en": "Main"},
    "checkpoint": {"zh": "主模型", "en": "Main"},
    "clip":      {"zh": "文本模型", "en": "Text(CLIP)"},
    "vae":       {"zh": "视频VAE", "en": "Video VAE"},
    "avae":      {"zh": "音频VAE", "en": "Audio VAE"},
    "lora":      {"zh": "LoRA",    "en": "LoRA"},
    "other":     {"zh": "其他",    "en": "Other"},
    "unknown":   {"zh": "未知",    "en": "Unknown"},
}

# 三色状态灯：绿=已加载(显存) / 蓝=未加载(CPU内存，未被显存加载) / 红=已卸载(硬盘)
# 主文字保持简短（避免与下拉内文件名文字重叠），位置提示由前端灯 tooltip 展示
STATE_INFO = {
    "gpu":     {"zh": "已加载", "en": "Loaded",       "color": "#4caf50"},
    "cpu":     {"zh": "未加载", "en": "Not loaded",   "color": "#2196f3"},
    "free":    {"zh": "已卸载", "en": "Unloaded",     "color": "#f44336"},
    "unknown": {"zh": "未知",   "en": "Unknown",      "color": "#9e9e9e"},
}

# 模型类型 + 分类（端口名显示：类型 + 序号 + 分类，如「主模型0 (Diffusion)」）
MODEL_TYPE_INFO = {
    "unet":       {"zh": "主模型 (Diffusion)",  "en": "Main (Diffusion)",   "color": "#b0722a"},
    "checkpoint": {"zh": "主模型 (Checkpoint)", "en": "Main (Checkpoint)",  "color": "#b0722a"},
    "clip":       {"zh": "文本模型 (CLIP)",     "en": "Text (CLIP)",        "color": "#8f6f2f"},
    "vae":        {"zh": "视频VAE (VAE)",       "en": "Video VAE (VAE)",    "color": "#2f6b8f"},
    "avae":       {"zh": "音频VAE (VAE)",       "en": "Audio VAE (VAE)",    "color": "#2f6b8f"},
    "lora":       {"zh": "LoRA (LoRA)",         "en": "LoRA",               "color": "#7a4fa0"},
    "other":      {"zh": "其他",                "en": "Other",              "color": "#9e9e9e"},
    "unknown":    {"zh": "未知",                "en": "Unknown",            "color": "#9e9e9e"},
}


def _now():
    return time.time()


def log_event(msg):
    with _LOCK:
        _REGISTRY["events"].append("[{}] {}".format(time.strftime("%H:%M:%S"), msg))
        del _REGISTRY["events"][:-_MAX_EVENTS]


def set_switch(on, eager=False):
    """设置全局卸载策略开关（低显存=彻底卸载 / CPU缓存=官方管理）。

    eager=True（前端手动切换）：切到低显存时，对已登记且不在显存的模型立即执行
    彻底卸载（释放权重，灯变红），让用户马上看到开关效果；显存中的模型不动
    （可能正在被使用）。eager=False（加载器 execute 同步）：不主动卸载，只登记策略。
    """
    with _LOCK:
        _REGISTRY["switch"] = bool(on)
        _REGISTRY["switch_ts"] = _now()
    if eager and on:
        with _LOCK:
            targets = [
                kind for kind, e in _REGISTRY["models"].items()
                if e.get("patcher") is not None and residency(e["patcher"]) != "gpu"
            ]
        for kind in targets:
            e = _REGISTRY["models"].get(kind)
            if e is not None:
                _do_unload_model(e, e.get("name", kind), _slot_label(kind, True, e=e), True)


def get_switch():
    with _LOCK:
        return _REGISTRY["switch"]


def _patcher_of(obj):
    return getattr(obj, "patcher", None) or obj


def residency(patcher):
    """返回 "gpu" / "cpu" / "free" / "unknown"（显存 / CPU缓存 / 完全卸载 / 未知）"""
    try:
        if patcher is None:
            return "unknown"
        vram = int(patcher.loaded_size())
        if vram > 0:
            return "gpu"
        if patcher.is_dynamic():
            ram = int(patcher.loaded_ram_size())
            return "cpu" if ram > 0 else "free"
        total = int(patcher.model_size())
        return "cpu" if total > 0 else "free"
    except Exception:
        return "unknown"


def register(kind, name, obj, model_type="", folder="", tkey=""):
    """节点1 每次执行后登记模型（同 kind 旧条目被替换，旧 patcher 交由 GC 回收）。

    folder/tkey 记录模型来源，供"模型加载开关"节点按需重新加载。
    """
    with _LOCK:
        patcher = _patcher_of(obj)
        _REGISTRY["models"][kind] = {
            "kind": kind,
            "name": name,
            "obj": obj,
            "patcher": patcher,
            "model_type": model_type,
            "folder": folder,
            "tkey": tkey,
            "state": residency(patcher),
            "ts": _now(),
        }
        _REGISTRY["loader_ts"] = _now()
    _attach_model_callbacks(patcher, kind)


def _attach_model_callbacks(patcher, kind):
    """在模型 patcher 上挂载位置检测回调（幂等，纯检测模型位置）：
    - ON_LOAD ：模型被官方加载到显存 → 实时更新状态（绿=显存）。
    - ON_DETACH：模型被官方模型管理卸载（其它模型需要显存时）→ 实时更新状态
      （蓝=CPU内存；低显存模式下动态模型再释放 CPU 权重 → 红=已卸载）。
      该机制与 guard 节点摆放位置无关，保证状态灯始终反映模型真实位置。
    """
    try:
        if getattr(patcher, "__zouyu_hooked", False):
            return
        patcher.__zouyu_hooked = True
        patcher.add_callback(CallbacksMP.ON_LOAD, lambda p, *a, **k: _on_model_load(kind, p))
        patcher.add_callback(CallbacksMP.ON_DETACH, lambda p, *a, **k: _on_model_detach(kind, p))
    except Exception:
        pass


def _on_model_load(kind, patcher):
    """模型被加载到显存：状态灯 → 绿（纯位置检测，与任何模式开关无关）。"""
    with _LOCK:
        e = _REGISTRY["models"].get(kind)
        if e is not None and e["patcher"] is patcher:
            e["state"] = residency(patcher)
            e["ts"] = _now()


def _on_model_detach(kind, patcher):
    """模型被官方模型管理卸载：状态灯更新为真实位置（纯检测，不依赖模式开关）。

    低显存模式下额外释放动态模型的 CPU 权重（动态模型可释放，状态 → 红）；
    CPU缓存模式下只更新位置（蓝=权重保留在内存），不主动干预。
    """
    freed = 0
    try:
        if get_switch() and patcher.is_dynamic():
            freed = int(patcher.partially_unload_ram(1e32) or 0)
    except Exception:
        freed = 0
    with _LOCK:
        e = _REGISTRY["models"].get(kind)
        if e is not None and e["patcher"] is patcher:
            # 位置在权重释放动作之后重新检测（低显存释放动态权重 → 真实位置变为 free/红）
            e["state"] = residency(patcher)
            e["ts"] = _now()
    if freed > 0:
        log_event("[{}] 空闲即卸载：已释放 CPU 内存".format(KIND_LABELS.get(kind, kind)))


def fully_unload_patcher(patcher):
    """把模型从显存卸载到 CPU（官方模型管理），DynamicVRAM 模型再释放 CPU 内存。
    返回释放的 CPU 内存字节数（0 = 传统模型权重常驻内存，无法释放）。"""
    freed_ram = 0
    try:
        model_management.unload_model_and_clones(patcher)
    except Exception:
        try:
            model_management.free_memory(1e30, model_management.get_torch_device())
        except Exception:
            pass
    try:
        if patcher.is_dynamic():
            freed_ram = int(patcher.partially_unload_ram(1e32) or 0)
    except Exception:
        freed_ram = 0
    return freed_ram


# ---------------------------------------------------------------------------
# 模型加载开关（节点2 → 新"模型加载开关"）：按信号经过向加载器发送加载/卸载任务
# ---------------------------------------------------------------------------

SWITCH_SLOT_COUNT = 8


def register_config(slots):
    """前端推送的加载器槽位配置（模型加载器在界面上配置后即可被开关识别，无需先运行）。

    slots: [{slot: int, tkey: str, folder: str, name: str}, ...]（空槽位传空 name 即清除）
    """
    with _LOCK:
        for s in slots or []:
            try:
                idx = int(s.get("slot", -1))
            except Exception:
                continue
            if not (0 <= idx < SWITCH_SLOT_COUNT):
                continue
            kind = "slot{}".format(idx)
            name = str(s.get("name") or "").strip()
            if not name or name in ("(未选择)", "(无文件)"):
                _REGISTRY["configured"].pop(kind, None)
                continue
            _REGISTRY["configured"][kind] = {
                "kind": kind,
                "tkey": str(s.get("tkey") or "other"),
                "folder": str(s.get("folder") or ""),
                "name": name,
            }
            # 仅当配置的模型与已登记模型不同时才使旧登记失效。
            # 前端每次 configure/reattach 都会重复推送相同配置，若无条件 pop，
            # 已加载/已登记的模型会被清成「已卸载」（红灯），蓝/绿灯无法保持。
            old = _REGISTRY["models"].get(kind)
            if old is not None and old.get("name") != name:
                _REGISTRY["models"].pop(kind, None)


def configured_model_kinds():
    """已登记/已配置的模型槽位 kind 列表（供开关下拉显示）。"""
    with _LOCK:
        kinds = set(k for k in _REGISTRY["models"] if k.startswith("slot"))
        kinds |= set(k for k in _REGISTRY["configured"] if k.startswith("slot"))
    return sorted(kinds, key=lambda k: (len(k), k))


def configured_model_options():
    """开关下拉校验选项：占位符 + 全部槽位 kind（任何时刻已保存的值都能通过校验）。
    前端只把"已配置"的 kind 放进下拉显示。"""
    return ["(未选择)"] + ["slot{}".format(i) for i in range(SWITCH_SLOT_COUNT)]


def _slot_label(kind, zh, e=None, cfg=None):
    mtype = ""
    if e is not None:
        mtype = e.get("model_type", "")
    elif cfg is not None:
        mtype = cfg.get("tkey", "")
    if not mtype and kind.startswith("slot"):
        with _LOCK:
            e2 = _REGISTRY["models"].get(kind)
            c2 = _REGISTRY["configured"].get(kind)
        mtype = (e2 or c2 or {}).get("model_type") or (c2 or {}).get("tkey") or ""
    if kind.startswith("slot") and mtype in SLOT_TYPE_LABELS:
        t = SLOT_TYPE_LABELS[mtype]
        return (t["zh"] if zh else t["en"]) + kind[len("slot"):]
    return kind


def load_or_unload_model(kind, do_load, zh):
    """模型加载开关：按信号执行 加载 / 卸载。

    卸载策略由「模型加载器」的低显存模式开关决定（加载器执行/前端切换时 set_switch 登记）：
    - 低显存模式开启（彻底卸载）：卸载信号 → 把模型从显存+CPU内存彻底卸载（红色「已卸载」）；
    - 低显存模式关闭（CPU缓存）：否决/屏蔽卸载信号，不主动干预模型，完全交由官方模型管理
      （官方在显存压力时自动卸载，权重保留在内存，蓝色「未加载」）。
    优先操作已登记（执行过加载器）的模型对象；只有配置未运行时：
    - 加载 → 直接从配置的文件加载并登记；
    - 卸载 → 提示尚未加载。
    """
    with _LOCK:
        e = _REGISTRY["models"].get(kind)
        cfg = _REGISTRY["configured"].get(kind)
    if e is None and cfg is None:
        return ("未找到模型 {}（请先在模型加载器中配置该槽位）".format(kind)
                if zh else "model {} not found (configure it in the loader first)".format(kind))
    name = (e or cfg).get("name", "") or kind
    label = _slot_label(kind, zh, e=e, cfg=cfg)
    if do_load:
        if e is not None:
            return _do_load_model(e, name, label, zh)
        return _do_load_config(kind, cfg, name, label, zh)
    # 卸载：CPU缓存模式（开关关闭）否决/屏蔽卸载信号——不主动干预模型，
    # 完全交由官方模型管理（官方在显存压力时自动卸载，权重保留在内存，蓝色「未加载」）；
    # 低显存模式（开关开启）才执行彻底卸载（显存+CPU权重释放，红色「已卸载」）。
    if e is None:
        return ("{} 尚未被加载器加载（未运行或加载失败），无法卸载".format(label)
                if zh else "{} not loaded by loader (not run or failed), cannot unload".format(label))
    if not get_switch():
        return ("{} CPU缓存模式：已屏蔽卸载信号，模型交由官方模型管理".format(label)
                if zh else "{} CPU-cache mode: unload signal vetoed, model managed by ComfyUI".format(label))
    return _do_unload_model(e, name, label, zh)


def _do_load_config(kind, cfg, name, label, zh):
    """仅有配置未运行：直接从配置的文件加载并登记到注册表。"""
    try:
        from .model_loader import _load_slot_model
        tkey = cfg.get("tkey") or "other"
        folder = cfg.get("folder", "")
        obj, actual, note = _load_slot_model(tkey, folder, cfg.get("name", ""))
        register(kind, cfg.get("name", ""), obj, model_type=actual, folder=folder, tkey=tkey)
        # 加载后立即载入显存，状态灯变绿（否则显示「未加载」蓝灯，加载看起来无效）
        try:
            patcher = _patcher_of(obj)
            if patcher is not None:
                model_management.load_model_gpu(patcher)
        except Exception:
            pass
        return ("已加载 {} {}（{}）".format(label, name, note or "显存")
                if zh else "loaded {} {} ({})".format(label, name, note or "VRAM"))
    except Exception as exc:
        return ("加载 {} 失败：{}".format(name, str(exc)[:80])
                if zh else "load {} failed: {}".format(name, str(exc)[:80]))


def _do_load_model(e, name, label, zh):
    """加载：优先把现有 patcher 载入显存（保持下游对象一致）；失败则按登记文件重载。"""
    kind = e["kind"]
    patcher = e.get("patcher")
    if patcher is not None:
        try:
            model_management.load_model_gpu(patcher)
            with _LOCK:
                cur = _REGISTRY["models"].get(kind)
                if cur is not None:
                    cur["state"] = residency(patcher)
                    cur["ts"] = _now()
            return ("已加载 {} {}（{}）".format(label, name, "显存")
                    if zh else "loaded {} {} (VRAM)".format(label, name))
        except Exception:
            pass
    try:
        from .model_loader import _load_slot_model
        tkey = e.get("tkey") or e.get("model_type") or "other"
        folder = e.get("folder", "")
        obj, actual, _note = _load_slot_model(tkey, folder, e.get("name", ""))
        register(kind, e.get("name", ""), obj, model_type=actual, folder=folder, tkey=tkey)
        # 重新加载后立即载入显存（与 _do_load_config 一致），状态灯变绿
        try:
            model_management.load_model_gpu(_patcher_of(obj))
        except Exception:
            pass
        return ("已重新加载 {}（{}）".format(label, _note)
                if zh else "reloaded {} ({})".format(label, _note))
    except Exception as exc:
        return ("重新加载 {} 失败：{}".format(name, str(exc)[:80])
                if zh else "reload {} failed: {}".format(name, str(exc)[:80]))


def _do_unload_model(e, name, label, zh):
    """彻底卸载（低显存模式）：从显存卸载到 CPU，并释放权重内存。

    DynamicVRAM 模型由 partially_unload_ram 释放 CPU 权重；传统模型（非动态）
    释放注册表对模型对象的引用，交由 GC 回收权重（模型已从官方缓存 unload_model_and_clones
    移除，无其他持有者时真正释放内存）。状态显示红色「已卸载」。
    重新加载时 patcher 为 None，会自动按登记文件重新加载。
    """
    kind = e["kind"]
    patcher = e.get("patcher")
    if patcher is None:
        return ("{} 已彻底卸载".format(label) if zh else "{} fully unloaded".format(label))
    freed = fully_unload_patcher(patcher)
    with _LOCK:
        cur = _REGISTRY["models"].get(kind)
        if cur is not None:
            # 释放对象引用：DynamicVRAM 权重已释放；传统模型权重随对象回收（GC）。
            cur["obj"] = None
            cur["patcher"] = None
            cur["state"] = "free"
            cur["ts"] = _now()
    if freed > 0:
        return ("已彻底卸载 {}（显存+CPU内存）".format(label)
                if zh else "fully unloaded {} (VRAM+RAM)".format(label))
    return ("已彻底卸载 {}（显存与CPU权重已释放）".format(label)
            if zh else "fully unloaded {} (VRAM+weights released)".format(label))


def status_payload():
    with _LOCK:
        models = []
        for kind, e in _REGISTRY["models"].items():
            # 低显存彻底卸载后对象引用已释放（patcher=None）→ 显示红色「已卸载」
            st = "free" if e.get("patcher") is None else residency(e["patcher"])
            info = STATE_INFO.get(st, STATE_INFO["unknown"])
            mtype = e.get("model_type", "")
            tinfo = MODEL_TYPE_INFO.get(mtype, MODEL_TYPE_INFO["unknown"]) if mtype else None
            # 标签：通用加载器槽位 → 类型+序号；经典 kind → KIND_LABELS
            if kind.startswith("slot") and mtype in SLOT_TYPE_LABELS:
                label_zh = SLOT_TYPE_LABELS[mtype]["zh"] + kind[len("slot"):]
                label_en = SLOT_TYPE_LABELS[mtype]["en"] + kind[len("slot"):]
            else:
                label_zh = KIND_LABELS.get(kind, kind)
                label_en = KIND_LABELS_EN.get(kind, kind)
            models.append({
                "kind": kind,
                "label": label_zh,
                "label_zh": label_zh,
                "label_en": label_en,
                "name": e.get("name", ""),
                "state": st,
                "color": info["color"],
                "zh": info["zh"],
                "en": info["en"],
                "type": mtype,
                "type_zh": tinfo["zh"] if tinfo else "",
                "type_en": tinfo["en"] if tinfo else "",
                "ts": e.get("ts", 0.0),
            })
        # 合并"仅配置未运行"的模型（加载器界面上配置后即可被开关下拉识别）
        # 状态按"free"（红色=模型位于硬盘，未加载）显示，与用户三色语义一致
        seen = {m["kind"] for m in models}
        free = STATE_INFO["free"]
        for kind, c in _REGISTRY["configured"].items():
            if kind in seen:
                continue
            mtype = c.get("tkey", "")
            tinfo = MODEL_TYPE_INFO.get(mtype, MODEL_TYPE_INFO["unknown"]) if mtype else None
            if kind.startswith("slot") and mtype in SLOT_TYPE_LABELS:
                label_zh = SLOT_TYPE_LABELS[mtype]["zh"] + kind[len("slot"):]
                label_en = SLOT_TYPE_LABELS[mtype]["en"] + kind[len("slot"):]
            else:
                label_zh = kind
                label_en = kind
            models.append({
                "kind": kind,
                "label": label_zh,
                "label_zh": label_zh,
                "label_en": label_en,
                "name": c.get("name", ""),
                "state": "free",
                "color": free["color"],
                "zh": free["zh"],
                "en": free["en"],
                "type": mtype,
                "type_zh": tinfo["zh"] if tinfo else "",
                "type_en": tinfo["en"] if tinfo else "",
                "ts": 0.0,
                "configured_only": True,
            })
        return {
            "switch": _REGISTRY["switch"],
            "switch_ts": _REGISTRY["switch_ts"],
            "loader_ts": _REGISTRY["loader_ts"],
            "models_root": _models_root(),
            "models": models,
            "configured_kinds": sorted(_REGISTRY["configured"].keys()),
            "events": list(_REGISTRY["events"][-15:]),
        }


# ---------------------------------------------------------------------------
# 自由文件夹解析 + 模型类型自动识别（前端"选择模型文件夹"用）
# ---------------------------------------------------------------------------

def _models_root():
    """ComfyUI/models 目录绝对路径（由 vae 分类根目录推导）。"""
    roots = folder_paths.get_folder_paths("vae")
    if roots:
        return os.path.dirname(os.path.abspath(roots[0]))
    roots = folder_paths.get_folder_paths("diffusion_models")
    return os.path.dirname(os.path.abspath(roots[0])) if roots else ""


# 合并文件列表缓存（短 TTL：拖入导入后数秒内自动刷新，无需重启）
_ALL_FILES_CACHE = {"ts": 0.0, "files": None}


def all_model_files():
    """全部模型文件合并列表（10 大官方分类 + models 根下任意自定义文件夹）。

    同时收录「相对 models 根的路径」与「纯文件名」两种形态：
    - 前端下拉按文件夹过滤后显示纯文件名，校验用纯文件名通过；
    - 相对路径形态兼容用户在文件夹输入框手填完整路径。
    拖入导入的新文件在导入后数秒内即被校验接受（无需重启）。
    """
    now = time.time()
    cached = _ALL_FILES_CACHE["files"]
    if cached is not None and now - _ALL_FILES_CACHE["ts"] < 5:
        return cached
    cats = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints",
            "clip_vision", "style_models", "upscale_models", "controlnet", "gligen"]
    files, seen = [], set()
    def _add(f):
        if f not in seen:
            seen.add(f)
            files.append(f)
    for c in cats:
        for f in folder_paths.get_filename_list(c):
            _add(f)
    models_root = _models_root()
    if models_root and os.path.isdir(models_root):
        for dirpath, dirnames, filenames in os.walk(models_root):
            rel = os.path.relpath(dirpath, models_root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 6:
                dirnames[:] = []
                continue
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in folder_paths.supported_pt_extensions:
                    relf = os.path.relpath(os.path.join(dirpath, fn), models_root).replace("\\", "/")
                    _add(relf)
                    _add(os.path.basename(relf))  # 纯文件名形态（下拉/校验主用）
    _add("(未选择)")
    _ALL_FILES_CACHE["files"] = files
    _ALL_FILES_CACHE["ts"] = now
    return files


def bust_model_files_cache():
    """拖入导入完成后立即失效缓存，让新文件马上可被校验。"""
    _ALL_FILES_CACHE["ts"] = 0.0


def _safe_join(base, rel):
    """把 rel 安全拼到 base 下，拒绝越界（防路径穿越）。"""
    base = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base, rel)) if rel else base
    if os.path.commonpath([base, target]) != base:
        raise ValueError("path out of root")
    return target


def _safe_join_soft(base, rel):
    try:
        return _safe_join(base, rel)
    except ValueError:
        return None


def _read_keys(path):
    """读取权重文件的键名列表（safetensors 只读元数据，很快；其余格式全量加载）。"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                return list(f.keys())
        sd = comfy.utils.load_torch_file(path)
        return list(sd.keys())
    except Exception:
        return None


_VAE_MARKERS = (
    "decoder.conv_in", "decoder.conv_out", "decoder.conv_post", "decoder.mid_block",
    "decoder.up_blocks", "decoder.norm_out", "decoder.proj_out", "decoder.mask_token",
    "decoder.register_tokens", "dec_in_proj", "enc_in_proj", "latents_mean",
    "latents_std", "first_stage_model.decoder",
)


def detect_model_type(abs_path):
    """根据权重键名识别模型类型：unet / checkpoint / clip / vae / lora / unknown。"""
    keys = _read_keys(abs_path)
    if not keys:
        return "unknown"
    joined = "|".join(keys)
    if any(k.startswith(("lora_unet.", "lora_te.")) for k in keys) \
            or ".lora_down." in joined or ".lora_up." in joined:
        return "lora"
    if "decoder." in joined and "encoder." in joined \
            and any(m in joined for m in _VAE_MARKERS):
        return "vae"
    if any(k.startswith("model.embed_tokens") for k in keys) \
            or "text_model.encoder" in joined \
            or "cond_stage_model." in joined \
            or ("visual." in joined and any(k.startswith("model.layers.") for k in keys)) \
            or ("shared.weight" in joined and "encoder.block." in joined):
        return "clip"
    if any(k.startswith("blocks.") for k in keys) \
            or "model.diffusion_model." in joined \
            or "diffusion_model." in joined \
            or "adaln_" in joined \
            or "patch_proj" in joined \
            or "pos_embed" in joined:
        return "checkpoint" if "cond_stage_model." in joined else "unet"
    return "unknown"


def _find_in_models(filename):
    """在整个 models 目录树里按文件名搜索，返回绝对路径列表。"""
    results = []
    seen = set()
    models_root = _models_root()
    if not models_root:
        return results
    for dirpath, dirnames, filenames in os.walk(models_root):
        if filename in filenames:
            p = os.path.join(dirpath, filename)
            if p not in seen:
                seen.add(p)
                results.append(p)
    return results


def _resolve_abs(slot_category, folder, name):
    """自由文件夹解析：folder 可为分类名/空/"."（=该槽位默认分类根），
    或任意相对 models 根的路径。按 basename 兜底全 models 搜索，
    兼容用户旧工作流中文件夹与文件不匹配的情况。"""
    name = str(name or "").strip().strip("/\\")
    if not name or name == "(无文件)":
        raise ValueError("[ZouyuModelLoader] 未选择模型文件")
    folder = str(folder or "").strip().strip("/\\")
    models_root = _models_root()
    candidates = []
    if folder in ("", ".", slot_category):
        for root in folder_paths.get_folder_paths(slot_category):
            candidates.append(os.path.join(root, name.replace("/", os.sep)))
            candidates.append(os.path.join(root, os.path.basename(name)))
    else:
        candidates.append(_safe_join_soft(models_root, folder + "/" + name))
        candidates.append(_safe_join_soft(models_root, folder + "/" + os.path.basename(name)))
    # 文件夹可能写错：兜底在槽位默认分类根下按文件名找
    for root in folder_paths.get_folder_paths(slot_category):
        candidates.append(os.path.join(root, os.path.basename(name)))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    found = _find_in_models(os.path.basename(name))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError("[ZouyuModelLoader] 多个目录存在同名文件: {}，请在文件夹中精确定位".format(os.path.basename(name)))
    raise ValueError("[ZouyuModelLoader] 找不到模型文件: {}（文件夹: {}）".format(name, folder or slot_category))


def list_files(category, folder):
    """列出模型文件：folder 为分类名/空/"." 时返回分类根全部文件；
    category="all" 时返回 models 目录下全部模型文件（去重纯文件名，供「请加载模型」初始下拉）；
    否则返回 models_root/folder 目录下的模型文件（相对该目录的路径）。"""
    if folder in ("", ".", category):
        if category == "all":
            files, seen = [], set()
            cats = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints",
                    "clip_vision", "style_models", "upscale_models", "controlnet", "gligen"]
            for c in cats:
                for f in folder_paths.get_filename_list(c):
                    base = os.path.basename(f)
                    if base not in seen:
                        seen.add(base)
                        files.append(base)
            models_root = _models_root()
            if models_root and os.path.isdir(models_root):
                for dirpath, dirnames, filenames in os.walk(models_root):
                    rel = os.path.relpath(dirpath, models_root)
                    depth = 0 if rel == "." else rel.count(os.sep) + 1
                    if depth > 6:
                        dirnames[:] = []
                        continue
                    for fn in filenames:
                        if os.path.splitext(fn)[1].lower() in folder_paths.supported_pt_extensions \
                                and fn not in seen:
                            seen.add(fn)
                            files.append(fn)
            return {"folder": folder or ".", "files": sorted(files)}
        return {"folder": folder or ".", "files": folder_paths.get_filename_list(category)}
    models_root = _models_root()
    base = _safe_join_soft(models_root, folder)
    if base is None or not os.path.isdir(base):
        return {"folder": folder, "files": []}
    files = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in folder_paths.supported_pt_extensions:
                files.append(os.path.relpath(os.path.join(dirpath, fn), base).replace("\\", "/"))
    return {"folder": folder, "files": sorted(files)}


def find_folder(name):
    """在 models 目录树中按文件夹名查找（限深 5 层），返回相对 models 根的路径列表。"""
    name = (name or "").strip()
    if not name:
        return {"found": []}
    models_root = _models_root()
    hits = []
    for dirpath, dirnames, _filenames in os.walk(models_root):
        rel = os.path.relpath(dirpath, models_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 5:
            dirnames[:] = []
            continue
        if name in dirnames:
            rel_hit = os.path.relpath(os.path.join(dirpath, name), models_root).replace("\\", "/")
            if rel_hit not in hits:
                hits.append(rel_hit)
        if len(hits) >= 20:
            break
    return {"found": hits}


def list_model_dirs(rel=""):
    """列出 models 根下 rel 相对路径的直接子目录（返回相对 models 根的路径列表）。

    供前端「模型文件夹选择」弹窗逐级浏览 models 目录树使用。
    rel="" 或 "." 表示 models 根目录。
    """
    models_root = _models_root()
    if not models_root:
        return {"root": "", "dirs": []}
    base = _safe_join_soft(models_root, rel or ".")
    if base is None or not os.path.isdir(base):
        return {"root": rel or ".", "dirs": []}
    dirs = []
    for name in sorted(os.listdir(base)):
        if name.startswith("."):
            continue
        p = os.path.join(base, name)
        if os.path.isdir(p):
            dirs.append(os.path.relpath(p, models_root).replace("\\", "/"))
    return {"root": rel or ".", "dirs": dirs}


def slot_action(kind, action):
    """手动控制槽位模型：action ∈ {"load", "unload"}，返回说明文字（卸载深度由加载器低显存模式决定）。"""
    do_load = action == "load"
    msg = load_or_unload_model(kind, do_load, True)
    log_event(msg)
    return msg


async def import_folder_files(folder_name, file_parts):
    """把拖入/选择的模型文件写入 models/<name>。

    规则（用户需求）：models 目录树中存在同名文件夹则直接复用；
    不存在则在 models 根下新建以拖入文件夹名命名的文件夹。
    每个文件只取 basename 落盘（防路径穿越），返回目标相对路径。
    """
    name = os.path.basename((folder_name or "").strip().strip("/\\"))
    if not name or name in (".", ".."):
        raise ValueError("无效的文件夹名")
    models_root = _models_root()
    if not models_root:
        raise ValueError("找不到 models 目录")
    # 在 models 树中查找同名文件夹（限深 3 层）
    target = None
    for dirpath, dirnames, _filenames in os.walk(models_root):
        rel = os.path.relpath(dirpath, models_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            dirnames[:] = []
            continue
        if name in dirnames:
            target = os.path.join(dirpath, name)
            break
    if target is None:
        target = _safe_join(models_root, name)
    os.makedirs(target, exist_ok=True)
    written = []
    for filename, part in file_parts:
        fn = os.path.basename(str(filename or "").strip().strip("/\\"))
        if not fn or fn in (".", ".."):
            continue
        dest = _safe_join(target, fn)
        with open(dest, "wb") as f:
            while True:
                chunk = await part.read_chunk(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        written.append(fn)
    if not written:
        raise ValueError("没有可写入的模型文件")
    rel = os.path.relpath(target, models_root).replace("\\", "/")
    return {"target": rel, "count": len(written), "files": sorted(set(written))}


def reveal_path(path):
    """在操作系统的文件资源管理器中打开（Windows explorer.exe）；path 为空时打开 models 根目录。"""
    try:
        models_root = _models_root()
        if not path:
            path = models_root
        if not path:
            return {"ok": False, "error": "empty path"}
        target = os.path.abspath(path)
        if os.path.commonpath([models_root, target]) != models_root:
            return {"ok": False, "error": "path outside models root"}
        if os.name != "nt":
            return {"ok": False, "error": "only supported on Windows"}
        import subprocess
        subprocess.Popen(["explorer.exe", target])
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        routes = getattr(instance, "routes", None) if instance is not None else None
        if routes is None:
            return
    except Exception:
        return

    @routes.get("/zouyu_model_loader/files")
    async def _files(request):
        return web.json_response(list_files(
            request.query.get("category", "diffusion_models"),
            request.query.get("folder", ".")))

    @routes.get("/zouyu_model_loader/find_folder")
    async def _find_folder(request):
        return web.json_response(find_folder(request.query.get("name", "")))

    @routes.get("/zouyu_model_loader/list_dirs")
    async def _list_dirs(request):
        """models 目录树浏览：返回 rel 路径下的直接子目录（相对 models 根）。"""
        return web.json_response(list_model_dirs(request.query.get("path", ".")))

    @routes.post("/zouyu_model_loader/slot_action")
    async def _slot_action(request):
        """手动加载/卸载指定槽位模型。body: {"kind": "slot0", "action": "load"|"unload"}"""
        try:
            data = await request.json()
            kind = str(data.get("kind") or "")
            action = str(data.get("action") or "")
            if not kind.startswith("slot") or action not in ("load", "unload"):
                return web.json_response({"ok": False, "error": "bad params"}, status=400)
            msg = slot_action(kind, action)
            return web.json_response({"ok": True, "message": msg})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)[:200]}, status=400)

    @routes.get("/zouyu_model_loader/reveal")
    async def _reveal(request):
        return web.json_response(reveal_path(request.query.get("path", "")))

    @routes.post("/zouyu_model_loader/import_folder")
    async def _import_folder(request):
        """接收拖入的文件夹：multipart 表单 folder_name + 若干 files，写入 models/<name>。"""
        try:
            reader = await request.multipart()
            folder_name = ""
            file_parts = []
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "folder_name":
                    folder_name = (await part.read()).decode("utf-8", "replace")
                elif part.filename:
                    file_parts.append((part.filename, part))
            if not folder_name:
                return web.json_response({"ok": False, "error": "缺少文件夹名"}, status=400)
            if not file_parts:
                return web.json_response({"ok": False, "error": "没有收到文件"}, status=400)
            result = await import_folder_files(folder_name, file_parts)
            bust_model_files_cache()  # 新文件立即可被下拉与校验接受
            result["ok"] = True
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)[:300]}, status=400)

    @routes.get("/zouyu_model_loader/status")
    async def _status(request):
        return web.json_response(status_payload())

    @routes.post("/zouyu_model_loader/register_config")
    async def _register_config(request):
        """前端推送加载器槽位配置：加载器在界面上配置后，开关下拉即可识别（无需先运行）。"""
        try:
            data = await request.json()
            register_config(data.get("slots") or [])
            return web.json_response({"ok": True, "count": len(_REGISTRY["configured"])})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)[:200]}, status=400)

    @routes.post("/zouyu_model_loader/set_switch")
    async def _set_switch(request):
        """前端同步加载器的「低显存/CPU缓存」开关到后端。

        eager=true（用户手动切换开关）：切到低显存时立即释放已登记的非显存模型
        （灯变红），让开关效果即时可见；eager=false（工作流配置恢复 reattach）：
        只同步策略值，不自动卸载任何模型（避免旧工作流低显存=true 恢复时模型被静默卸载）。"""
        try:
            data = await request.json()
            on = bool(data.get("on", False))
            eager = bool(data.get("eager", False))
            set_switch(on, eager=eager)
            return web.json_response({"ok": True, "switch": get_switch()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)[:200]}, status=400)


# ---------------------------------------------------------------------------
# 节点2：模型加载开关（导线式信号检测 → 加载/卸载任务）
# ---------------------------------------------------------------------------

class ZouyuModelSwitch(io.ComfyNode):
    """模型加载开关：左右各一个端口，信号直接透传，仅检测是否有数据经过。
    一旦有信号经过，按『动作』开关向后端发送加载/卸载任务，由加载器完成并显示。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuModelSwitch",
            display_name="模型加载开关 (Model Load Switch)",
            category="ZouyuAI/SeedTensor",
            description=(
                "导线式信号检测：左右各一个端口（* 任意类型），可插入任何插件之间的连线，"
                "信号直接透传。当检测到信号（数据）经过本节点时，按『动作』开关执行任务——"
                "打开（加载）：把下拉所选模型加载进显存/CPU内存；关闭（卸载）：向模型加载器发送卸载信号。"
                "行为由加载器的『低显存模式』开关决定——低显存：开关正常执行加载/卸载"
                "（卸载=把模型从显存与CPU内存彻底卸载）；CPU缓存（默认）：开关退化为『转接点』，"
                "完全否决加载/卸载信号，只透传信号、不干预模型，模型完全交由官方模型管理。"
                "模型下拉自动识别模型加载器在界面上已配置的模型（无需先运行加载器）。"
                "副标题实时显示当前动作（加载/卸载/转接）。"
            ),
            inputs=[
                io.AnyType.Input("signal", optional=True, tooltip="信号输入（任意类型，直接透传；有数据经过时触发任务）"),
                io.Combo.Input("model", options=configured_model_options(), default="(未选择)",
                               tooltip="选择要控制的模型（自动列出模型加载器已配置的槽位）"),
                io.Boolean.Input("action", default=True, label_on="加载", label_off="卸载",
                                 tooltip="加载=信号经过时把模型加载进显存；卸载=信号经过时向加载器发送卸载信号。"
                                         "低显存模式：正常执行；CPU缓存模式：信号被完全否决（开关退化为转接点，只透传不干预）"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[
                io.AnyType.Output("signal", display_name="信号"),
            ],
        )

    @classmethod
    def execute(cls, signal=None, model="(未选择)", action=True, language="中文") -> io.NodeOutput:
        zh = (language != "English")
        # 信号检测：只有真正有数据经过（signal 非空）才触发加载/卸载任务
        if signal is None:
            text = ("无信号经过：仅透传" if zh else "no signal: pass-through only")
            return io.NodeOutput(signal, ui={"text": [text]})
        if model and model != "(未选择)":
            if not get_switch():
                # CPU缓存模式（开关关闭）：开关退化为「转接点」——完全否决加载/卸载信号，
                # 只透传信号，不干预模型（模型完全交由官方模型管理）
                label = _slot_label(model, zh)
                msg = ("{} CPU缓存模式：开关已禁用（转接点），仅透传信号，不干预模型".format(label)
                       if zh else "{} CPU-cache mode: switch disabled (passthrough), signal only, no model action".format(label))
                log_event(msg)
                text = msg
            else:
                msg = load_or_unload_model(model, bool(action), zh)
                log_event(msg)
                text = msg
        else:
            text = ("未选择要控制的模型" if zh else "no model selected")
        return io.NodeOutput(signal, ui={"text": [text]})
