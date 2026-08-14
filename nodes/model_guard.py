"""
Zouyu Model Guard — 模型占用检测节点 + 两个新节点共享的模型状态注册表。

节点2（ZouyuModelGuard）：接在模型连线上（MODEL/CLIP/VAE 均可，多个可同时接）。
本节点执行时，其所有下游消费者都已经运行完毕，此刻可判定模型"空闲"（未被调用），
随后把"闲置信号"写入共享注册表，并按节点1（ZouyuModelLoader）写入的低显存开关决定动作：

- 开关关闭（默认）：不干预。由 ComfyUI 官方模型管理在需要显存时自动把闲置模型
  卸载到 CPU 内存（状态灯 → 蓝色）。
- 开关开启（极低显存）：立即把该模型从显存彻底卸载；若运行在 DynamicVRAM 模式
  （ModelPatcherDynamic），还会释放 CPU 内存（状态灯 → 红色）；传统 ModelPatcher
  权重常驻内存、无法释放，只能卸载到 CPU（状态灯 → 蓝色，并给出提示）。

节点1 在下一次执行时消费这些信号，用于状态展示与日志。注册表为同一进程内的
模块级状态，两个节点与 HTTP 路由（前端轮询红/绿/蓝状态灯）互通。
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
    "switch": False,      # 节点1 的低显存开关（True=极低显存，彻底卸载；False=交给官方管理）
    "switch_ts": 0.0,
    "loader_ts": 0.0,     # 节点1 最近一次执行时间
    "models": {},         # kind -> {kind, name, obj, patcher, state, ts}  （节点1 的四个模型）
    "watched": {},        # id(patcher) -> {kind, patcher, ts}             （节点2 观测到的任意模型）
    "signals": [],        # (kind, state, ts) 闲置信号（节点2 → 节点1）
    "events": [],         # 日志环形缓冲
}

_MAX_EVENTS = 40
_MAX_SIGNALS = 60
_MAX_WATCHED = 200
_WATCHED_TTL = 3600.0

KIND_LABELS = {"unet": "UNET", "clip": "CLIP", "vae": "视频VAE", "audio_vae": "音频VAE"}

KIND_LABELS_EN = {"unet": "UNET", "clip": "CLIP", "vae": "Video VAE", "audio_vae": "Audio VAE"}

SLOT_TYPE_LABELS = {
    "main":  {"zh": "主模型",  "en": "Main"},
    "clip":  {"zh": "文本模型", "en": "Text(CLIP)"},
    "vae":   {"zh": "视频VAE", "en": "Video VAE"},
    "avae":  {"zh": "音频VAE", "en": "Audio VAE"},
    "lora":  {"zh": "LoRA",    "en": "LoRA"},
    "other": {"zh": "其他",    "en": "Other"},
}

STATE_INFO = {
    "gpu":     {"zh": "已加载(GPU)", "en": "Loaded (GPU)", "color": "#4caf50"},
    "cpu":     {"zh": "CPU缓存",     "en": "CPU cached",   "color": "#2196f3"},
    "free":    {"zh": "未加载",      "en": "Not loaded",   "color": "#f44336"},
    "unknown": {"zh": "未知",        "en": "Unknown",      "color": "#9e9e9e"},
}

MODEL_TYPE_INFO = {
    "unet":       {"zh": "主模型(Diffusion)",  "en": "Main (Diffusion)",  "color": "#b0722a"},
    "checkpoint": {"zh": "主模型(Checkpoint)", "en": "Main (Checkpoint)", "color": "#b0722a"},
    "clip":       {"zh": "文本模型(CLIP)",     "en": "Text (CLIP)",       "color": "#8f6f2f"},
    "vae":        {"zh": "VAE 模型",           "en": "VAE",               "color": "#2f6b8f"},
    "lora":       {"zh": "LoRA",               "en": "LoRA",              "color": "#7a4fa0"},
    "unknown":    {"zh": "未知",               "en": "Unknown",           "color": "#9e9e9e"},
}


def _now():
    return time.time()


def log_event(msg):
    with _LOCK:
        _REGISTRY["events"].append("[{}] {}".format(time.strftime("%H:%M:%S"), msg))
        del _REGISTRY["events"][:-_MAX_EVENTS]


def set_switch(on):
    with _LOCK:
        _REGISTRY["switch"] = bool(on)
        _REGISTRY["switch_ts"] = _now()


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


def register(kind, name, obj, model_type=""):
    """节点1 每次执行后登记四个模型（同 kind 旧条目被替换，旧 patcher 交由 GC 回收）"""
    with _LOCK:
        patcher = _patcher_of(obj)
        _REGISTRY["models"][kind] = {
            "kind": kind,
            "name": name,
            "obj": obj,
            "patcher": patcher,
            "model_type": model_type,
            "state": residency(patcher),
            "ts": _now(),
        }
        _REGISTRY["loader_ts"] = _now()
    _attach_model_callbacks(patcher, kind)


def _attach_model_callbacks(patcher, kind):
    """在模型 patcher 上挂载状态回调（幂等）：
    - ON_LOAD ：模型被加载/使用 → 记录"曾被使用"并实时更新状态（绿）。
    - ON_DETACH：模型被官方模型管理卸载（其它模型需要显存时）→ 低显存模式下
      立即释放其 CPU 内存（DynamicVRAM 模型）+ 实时更新状态（红/蓝）。
      该机制与 guard 节点摆放位置无关，保证"工作模型运行时其它模型被自动卸载"。
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
    with _LOCK:
        e = _REGISTRY["models"].get(kind)
        if e is not None and e["patcher"] is patcher:
            e["ever_loaded"] = True
            e["state"] = "gpu"
            e["ts"] = _now()


def _on_model_detach(kind, patcher):
    if not get_switch():
        return
    freed = 0
    try:
        if patcher.is_dynamic():
            freed = int(patcher.partially_unload_ram(1e32) or 0)
    except Exception:
        freed = 0
    with _LOCK:
        e = _REGISTRY["models"].get(kind)
        if e is not None and e["patcher"] is patcher:
            e["state"] = residency(patcher)
            e["ts"] = _now()
    if freed > 0:
        log_event("[{}] 空闲即卸载：已释放 CPU 内存".format(KIND_LABELS.get(kind, kind)))


def watch(kind, patcher):
    """节点2 观测模型（通用于任意工作流、任意模型类型）"""
    with _LOCK:
        prev = _REGISTRY["watched"].get(id(patcher))
        ever_loaded = bool(prev and prev.get("ever_loaded"))
        if residency(patcher) == "gpu":
            ever_loaded = True
        _REGISTRY["watched"][id(patcher)] = {
            "kind": kind, "patcher": patcher, "ts": _now(), "ever_loaded": ever_loaded,
        }
        cutoff = _now() - _WATCHED_TTL
        for k in [k for k, v in _REGISTRY["watched"].items() if v["ts"] < cutoff]:
            _REGISTRY["watched"].pop(k, None)
        if len(_REGISTRY["watched"]) > _MAX_WATCHED:
            _REGISTRY["watched"] = dict(list(_REGISTRY["watched"].items())[-_MAX_WATCHED:])
        if kind in _REGISTRY["models"] and _REGISTRY["models"][kind]["patcher"] is patcher:
            _REGISTRY["models"][kind]["state"] = residency(patcher)
            _REGISTRY["models"][kind]["ts"] = _now()
        # 通用：按 patcher 匹配更新所有条目（通用加载器的槽位条目 kind 为 slot{i}）
        for k, e in _REGISTRY["models"].items():
            if k != kind and e.get("patcher") is patcher:
                e["state"] = residency(patcher)
                e["ts"] = _now()
    _attach_model_callbacks(patcher, kind)
    return ever_loaded


def record_signal(kind, state):
    with _LOCK:
        _REGISTRY["signals"].append((kind, state, _now()))
        del _REGISTRY["signals"][:-_MAX_SIGNALS]


def consume_signals():
    """节点1 消费自上次执行以来的闲置信号，返回 [(kind, state, ts), ...]"""
    with _LOCK:
        out = list(_REGISTRY["signals"])
        _REGISTRY["signals"] = []
    return out


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


def evaluate_idle(kind, patcher, zh, has_trigger=False):
    """节点2 的核心判定：模型此刻是否空闲 → 按开关决定动作。

    返回 (说明文本, 动作后状态, action)，action ∈ {"none", "official", "unload"}。
    判定条件：
    - 开关关闭：只记录闲置信号，交给官方管理（卸载到 CPU，状态→蓝色）。
    - 开关开启 + 有 trigger（阶段边界，本节点之后不再使用该模型）：
      - 模型在显存（gpu）→ 彻底卸载（显存+CPU内存，状态→红色）；
      - 模型在 CPU 缓存（cpu）→ 动态模型继续释放 CPU 内存；
      - 完全卸载（free）/从未加载 → 无需处理。
    - 开关开启 + 无 trigger（纯监测透传）：不动显存中的模型（可能即将被使用），
      只对已在 CPU 缓存的动态模型释放内存、并记录状态。
    """
    switch = get_switch()
    st = residency(patcher)
    ever_loaded = watch(kind, patcher)
    record_signal(kind, st)
    label = KIND_LABELS.get(kind, kind)
    if not switch:
        return ("[{}] 空闲，开关关闭：交由官方管理自动卸载到CPU内存".format(label), st, "official")
    if st == "gpu" and not has_trigger:
        return ("[{}] 在显存中（可能即将被使用），未触发卸载，仅记录状态".format(label), st, "none")
    if st == "gpu":
        freed_ram = fully_unload_patcher(patcher)
        st2 = residency(patcher)
        if freed_ram > 0:
            return ("[{}] 空闲 → 已彻底卸载（显存+CPU内存）".format(label), st2, "unload")
        return ("[{}] 空闲 → 已卸载到CPU内存（传统模型不支持释放内存）".format(label), st2, "unload")
    if st == "cpu":
        freed_ram = 0
        try:
            if patcher.is_dynamic():
                freed_ram = int(patcher.partially_unload_ram(1e32) or 0)
        except Exception:
            freed_ram = 0
        st2 = residency(patcher)
        if freed_ram > 0:
            return ("[{}] 已在CPU缓存 → 已释放CPU内存".format(label), st2, "unload")
        if ever_loaded:
            return ("[{}] 已卸载到CPU（传统模型无法释放内存）".format(label), st2, "none")
        return ("[{}] 尚未被调用（或已在CPU缓存），跳过".format(label), st2, "none")
    return ("[{}] 已完全卸载，无需处理".format(label), st, "none")


def status_payload():
    with _LOCK:
        models = []
        for kind, e in _REGISTRY["models"].items():
            st = residency(e["patcher"])
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
        return {
            "switch": _REGISTRY["switch"],
            "switch_ts": _REGISTRY["switch_ts"],
            "loader_ts": _REGISTRY["loader_ts"],
            "models_root": _models_root(),
            "models": models,
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
    否则返回 models_root/folder 目录下的模型文件（相对该目录的路径）。"""
    if folder in ("", ".", category):
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


def reveal_path(path):
    """在操作系统的文件资源管理器中打开（Windows explorer.exe）。"""
    try:
        if not path:
            return {"ok": False, "error": "empty path"}
        models_root = _models_root()
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

    @routes.get("/zouyu_model_loader/reveal")
    async def _reveal(request):
        return web.json_response(reveal_path(request.query.get("path", "")))

    @routes.get("/zouyu_model_loader/status")
    async def _status(request):
        return web.json_response(status_payload())


# ---------------------------------------------------------------------------
# 节点2：模型占用检测（透传 + 空闲判定）
# ---------------------------------------------------------------------------

class ZouyuModelGuard(io.ComfyNode):
    """模型占用检测：接在模型连线上，检测模型是否空闲并按低显存开关执行卸载策略。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuModelGuard",
            display_name="Zouyu 模型占用检测 (Model Guard)",
            category="ZouyuAI/SeedTensor",
            description=(
                "接在每个模型连线上（MODEL/CLIP/VAE 可同时接多个），当本节点执行时判定模型是否"
                "空闲并按低显存开关执行卸载。『触发』输入（任意类型，可接 conditioning/采样结果等）"
                "用于控制执行时机：例如接在文本编码节点之后、采样节点之前，即可在编码完成后立刻把"
                "CLIP/VAE 彻底卸载，让采样模型独享显存；采样结束后解码节点会自动重新加载所需 VAE。"
                "开关关闭则交由官方模型管理自动卸载到 CPU 内存。"
            ),
            inputs=[
                io.Model.Input("model", optional=True, tooltip="视频模型（DiT）"),
                io.Clip.Input("clip", optional=True, tooltip="文本编码器"),
                io.Vae.Input("vae", optional=True, tooltip="视频 VAE"),
                io.Vae.Input("audio_vae", optional=True, tooltip="音频 VAE"),
                io.AnyType.Input("trigger", optional=True,
                                 tooltip="触发（任意类型）：控制本节点执行时机，值会被忽略"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[
                io.Model.Output(display_name="模型"),
                io.Clip.Output(display_name="CLIP"),
                io.Vae.Output(display_name="VAE"),
                io.Vae.Output(display_name="音频VAE"),
                io.String.Output(display_name="日志"),
            ],
        )

    @classmethod
    def execute(cls, model=None, clip=None, vae=None, audio_vae=None, trigger=None,
                language="中文") -> io.NodeOutput:
        zh = (language != "English")
        has_trigger = trigger is not None
        lines = []
        for kind, obj in (("unet", model), ("clip", clip), ("vae", vae), ("audio_vae", audio_vae)):
            if obj is None:
                continue
            patcher = _patcher_of(obj)
            msg, _st, _action = evaluate_idle(kind, patcher, zh, has_trigger=has_trigger)
            log_event(msg)
            lines.append(msg)
        if has_trigger and lines:
            lines.insert(0, ("触发已就绪：当前阶段结束，执行空闲卸载策略" if zh
                             else "trigger ready: phase done, applying idle policy"))
        text = "\n".join(lines) if lines else (
            "（未连接任何模型，仅透传）" if zh else "(no model connected, pass-through only)")
        return io.NodeOutput(model, clip, vae, audio_vae, text, ui={"text": [text]})
