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
import folder_paths
from comfy_api.latest import io

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

STATE_INFO = {
    "gpu":     {"zh": "已加载(GPU)", "en": "Loaded (GPU)", "color": "#4caf50"},
    "cpu":     {"zh": "CPU缓存",     "en": "CPU cached",   "color": "#2196f3"},
    "free":    {"zh": "未加载",      "en": "Not loaded",   "color": "#f44336"},
    "unknown": {"zh": "未知",        "en": "Unknown",      "color": "#9e9e9e"},
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


def register(kind, name, obj):
    """节点1 每次执行后登记四个模型（同 kind 旧条目被替换，旧 patcher 交由 GC 回收）"""
    with _LOCK:
        patcher = _patcher_of(obj)
        _REGISTRY["models"][kind] = {
            "kind": kind,
            "name": name,
            "obj": obj,
            "patcher": patcher,
            "state": residency(patcher),
            "ts": _now(),
        }
        _REGISTRY["loader_ts"] = _now()


def watch(kind, patcher):
    """节点2 观测模型（通用于任意工作流、任意模型类型）"""
    with _LOCK:
        _REGISTRY["watched"][id(patcher)] = {"kind": kind, "patcher": patcher, "ts": _now()}
        cutoff = _now() - _WATCHED_TTL
        for k in [k for k, v in _REGISTRY["watched"].items() if v["ts"] < cutoff]:
            _REGISTRY["watched"].pop(k, None)
        if len(_REGISTRY["watched"]) > _MAX_WATCHED:
            _REGISTRY["watched"] = dict(list(_REGISTRY["watched"].items())[-_MAX_WATCHED:])
        if kind in _REGISTRY["models"] and _REGISTRY["models"][kind]["patcher"] is patcher:
            _REGISTRY["models"][kind]["state"] = residency(patcher)
            _REGISTRY["models"][kind]["ts"] = _now()


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


def evaluate_idle(kind, patcher, zh):
    """节点2 的核心判定：模型此刻是否空闲 → 按开关决定动作。

    返回 (说明文本, 动作后状态, action)，action ∈ {"none", "official", "unload"}。
    - 开关关闭：只记录闲置信号，交给官方管理（状态会由官方卸载变为 cpu/蓝色）。
    - 开关开启且模型在显存（gpu）：说明已被调用过且已空闲 → 彻底卸载。
    - 从未加载 / 已在 CPU：无事可做，跳过。
    """
    switch = get_switch()
    st = residency(patcher)
    watch(kind, patcher)
    record_signal(kind, st)
    label = KIND_LABELS.get(kind, kind)
    if not switch:
        return ("[{}] 空闲，开关关闭：交由官方管理自动卸载到CPU内存".format(label), st, "official")
    if st == "gpu":
        freed_ram = fully_unload_patcher(patcher)
        st2 = residency(patcher)
        if freed_ram > 0:
            return ("[{}] 空闲 → 已彻底卸载（显存+CPU内存）".format(label), st2, "unload")
        return ("[{}] 空闲 → 已卸载到CPU内存（传统模型不支持释放内存）".format(label), st2, "unload")
    if st == "free":
        return ("[{}] 已完全卸载，无需处理".format(label), st, "none")
    return ("[{}] 尚未被调用（或已在CPU缓存），跳过".format(label), st, "none")


def status_payload():
    with _LOCK:
        models = []
        for kind, e in _REGISTRY["models"].items():
            st = residency(e["patcher"])
            info = STATE_INFO.get(st, STATE_INFO["unknown"])
            models.append({
                "kind": kind,
                "label": KIND_LABELS.get(kind, kind),
                "name": e.get("name", ""),
                "state": st,
                "color": info["color"],
                "zh": info["zh"],
                "en": info["en"],
                "ts": e.get("ts", 0.0),
            })
        return {
            "switch": _REGISTRY["switch"],
            "switch_ts": _REGISTRY["switch_ts"],
            "loader_ts": _REGISTRY["loader_ts"],
            "models": models,
            "events": list(_REGISTRY["events"][-15:]),
        }


# ---------------------------------------------------------------------------
# 文件夹 / 文件列表（前端"选择模型文件夹"用）
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


def browse_dir(category, folder):
    """浏览 models 目录：folder 为相对 models 根的空/相对路径（"" = models 根）。
    返回当前目录绝对路径、一级子文件夹、上级目录相对路径。"""
    models_root = _models_root()
    rel = (folder or "").strip().strip("/\\")
    current = _safe_join(models_root, rel)
    folders = []
    try:
        for d in sorted(os.listdir(current)):
            if os.path.isdir(os.path.join(current, d)):
                folders.append(d)
    except OSError:
        pass
    up = "/".join(rel.split("/")[:-1]) if rel else ""
    return {
        "category": category,
        "models_root": models_root,
        "current": current,
        "rel": rel,
        "up": up,
        "folders": folders,
    }


def pick_category_rel(category, rel_from_models):
    """把 models 根下的相对路径换算为该分类根下的相对路径。
    选中分类根目录本身返回分类名；不在分类搜索目录内返回 None。"""
    try:
        abs_target = _safe_join(_models_root(), rel_from_models)
    except ValueError:
        return None
    for root in folder_paths.get_folder_paths(category):
        root_abs = os.path.abspath(root)
        try:
            r = os.path.relpath(abs_target, root_abs)
        except ValueError:
            continue
        if r == ".":
            return category
        if not r.startswith(".."):
            return r.replace("\\", "/")
    return None


def list_files(category, folder):
    """列出分类根目录（folder 为分类名/空/"."）或某子文件夹下的模型文件。"""
    allowed = folder_paths.get_filename_list(category)
    folder = (folder or "").strip().strip("/\\")
    if folder in ("", ".", category):
        out = [f for f in allowed if "/" not in f and "\\" not in f]
    else:
        prefix = folder.replace("\\", "/").rstrip("/") + "/"
        out = [f for f in allowed if f.replace("\\", "/").startswith(prefix)]
    return {"category": category, "folder": folder or ".", "files": out}


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

    @routes.get("/zouyu_model_loader/browse")
    async def _browse(request):
        return web.json_response(browse_dir(
            request.query.get("category", "diffusion_models"),
            request.query.get("folder", "")))

    @routes.get("/zouyu_model_loader/pick")
    async def _pick(request):
        folder = pick_category_rel(
            request.query.get("category", "diffusion_models"),
            request.query.get("rel", ""))
        if folder is None:
            return web.json_response({"ok": False, "error": "所选文件夹不在该模型分类的搜索目录内"})
        return web.json_response({"ok": True, "folder": folder})

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
                "接在每个模型连线上（MODEL/CLIP/VAE 可同时接多个）。当本节点执行时（所有下游"
                "消费者已运行完毕），判定模型空闲并通知模型加载器。若模型加载器的『低显存模式』"
                "开启，立即把该模型从显存（DynamicVRAM 模型含 CPU 内存）彻底卸载；关闭则交由"
                "官方模型管理自动卸载到 CPU 内存。"
            ),
            inputs=[
                io.Model.Input("model", optional=True, tooltip="视频模型（DiT）"),
                io.Clip.Input("clip", optional=True, tooltip="文本编码器"),
                io.Vae.Input("vae", optional=True, tooltip="视频 VAE"),
                io.Vae.Input("audio_vae", optional=True, tooltip="音频 VAE"),
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
    def execute(cls, model=None, clip=None, vae=None, audio_vae=None, language="中文") -> io.NodeOutput:
        zh = (language != "English")
        lines = []
        for kind, obj in (("unet", model), ("clip", clip), ("vae", vae), ("audio_vae", audio_vae)):
            if obj is None:
                continue
            patcher = _patcher_of(obj)
            msg, _st, _action = evaluate_idle(kind, patcher, zh)
            log_event(msg)
            lines.append(msg)
        text = "\n".join(lines) if lines else (
            "（未连接任何模型，仅透传）" if zh else "(no model connected, pass-through only)")
        return io.NodeOutput(model, clip, vae, audio_vae, text, ui={"text": [text]})
