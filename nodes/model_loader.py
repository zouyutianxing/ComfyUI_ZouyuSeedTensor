"""
Zouyu Model Loader — 模型加载节点（集成官方四个加载器 + 自由文件夹 + 类型自动识别）。

与官方节点完全等价：
- UNET   ← UNETLoader（diffusion_models 目录，weight_dtype 精度选项）
- CLIP   ← CLIPLoader（text_encoders 目录，type/device 选项，默认 minimax）
- VAE    ← VAELoader（vae 目录）
- 音频VAE ← VAELoader（vae 目录）

增强功能：
- 每个模型槽位的"文件夹"输入不限制分类：默认值即官方各加载器使用的目录
  （diffusion_models / text_encoders / vae / vae），也可填任意相对 models 根的路径；
  『选择模型文件夹』按钮直接打开系统资源管理器（原生目录对话框），选择后自动识别
  文件夹名并刷新模型下拉。
- 自动识别所选模型的类型（主模型/文本模型/VAE/LoRA），按类型选择对应加载函数，
  并在 UI 状态行显示类型；槽位与类型不匹配时报清晰错误。
- 默认值开箱即用：文件下拉默认选中 minimax 相关文件（无则第一个），精度/类型等
  均有默认项；文件夹与文件不匹配时按文件名全 models 兜底查找，兼容旧工作流。
- 每个模型旁的状态灯：绿=已加载(GPU)、蓝=CPU缓存、红=未加载（完全卸载）。
- 低显存开关（low_vram_mode）：开启后，节点2（ZouyuModelGuard）检测到模型空闲时，
  把该模型从显存+CPU内存彻底卸载；关闭则交给官方模型管理卸载到 CPU 内存。
"""

import os

import torch
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .model_guard import (
    KIND_LABELS,
    STATE_INFO,
    MODEL_TYPE_INFO,
    set_switch,
    register,
    consume_signals,
    status_payload,
    log_event,
    _resolve_abs,
    detect_model_type,
)

CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart",
    "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image",
    "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit",
    "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

# 槽位 → 默认分类 / 期望的模型类型
SLOT_INFO = {
    "unet":      {"category": "diffusion_models", "expect": ("unet", "checkpoint"),
                  "label": "主模型", "en_label": "main model"},
    "clip":      {"category": "text_encoders", "expect": ("clip",),
                  "label": "文本模型", "en_label": "text model"},
    "vae":       {"category": "vae", "expect": ("vae",),
                  "label": "VAE", "en_label": "VAE"},
    "audio_vae": {"category": "vae", "expect": ("vae",),
                  "label": "VAE", "en_label": "VAE"},
}


def _file_options(category):
    files = folder_paths.get_filename_list(category)
    return files if files else ["(无文件)"]


def _default_file(category):
    """默认文件：优先 minimax 相关文件（本插件面向 MiniMax H3），否则取第一个。"""
    files = folder_paths.get_filename_list(category)
    if not files:
        return "(无文件)"
    for f in files:
        if "minimax" in os.path.basename(f).lower():
            return f
    return files[0]


def _try_resolve(slot, folder, name):
    """容错解析 + 类型识别；失败返回 None（不抛错，交由自动恢复）。"""
    try:
        path = _resolve_abs(SLOT_INFO[slot]["category"], folder, name)
        mtype = detect_model_type(path)
        return {"path": path, "type": mtype, "name": os.path.basename(path)}
    except Exception:
        return None


def _fits(r, slot):
    return r is not None and r["type"] in SLOT_INFO[slot]["expect"]


def _auto_assign(inputs, zh):
    """自动识别并分配四个槽位的模型文件（把报错变成自动恢复）：
    1) 类型匹配槽位的直接保留；
    2) 类型不符的槽位优先与其它槽位文件成对互换（自动纠正放错位置的模型）；
    3) 仍不符/无法解析的槽位自动改用默认文件；
    4) 多余且类型不匹配任何槽位的文件（如 LoRA）丢弃并提示。
    返回 (assigned: {slot: r}, notes: [str])，r = {"path","type","name"}。
    """
    slots = list(SLOT_INFO)
    resolved = {s: _try_resolve(s, *inputs[s]) for s in slots}
    notes = []

    label = lambda s: SLOT_INFO[s]["label"] if zh else SLOT_INFO[s]["en_label"]

    # 类型匹配 → 保留
    claimed = {s for s in slots if _fits(resolved[s], s)}
    # 类型不符的槽位：先尝试成对互换
    for s in slots:
        if _fits(resolved[s], s):
            continue
        swapped = False
        for s2 in slots:
            if s2 == s or not _fits(resolved[s2], s):
                continue
            if _fits(resolved[s], s2):  # s 的文件恰好适合 s2 → 成对互换
                resolved[s], resolved[s2] = resolved[s2], resolved[s]
                claimed.add(s)
                claimed.discard(s2)
                notes.append(("{} 与 {} 的模型文件类型不匹配，已自动互换"
                              .format(label(s), label(s2)))
                             if zh else
                             ("{} and {} files had mismatched types; auto-swapped"
                              .format(label(s), label(s2))))
                swapped = True
                break
        if swapped:
            continue
        # 无成对互换：从其它槽位借用类型匹配且未被占用的文件
        for s2 in slots:
            if s2 == s or s2 in claimed or not _fits(resolved[s2], s):
                continue
            resolved[s] = resolved[s2]
            resolved[s2] = None
            claimed.add(s)
            notes.append(("{} 的模型文件类型不匹配，已自动改用 {} 槽位的文件"
                          .format(label(s), label(s2)))
                         if zh else
                         ("{} file type mismatch; using {} slot's file instead"
                          .format(label(s), label(s2))))
            swapped = True
            break
        if not swapped:
            notes.append(("{} 的模型文件类型不匹配，自动改用默认文件".format(label(s)))
                         if zh else
                         ("{} file type mismatch; falling back to default".format(label(s))))

    # 回退：仍未匹配/无法解析的槽位 → 默认文件
    for s in slots:
        if _fits(resolved[s], s):
            continue
        default_name = _default_file(SLOT_INFO[s]["category"])
        r = _try_resolve(s, SLOT_INFO[s]["category"], default_name)
        if r is not None and _fits(r, s):
            resolved[s] = r
            notes.append(("{} 已自动改用默认文件 {}".format(label(s), r["name"]))
                         if zh else
                         ("{} using default file {}".format(label(s), r["name"])))
        else:
            resolved[s] = None
    return resolved, notes


def _load_assigned(slot, r, model_options):
    """按已识别类型加载模型。"""
    mtype = r["type"]
    abs_path = r["path"]
    if mtype in ("unet", "checkpoint"):
        return comfy.sd.load_diffusion_model(abs_path, model_options=model_options)
    if mtype == "clip":
        return comfy.sd.load_clip(
            ckpt_paths=[abs_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=model_options["clip_type"],
            model_options=model_options.get("clip_model_options", {}),
        )
    sd, metadata = comfy.utils.load_torch_file(abs_path, return_metadata=True)
    obj = comfy.sd.VAE(sd=sd, metadata=metadata)
    obj.throw_exception_if_invalid()
    obj.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (abs_path, metadata, None))
    return obj


class ZouyuModelLoader(io.ComfyNode):
    """模型加载器：集成 UNETLoader / CLIPLoader / VAELoader(×2)，自由文件夹 + 类型识别。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuModelLoader",
            display_name="Zouyu 模型加载器 (Model Loader)",
            category="ZouyuAI/SeedTensor",
            description=(
                "集成 MiniMax H3 示例工作流的四个模型加载（UNET/CLIP/视频VAE/音频VAE），与官方 "
                "UNETLoader/CLIPLoader/VAELoader 完全等价，可直接连接 MiniMax H3 Reference to "
                "Video 等节点使用（各下拉均有默认项）。『选择模型文件夹』直接打开系统文件夹选择"
                "对话框，不限制分类，可自由选择 models 下任意目录；加载时自动识别模型类型"
                "（主模型/文本模型/VAE/LoRA）并按类型加载，状态行显示类型。状态灯：绿=已加载"
                "(GPU)、蓝=CPU缓存、红=未加载。『低显存模式』开启后，节点2检测到模型空闲会从"
                "显存+CPU内存彻底卸载；关闭则交由官方管理卸载到CPU内存。"
            ),
            inputs=[
                io.String.Input("unet_folder", default="diffusion_models",
                                tooltip="UNET 文件夹：默认 diffusion_models，也可填任意相对 models 的路径"),
                io.Combo.Input("unet_name", options=_file_options("diffusion_models"),
                               default=_default_file("diffusion_models"),
                               tooltip="主模型文件（自动识别类型）"),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default",
                               tooltip="权重精度（同官方 UNETLoader）"),
                io.String.Input("clip_folder", default="text_encoders",
                                tooltip="文本编码器文件夹：默认 text_encoders，也可填任意相对 models 的路径"),
                io.Combo.Input("clip_name", options=_file_options("text_encoders"),
                               default=_default_file("text_encoders"),
                               tooltip="文本模型文件（自动识别类型）"),
                io.Combo.Input("clip_type", options=CLIP_TYPES, default="minimax",
                               tooltip="文本编码器类型（MiniMax H3 用 minimax）"),
                io.Combo.Input("clip_device", options=["default", "cpu"], default="default",
                               tooltip="编码器设备（cpu=强制 CPU 加载）"),
                io.String.Input("vae_folder", default="vae",
                                tooltip="视频VAE文件夹：默认 vae，也可填任意相对 models 的路径"),
                io.Combo.Input("vae_name", options=_file_options("vae"), default=_default_file("vae"),
                               tooltip="视频VAE文件（自动识别类型）"),
                io.String.Input("audio_vae_folder", default="vae",
                                tooltip="音频VAE文件夹：默认 vae，也可填任意相对 models 的路径"),
                io.Combo.Input("audio_vae_name", options=_file_options("vae"), default=_default_file("vae"),
                               tooltip="音频VAE文件（自动识别类型）"),
                io.Boolean.Input("low_vram_mode", default=False,
                                 label_on="彻底卸载", label_off="CPU缓存",
                                 tooltip="开启=极低显存：模型空闲时从显存+CPU内存彻底卸载；"
                                         "关闭=交给官方模型管理卸载到CPU内存"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[
                io.Model.Output(display_name="模型(UNET)"),
                io.Clip.Output(display_name="文本编码器(CLIP)"),
                io.Vae.Output(display_name="视频VAE"),
                io.Vae.Output(display_name="音频VAE"),
                io.String.Output(display_name="状态"),
            ],
        )

    @classmethod
    def execute(cls, unet_folder, unet_name, weight_dtype, clip_folder, clip_name, clip_type,
                clip_device, vae_folder, vae_name, audio_vae_folder, audio_vae_name,
                low_vram_mode, language) -> io.NodeOutput:
        zh = (language or "中文") != "English"
        switch = bool(low_vram_mode)
        set_switch(switch)
        log_event("模型加载器执行（低显存模式={}）".format("开启" if switch else "关闭"))

        # 空值兜底：未选择/空值一律自动用默认项（开箱即用，与官方加载器一致）
        inputs = {
            "unet": (unet_folder or "diffusion_models", unet_name or _default_file("diffusion_models")),
            "clip": (clip_folder or "text_encoders", clip_name or _default_file("text_encoders")),
            "vae": (vae_folder or "vae", vae_name or _default_file("vae")),
            "audio_vae": (audio_vae_folder or "vae", audio_vae_name or _default_file("vae")),
        }
        for s in SLOT_INFO:
            if not inputs[s][1] or inputs[s][1] == "(无文件)":
                inputs[s] = (SLOT_INFO[s]["category"], _default_file(SLOT_INFO[s]["category"]))

        # 自动识别 + 自动纠正（类型不符互换/回退默认，不再报错）
        assigned, notes = _auto_assign(inputs, zh)
        for n in notes:
            log_event("自动调整: " + n)

        # 按槽位组装加载选项
        unet_opts = {}
        if weight_dtype == "fp8_e4m3fn":
            unet_opts["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            unet_opts["dtype"] = torch.float8_e4m3fn
            unet_opts["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            unet_opts["dtype"] = torch.float8_e5m2
        clip_type_obj = getattr(comfy.sd.CLIPType, str(clip_type or "minimax").upper(),
                                comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_model_options = {}
        if (clip_device or "default") == "cpu":
            clip_model_options["load_device"] = clip_model_options["offload_device"] = torch.device("cpu")
        clip_opts = {"clip_type": clip_type_obj, "clip_model_options": clip_model_options}

        # 加载
        loaded = {}
        for s in SLOT_INFO:
            r = assigned[s]
            if r is None:
                raise ValueError("[ZouyuModelLoader] {} 槽位没有可用模型文件，请检查 models 目录"
                                 .format(SLOT_INFO[s]["label"]))
            opts = unet_opts if s == "unet" else (clip_opts if s == "clip" else {})
            obj = _load_assigned(s, r, opts)
            loaded[s] = obj
            tinfo = MODEL_TYPE_INFO.get(r["type"], MODEL_TYPE_INFO["unknown"])
            log_event("{} ← {}（{}）".format(KIND_LABELS.get(s, s), r["name"], tinfo["zh"]))

        # 登记（含检测类型）供状态灯显示
        for s in SLOT_INFO:
            r = assigned[s]
            register(s, r["name"], loaded[s], model_type=r["type"])

        # 消费节点2（ZouyuModelGuard）传来的闲置信号，更新状态/日志
        for kind, st, _ts in consume_signals():
            log_event("收到闲置信号: {} 状态={}".format(KIND_LABELS.get(kind, kind), st))

        payload = status_payload()
        lines = [
            ("低显存模式: " + ("开启（彻底卸载闲置模型）" if switch else "关闭（官方管理→CPU缓存）"))
            if zh else
            ("Low VRAM: " + ("ON (fully unload idle)" if switch else "OFF (official → CPU cache)"))
        ]
        lines.extend(notes)
        for m in payload["models"]:
            info = STATE_INFO.get(m["state"], STATE_INFO["unknown"])
            label = info["zh"] if zh else info["en"]
            tlabel = m["type_zh"] if zh else m["type_en"]
            lines.append("{}: {} {}  ({})".format(m["label"], tlabel, label, m["name"]))

        text = "\n".join(lines)
        return io.NodeOutput(loaded["unet"], loaded["clip"], loaded["vae"], loaded["audio_vae"],
                             text, ui={"text": [text], "zouyu_status": payload})
