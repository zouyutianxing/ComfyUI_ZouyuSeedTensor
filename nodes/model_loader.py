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


def _load_slot(kind, folder, name, model_options):
    """自由文件夹解析 → 模型类型识别 → 按槽位加载。返回 (obj, 类型标签)。"""
    slot = SLOT_INFO[kind]
    abs_path = _resolve_abs(slot["category"], folder, name)
    mtype = detect_model_type(abs_path)
    tinfo = MODEL_TYPE_INFO.get(mtype, MODEL_TYPE_INFO["unknown"])

    if mtype == "lora":
        raise ValueError("[ZouyuModelLoader] {} 是 LoRA 模型，不能作为独立模型加载，请选择主模型/文本模型/VAE 文件"
                         .format(os.path.basename(abs_path)))
    if mtype not in slot["expect"]:
        raise ValueError(
            "[ZouyuModelLoader] {} 是「{}」，不是{}，请选择对应类型的模型文件"
            .format(os.path.basename(abs_path), tinfo["zh"], slot["label"]))

    if mtype in ("unet", "checkpoint"):
        obj = comfy.sd.load_diffusion_model(abs_path, model_options=model_options)
    elif mtype == "clip":
        obj = comfy.sd.load_clip(
            ckpt_paths=[abs_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=model_options["clip_type"],
            model_options=model_options.get("clip_model_options", {}),
        )
    else:  # vae
        sd, metadata = comfy.utils.load_torch_file(abs_path, return_metadata=True)
        obj = comfy.sd.VAE(sd=sd, metadata=metadata)
        obj.throw_exception_if_invalid()
        obj.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (abs_path, metadata, None))
    return obj, mtype


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

        # 空值兜底（前端未选择时用默认项，保证开箱即用，与官方加载器一致）
        unet_name = unet_name or _default_file("diffusion_models")
        clip_name = clip_name or _default_file("text_encoders")
        vae_name = vae_name or _default_file("vae")
        audio_vae_name = audio_vae_name or _default_file("vae")
        weight_dtype = weight_dtype or "default"
        clip_type = clip_type or "minimax"
        clip_device = clip_device or "default"

        # 主模型（UNET 槽位）
        unet_opts = {}
        if weight_dtype == "fp8_e4m3fn":
            unet_opts["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            unet_opts["dtype"] = torch.float8_e4m3fn
            unet_opts["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            unet_opts["dtype"] = torch.float8_e5m2
        unet, unet_type = _load_slot("unet", unet_folder, unet_name, unet_opts)

        # 文本模型（CLIP 槽位）
        clip_type_obj = getattr(comfy.sd.CLIPType, str(clip_type).upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_model_options = {}
        if clip_device == "cpu":
            clip_model_options["load_device"] = clip_model_options["offload_device"] = torch.device("cpu")
        clip, clip_type_det = _load_slot("clip", clip_folder, clip_name, {
            "clip_type": clip_type_obj, "clip_model_options": clip_model_options})

        # 视频 VAE / 音频 VAE 槽位
        vae, vae_type = _load_slot("vae", vae_folder, vae_name, {})
        audio_vae, audio_vae_type = _load_slot("audio_vae", audio_vae_folder, audio_vae_name, {})

        for kind, name, obj, mtype in (("unet", unet_name, unet, unet_type),
                                       ("clip", clip_name, clip, clip_type_det),
                                       ("vae", vae_name, vae, vae_type),
                                       ("audio_vae", audio_vae_name, audio_vae, audio_vae_type)):
            register(kind, name, obj, model_type=mtype)
            tinfo = MODEL_TYPE_INFO.get(mtype, MODEL_TYPE_INFO["unknown"])
            log_event("{} ← {}（{}）".format(KIND_LABELS.get(kind, kind), name, tinfo["zh"]))

        # 消费节点2（ZouyuModelGuard）传来的闲置信号，更新状态/日志
        for kind, st, _ts in consume_signals():
            log_event("收到闲置信号: {} 状态={}".format(KIND_LABELS.get(kind, kind), st))

        payload = status_payload()
        lines = [
            ("低显存模式: " + ("开启（彻底卸载闲置模型）" if switch else "关闭（官方管理→CPU缓存）"))
            if zh else
            ("Low VRAM: " + ("ON (fully unload idle)" if switch else "OFF (official → CPU cache)"))
        ]
        for m in payload["models"]:
            info = STATE_INFO.get(m["state"], STATE_INFO["unknown"])
            label = info["zh"] if zh else info["en"]
            tlabel = m["type_zh"] if zh else m["type_en"]
            lines.append("{}: {} {}  ({})".format(m["label"], tlabel, label, m["name"]))

        text = "\n".join(lines)
        return io.NodeOutput(unet, clip, vae, audio_vae, text,
                             ui={"text": [text], "zouyu_status": payload})
