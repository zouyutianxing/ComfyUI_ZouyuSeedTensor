"""
Zouyu Model Loader — 模型加载节点（集成官方四个加载器 + 文件夹选择 + 状态灯）。

与官方节点完全等价：
- UNET   ← UNETLoader（diffusion_models 目录，weight_dtype 精度选项）
- CLIP   ← CLIPLoader（text_encoders 目录，type/device 选项，默认 minimax）
- VAE    ← VAELoader（vae 目录）
- 音频VAE ← VAELoader（vae 目录）

增强功能：
- 每个模型上方有一个"文件夹"输入：默认值即官方各加载器使用的目录（diffusion_models /
  text_encoders / vae / vae）；前端『📁』按钮可跳转到 ComfyUI models 目录选择子文件夹，
  文件下拉随之刷新（可选"根目录"）。
- 每个模型旁的状态灯（前端徽章轮询 /zouyu_model_loader/status）：
  绿=已加载(GPU)、蓝=CPU缓存、红=未加载（完全卸载）。
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
    set_switch,
    register,
    consume_signals,
    status_payload,
    log_event,
)

CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart",
    "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image",
    "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit",
    "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _file_options(category):
    files = folder_paths.get_filename_list(category)
    return files if files else ["(无文件)"]


def _folder_is_root(folder, category):
    return (folder or "").strip().strip("/\\") in ("", ".", category)


def _resolve(category, folder, name):
    """按 folder+name 解析出可加载的模型相对路径，并做目录包含校验（防路径穿越）。"""
    if name == "(无文件)":
        raise ValueError("[ZouyuModelLoader] {} 目录下没有可用模型文件".format(category))
    rel = str(name).replace("\\", "/")
    if "/" not in rel and not _folder_is_root(folder, category):
        rel = "{}/{}".format(str(folder).strip().strip("/\\"), rel)
    allowed = {f.replace("\\", "/") for f in folder_paths.get_filename_list(category)}
    if rel not in allowed:
        raise ValueError("[ZouyuModelLoader] 模型不在可加载列表中: {}/{}（请先点『🔄 刷新文件』）".format(category, rel))
    return folder_paths.get_full_path_or_raise(category, rel)


def _load_unet(unet_folder, unet_name, weight_dtype):
    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2
    path = _resolve("diffusion_models", unet_folder, unet_name)
    return comfy.sd.load_diffusion_model(path, model_options=model_options)


def _load_clip(clip_folder, clip_name, clip_type, clip_device):
    clip_type_obj = getattr(comfy.sd.CLIPType, str(clip_type).upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    model_options = {}
    if clip_device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    path = _resolve("text_encoders", clip_folder, clip_name)
    return comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type_obj,
        model_options=model_options,
    )


def _load_vae(vae_folder, vae_name):
    path = _resolve("vae", vae_folder, vae_name)
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (path, metadata, None))
    return vae


class ZouyuModelLoader(io.ComfyNode):
    """模型加载器：集成 UNETLoader / CLIPLoader / VAELoader(×2)，含文件夹选择与状态显示。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuModelLoader",
            display_name="模型加载器 (Zouyu Loader)",
            category="ZouyuAI/SeedTensor",
            description=(
                "集成 MiniMax H3 示例工作流的四个模型加载（UNET/CLIP/视频VAE/音频VAE），与官方 "
                "UNETLoader/CLIPLoader/VAELoader 完全等价。每个模型的『文件夹』默认值即官方加载器"
                "使用的目录；前端『📁』按钮可跳转到 models 目录选择子文件夹。状态灯：绿=已加载(GPU)、"
                "蓝=CPU缓存、红=未加载。『低显存模式』开启后，节点2检测到模型空闲会从显存+CPU内存"
                "彻底卸载；关闭则交由官方管理卸载到CPU内存。"
            ),
            inputs=[
                io.String.Input("unet_folder", default="diffusion_models",
                                tooltip="UNET 子文件夹（相对 ComfyUI/models 目录）"),
                io.Combo.Input("unet_name", options=_file_options("diffusion_models"),
                               tooltip="扩散模型文件（含子文件夹路径）"),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default", advanced=True,
                               tooltip="权重精度（同官方 UNETLoader）"),
                io.String.Input("clip_folder", default="text_encoders",
                                tooltip="文本编码器子文件夹（相对 models 目录）"),
                io.Combo.Input("clip_name", options=_file_options("text_encoders"),
                               tooltip="文本编码器文件"),
                io.Combo.Input("clip_type", options=CLIP_TYPES, default="minimax",
                               tooltip="文本编码器类型（MiniMax H3 用 minimax）"),
                io.Combo.Input("clip_device", options=["default", "cpu"], default="default", advanced=True),
                io.String.Input("vae_folder", default="vae", tooltip="视频VAE子文件夹（相对 models 目录）"),
                io.Combo.Input("vae_name", options=_file_options("vae"), tooltip="视频VAE文件"),
                io.String.Input("audio_vae_folder", default="vae", tooltip="音频VAE子文件夹"),
                io.Combo.Input("audio_vae_name", options=_file_options("vae"), tooltip="音频VAE文件"),
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
        zh = (language != "English")
        switch = bool(low_vram_mode)
        set_switch(switch)
        log_event("模型加载器执行（低显存模式={}）".format("开启" if switch else "关闭"))

        unet = _load_unet(unet_folder, unet_name, weight_dtype)
        clip = _load_clip(clip_folder, clip_name, clip_type, clip_device)
        vae = _load_vae(vae_folder, vae_name)
        audio_vae = _load_vae(audio_vae_folder, audio_vae_name)

        for kind, name, obj in (("unet", unet_name, unet), ("clip", clip_name, clip),
                                ("vae", vae_name, vae), ("audio_vae", audio_vae_name, audio_vae)):
            register(kind, name, obj)
            log_event("{} ← {}".format(KIND_LABELS.get(kind, kind), name))

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
            lines.append("{}: {}  ({})".format(m["label"], label, m["name"]))

        text = "\n".join(lines)
        return io.NodeOutput(unet, clip, vae, audio_vae, text,
                             ui={"text": [text], "zouyu_status": payload})
