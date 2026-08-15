"""
Zouyu Model Loader — 通用多模型加载节点（动态 UI：类型下拉 → 文件选择 → 自动加载）。

与官方模型加载节点对齐：用户先选择模型类型（主模型/文本模型/视频VAE/音频VAE/LoRA/其他），
再选择模型文件；节点按所选类型自动适配对应的加载器（load_diffusion_model / load_clip /
VAE / LoRA 元数据），并自动识别文件的真实类型（自动纠错互换/回退默认，不报错）。

动态 UI（前端实现）：
- 开局只显示一个"请选择模型"；选择并填好一个后自动弹出下一个；
- 每个模型一行：类型下拉 + 文件夹 + 文件下拉 + 三色状态灯 + 对应输出端口；
- 输出端口随模型动态增减，按类型编号（主模型0/主模型1/文本模型0/lora0/...）；
- 「集成模式」开关：只保留各模型下拉、状态灯、输出端口与语言/低显存开关。

输出全部为 *（任意）类型，可与 MODEL/CLIP/VAE 等任意类型端口直连。
"""

import os

import torch
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .model_guard import (
    STATE_INFO,
    set_switch,
    register,
    consume_signals,
    status_payload,
    log_event,
    _resolve_abs,
    detect_model_type,
    _read_keys,
    all_model_files,
)

MAX_MODELS = 8

# 类型选项（前端下拉显示值 = 后端键值，保持稳定）
MODEL_TYPE_OPTIONS = ["未使用", "主模型", "文本模型", "视频VAE", "音频VAE", "LoRA", "其他"]

TYPE_KEY = {
    "未使用": "",
    "主模型": "main",
    "文本模型": "clip",
    "视频VAE": "vae",
    "音频VAE": "avae",
    "LoRA": "lora",
    "其他": "other",
}

TYPE_LABELS = {
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

# 每类模型的默认分类（供"自由文件夹"回退与默认文件）
TYPE_CATEGORY = {
    "main": "diffusion_models",
    "clip": "text_encoders",
    "vae": "vae",
    "avae": "vae",
    "lora": "loras",
    "other": "diffusion_models",
}


def _file_options(category):
    files = folder_paths.get_filename_list(category) if category else []
    return files if files else ["(未选择)"]


def _default_file(category):
    files = folder_paths.get_filename_list(category) if category else []
    if not files:
        return "(未选择)"
    for f in files:
        if "minimax" in os.path.basename(f).lower():
            return f
    return files[0]


# ---------------------------------------------------------------------------
# 按类型加载（自动适应该模型的加载器）
# ---------------------------------------------------------------------------

def _load_clip_auto(path):
    """按权重键名自动选择文本编码器类型（minimax/qwen/wan/...），失败回退 stable_diffusion。"""
    keys = _read_keys(path) or []
    joined = "|".join(keys)
    base = os.path.basename(path).lower()
    clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
    if "model.embed_tokens" in joined and "visual." in joined:
        if "minimax" in base:
            clip_type = getattr(comfy.sd.CLIPType, "MINIMAX", clip_type)
        else:
            clip_type = getattr(comfy.sd.CLIPType, "QWEN_IMAGE", clip_type)
    elif "umt5" in base or "t5" in base:
        clip_type = getattr(comfy.sd.CLIPType, "WAN", clip_type)
    elif "text_model.encoder" in joined or "shared.weight" in joined:
        clip_type = getattr(comfy.sd.CLIPType, "STABLE_DIFFUSION", clip_type)
    try:
        return comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={},
        )
    except Exception:
        return comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
            model_options={},
        )


def _load_vae(path):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    obj = comfy.sd.VAE(sd=sd, metadata=metadata)
    obj.throw_exception_if_invalid()
    obj.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (path, metadata, None))
    return obj


def _load_slot_model(tkey, folder, name):
    """按用户选择类型加载；若文件真实类型与选择不符则自动适配（不报错）。"""
    category = TYPE_CATEGORY[tkey]
    path = _resolve_abs(category, folder, name)
    detected = detect_model_type(path)

    if tkey == "main":
        if detected == "clip":
            return _load_clip_auto(path), "clip", "文件实为文本模型，已自动按文本模型加载"
        if detected == "vae":
            return _load_vae(path), "vae", "文件实为VAE，已自动按VAE加载"
        return comfy.sd.load_diffusion_model(path, model_options={}), detected, ""
    if tkey == "clip":
        if detected in ("unet", "checkpoint"):
            return comfy.sd.load_diffusion_model(path, model_options={}), detected, "文件实为主模型，已自动按主模型加载"
        if detected == "vae":
            return _load_vae(path), "vae", "文件实为VAE，已自动按VAE加载"
        return _load_clip_auto(path), "clip", ""
    if tkey in ("vae", "avae"):
        if detected == "clip":
            return _load_clip_auto(path), "clip", "文件实为文本模型，已自动按文本模型加载"
        if detected in ("unet", "checkpoint"):
            return comfy.sd.load_diffusion_model(path, model_options={}), detected, "文件实为主模型，已自动按主模型加载"
        return _load_vae(path), "vae", ""
    if tkey == "lora":
        try:
            sd = comfy.utils.load_torch_file(path)
        except Exception:
            sd = {}
        return {"path": path, "name": os.path.basename(path), "state_dict": sd}, "lora", ""
    # other：自动识别后按识别结果加载，识别不出则输出文件信息
    if detected in ("unet", "checkpoint"):
        return comfy.sd.load_diffusion_model(path, model_options={}), detected, "已自动识别为主模型"
    if detected == "clip":
        return _load_clip_auto(path), "clip", "已自动识别为文本模型"
    if detected == "vae":
        return _load_vae(path), "vae", "已自动识别为VAE"
    return {"path": path, "name": os.path.basename(path), "type": "other"}, "other", "未识别出已知类型，输出文件信息"


class ZouyuModelLoader(io.ComfyNode):
    """通用模型加载器：动态多模型（类型下拉 → 文件选择 → 自动适配加载器）。"""

    @classmethod
    def define_schema(cls):
        inputs = []
        _all_files = all_model_files()
        for i in range(MAX_MODELS):
            inputs.extend([
                io.Combo.Input("model_{}_type".format(i), options=MODEL_TYPE_OPTIONS, default="未使用",
                               optional=True, tooltip="模型类型（未使用=该槽位空置）"),
                io.String.Input("model_{}_folder".format(i), default="", optional=True,
                                tooltip="模型文件夹（相对 models 目录，可留空=按类型默认）"),
                # 文件下拉选项 = 全模型合并列表（官方分类 + models 下自定义文件夹）；
                # 前端按类型/文件夹过滤显示，后端校验对任何真实文件都能通过，
                # 拖入导入的新文件数秒内即可被校验接受（无需重启）
                io.Combo.Input("model_{}_name".format(i), options=_all_files, default="(未选择)",
                               optional=True, tooltip="模型文件"),
            ])
        inputs.extend([
            io.Boolean.Input("compact_mode", default=False, label_on="集成", label_off="展开",
                             optional=True, tooltip="（已由『精简显示』开关取代，保留以兼容旧工作流）"),
            io.Boolean.Input("compact_view", default=False, label_on="完整", label_off="简洁",
                             optional=True,
                             tooltip="关闭（简洁，默认）：只显示模型下拉/三色灯/加载提示/端口/低显存/语言/本开关，"
                                     "其余全部隐藏，界面自动收缩；打开（完整）：额外显示每个下拉下方的模型文件夹选择、"
                                     "手动加载/卸载按钮与底部拖入导入条"),
            io.Boolean.Input("low_vram_mode", default=False, label_on="低显存", label_off="CPU缓存",
                             optional=True,
                             tooltip="卸载深度控制：低显存=收到模型加载开关的卸载信号时，把对应模型从显存+CPU内存彻底卸载至硬盘"
                                     "（DynamicVRAM 模型释放内存，红『已卸载』）；CPU缓存=不主动控制卸载，完全交给官方模型管理"
                                     "（官方把模型从显存卸载到 CPU 内存，蓝『未加载』，权重保留在内存）"),
            io.Combo.Input("language", options=["中文", "English"], default="中文", optional=True),
        ])
        outputs = [io.AnyType.Output("model_{}".format(i)) for i in range(MAX_MODELS)]
        return io.Schema(
            node_id="ZouyuModelLoader",
            display_name="Zouyu 模型加载器 (Model Loader)",
            category="ZouyuAI/SeedTensor",
            description=(
                "通用模型加载器（简洁模式）：最开始时只显示一个『请加载模型』下拉菜单 + 中英文切换开关 + "
                "低显存模式切换开关；在下拉中选择模型后自动在下方出现下一个『请加载模型』下拉，最多可选 8 个模型。"
                "模型类型完全自动识别（主模型/文本模型/视频VAE/音频VAE/LoRA 等），并自动适配官方加载器底层原理加载，"
                "无需手动选择类型。每个已选模型的下拉下方有一个模型文件夹选择开关（打开 models 目录浏览器，"
                "选定文件夹后下拉自动更新其中的模型文件）。每个下拉行尾有三色状态灯 + 状态提示（绿=已加载/显存、"
                "蓝=未加载/CPU内存、红=已卸载/硬盘），输出端口与该行处于同一水平线，端口名显示『类型+序号+分类』"
                "（如 主模型0 (Diffusion)）。可通过后端自动接收『模型加载控制开关』的加载/卸载信号，"
                "也可以点击行尾按钮手动加载/卸载。"
            ),
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def execute(cls, compact_mode=False, compact_view=False, low_vram_mode=False, language="中文",
                **kwargs) -> io.NodeOutput:
        zh = (language or "中文") != "English"
        switch = bool(low_vram_mode)
        set_switch(switch)
        log_event("模型加载器执行（低显存模式={}，精简显示={}）"
                  .format("开启" if switch else "关闭", "关闭(简洁)" if not compact_view else "打开(完整)"))

        # 收集已使用的槽位
        slots = []
        for i in range(MAX_MODELS):
            t = kwargs.get("model_{}_type".format(i)) or "未使用"
            name = kwargs.get("model_{}_name".format(i)) or ""
            # 集成模式：只选文件不选类型时自动识别（按"其他"处理）
            if name and name != "(未选择)":
                if t == "未使用":
                    t = "其他"
                folder = kwargs.get("model_{}_folder".format(i)) or ""
                slots.append((i, t, folder, name))

        loaded = [None] * MAX_MODELS
        actual_keys = {}
        notes = []
        for i, t, folder, name in slots:
            tkey = TYPE_KEY.get(t, "other")
            try:
                obj, actual_key, note = _load_slot_model(tkey, folder, name)
            except Exception as exc:
                # 自动恢复：改用该类型默认文件
                try:
                    default = _default_file(TYPE_CATEGORY[tkey])
                    obj, actual_key, note = _load_slot_model(tkey, TYPE_CATEGORY[tkey], default)
                    notes.append("{}槽位所选文件不可用（{}），已自动改用默认文件 {}".format(
                        TYPE_LABELS.get(tkey, {}).get("zh", t), str(exc)[:60], default))
                except Exception as exc2:
                    raise ValueError("[ZouyuModelLoader] {}槽位没有可用模型文件: {}".format(
                        TYPE_LABELS.get(tkey, {}).get("zh", t), exc2)) from exc2
            loaded[i] = obj
            actual_keys[i] = actual_key
            if note:
                notes.append(note)
            tlabel = TYPE_LABELS.get(actual_key, TYPE_LABELS["other"])
            log_event("槽位{}：{} ← {}（{}）".format(i, tlabel["zh"], os.path.basename(name), note or "正常"))

        # 登记（按槽位 + 实际识别类型），供状态灯/guard 使用
        for i, t, folder, name in slots:
            obj = loaded[i]
            if obj is None:
                continue
            register("slot{}".format(i), os.path.basename(name), obj,
                     model_type=actual_keys.get(i, TYPE_KEY.get(t, "other")),
                     folder=folder, tkey=TYPE_KEY.get(t, "other"))

        # 消费节点2（模型加载开关）传来的闲置信号
        for kind, st, _ts in consume_signals():
            log_event("收到闲置信号: {} 状态={}".format(kind, st))

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
        return io.NodeOutput(*loaded, ui={"text": [text], "zouyu_status": payload})
