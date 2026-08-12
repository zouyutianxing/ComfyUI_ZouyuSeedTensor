"""
ZouyuSeedTensor -- 种子张量缓存与混合系统（MiniMax H3 视频生成）

将 MiniMax H3 视频生成过程中的 conditioning 张量 + 种子 + 参考媒体（图片/视频/音频）
+ LoRA 栈打包保存到插件内部 seeds/ 目录，支持通过提示词中的 @文件名 引用多个已保存的
张量种子文件进行混合生成，从而节约显存占用。

主要能力：
1.  通用图像缩放：保存前将参考图统一缩放到目标尺寸（match/max 策略）
2.  计算适配 MiniMax H3 的画布尺寸（32 倍数、768 短边、面积上限），写入元数据分辨率
3.  图像裁剪 + 缩放：预处理参考图（disabled/center/contain/stretch）
4.  张量与种子除提示词/图片外，同时打包视频帧与音频波形张量
5.  自动更新的目录 catalog.json（所有张量+种子文件的索引）
6.  前端 @ 引用 / 加载时自动显示文件下拉菜单
7.  批量处理文件时显示进度条
8.  保存时记录 GPU 环境信息
9.  写入种子元数据作为溯源信息（provenance）
10. 前端动态 LoRA 槽管理（配合 ZouyuLoraStack）

节点:
- ZouyuSaveSeedConditioning : 保存 conditioning + 种子 + 参考媒体 + LoRA + 溯源元数据
- ZouyuLoadSeedConditioning : 加载单个种子张量文件（含完整元数据）
- ZouyuSeedBlender          : 解析提示词中的 @引用，混合多个种子张量
- ZouyuLoraStack           : 动态 LoRA 槽位栈（名称 + 强度）
- ZouyuExtractSeedMedia    : 从种子文件中提取参考图/视频/音频/LoRA
- ZouyuSeedCatalog         : 查看 / 刷新自动目录
"""

import os
import re
import io
import json
import math
import hashlib
import datetime

import torch
import folder_paths
import comfy.model_management as model_management
from pathlib import Path
from comfy.nested_tensor import NestedTensor

try:
    import comfy.utils as comfy_utils
    _PROGRESS_AVAILABLE = hasattr(comfy_utils, "ProgressBar")
except Exception:
    _PROGRESS_AVAILABLE = False


PLUGIN_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# MiniMax H3 画布常量（与官方 nodes_minimax_h3.py 保持一致）
# ---------------------------------------------------------------------------
CANVAS_MULTIPLE = 32            # VAE 16x 下采样 + 2x2 patchify -> 画布须为 32 倍数
BASE_SHORT_EDGE = 768           # 短边基准
MAX_PIXELS = 768 * 1344         # 面积上限
REF_IMAGE_SHORT_EDGE = 2048     # 参考图短边上限（"max" 策略）
FPS = 24
AUDIO_LATENT_FPS = 40

# 参考槽位上限（与官方 Autogrow max 一致）
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_LORA_SLOTS = 8

# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent
_SEEDS_DIR = _PLUGIN_DIR / "seeds"
_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
_CATALOG_PATH = _SEEDS_DIR / "catalog.json"

# Windows 文件名中不允许的字符
_FORBIDDEN_CHARS = set('< > : " / \\ | ? *'.split())


def _now_iso() -> str:
    try:
        return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception:
        return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_filename(name: str) -> str:
    """清理文件名，移除 Windows 不允许的字符，保留中文等 Unicode 字符"""
    safe = "".join(c for c in str(name) if c not in _FORBIDDEN_CHARS).strip()
    safe = safe.rstrip(". ")
    if not safe:
        safe = "seed_tensor"
    return safe


def get_seeds_dir() -> str:
    """返回插件内部 seeds/ 目录的绝对路径"""
    return str(_SEEDS_DIR)


def scan_seed_files():
    """扫描 seeds/ 目录下所有 .pt 文件，返回文件名列表（含 .pt 扩展名）"""
    seen = set()
    files = []
    try:
        for name in sorted(os.listdir(_SEEDS_DIR)):
            if name.endswith(".pt") and name not in seen:
                seen.add(name)
                files.append(name)
    except OSError:
        pass
    return files


def _seed_meta_path(seed_name: str) -> Path:
    """返回某个种子的轻量元数据 sidecar 路径（用于快速重建目录，避免重载大张量）"""
    base = os.path.splitext(seed_name)[0]
    return _SEEDS_DIR / f"{base}.meta.json"


# ---------------------------------------------------------------------------
# 序列化 / 反序列化（处理 NestedTensor）
# ---------------------------------------------------------------------------

def _convert_to_serializable(obj):
    """递归将 NestedTensor 和普通 tensor 转为可 pickle 的纯结构（CPU）"""
    if isinstance(obj, NestedTensor):
        return {"__nested__": [_convert_to_serializable(t) for t in obj.tensors]}
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _convert_from_serializable(obj):
    """递归将纯结构还原为 NestedTensor / tensor（仍在 CPU 上）"""
    if isinstance(obj, dict):
        if "__nested__" in obj:
            return NestedTensor([_convert_from_serializable(t) for t in obj["__nested__"]])
        return {k: _convert_from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_from_serializable(v) for v in obj]
    return obj


def _move_to_device(obj, device):
    """递归将 tensor / NestedTensor 搬到指定设备"""
    if isinstance(obj, NestedTensor):
        return NestedTensor([_move_to_device(t, device) for t in obj.tensors])
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _extract_structure(data):
    """从加载的 .pt 中提取 conditioning、元数据和种子"""
    if isinstance(data, dict):
        cond = data.get("conditioning")
        meta = data.get("metadata", {})
        seed = data.get("seed", 0)
        return cond, meta, seed
    return data, {}, 0


def _extract_media(data):
    """从加载的 .pt 中提取参考媒体（图片/视频/音频）"""
    if isinstance(data, dict):
        return data.get("media", {})
    return {}


# ---------------------------------------------------------------------------
# MiniMax H3 画布尺寸计算（功能 2）
# ---------------------------------------------------------------------------

def snap_dimension(value: int, stride: int = CANVAS_MULTIPLE) -> int:
    """将 value 取整到最接近的 stride 倍数，且不低于 stride"""
    return max(stride, round(int(value) / stride) * stride)


def adapt_canvas(width: int, height: int):
    """768 短边画布 + 768*1344 面积上限，逐轴取整到 32（与官方一致）"""
    if width <= 0 or height <= 0:
        return 0, 0
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def align_frame_count(n: int) -> int:
    """帧数对齐到 17k+5 网格"""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int):
    """返回 (frame_count, video_latent_t, audio_t)"""
    frame_count = align_frame_count(max(5, int(length)))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _resolve_canvas(canvas_mode: str, width: int, height: int, ref_images):
    """根据模式计算 MiniMax H3 适配画布尺寸。

    返回 (w, h, mode)。
    - custom: 使用给定 width/height（取整到 32）
    - auto  : 依据第一张参考图宽高比，用 adapt_canvas 计算
    - max   : 使用给定宽高，或回退到 1344x768
    """
    canvas_mode = (canvas_mode or "auto").lower()

    if canvas_mode == "custom":
        w = snap_dimension(width) if width else 0
        h = snap_dimension(height) if height else 0
        return w, h, "custom"

    src_w = src_h = 0
    if ref_images:
        for img in ref_images:
            if img is not None and getattr(img, "shape", None) and img.shape[0] > 0:
                src_h, src_w = int(img.shape[1]), int(img.shape[2])
                break

    if canvas_mode == "max" or (src_w <= 0 or src_h <= 0):
        if width and height:
            w, h = snap_dimension(width), snap_dimension(height)
        elif src_w > 0 and src_h > 0:
            w, h = adapt_canvas(src_w, src_h)
        else:
            w, h = 1344, 768
        return w, h, "max"

    w, h = adapt_canvas(src_w, src_h)
    return w, h, "auto"


# ---------------------------------------------------------------------------
# 图像预处理：缩放 + 裁剪（功能 1、3）
# ---------------------------------------------------------------------------

def _upscale(image, width, height, crop):
    return comfy_utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", crop).movedim(1, -1)


def _ref_target_dims(img_h, img_w, canvas_w, canvas_h, ref_image_size):
    """根据 match/max 策略计算参考图的目标宽高（保持宽高比，向下缩放，32 对齐）"""
    if ref_image_size == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(img_w, img_h))
    else:  # match
        scale = 1.0
        if canvas_w > 0 and canvas_h > 0 and img_w * img_h > 0:
            scale = min(1.0, math.sqrt((canvas_w * canvas_h) / (img_w * img_h)))
    tw = max(CANVAS_MULTIPLE, round(img_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(img_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def preprocess_image(img, width, height, crop_mode="disabled"):
    """将参考图 [B,H,W,C] 缩放/裁剪到目标 (width, height)。

    crop_mode:
      disabled / stretch : 直接缩放到目标 (width, height)，不裁剪（配合 _ref_target_dims 保持宽高比）
      center / cover     : 缩放铺满目标框后居中裁剪到精确目标
      contain / pad      : 保持宽高比适配后 letterbox 填充到精确目标
    """
    img = img[..., :3]
    b, h, w, c = img.shape
    mode = (crop_mode or "disabled").lower()

    if mode in ("stretch", "disabled", "none"):
        return _upscale(img, width, height, "disabled")

    if mode in ("center", "cover", "crop"):
        scale = max(width / w, height / h)
        tw = max(1, round(w * scale))
        th = max(1, round(h * scale))
        img = _upscale(img, tw, th, "disabled")
        x0 = max(0, (tw - width) // 2)
        y0 = max(0, (th - height) // 2)
        return img[:, y0:y0 + height, x0:x0 + width, :]

    if mode in ("contain", "pad", "letterbox"):
        scale = min(width / w, height / h)
        tw = max(1, round(w * scale))
        th = max(1, round(h * scale))
        img = _upscale(img, tw, th, "disabled")
        out = torch.zeros((b, height, width, c), dtype=img.dtype, device=img.device)
        y0 = max(0, (height - th) // 2)
        x0 = max(0, (width - tw) // 2)
        copy_h = min(th, height - y0)
        copy_w = min(tw, width - x0)
        out[:, y0:y0 + copy_h, x0:x0 + copy_w, :] = img[:, :copy_h, :copy_w, :]
        return out

    # 默认：直接缩放到目标尺寸
    return _upscale(img, width, height, "disabled")


# ---------------------------------------------------------------------------
# 图像压缩（用于把参考图以字节存进 .pt，便于跨分辨率重编码 / 云迁移）
# ---------------------------------------------------------------------------

def _image_to_bytes(image_tensor, fmt="jpeg", quality=95):
    from PIL import Image
    results = []
    arr = (image_tensor.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i])
        buf = io.BytesIO()
        if fmt.lower() == "png":
            img.save(buf, format="PNG", optimize=True)
        else:
            img.save(buf, format="JPEG", quality=quality)
        results.append(buf.getvalue())
    return results


def _bytes_to_image(bytes_list, fmt="jpeg"):
    from PIL import Image
    import numpy as np
    images = []
    for b in bytes_list:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        images.append(np.array(img, dtype=np.float32) / 255.0)
    if not images:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(images, axis=0))


# ---------------------------------------------------------------------------
# GPU 环境信息（功能 8）
# ---------------------------------------------------------------------------

def collect_gpu_info():
    info = {
        "torch_version": torch.__version__,
        "compute_device": str(model_management.get_torch_device()),
    }
    try:
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["cuda_available"] = False

    try:
        if torch.cuda.is_available():
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": getattr(props, "name", "unknown"),
                    "total_memory_mb": int(getattr(props, "total_memory", 0) // (1024 * 1024)),
                    "compute_capability": [getattr(props, "major", None), getattr(props, "minor", None)],
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
            info["devices"] = devices
            try:
                info["current_device"] = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info()
                info["vram_free_mb"] = int(free // (1024 * 1024))
                info["vram_total_mb"] = int(total // (1024 * 1024))
            except Exception:
                pass
        else:
            info["hip_version"] = getattr(torch.version, "hip", None)
            info["devices"] = []
    except Exception as exc:  # noqa: BLE001
        info["gpu_error"] = str(exc)
    return info


# ---------------------------------------------------------------------------
# LoRA 解析（功能 10 后端）
# ---------------------------------------------------------------------------

def list_lora_files():
    try:
        return sorted(folder_paths.get_filename_list("loras"))
    except Exception:
        return []


def _parse_lora_stack(lora_stack):
    """把 LoRA 栈解析为 [{'name', 'strength'}]。支持 dict / list / JSON 字符串。"""
    out = []
    if lora_stack is None:
        return out
    data = lora_stack
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return out
    if isinstance(data, dict):
        data = data.get("loras", data.get("items", []))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("name", "")
                if not name:
                    continue
                try:
                    strength = float(item.get("strength", item.get("strength_model", 1.0)))
                except (TypeError, ValueError):
                    strength = 1.0
                out.append({"name": str(name), "strength": strength})
    return out


# ---------------------------------------------------------------------------
# 自动更新目录 catalog.json（功能 5）
# ---------------------------------------------------------------------------

def load_catalog():
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "updated_at": "", "count": 0, "files": []}


def _save_catalog(catalog):
    catalog = dict(catalog)
    catalog["version"] = 1
    catalog["count"] = len(catalog.get("files", []))
    catalog["updated_at"] = _now_iso()
    try:
        with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 更新目录失败: {exc}")
    return catalog


def _update_catalog_entry(name, entry):
    cat = load_catalog()
    files = [f for f in cat.get("files", []) if f.get("name") != name]
    files.append(entry)
    files.sort(key=lambda x: x.get("name", ""))
    cat["files"] = files
    _save_catalog(cat)


def _remove_catalog_entry(name):
    cat = load_catalog()
    cat["files"] = [f for f in cat.get("files", []) if f.get("name") != name]
    _save_catalog(cat)


def rebuild_catalog():
    """扫描 seeds/ 目录，依据 sidecar 元数据快速重建目录（不加载大张量）"""
    entries = []
    for fname in scan_seed_files():
        base = os.path.splitext(fname)[0]
        side = _seed_meta_path(fname)
        entry = None
        if side.exists():
            try:
                with open(side, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except Exception:
                entry = None
        if entry is None:
            # 无 sidecar 的旧文件：尽力用轻量信息兜底
            try:
                p = _SEEDS_DIR / fname
                entry = {
                    "name": base,
                    "file": fname,
                    "seed": 0,
                    "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                    "saved_at": "",
                }
            except OSError:
                continue
        entry.setdefault("name", base)
        entry.setdefault("file", fname)
        entries.append(entry)
    entries.sort(key=lambda x: x.get("name", ""))
    cat = {"version": 1, "updated_at": "", "count": 0, "files": entries}
    return _save_catalog(cat)


def _write_sidecar_meta(name, entry):
    try:
        side = _seed_meta_path(name)
        with open(side, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 写 sidecar 元数据失败: {exc}")


# ---------------------------------------------------------------------------
# 混合逻辑
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(f"[ZouyuSeedTensor] {msg}")


def blend_conditionings(cond_list, weights=None):
    """将多个 conditioning 按权重混合（加权平均 + 合并 minimax_refs）"""
    if not cond_list:
        raise ValueError("[ZouyuSeedTensor] 没有可混合的 conditioning")

    if len(cond_list) == 1:
        return cond_list[0]

    if weights is None:
        weights = [1.0 / len(cond_list)] * len(cond_list)

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    base = cond_list[0]
    result = []

    for batch_idx in range(len(base)):
        base_entry = base[batch_idx]
        base_tensor = base_entry[0]
        base_dict = dict(base_entry[1])

        blended_tensor = base_tensor.clone() * weights[0]
        for i in range(1, len(cond_list)):
            other_tensor = cond_list[i][batch_idx][0]
            if other_tensor.shape != blended_tensor.shape:
                _log(f"警告: conditioning[{i}] tensor 形状 {list(other_tensor.shape)} "
                     f"与基准 {list(blended_tensor.shape)} 不一致，将跳过")
                continue
            blended_tensor += other_tensor * weights[i]

        all_refs = []
        for i, cond in enumerate(cond_list):
            entry = cond[batch_idx]
            entry_dict = entry[1] if isinstance(entry, list) and len(entry) >= 2 else {}
            refs = entry_dict.get("minimax_refs", [])
            if refs:
                all_refs.extend(refs)
        if all_refs:
            base_dict["minimax_refs"] = all_refs

        if "pooled_output" in base_dict:
            pooled = base_dict["pooled_output"].clone() * weights[0]
            for i in range(1, len(cond_list)):
                entry = cond_list[i][batch_idx]
                entry_dict = entry[1] if isinstance(entry, list) and len(entry) >= 2 else {}
                if "pooled_output" in entry_dict:
                    pooled += entry_dict["pooled_output"] * weights[i]
            base_dict["pooled_output"] = pooled

        result.append([blended_tensor, base_dict])

    return result


def _make_progress(total, label=""):
    """创建进度条（功能 7），不可用则返回一个空对象"""
    if _PROGRESS_AVAILABLE and total and total > 1:
        try:
            pbar = comfy_utils.ProgressBar(total)
            return pbar
        except Exception:
            return None
    return None


def _progress_update(pbar, step=1):
    if pbar is not None:
        try:
            pbar.update(step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 节点: ZouyuSaveSeedConditioning
# ---------------------------------------------------------------------------

class ZouyuSaveSeedConditioning:
    """将 conditioning 张量 + 种子 + 参考媒体 + LoRA + 溯源元数据打包保存。

    输入:
    - conditioning: MiniMaxH3ReferenceToVideo / Director Conditioning 输出
    - seed / filename / language
    - canvas_mode / width / height / ref_image_size / crop_mode（画布与图像预处理）
    - reference_image_0..8 / ref_video_0..2 / ref_video_audio_0..2 / ref_audio_0..2
    - lora_stack: 来自 ZouyuLoraStack 的 LoRA 栈
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "来自 MiniMaxH3ReferenceToVideo 或 Director Conditioning 的 conditioning 输出"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "当前使用的随机种子"
                }),
                "filename": ("STRING", {
                    "default": "my_seed",
                    "tooltip": "保存文件名（不含扩展名），如 shot_001_black_cat"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
                "canvas_mode": (["auto", "max", "custom"], {
                    "default": "auto",
                    "tooltip": "画布计算模式：auto=按参考图宽高比自适应；max=使用给定/默认尺寸；custom=使用 width/height"
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "目标宽度（0=自动）。MiniMax H3 要求 32 的倍数"
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "目标高度（0=自动）。MiniMax H3 要求 32 的倍数"
                }),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "参考图统一缩放策略：match=按生成画布面积；max=短边 2048 高保真"
                }),
                "crop_mode": (["disabled", "center", "contain"], {
                    "default": "disabled",
                    "tooltip": "参考图裁剪+缩放方式：disabled=按宽高比缩放不裁剪；center=铺满画布居中裁剪；contain=letterbox 填充"
                }),
            },
            "optional": {
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "lora_stack": ("ZOUYU_LORA_STACK", {
                    "tooltip": "来自 ZouyuLoraStack 的 LoRA 栈，写入溯源元数据"
                }),
                "ref_image_format": (["jpeg", "png"], {"default": "jpeg"}),
                "reference_image_0": ("IMAGE", {"tooltip": "参考图 1"}),
                "reference_image_1": ("IMAGE", {"tooltip": "参考图 2"}),
                "reference_image_2": ("IMAGE", {"tooltip": "参考图 3"}),
                "reference_image_3": ("IMAGE", {"tooltip": "参考图 4"}),
                "reference_image_4": ("IMAGE", {"tooltip": "参考图 5"}),
                "reference_image_5": ("IMAGE", {"tooltip": "参考图 6"}),
                "reference_image_6": ("IMAGE", {"tooltip": "参考图 7"}),
                "reference_image_7": ("IMAGE", {"tooltip": "参考图 8"}),
                "reference_image_8": ("IMAGE", {"tooltip": "参考图 9"}),
                "ref_video_0": ("IMAGE", {"tooltip": "参考视频 1（帧序列，24fps，2-15s）"}),
                "ref_video_1": ("IMAGE", {"tooltip": "参考视频 2"}),
                "ref_video_2": ("IMAGE", {"tooltip": "参考视频 3"}),
                "ref_video_audio_0": ("AUDIO", {"tooltip": "参考视频 1 的配乐"}),
                "ref_video_audio_1": ("AUDIO", {"tooltip": "参考视频 2 的配乐"}),
                "ref_video_audio_2": ("AUDIO", {"tooltip": "参考视频 3 的配乐"}),
                "ref_audio_0": ("AUDIO", {"tooltip": "独立参考音频 1"}),
                "ref_audio_1": ("AUDIO", {"tooltip": "独立参考音频 2"}),
                "ref_audio_2": ("AUDIO", {"tooltip": "独立参考音频 3"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ZouyuAI/SeedTensor"

    def _collect_ref_images(self, kwargs):
        return [kwargs.get(f"reference_image_{i}") for i in range(MAX_REFERENCE_IMAGES)]

    def _collect_ref_videos(self, kwargs):
        return [kwargs.get(f"ref_video_{i}") for i in range(MAX_REFERENCE_VIDEOS)]

    def _collect_ref_audios(self, kwargs):
        return [kwargs.get(f"ref_audio_{i}") for i in range(MAX_REFERENCE_AUDIOS)]

    def _collect_ref_video_audios(self, kwargs):
        return [kwargs.get(f"ref_video_audio_{i}") for i in range(MAX_REFERENCE_VIDEOS)]

    def save(self, conditioning, seed, filename, language, canvas_mode="auto",
             width=0, height=0, ref_image_size="match", crop_mode="disabled",
             prompt_text="", duration=0.0, lora_stack=None, ref_image_format="jpeg",
             **kwargs):
        safe_name = _safe_filename(filename)
        path = os.path.join(get_seeds_dir(), f"{safe_name}.pt")

        zh = (language != "English")

        ref_images = self._collect_ref_images(kwargs)
        ref_videos = self._collect_ref_videos(kwargs)
        ref_video_audios = self._collect_ref_video_audios(kwargs)
        ref_audios = self._collect_ref_audios(kwargs)

        # ---- 功能 2：计算适配 MiniMax H3 的画布尺寸 ----
        canvas_w, canvas_h, canvas_used_mode = _resolve_canvas(canvas_mode, width, height, ref_images)

        # ---- 功能 1 + 3：预处理参考图（统一缩放 + 裁剪缩放）----
        preproc_images = []
        preproc_shapes = []
        for idx, img in enumerate(ref_images):
            if img is None or getattr(img, "shape", None) is None or img.shape[0] == 0:
                continue
            if canvas_w > 0 and canvas_h > 0:
                if crop_mode in ("center", "contain"):
                    # 裁剪/填充到精确画布尺寸
                    proc = preprocess_image(img, canvas_w, canvas_h, crop_mode)
                else:
                    # 保持宽高比缩放到 32 对齐目标尺寸（无裁剪）
                    tw, th = _ref_target_dims(img.shape[1], img.shape[2], canvas_w, canvas_h, ref_image_size)
                    proc = preprocess_image(img, tw, th, "disabled")
            else:
                proc = img[..., :3]
            preproc_images.append(proc)
            preproc_shapes.append([int(proc.shape[1]), int(proc.shape[2])])

        # 参考图压缩为字节
        ref_image_bytes = []
        ref_image_shapes = []
        for proc in preproc_images:
            data_list = _image_to_bytes(proc, fmt=ref_image_format)
            for d, s in zip(data_list, [[proc.shape[1], proc.shape[2]]] * len(data_list)):
                ref_image_bytes.append(d)
                ref_image_shapes.append(s)

        # ---- 功能 4：参考视频帧 + 音频波形张量 ----
        ref_video_tensors = []
        for v in ref_videos:
            if v is None or getattr(v, "shape", None) is None or v.shape[0] == 0:
                continue
            ref_video_tensors.append(v[..., :3].detach().to(torch.float16).cpu())

        def _serialize_audio(audio):
            if not isinstance(audio, dict):
                return None
            wave = audio.get("waveform")
            if wave is None:
                return None
            return {
                "waveform": wave.detach().cpu(),
                "sample_rate": int(audio.get("sample_rate", 44100)),
            }

        ref_video_audio_tensors = []
        for a in ref_video_audios:
            sa = _serialize_audio(a)
            if sa is not None:
                ref_video_audio_tensors.append(sa)

        ref_audio_tensors = []
        for a in ref_audios:
            sa = _serialize_audio(a)
            if sa is not None:
                ref_audio_tensors.append(sa)

        # ---- 功能 10：LoRA 栈 ----
        loras = _parse_lora_stack(lora_stack)

        # ---- 功能 8：GPU 环境信息 ----
        gpu_info = collect_gpu_info()

        # ---- 序列化 conditioning ----
        cond_data = _convert_to_serializable(conditioning)

        frame_count, latent_t, audio_t = (0, 0, 0)
        if duration and duration > 0:
            frame_count, latent_t, audio_t = temporal_shape(round(duration * FPS))

        # ---- 功能 9：溯源元数据 ----
        metadata = {
            "seed": int(seed),
            "prompt_text": prompt_text,
            "duration": float(duration) if duration else 0.0,
            "width": int(canvas_w),
            "height": int(canvas_h),
            "resolution": {"width": int(canvas_w), "height": int(canvas_h), "canvas_mode": canvas_used_mode},
            "canvas": {"stride": CANVAS_MULTIPLE, "short_edge": BASE_SHORT_EDGE, "max_pixels": MAX_PIXELS},
            "frame_rate": FPS,
            "frame_count": int(frame_count),
            "latent_t": int(latent_t),
            "audio_t": int(audio_t),
            "ref_image_size": ref_image_size,
            "crop_mode": crop_mode,
            "ref_image_format": ref_image_format,
            "ref_image_count": len(ref_image_bytes),
            "ref_image_shapes": ref_image_shapes,
            "ref_video_count": len(ref_video_tensors),
            "ref_video_audio_count": len(ref_video_audio_tensors),
            "ref_audio_count": len(ref_audio_tensors),
            "loras": loras,
            "gpu": gpu_info,
            "provenance": {
                "plugin": "ComfyUI_ZouyuSeedTensor",
                "plugin_version": PLUGIN_VERSION,
                "saved_at": _now_iso(),
                "model": "MiniMax H3",
                "compatible_models": ["MiniMax H3", "generic CONDITIONING (comfy.nested_tensor)"],
                "format_version": 2,
                "content_hash": "",
            },
        }

        # 计算内容指纹（轻量、稳定）
        try:
            fingerprint = {
                "seed": int(seed),
                "prompt": prompt_text,
                "resolution": [int(canvas_w), int(canvas_h)],
                "duration": float(duration) if duration else 0.0,
                "ref_image_count": len(ref_image_bytes),
                "ref_video_count": len(ref_video_tensors),
                "ref_audio_count": len(ref_audio_tensors),
                "loras": loras,
            }
            metadata["provenance"]["content_hash"] = hashlib.sha256(
                json.dumps(fingerprint, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:16]
        except Exception:
            pass

        media = {
            "ref_images": {"format": ref_image_format, "bytes": ref_image_bytes, "shapes": ref_image_shapes},
            "ref_videos": ref_video_tensors,
            "ref_video_audios": ref_video_audio_tensors,
            "ref_audios": ref_audio_tensors,
        }

        wrapper = {
            "conditioning": cond_data,
            "seed": int(seed),
            "metadata": metadata,
            "media": media,
        }

        # ---- 功能 7：进度（序列化 + 保存）----
        pbar = _make_progress(1 + max(1, len(ref_image_bytes)), label="保存种子张量")
        _progress_update(pbar, 1)

        torch.save(wrapper, path)
        _progress_update(pbar, len(ref_image_bytes))

        mb = os.path.getsize(path) / (1024 * 1024)

        # ---- 功能 5：自动更新目录 + sidecar ----
        file_sha = ""
        if mb < 512:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                file_sha = h.hexdigest()
            except Exception:
                pass

        entry = {
            "name": safe_name,
            "file": f"{safe_name}.pt",
            "seed": int(seed),
            "prompt": (prompt_text or "")[:200],
            "width": int(canvas_w),
            "height": int(canvas_h),
            "canvas_mode": canvas_used_mode,
            "duration": float(duration) if duration else 0.0,
            "size_mb": round(mb, 2),
            "sha256": file_sha,
            "content_hash": metadata["provenance"]["content_hash"],
            "saved_at": metadata["provenance"]["saved_at"],
            "gpu": gpu_info,
            "ref_image_count": len(ref_image_bytes),
            "ref_video_count": len(ref_video_tensors),
            "ref_audio_count": len(ref_audio_tensors),
            "loras": loras,
            "plugin_version": PLUGIN_VERSION,
        }
        _write_sidecar_meta(f"{safe_name}.pt", entry)
        _update_catalog_entry(safe_name, entry)

        if zh:
            _log(f"已保存种子张量 -> {path} ({mb:.1f} MB, seed={seed}, "
                 f"画布={canvas_w}x{canvas_h}({canvas_used_mode}), 参考图={len(ref_image_bytes)}, "
                 f"视频={len(ref_video_tensors)}, 音频={len(ref_audio_tensors)}, LoRA={len(loras)})")
        else:
            _log(f"Saved seed tensor -> {path} ({mb:.1f} MB, seed={seed}, "
                 f"canvas={canvas_w}x{canvas_h}, refs={len(ref_image_bytes)}, "
                 f"videos={len(ref_video_tensors)}, audios={len(ref_audio_tensors)}, loras={len(loras)})")

        # 通知前端刷新下拉菜单
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
        except Exception:
            pass

        summary = (
            f"文件: {safe_name}.pt\n"
            f"种子: {seed}\n"
            f"画布: {canvas_w}x{canvas_h} ({canvas_used_mode})\n"
            f"参考图: {len(ref_image_bytes)}  视频: {len(ref_video_tensors)}  音频: {len(ref_audio_tensors)}\n"
            f"LoRA: {len(loras)}  大小: {mb:.1f} MB"
        )

        return {"ui": {"text": [summary]}, "result": (path,)}


# ---------------------------------------------------------------------------
# 节点: ZouyuLoadSeedConditioning
# ---------------------------------------------------------------------------

class ZouyuLoadSeedConditioning:
    """加载单个种子张量文件，输出 conditioning + 种子 + 完整元数据。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = scan_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return {
            "required": {
                "file_name": (files, {"tooltip": "选择 seeds/ 目录下的种子张量文件"}),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "metadata")
    FUNCTION = "load"
    CATEGORY = "ZouyuAI/SeedTensor"

    def load(self, file_name, language):
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件，请先使用 ZouyuSaveSeedConditioning 保存")

        path = os.path.join(get_seeds_dir(), file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ZouyuSeedTensor] 文件不存在: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        cond_data, meta, seed = _extract_structure(data)

        cond = _convert_from_serializable(cond_data)
        device = model_management.get_torch_device()
        cond = _move_to_device(cond, device)

        mb = os.path.getsize(path) / (1024 * 1024)

        # 用 JSON 输出完整元数据（含溯源 + GPU + LoRA）
        try:
            meta_display = json.dumps(meta, ensure_ascii=False, indent=2, default=str)
        except Exception:
            meta_display = str(meta)

        if language == "English":
            _log(f"Loaded seed tensor <- {file_name} ({mb:.1f} MB, seed={seed}) -> {device}")
        else:
            _log(f"已加载种子张量 <- {file_name} ({mb:.1f} MB, seed={seed}) -> {device}")

        return (cond, int(seed), meta_display)


# ---------------------------------------------------------------------------
# 节点: ZouyuSeedBlender
# ---------------------------------------------------------------------------

class ZouyuSeedBlender:
    """解析提示词中的 @文件名 引用，加载多个种子张量文件并混合。

    用法：在 prompt 中使用 @文件名 引用已保存的种子张量（不含 .pt 扩展名）。
    例如: "@shot_001 @shot_002 一个拿着剑的角色在森林中行走"
    前端会在输入 @ 时自动弹出可用的种子文件名下拉菜单。
    """

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "提示词文本。使用 @文件名 引用 seeds/ 目录下的种子张量文件（输入 @ 会弹出下拉菜单）"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
            "optional": {
                "weights": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "可选权重: @name=0.7,@name2=0.3。留空则等权平均。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "source_names", "cleaned_prompt")
    FUNCTION = "blend"
    CATEGORY = "ZouyuAI/SeedTensor"

    def _parse_weight_map(self, weights_str):
        wmap = {}
        if not weights_str or not weights_str.strip():
            return wmap
        wpat = re.compile(r'@([\w\-.\u4e00-\u9fff]+)\s*=\s*([\d.]+)')
        for m in wpat.finditer(weights_str):
            try:
                w = float(m.group(2))
                if w > 0:
                    wmap[m.group(1)] = w
            except ValueError:
                pass
        return wmap

    def blend(self, prompt, language, weights=""):
        prompt = prompt or ""
        zh = (language != "English")

        refs = self._AT_PATTERN.findall(prompt)
        if not refs:
            raise ValueError(
                "[ZouyuSeedTensor] 提示词中未找到任何 @文件名 引用。请使用 @文件名 引用种子张量文件。"
                if zh else
                "[ZouyuSeedTensor] No @filename references found in prompt."
            )

        seen = set()
        unique_refs = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                unique_refs.append(r)

        weight_map = self._parse_weight_map(weights)

        cond_list = []
        seeds = []
        loaded_names = []
        device = model_management.get_torch_device()

        # ---- 功能 7：批量加载进度 ----
        pbar = _make_progress(len(unique_refs), label="加载种子")

        for ref_name in unique_refs:
            fname = ref_name if ref_name.endswith(".pt") else f"{ref_name}.pt"
            path = os.path.join(get_seeds_dir(), fname)
            if not os.path.isfile(path):
                _log(f"警告: 文件 {fname} 不存在，跳过" if zh else f"Warning: file {fname} not found")
                _progress_update(pbar, 1)
                continue

            data = torch.load(path, map_location="cpu", weights_only=False)
            cond_data, meta, seed = _extract_structure(data)
            cond = _convert_from_serializable(cond_data)
            cond = _move_to_device(cond, device)

            cond_list.append(cond)
            seeds.append(seed)
            loaded_names.append(ref_name)
            _progress_update(pbar, 1)

        if not cond_list:
            raise FileNotFoundError(
                f"[ZouyuSeedTensor] 所有 @引用 的文件都不存在于 {get_seeds_dir()} 目录下。"
                if zh else f"[ZouyuSeedTensor] None of the @referenced files exist."
            )

        if weight_map and loaded_names:
            blend_weights = [weight_map.get(name, 1.0) for name in loaded_names]
        else:
            blend_weights = [1.0] * len(cond_list)

        blended_cond = blend_conditionings(cond_list, blend_weights)

        total_w = sum(blend_weights)
        if total_w > 0 and seeds:
            blended_seed = int(round(sum(s * w for s, w in zip(seeds, blend_weights)) / total_w))
        else:
            blended_seed = seeds[0] if seeds else 0

        cleaned_prompt = self._AT_PATTERN.sub('', prompt).strip()
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        source_names_str = (
            "源: " + ", ".join(f"@{n}(seed={s})" for n, s in zip(loaded_names, seeds))
            if zh else
            "Sources: " + ", ".join(f"@{n}(seed={s})" for n, s in zip(loaded_names, seeds))
        )

        return (blended_cond, blended_seed, source_names_str, cleaned_prompt)


# ---------------------------------------------------------------------------
# 节点: ZouyuLoraStack（功能 10 后端：动态 LoRA 槽）
# ---------------------------------------------------------------------------

class ZouyuLoraStack:
    """动态 LoRA 槽位栈：名称 + 强度。前端提供 +/- 动态增删槽位。

    输出 ZOUYU_LORA_STACK 供 ZouyuSaveSeedConditioning 的 lora_stack 输入使用，
    同时输出 lora_json 字符串供其它模型/节点参考。
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "lora_count": ("INT", {
                    "default": 1, "min": 0, "max": MAX_LORA_SLOTS,
                    "tooltip": "当前激活的 LoRA 槽位数量（前端 +/- 按钮会同步此值）"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
            "optional": {},
        }
        for i in range(1, MAX_LORA_SLOTS + 1):
            inputs["optional"][f"lora_name_{i}"] = ("STRING", {
                "default": "",
                "tooltip": "LoRA 文件名（如 my_lora.safetensors）。可在目录 loras/ 下查看可用文件"
            })
            inputs["optional"][f"lora_strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })
        return inputs

    RETURN_TYPES = ("ZOUYU_LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_json")
    FUNCTION = "build"
    CATEGORY = "ZouyuAI/SeedTensor"

    def build(self, lora_count=0, language="中文", **kwargs):
        zh = (language != "English")
        loras = []
        for i in range(1, MAX_LORA_SLOTS + 1):
            name = kwargs.get(f"lora_name_{i}")
            strength = kwargs.get(f"lora_strength_{i}", 1.0)
            if name in (None, "", "(无可用 lora)"):
                continue
            try:
                strength = float(strength)
            except (TypeError, ValueError):
                strength = 1.0
            loras.append({"name": str(name), "strength": strength})

        stack = {"loras": loras}
        lora_json = json.dumps(stack, ensure_ascii=False)
        desc = ", ".join(f"{l['name']}:{l['strength']}" for l in loras)
        if zh:
            _log(f"LoRA 栈: {len(loras)} 个 -> [{desc}]")
        else:
            _log(f"LoRA stack: {len(loras)} -> [{desc}]")
        return (stack, lora_json)


# ---------------------------------------------------------------------------
# 节点: ZouyuExtractSeedMedia（功能 4 加载侧：提取参考图/视频/音频/LoRA）
# ---------------------------------------------------------------------------

class ZouyuExtractSeedMedia:
    """从已保存的种子文件中提取参考媒体与 LoRA 信息。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = scan_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return {
            "required": {
                "file_name": (files, {"tooltip": "选择种子张量文件"}),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("ref_images", "ref_videos", "ref_audio", "lora_json")
    FUNCTION = "extract"
    CATEGORY = "ZouyuAI/SeedTensor"

    def extract(self, file_name, language):
        zh = (language != "English")
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件")

        path = os.path.join(get_seeds_dir(), file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ZouyuSeedTensor] 文件不存在: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        _, meta, _ = _extract_structure(data)
        media = _extract_media(data)

        # 参考图
        ref_images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        img_data = media.get("ref_images", {}) if isinstance(media, dict) else {}
        if isinstance(img_data, dict) and img_data.get("bytes"):
            ref_images = _bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))

        # 参考视频（帧序列）
        ref_videos = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        videos = media.get("ref_videos", []) if isinstance(media, dict) else []
        if videos:
            try:
                ref_videos = torch.cat([v.float() for v in videos], dim=0)
            except Exception:
                ref_videos = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

        # 参考音频（取第一个）
        ref_audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
        audios = media.get("ref_audios", []) if isinstance(media, dict) else []
        if audios and isinstance(audios[0], dict) and "waveform" in audios[0]:
            ref_audio = audios[0]

        loras = meta.get("loras", []) if isinstance(meta, dict) else []
        lora_json = json.dumps({"loras": loras}, ensure_ascii=False)

        if zh:
            _log(f"提取种子媒体 <- {file_name}: 参考图={ref_images.shape[0]}, 视频帧={ref_videos.shape[0]}, LoRA={len(loras)}")
        else:
            _log(f"Extracted seed media <- {file_name}: images={ref_images.shape[0]}, frames={ref_videos.shape[0]}, loras={len(loras)}")

        return (ref_images, ref_videos, ref_audio, lora_json)


# ---------------------------------------------------------------------------
# 节点: ZouyuSeedCatalog（功能 5：查看/刷新自动目录）
# ---------------------------------------------------------------------------

class ZouyuSeedCatalog:
    """查看 / 刷新 seeds/ 目录的自动索引 catalog.json。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rebuild": ("BOOLEAN", {
                    "default": False,
                    "label_on": "重建目录",
                    "label_off": "读取目录",
                    "tooltip": "True 时重新扫描 seeds/ 目录重建索引"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("catalog_json", "count")
    FUNCTION = "catalog"
    CATEGORY = "ZouyuAI/SeedTensor"

    def catalog(self, rebuild=False, language="中文"):
        zh = (language != "English")
        if rebuild:
            cat = rebuild_catalog()
        else:
            cat = load_catalog()
        files = cat.get("files", [])
        cat_json = json.dumps(cat, ensure_ascii=False, indent=2, default=str)
        _log(f"目录索引: {len(files)} 个种子文件" if zh else f"Catalog: {len(files)} seed files")
        return (cat_json, len(files))


# ---------------------------------------------------------------------------
# 节点注册
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ZouyuSaveSeedConditioning": ZouyuSaveSeedConditioning,
    "ZouyuLoadSeedConditioning": ZouyuLoadSeedConditioning,
    "ZouyuSeedBlender": ZouyuSeedBlender,
    "ZouyuLoraStack": ZouyuLoraStack,
    "ZouyuExtractSeedMedia": ZouyuExtractSeedMedia,
    "ZouyuSeedCatalog": ZouyuSeedCatalog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZouyuSaveSeedConditioning": "Zouyu Save Seed+Tensor (保存种子张量)",
    "ZouyuLoadSeedConditioning": "Zouyu Load Seed+Tensor (加载种子张量)",
    "ZouyuSeedBlender": "Zouyu Seed Blender (多种子混合器)",
    "ZouyuLoraStack": "Zouyu LoRA Stack (动态 LoRA 槽)",
    "ZouyuExtractSeedMedia": "Zouyu Extract Seed Media (提取参考媒体)",
    "ZouyuSeedCatalog": "Zouyu Seed Catalog (种子目录)",
}
