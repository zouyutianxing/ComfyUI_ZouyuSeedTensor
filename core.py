"""
ZouyuSeedTensor 核心共享模块。

提供所有节点共用的工具函数与常量：
- 路径管理（永久 seeds/ 目录 + 临时 temp/ 目录）
- 序列化 / 反序列化（处理 comfy.nested_tensor）
- MiniMax H3 画布计算 + 图像预处理
- 图像压缩 / 解压
- GPU 环境信息
- 自动更新目录 catalog.json
- 混合逻辑 + 进度条 + 日志
"""

import os
import re
import io
import json
import math
import shutil
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


PLUGIN_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# MiniMax H3 画布常量（与官方 nodes_minimax_h3.py 保持一致）
# ---------------------------------------------------------------------------
CANVAS_MULTIPLE = 32            # VAE 16x 下采样 + 2x2 patchify -> 画布须为 32 倍数
BASE_SHORT_EDGE = 768           # 短边基准
MAX_PIXELS = 768 * 1344         # 面积上限
REF_IMAGE_SHORT_EDGE = 2048     # 参考图短边上限（"max" 策略）
FPS = 24
AUDIO_LATENT_FPS = 40

# 参考槽位上限（动态端口：参考图最多 50 个，视频/音频各 3 个）
MAX_REFERENCE_IMAGES = 50
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3

# ---------------------------------------------------------------------------
# 中英文选项归一化（combo 值默认用中文，后端兼容中英文）
# ---------------------------------------------------------------------------
_CHOICE_MAPS = {
    "canvas_mode": {
        "自动": "auto", "auto": "auto", "Auto": "auto",
        "最大": "max", "max": "max", "Max": "max",
        "自定义": "custom", "custom": "custom", "Custom": "custom",
    },
    "ref_image_size": {
        "匹配画布": "match", "match": "match", "Match": "match",
        "短边2048": "max", "max": "max", "Max": "max", "Max(2048)": "max",
    },
    "crop_mode": {
        "不裁剪": "disabled", "disabled": "disabled", "Disabled": "disabled",
        "居中裁剪": "center", "center": "center", "Center": "center",
        "等比填充": "contain", "contain": "contain", "Contain": "contain",
    },
    "storage": {
        "永久存储": "permanent", "permanent": "permanent", "Permanent": "permanent",
        "临时存储": "temp", "temp": "temp", "Temporary": "temp",
    },
    "backup": {
        "永久备份": "permanent", "permanent": "permanent", "Permanent": "permanent",
        "临时备份": "temp", "temp": "temp", "Temporary": "temp",
        "不备份": "none", "none": "none", "None": "none",
    },
}


def normalize_choice(key, value, default):
    """把中/英文 combo 值归一化为内部英文键。未知值回退 default。"""
    m = _CHOICE_MAPS.get(key, {})
    v = str(value)
    if v in m:
        return m[v]
    # 兜底：大小写不敏感
    low = v.lower()
    for k, mapped in m.items():
        if k.lower() == low:
            return mapped
    return default

# ---------------------------------------------------------------------------
# 路径管理：永久目录（seeds/）+ 临时目录（temp/）
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent
_SEEDS_DIR = _PLUGIN_DIR / "seeds"       # 永久存储
_TEMP_DIR = _PLUGIN_DIR / "temp"         # 临时存储（生成完成后清空）
_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

_CATALOG_PATH = _SEEDS_DIR / "catalog.json"

# Windows 文件名中不允许的字符
_FORBIDDEN_CHARS = set('< > : " / \\ | ? *'.split())


def now_iso() -> str:
    try:
        return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception:
        return datetime.datetime.now().isoformat(timespec="seconds")


def safe_filename(name) -> str:
    """清理文件名，移除 Windows 不允许的字符，保留中文等 Unicode 字符"""
    safe = "".join(c for c in str(name) if c not in _FORBIDDEN_CHARS).strip()
    safe = safe.rstrip(". ")
    if not safe:
        safe = "seed_tensor"
    return safe


def get_seeds_dir() -> str:
    """返回永久存储目录 seeds/ 的绝对路径"""
    return str(_SEEDS_DIR)


def get_temp_dir() -> str:
    """返回临时存储目录 temp/ 的绝对路径"""
    return str(_TEMP_DIR)


def _scan_pt(directory: Path):
    seen = set()
    files = []
    try:
        for name in sorted(os.listdir(directory)):
            if name.endswith(".pt") and name not in seen:
                seen.add(name)
                files.append(name)
    except OSError:
        pass
    return files


def scan_seed_files():
    """扫描永久目录 seeds/ 下所有 .pt 文件"""
    return _scan_pt(_SEEDS_DIR)


def scan_temp_files():
    """扫描临时目录 temp/ 下所有 .pt 文件"""
    return _scan_pt(_TEMP_DIR)


def scan_all_seed_files():
    """永久 + 临时目录的所有 .pt 文件名（永久优先，去重）"""
    result = scan_seed_files()
    seen = set(result)
    for name in scan_temp_files():
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def resolve_seed_path(name):
    """解析种子文件路径：先在永久目录找，再在临时目录找。

    返回 (path, location)，location ∈ {"permanent", "temp"}；找不到抛 FileNotFoundError。
    """
    if not name:
        raise ValueError("[ZouyuSeedTensor] 未指定种子文件名")

    fname = name if name.endswith(".pt") else f"{name}.pt"

    permanent = _SEEDS_DIR / fname
    if permanent.is_file():
        return str(permanent), "permanent"

    temp = _TEMP_DIR / fname
    if temp.is_file():
        return str(temp), "temp"

    raise FileNotFoundError(
        f"[ZouyuSeedTensor] 文件不存在: {fname}（已搜索 {_SEEDS_DIR} 与 {_TEMP_DIR}）"
    )


def clear_temp_dir():
    """清空临时目录 temp/ 内所有文件，返回删除的文件/目录数量。"""
    removed = 0
    try:
        if _TEMP_DIR.exists():
            for f in _TEMP_DIR.iterdir():
                try:
                    if f.is_symlink() or f.is_file():
                        f.unlink()
                        removed += 1
                    elif f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 清空临时目录失败: {exc}")
    return removed


def clear_temp_except(keep_name):
    """清空临时目录内所有文件，仅保留 keep_name（文件名，如 'xxx.pt'）。返回删除数量。"""
    removed = 0
    try:
        if _TEMP_DIR.exists():
            for f in _TEMP_DIR.iterdir():
                if f.name == keep_name:
                    continue
                try:
                    if f.is_symlink() or f.is_file():
                        f.unlink()
                        removed += 1
                    elif f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 清空临时目录（保留 {keep_name}）失败: {exc}")
    return removed


def copy_to_temp(src_path):
    """把 src_path 文件复制到临时目录，返回临时目录中的绝对路径。"""
    try:
        src = Path(src_path)
        if not src.is_file():
            return None
        dst = _TEMP_DIR / src.name
        if src.resolve() == dst.resolve():
            return str(dst)  # 已在临时目录，无需复制
        shutil.copy2(str(src), str(dst))
        return str(dst)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 复制到临时目录失败: {exc}")
        return None


def free_memory():
    """卸载显存与内存占用（中间变量释放后调用）。"""
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 序列化 / 反序列化（处理 NestedTensor）
# ---------------------------------------------------------------------------

def convert_to_serializable(obj):
    """递归将 NestedTensor 和普通 tensor 转为可 pickle 的纯结构（CPU）"""
    if isinstance(obj, NestedTensor):
        return {"__nested__": [convert_to_serializable(t) for t in obj.tensors]}
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


def convert_from_serializable(obj):
    """递归将纯结构还原为 NestedTensor / tensor（仍在 CPU 上）"""
    if isinstance(obj, dict):
        if "__nested__" in obj:
            return NestedTensor([convert_from_serializable(t) for t in obj["__nested__"]])
        return {k: convert_from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_from_serializable(v) for v in obj]
    return obj


def move_to_device(obj, device):
    """递归将 tensor / NestedTensor 搬到指定设备"""
    if isinstance(obj, NestedTensor):
        return NestedTensor([move_to_device(t, device) for t in obj.tensors])
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def extract_structure(data):
    """从加载的 .pt 中提取 conditioning、元数据和种子"""
    if isinstance(data, dict):
        cond = data.get("conditioning")
        meta = data.get("metadata", {})
        seed = data.get("seed", 0)
        return cond, meta, seed
    return data, {}, 0


def extract_media(data):
    """从加载的 .pt 中提取参考媒体（图片/视频/音频）"""
    if isinstance(data, dict):
        return data.get("media", {})
    return {}


# ---------------------------------------------------------------------------
# MiniMax H3 画布尺寸计算
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


def resolve_canvas(canvas_mode: str, width: int, height: int, ref_images):
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
# 图像预处理：缩放 + 裁剪
# ---------------------------------------------------------------------------

def _upscale(image, width, height, crop):
    return comfy_utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", crop).movedim(1, -1)


def ref_target_dims(img_h, img_w, canvas_w, canvas_h, ref_image_size):
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
      disabled / stretch : 直接缩放到目标 (width, height)，不裁剪
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

def image_to_bytes(image_tensor, fmt="jpeg", quality=95):
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


def bytes_to_image(bytes_list, fmt="jpeg"):
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
# GPU 环境信息
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
# 自动更新目录 catalog.json（仅索引永久目录）
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
    catalog["updated_at"] = now_iso()
    try:
        with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 更新目录失败: {exc}")
    return catalog


def update_catalog_entry(name, entry):
    cat = load_catalog()
    files = [f for f in cat.get("files", []) if f.get("name") != name]
    files.append(entry)
    files.sort(key=lambda x: x.get("name", ""))
    cat["files"] = files
    _save_catalog(cat)


def rebuild_catalog():
    """扫描永久目录，依据 sidecar 元数据快速重建目录（不加载大张量）"""
    entries = []
    for fname in scan_seed_files():
        base = os.path.splitext(fname)[0]
        side = _SEEDS_DIR / f"{base}.meta.json"
        entry = None
        if side.exists():
            try:
                with open(side, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except Exception:
                entry = None
        if entry is None:
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


def write_sidecar_meta(name, entry):
    try:
        side = _SEEDS_DIR / f"{os.path.splitext(name)[0]}.meta.json"
        with open(side, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 写 sidecar 元数据失败: {exc}")


# ---------------------------------------------------------------------------
# 混合逻辑 + 进度条 + 日志
# ---------------------------------------------------------------------------

def log(msg: str):
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
                log(f"警告: conditioning[{i}] tensor 形状 {list(other_tensor.shape)} "
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


def make_progress(total, label=""):
    """创建进度条，不可用则返回 None"""
    if _PROGRESS_AVAILABLE and total and total > 1:
        try:
            return comfy_utils.ProgressBar(total)
        except Exception:
            return None
    return None


def progress_update(pbar, step=1):
    if pbar is not None:
        try:
            pbar.update(step)
        except Exception:
            pass


def notify_files_refresh():
    """通知前端刷新文件下拉菜单"""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
    except Exception:
        pass
