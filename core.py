"""
ZouyuSeedTensor 核心共享模块。

提供所有节点共用的工具函数与常量：
- 路径管理（永久 seeds/ 目录 + 临时 temp/ 目录）
- 序列化 / 反序列化（处理 comfy.nested_tensor）
- 图像压缩 / 解压
- GPU 环境信息
- 自动更新目录 catalog.json
- 混合逻辑 + 进度条 + 日志
"""

import os
import io
import json
import shutil
import datetime
import gc

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

# 帧率常量（temporal_shape 计算用）
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
    "ref_image_size": {
        "匹配画布": "match", "match": "match", "Match": "match",
        "短边2048": "max", "max": "max", "Max": "max", "Max(2048)": "max",
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
_TEMP_DIR = _PLUGIN_DIR / "temp"         # 临时存储（生成完成后自动清理）
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


def extract_media(data):
    """从加载的 .pt 中提取参考媒体（图片/视频/音频）"""
    if isinstance(data, dict):
        return data.get("media", {})
    return {}


# ---------------------------------------------------------------------------
# 帧数 / 时长计算（MiniMax H3 训练网格 17k+5）
# ---------------------------------------------------------------------------

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
    # 不同尺寸的图像无法直接 np.stack，统一 pad 到最大 H/W（黑边用 0 填充）。
    # 调用方若持有原始 shapes（media.ref_images.shapes），可据此裁剪回原尺寸，无损。
    max_h = max(int(im.shape[0]) for im in images)
    max_w = max(int(im.shape[1]) for im in images)
    padded = []
    for im in images:
        if im.shape[0] == max_h and im.shape[1] == max_w:
            padded.append(im)
        else:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.float32)
            canvas[:im.shape[0], :im.shape[1], :] = im
            padded.append(canvas)
    return torch.from_numpy(np.stack(padded, axis=0))


# ---------------------------------------------------------------------------
# 视频帧编解码（H.264，用于压缩种子文件中的参考视频）
# ---------------------------------------------------------------------------

def frames_to_video_bytes(frames, fps=FPS, quality=8):
    """把视频帧序列 [N,H,W,3] float(0-1) 编码为 H.264 mp4 字节流。

    相比直接存 float16 原始帧，H.264 可缩小 50~100 倍；参考视频在重新编码时
    本就会缩放到画布尺寸再进 VAE，轻微有损对最终生成质量几乎无影响。
    返回 bytes；失败时回退为 uint8 张量（仍比 float16 小一半）。
    """
    if frames is None:
        return None
    try:
        import numpy as np
        import imageio.v2 as imageio
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 视频编码库不可用，回退 uint8 存储: {exc}")
        return _frames_to_uint8(frames)

    frames = frames[..., :3].detach().cpu()
    n = int(frames.shape[0])
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if n == 0:
        return None
    # H.264 yuv420p 需要偶数尺寸；imageio 默认 macro_block_size=16，
    # 为避免其内部 resize 导致像素错位，这里预先 pad 到 16 的倍数，
    # 解码后由 original_shape 裁剪回原尺寸（黑边被裁掉，无损）。
    out_h = ((h + 15) // 16) * 16
    out_w = ((w + 15) // 16) * 16

    fd, tmp = None, None
    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        writer = imageio.get_writer(
            tmp, format="FFMPEG", fps=int(fps), codec="libx264",
            quality=quality, pixelformat="yuv420p", macro_block_size=16,
            # veryfast 预设：烘焙种子里的参考视频时编码快 2~3 倍，画质差异可忽略
            ffmpeg_params=["-preset", "veryfast"],
        )
        for i in range(n):
            frame = frames[i]
            if out_h != h or out_w != w:
                canvas = torch.zeros((out_h, out_w, 3), dtype=frame.dtype)
                canvas[:h, :w, :] = frame
                frame = canvas
            arr = (frame.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).numpy()
            writer.append_data(arr)
        writer.close()
        with open(tmp, "rb") as f:
            return f.read()
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 视频 H.264 编码失败，回退 uint8 存储: {exc}")
        return _frames_to_uint8(frames)
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _frames_to_uint8(frames):
    """回退方案：uint8 存储（比 float16 小一半）"""
    return (frames[..., :3].clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).detach().cpu()


def video_bytes_to_frames(data, original_shape=None):
    """把 H.264 mp4 字节流（或旧版 uint8/float16 张量）解码回帧序列 [N,H,W,3] float(0-1)。

    original_shape: (h, w)，若提供则裁剪回原始尺寸（H.264 会 pad 到偶数）。
    """
    if data is None:
        return None
    # 向后兼容旧格式：直接存张量
    if torch.is_tensor(data):
        return data[..., :3].float()
    if isinstance(data, (bytes, bytearray)):
        try:
            import numpy as np
            import imageio.v2 as imageio
            fd, tmp = None, None
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                reader = imageio.get_reader(tmp, format="FFMPEG")
                frames = [f for f in reader]
                reader.close()
                if not frames:
                    return None
                arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
                out = torch.from_numpy(arr)
                if original_shape:
                    h, w = int(original_shape[0]), int(original_shape[1])
                    out = out[:, :h, :w, :]
                return out
            finally:
                if tmp:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        except Exception as exc:  # noqa: BLE001
            print(f"[ZouyuSeedTensor] 视频解码失败: {exc}")
            return None
    return None


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
# 自动更新目录 catalog.json（仅索引永久目录，供 @引用识别）
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


def write_sidecar_meta(name, entry):
    try:
        side = _SEEDS_DIR / f"{os.path.splitext(name)[0]}.meta.json"
        with open(side, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[ZouyuSeedTensor] 写 sidecar 元数据失败: {exc}")


# ---------------------------------------------------------------------------
# 进度条 + 日志
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"[ZouyuSeedTensor] {msg}")


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
