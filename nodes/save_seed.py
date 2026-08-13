"""
节点: ZouyuSaveSeedConditioning（保存种子张量）

将 conditioning 张量 + 种子 + 参考媒体（图片/视频/音频）+ 溯源元数据打包保存。
支持保存到永久目录（seeds/）或临时目录（temp/）。

参考图/视频/音频端口由前端动态增减（参考图最多 50 个）。
combo 值使用英文内部键（后端稳定），显示文字由前端 i18n 切换中英文。
"""

import os
import json
import hashlib

import torch

from ..core import (
    PLUGIN_VERSION,
    CANVAS_MULTIPLE, BASE_SHORT_EDGE, MAX_PIXELS, FPS,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, MAX_REFERENCE_AUDIOS,
    safe_filename, now_iso,
    get_seeds_dir, get_temp_dir,
    convert_to_serializable,
    resolve_canvas, ref_target_dims, preprocess_image,
    image_to_bytes, collect_gpu_info, temporal_shape,
    normalize_choice, update_catalog_entry, write_sidecar_meta,
    make_progress, progress_update, notify_files_refresh, log,
)


class ZouyuSaveSeedConditioning:
    """保存 conditioning + 种子 + 参考媒体 + 溯源元数据。"""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "prompt_text": ("STRING", {"default": "", "multiline": True}),
            "duration": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "ref_image_format": (["jpeg", "png"], {"default": "jpeg"}),
        }
        # 参考端口统一 0 起点，由前端动态扩展；仅 0 号端口提供说明
        optional.update({f"reference_image_{i}": ("IMAGE", {}) for i in range(MAX_REFERENCE_IMAGES)})
        optional.update({f"ref_video_{i}": ("IMAGE", {}) for i in range(MAX_REFERENCE_VIDEOS)})
        optional.update({f"ref_video_audio_{i}": ("AUDIO", {}) for i in range(MAX_REFERENCE_VIDEOS)})
        optional.update({f"ref_audio_{i}": ("AUDIO", {}) for i in range(MAX_REFERENCE_AUDIOS)})
        optional["reference_image_0"] = ("IMAGE", {"tooltip": "参考图 1（连接后自动出现参考图 2，最多 50 张）"})
        optional["ref_video_0"] = ("IMAGE", {"tooltip": "参考视频 1（帧序列，24fps，最多 3 个）"})
        optional["ref_video_audio_0"] = ("AUDIO", {"tooltip": "参考视频 1 的配乐（最多 3 个）"})
        optional["ref_audio_0"] = ("AUDIO", {"tooltip": "独立参考音频 1（最多 3 个）"})

        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "来自 MiniMaxH3ReferenceToVideo 或融合节点的 conditioning 输出"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "当前使用的随机种子"
                }),
                "filename": ("STRING", {
                    "default": "my_seed",
                    "tooltip": "保存文件名（不含扩展名），如 shot_001_black_cat"
                }),
                "storage": (["permanent", "temp"], {
                    "default": "permanent",
                    "tooltip": "存储位置：permanent=seeds/ 永久目录；temp=temp/ 临时目录"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
                "canvas_mode": (["auto", "max", "custom"], {
                    "default": "auto",
                    "tooltip": "画布计算模式：auto=按参考图宽高比自适应；max=使用给定/默认尺寸；custom=使用宽度/高度"
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
                    "tooltip": "参考图裁剪+缩放：disabled=按宽高比缩放；center=铺满居中裁剪；contain=letterbox 填充"
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ZouyuAI/SeedTensor"

    def _collect(self, kwargs, prefix, count):
        return [kwargs.get(f"{prefix}{i}") for i in range(count)]

    def save(self, conditioning, seed, filename, language, canvas_mode="auto",
             width=0, height=0, ref_image_size="match", crop_mode="disabled",
             storage="permanent", prompt_text="", duration=0.0, ref_image_format="jpeg",
             **kwargs):
        canvas_mode = normalize_choice("canvas_mode", canvas_mode, "auto")
        ref_image_size = normalize_choice("ref_image_size", ref_image_size, "match")
        crop_mode = normalize_choice("crop_mode", crop_mode, "disabled")
        storage = normalize_choice("storage", storage, "permanent")

        safe_name = safe_filename(filename)
        target_dir = get_temp_dir() if storage == "temp" else get_seeds_dir()
        path = os.path.join(target_dir, f"{safe_name}.pt")

        zh = (language != "English")

        ref_images = self._collect(kwargs, "reference_image_", MAX_REFERENCE_IMAGES)
        ref_videos = self._collect(kwargs, "ref_video_", MAX_REFERENCE_VIDEOS)
        ref_video_audios = self._collect(kwargs, "ref_video_audio_", MAX_REFERENCE_VIDEOS)
        ref_audios = self._collect(kwargs, "ref_audio_", MAX_REFERENCE_AUDIOS)

        # 画布尺寸
        canvas_w, canvas_h, canvas_used_mode = resolve_canvas(canvas_mode, width, height, ref_images)

        # 预处理参考图（统一缩放 + 裁剪）
        preproc_images = []
        preproc_shapes = []
        for img in ref_images:
            if img is None or getattr(img, "shape", None) is None or img.shape[0] == 0:
                continue
            if canvas_w > 0 and canvas_h > 0:
                if crop_mode in ("center", "contain"):
                    proc = preprocess_image(img, canvas_w, canvas_h, crop_mode)
                else:
                    tw, th = ref_target_dims(img.shape[1], img.shape[2], canvas_w, canvas_h, ref_image_size)
                    proc = preprocess_image(img, tw, th, "disabled")
            else:
                proc = img[..., :3]
            preproc_images.append(proc)
            preproc_shapes.append([int(proc.shape[1]), int(proc.shape[2])])

        ref_image_bytes = []
        ref_image_shapes = []
        for proc in preproc_images:
            data_list = image_to_bytes(proc, fmt=ref_image_format)
            for d in data_list:
                ref_image_bytes.append(d)
            ref_image_shapes.extend([[int(proc.shape[1]), int(proc.shape[2])]] * len(data_list))

        # 参考视频帧 + 音频波形张量
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

        ref_video_audio_tensors = [sa for a in ref_video_audios if (sa := _serialize_audio(a)) is not None]
        ref_audio_tensors = [sa for a in ref_audios if (sa := _serialize_audio(a)) is not None]

        gpu_info = collect_gpu_info()
        cond_data = convert_to_serializable(conditioning)

        frame_count, latent_t, audio_t = (0, 0, 0)
        if duration and duration > 0:
            frame_count, latent_t, audio_t = temporal_shape(round(duration * FPS))

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
            "storage": storage,
            "ref_image_count": len(ref_image_bytes),
            "ref_image_shapes": ref_image_shapes,
            "ref_video_count": len(ref_video_tensors),
            "ref_video_audio_count": len(ref_video_audio_tensors),
            "ref_audio_count": len(ref_audio_tensors),
            "gpu": gpu_info,
            "provenance": {
                "plugin": "ComfyUI_ZouyuSeedTensor",
                "plugin_version": PLUGIN_VERSION,
                "saved_at": now_iso(),
                "model": "MiniMax H3",
                "compatible_models": ["MiniMax H3", "generic CONDITIONING (comfy.nested_tensor)"],
                "format_version": 2,
                "content_hash": "",
            },
        }

        try:
            fingerprint = {
                "seed": int(seed),
                "prompt": prompt_text,
                "resolution": [int(canvas_w), int(canvas_h)],
                "duration": float(duration) if duration else 0.0,
                "ref_image_count": len(ref_image_bytes),
                "ref_video_count": len(ref_video_tensors),
                "ref_audio_count": len(ref_audio_tensors),
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

        pbar = make_progress(1 + max(1, len(ref_image_bytes)), label="保存种子张量")
        progress_update(pbar, 1)

        torch.save(wrapper, path)
        progress_update(pbar, len(ref_image_bytes))

        mb = os.path.getsize(path) / (1024 * 1024)

        # 永久存储才写目录/sidecar；临时存储不索引
        if storage == "permanent":
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
                "plugin_version": PLUGIN_VERSION,
            }
            write_sidecar_meta(f"{safe_name}.pt", entry)
            update_catalog_entry(safe_name, entry)

        loc = "temp" if storage == "temp" else "permanent"
        if zh:
            log(f"已保存种子张量[{loc}] -> {path} ({mb:.1f} MB, seed={seed}, "
                f"画布={canvas_w}x{canvas_h}({canvas_used_mode}), 参考图={len(ref_image_bytes)}, "
                f"视频={len(ref_video_tensors)}, 音频={len(ref_audio_tensors)})")
        else:
            log(f"Saved seed tensor[{loc}] -> {path} ({mb:.1f} MB, seed={seed}, "
                f"canvas={canvas_w}x{canvas_h}, refs={len(ref_image_bytes)}, "
                f"videos={len(ref_video_tensors)}, audios={len(ref_audio_tensors)})")

        notify_files_refresh()

        summary = (
            f"文件: {safe_name}.pt\n"
            f"位置: {'临时' if storage == 'temp' else '永久'}存储\n"
            f"种子: {seed}\n"
            f"画布: {canvas_w}x{canvas_h} ({canvas_used_mode})\n"
            f"参考图: {len(ref_image_bytes)}  视频: {len(ref_video_tensors)}  音频: {len(ref_audio_tensors)}\n"
            f"大小: {mb:.1f} MB"
        )

        return {"ui": {"text": [summary]}, "result": (path,)}
