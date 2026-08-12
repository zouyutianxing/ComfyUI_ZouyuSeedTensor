"""
节点: ZouyuSeedLoader（融合加载器 / 引导器）

融合「加载种子 + 多种子混合 + MiniMax H3 Reference to Video」为一个节点：

1. 解析提示词中的 @文件名，自动将永久目录(seeds/)中的张量文件复制到临时目录(temp/)
2. 从临时目录所有 .pt 提取参考媒体（图片/视频/音频）
3. 将新提供的参考提示词 / 图片 / 视频 / 音频 打包成张量文件（自动存入临时目录）
4. 用内置 MiniMax H3 Reference to Video 根据新提示词重新解码编码
5. 将重新编码结果全部打包成一个张量文件，清空临时目录内其他文件，卸载显存/内存
6. 输出「引导器(conditioning)」+「Latent」给自定义采样器

输入 = 模型(clip / vae / audio_vae)；输出 = 引导器 + Latent。
"""

import os
import re

import torch

from ..core import (
    PLUGIN_VERSION,
    FPS,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, MAX_REFERENCE_AUDIOS,
    safe_filename, now_iso,
    get_temp_dir, scan_temp_files, resolve_seed_path, copy_to_temp,
    clear_temp_except, free_memory,
    convert_to_serializable, extract_media,
    bytes_to_image, image_to_bytes, collect_gpu_info, temporal_shape,
    normalize_choice, make_progress, progress_update, log,
)


class ZouyuSeedLoader:
    """融合加载器：@复制 + 参考媒体提取 + Reference to Video 重新编码 + 打包 + 清空 + 卸载。"""

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # 参考图动态端口（最多 50 个）
        for i in range(MAX_REFERENCE_IMAGES):
            optional[f"reference_image_{i}"] = ("IMAGE", {"tooltip": f"新参考图 {i + 1}"})
        for i in range(MAX_REFERENCE_VIDEOS):
            optional[f"ref_video_{i}"] = ("IMAGE", {"tooltip": f"新参考视频 {i + 1}（帧序列，24fps）"})
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": f"新参考视频 {i + 1} 的配乐"})
        for i in range(MAX_REFERENCE_AUDIOS):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": f"新参考音频 {i + 1}"})

        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 文本编码器（Qwen3-VL）"}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 视频 VAE"}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 音频 VAE（有参考时必需）"}),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "提示词。使用 @文件名 引用永久目录中的种子张量（会自动复制到临时目录并重新编码）"
                }),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename": ("STRING", {"default": "fused_seed", "tooltip": "重新编码后打包的输出文件名（存入临时目录）"}),
                "language": (["中文", "English"], {"default": "中文"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "execute"
    CATEGORY = "ZouyuAI/SeedTensor"

    @staticmethod
    def _unpack(out):
        args = getattr(out, "args", None)
        if args is None and isinstance(out, (tuple, list)):
            args = out
        if args and len(args) >= 2:
            return args[0], args[1]
        raise RuntimeError(f"MiniMax H3 conditioning 返回异常: {type(out)!r}")

    def _collect_new_refs(self, kwargs):
        images, videos, video_audios, audios = [], [], [], []
        for i in range(MAX_REFERENCE_IMAGES):
            img = kwargs.get(f"reference_image_{i}")
            if img is not None and getattr(img, "shape", None) and img.shape[0] > 0:
                images.append(img)
        for i in range(MAX_REFERENCE_VIDEOS):
            v = kwargs.get(f"ref_video_{i}")
            if v is not None and getattr(v, "shape", None) and v.shape[0] > 0:
                videos.append(v)
            va = kwargs.get(f"ref_video_audio_{i}")
            if isinstance(va, dict) and va.get("waveform") is not None:
                video_audios.append(va)
        for i in range(MAX_REFERENCE_AUDIOS):
            a = kwargs.get(f"ref_audio_{i}")
            if isinstance(a, dict) and a.get("waveform") is not None:
                audios.append(a)
        return images, videos, video_audios, audios

    def _collect_from_temp(self):
        """从临时目录所有 .pt 提取参考媒体（图片/视频/音频），返回合并列表。"""
        images, videos, video_audios, audios = [], [], [], []
        for fname in scan_temp_files():
            path = os.path.join(get_temp_dir(), fname)
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                log(f"跳过无法读取的临时文件 {fname}: {exc}")
                continue
            media = extract_media(data)
            if not isinstance(media, dict):
                continue
            img_data = media.get("ref_images", {})
            if isinstance(img_data, dict) and img_data.get("bytes"):
                try:
                    imgs = bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))
                    for i in range(imgs.shape[0]):
                        images.append(imgs[i:i + 1])
                except Exception as exc:  # noqa: BLE001
                    log(f"解码参考图失败 {fname}: {exc}")
            for v in media.get("ref_videos", []):
                if isinstance(v, torch.Tensor):
                    videos.append(v.float())
            for va in media.get("ref_video_audios", []):
                if isinstance(va, dict) and va.get("waveform") is not None:
                    video_audios.append(va)
            for a in media.get("ref_audios", []):
                if isinstance(a, dict) and a.get("waveform") is not None:
                    audios.append(a)
        return images, videos, video_audios, audios

    def execute(self, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", seed=0, filename="fused_seed", language="中文",
                **kwargs):
        zh = (language != "English")
        ref_image_size = normalize_choice("ref_image_size", ref_image_size, "match")

        prompt = prompt or ""

        # ---- 1. 解析 @引用，复制永久 -> 临时 ----
        refs = self._AT_PATTERN.findall(prompt)
        copied = []
        if refs:
            seen = set()
            for r in refs:
                if r in seen:
                    continue
                seen.add(r)
                try:
                    src, loc = resolve_seed_path(r)
                except FileNotFoundError:
                    log(f"警告: @文件 {r} 不存在，跳过" if zh else f"Warning: @file {r} not found")
                    continue
                if loc == "permanent":
                    dst = copy_to_temp(src)
                    if dst:
                        copied.append(r)
            if copied and zh:
                log(f"已复制 {len(copied)} 个永久种子到临时目录: {copied}")
            elif copied:
                log(f"Copied {len(copied)} permanent seeds to temp: {copied}")

        # ---- 2. 从临时目录提取参考媒体 ----
        temp_images, temp_videos, temp_video_audios, temp_audios = self._collect_from_temp()

        # ---- 3. 加上新参考 ----
        new_images, new_videos, new_video_audios, new_audios = self._collect_new_refs(kwargs)

        all_images = temp_images + new_images
        all_videos = temp_videos + new_videos
        all_video_audios = temp_video_audios + new_video_audios
        all_audios = temp_audios + new_audios

        has_ref = bool(all_images or all_videos or all_video_audios or all_audios)

        # ---- 4. 清理提示词 ----
        cleaned_prompt = self._AT_PATTERN.sub('', prompt).strip()
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        # ---- 5. 组装官方 ReferenceToVideo 的 dict 参数 ----
        ref_images_dict = {f"ref_image_{i}": img for i, img in enumerate(all_images)} or None
        ref_videos_dict = {f"ref_video_{i}": v for i, v in enumerate(all_videos)} or None
        ref_video_audios_dict = {f"ref_video_audio_{i}": a for i, a in enumerate(all_video_audios)} or None
        ref_audios_dict = {f"ref_audio_{i}": a for i, a in enumerate(all_audios)} or None

        # ---- 6. 用内置 Reference to Video 重新解码编码 ----
        pbar = make_progress(2, label="重新编码")
        progress_update(pbar, 1)

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo

        if has_ref:
            if audio_vae is None:
                raise ValueError("[ZouyuSeedTensor] 参考模式需要 audio_vae（音频 VAE）")
            out = MiniMaxH3ReferenceToVideo.execute(
                clip, vae, audio_vae, cleaned_prompt, width, height, length, ref_image_size,
                ref_images=ref_images_dict,
                ref_videos=ref_videos_dict,
                ref_video_audios=ref_video_audios_dict,
                ref_audios=ref_audios_dict,
            )
        else:
            out = MiniMaxH3ImageToVideo.execute(clip, vae, cleaned_prompt, width, height, length)

        cond, latent = self._unpack(out)
        progress_update(pbar, 1)

        # ---- 7. 打包重新编码结果成一个张量文件（存入临时目录）----
        safe_name = safe_filename(filename)
        temp_path = os.path.join(get_temp_dir(), f"{safe_name}.pt")

        ref_image_bytes = []
        ref_image_shapes = []
        for img in all_images:
            try:
                data_list = image_to_bytes(img[..., :3], fmt="jpeg")
            except Exception:
                continue
            ref_image_bytes.extend(data_list)
            ref_image_shapes.extend([[int(img.shape[1]), int(img.shape[2])]] * len(data_list))

        frame_count, latent_t, audio_t = temporal_shape(length)

        metadata = {
            "seed": int(seed),
            "prompt_text": cleaned_prompt,
            "duration": round(frame_count / FPS, 2),
            "width": int(width),
            "height": int(height),
            "resolution": {"width": int(width), "height": int(height), "canvas_mode": "custom"},
            "frame_rate": FPS,
            "frame_count": int(frame_count),
            "latent_t": int(latent_t),
            "audio_t": int(audio_t),
            "ref_image_size": ref_image_size,
            "storage": "temp",
            "ref_image_count": len(ref_image_bytes),
            "ref_image_shapes": ref_image_shapes,
            "ref_video_count": len(all_videos),
            "ref_video_audio_count": len(all_video_audios),
            "ref_audio_count": len(all_audios),
            "gpu": collect_gpu_info(),
            "provenance": {
                "plugin": "ComfyUI_ZouyuSeedTensor",
                "plugin_version": PLUGIN_VERSION,
                "saved_at": now_iso(),
                "model": "MiniMax H3",
                "format_version": 2,
                "content_hash": "",
                "fused_sources": copied,
            },
        }

        media = {
            "ref_images": {"format": "jpeg", "bytes": ref_image_bytes, "shapes": ref_image_shapes},
            "ref_videos": [v[..., :3].detach().to(torch.float16).cpu() for v in all_videos],
            "ref_video_audios": all_video_audios,
            "ref_audios": all_audios,
        }

        wrapper = {
            "conditioning": convert_to_serializable(cond),
            "seed": int(seed),
            "metadata": metadata,
            "media": media,
        }
        torch.save(wrapper, temp_path)

        # ---- 8. 清空临时目录其他文件（仅保留新打包的张量）----
        removed = clear_temp_except(f"{safe_name}.pt")

        # ---- 9. 卸载显存与内存 ----
        free_memory()

        mb = os.path.getsize(temp_path) / (1024 * 1024)

        if zh:
            log(f"融合完成 -> {temp_path} ({mb:.1f} MB, 参考图={len(ref_image_bytes)}, "
                f"视频={len(all_videos)}, 音频={len(all_audios)}, 已清空临时 {removed} 项)")
        else:
            log(f"Fusion done -> {temp_path} ({mb:.1f} MB, images={len(ref_image_bytes)}, "
                f"videos={len(all_videos)}, audios={len(all_audios)}, cleared {removed})")

        return (cond, latent)
