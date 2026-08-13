"""
节点: ZouyuSeedLoader（融合加载器 / 引导器）

融合「加载种子 + 多种子混合 + MiniMax H3 Reference to Video」为一个节点：

1. 解析提示词中的 @文件名，自动将永久目录(seeds/)中的张量文件复制到临时目录(temp/)
2. 从临时目录所有 .pt 提取参考媒体（图片/视频/音频）
3. 将新提供的参考提示词 / 图片 / 视频 / 音频 打包成张量文件（自动存入临时目录）
4. 用内置 MiniMax H3 Reference to Video 根据新提示词重新解码编码
5. 将重新编码结果打包成一个张量文件，清空临时目录其他文件，卸载显存/内存
6. 输出「张量输出(conditioning)」+「引导器(GUIDER)」+「Latent」+「日志(logs)」给自定义采样器

- 时长(duration) + 帧率(fps) 控制，替代固定帧数
- 视频端口直接接收 VIDEO（内部提取帧 + 音轨）
- 打包前可备份到永久/临时目录
- 输出前卸载 clip/vae/audio_vae 显存
- 新增「日志」输出端口：运行全过程日志以中文描述输出
- 提示词中支持 @参考图N / @参考视频N / @参考音频N 引用已连接的参考端口
"""

import os
import re
import gc

import torch
import comfy.model_management as model_management

from ..core import (
    PLUGIN_VERSION,
    FPS,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, MAX_REFERENCE_AUDIOS,
    safe_filename, now_iso,
    get_seeds_dir, get_temp_dir, scan_temp_files, resolve_seed_path, copy_to_temp,
    clear_temp_except, free_memory,
    convert_to_serializable, extract_media,
    bytes_to_image, image_to_bytes, collect_gpu_info, temporal_shape,
    normalize_choice, update_catalog_entry, write_sidecar_meta,
    make_progress, progress_update, log,
)


class ZouyuSeedLoader:
    """融合加载器：@复制 + 媒体提取 + Reference to Video 重新编码 + 打包 + 备份 + 清空 + 卸载。"""

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')
    # 参考端口 @ 提及（@参考图N / @参考视频N / @参考音频N），参考媒体由端口连接自动收集
    _REF_PORT_MENTION = re.compile(r'@\s*参考(?:图|视频|音频)\s*\d*')

    @classmethod
    def INPUT_TYPES(cls):
        # 参考端口统一 0 起点，由前端动态扩展（仅 0 号端口提供说明）
        optional = {
            **{f"reference_image_{i}": ("IMAGE", {}) for i in range(MAX_REFERENCE_IMAGES)},
            **{f"ref_video_{i}": ("VIDEO", {}) for i in range(MAX_REFERENCE_VIDEOS)},
            **{f"ref_audio_{i}": ("AUDIO", {}) for i in range(MAX_REFERENCE_AUDIOS)},
        }
        optional["reference_image_0"] = ("IMAGE", {
            "tooltip": "参考图 1（连接后自动出现参考图 2，最多 50 张，同类端口连续排列）"
        })
        optional["ref_video_0"] = ("VIDEO", {
            "tooltip": "参考视频 1（直连 LoadVideo，内部提取帧与音轨，最多 3 个）"
        })
        optional["ref_audio_0"] = ("AUDIO", {
            "tooltip": "参考音频 1（直接上传，内部用音频 VAE 编码，最多 3 个）"
        })

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 模型（用于构建引导器 GUIDER）"}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 文本编码器（Qwen3-VL）"}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 视频 VAE"}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 音频 VAE（有参考时必需）"}),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "提示词。@文件名 引用永久目录种子张量（自动复制到临时目录重新编码）；@参考图N/@参考视频N/@参考音频N 引用已连接参考端口"
                }),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 3600.0, "step": 0.1,
                                       "tooltip": "视频总时长（秒），优先于提示词中的时长"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                                  "tooltip": "最终输出视频帧率"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename": ("STRING", {"default": "fused_seed", "tooltip": "重新编码后打包的输出文件名"}),
                "backup": (["permanent", "temp", "none"], {
                    "default": "permanent",
                    "tooltip": "打包前备份位置：permanent=永久目录(长期保留)；temp=临时目录(视频生成结束后清空)；none=不备份"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "GUIDER", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "guider", "latent", "logs")
    FUNCTION = "execute"
    CATEGORY = "ZouyuAI/SeedTensor"

    # ------------------------------------------------------------------
    # 日志（中文描述，累积到 self._log_lines，同时打印到控制台）
    # ------------------------------------------------------------------
    def _logz(self, msg):
        self._log_lines.append(msg)
        log(msg)

    @staticmethod
    def _unpack(out):
        args = getattr(out, "args", None)
        if args is None and isinstance(out, (tuple, list)):
            args = out
        if args and len(args) >= 2:
            return args[0], args[1]
        raise RuntimeError(f"MiniMax H3 conditioning 返回异常: {type(out)!r}")

    def _video_to_frames(self, v):
        """把 VIDEO 对象（或 IMAGE 帧序列）转为 (frames, audio)。"""
        if v is None:
            return None, None
        # Video 对象（LoadVideo 输出）
        if hasattr(v, "get_components"):
            try:
                comps = v.get_components()
                frames = comps.images  # [F,H,W,C]
                audio = getattr(comps, "audio", None)
                return frames, audio
            except Exception as exc:  # noqa: BLE001
                self._logz(f"提取视频帧失败: {exc}")
                return None, None
        # IMAGE 帧序列
        if torch.is_tensor(v):
            return v, None
        return None, None

    def _collect_new_refs(self, kwargs):
        """收集新连接的参考端口（缺失/未连接的端口自动跳过）。"""
        images, videos, video_audios, audios = [], [], [], []
        for i in range(MAX_REFERENCE_IMAGES):
            img = kwargs.get(f"reference_image_{i}")
            if img is None or getattr(img, "shape", None) is None or img.shape[0] == 0:
                continue  # 跳过缺失链接的参考端口
            images.append(img)
        for i in range(MAX_REFERENCE_VIDEOS):
            v = kwargs.get(f"ref_video_{i}")
            if v is None:
                continue  # 跳过缺失链接的参考端口
            frames, audio = self._video_to_frames(v)
            if frames is not None and getattr(frames, "shape", None) and frames.shape[0] > 0:
                videos.append(frames)
                if audio is not None:
                    video_audios.append(audio)
        for i in range(MAX_REFERENCE_AUDIOS):
            a = kwargs.get(f"ref_audio_{i}")
            if a is None:
                continue  # 跳过缺失链接的参考端口
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
                self._logz(f"跳过无法读取的临时文件 {fname}: {exc}")
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
                    self._logz(f"解码参考图失败 {fname}: {exc}")
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

    def execute(self, clip, vae, audio_vae, model, prompt, width, height, duration, fps,
                ref_image_size="match", seed=0, filename="fused_seed", backup="permanent",
                language="中文", **kwargs):
        self._log_lines = []
        ref_image_size = normalize_choice("ref_image_size", ref_image_size, "match")
        backup = normalize_choice("backup", backup, "permanent")

        prompt = prompt or ""
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = FPS
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 5.0

        # 帧数 = 时长 × 帧率
        frame_count = max(5, int(round(duration * fps)))

        self._logz(f"开始融合：时长={duration}s @ {fps}fps，画布={width}x{height}，种子={seed}")

        # ---- 0. 剥离参考端口 @ 提及（参考媒体由端口连接自动收集）----
        ref_port_mentions = self._REF_PORT_MENTION.findall(prompt)
        if ref_port_mentions:
            self._logz(f"识别到参考端口 @ 提及: {[m.strip() for m in ref_port_mentions]}")
            prompt = self._REF_PORT_MENTION.sub('', prompt)

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
                    self._logz(f"警告: @文件 {r} 不存在，跳过")
                    continue
                if loc == "permanent":
                    dst = copy_to_temp(src)
                    if dst:
                        copied.append(r)
            if copied:
                self._logz(f"已复制 {len(copied)} 个永久种子到临时目录: {copied}")

        # ---- 2. 从临时目录提取参考媒体 ----
        temp_images, temp_videos, temp_video_audios, temp_audios = self._collect_from_temp()

        # ---- 3. 加上新参考（缺失端口自动跳过）----
        new_images, new_videos, new_video_audios, new_audios = self._collect_new_refs(kwargs)

        all_images = temp_images + new_images
        all_videos = temp_videos + new_videos
        all_video_audios = temp_video_audios + new_video_audios
        all_audios = temp_audios + new_audios

        has_ref = bool(all_images or all_videos or all_video_audios or all_audios)

        self._logz(f"参考媒体：图片={len(all_images)}，视频={len(all_videos)}，音频={len(all_audios)}")

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

        self._logz("调用 MiniMax H3 Reference to Video 重新解码编码…")

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo

        if has_ref:
            if audio_vae is None:
                raise ValueError("[ZouyuSeedTensor] 参考模式需要 audio_vae（音频 VAE）")
            out = MiniMaxH3ReferenceToVideo.execute(
                clip, vae, audio_vae, cleaned_prompt, width, height, frame_count, ref_image_size,
                ref_images=ref_images_dict,
                ref_videos=ref_videos_dict,
                ref_video_audios=ref_video_audios_dict,
                ref_audios=ref_audios_dict,
            )
        else:
            out = MiniMaxH3ImageToVideo.execute(clip, vae, cleaned_prompt, width, height, frame_count)

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

        aligned_frames, latent_t, audio_t = temporal_shape(frame_count)

        # 模型溯源信息
        model_info = ""
        if model is not None:
            try:
                model_info = str(getattr(model, "model", model).__class__.__name__)
            except Exception:
                model_info = str(type(model))

        metadata = {
            "seed": int(seed),
            "prompt_text": cleaned_prompt,
            "duration": round(float(duration), 2),
            "fps": round(float(fps), 2),
            "width": int(width),
            "height": int(height),
            "resolution": {"width": int(width), "height": int(height), "canvas_mode": "custom"},
            "frame_rate": FPS,
            "frame_count": int(aligned_frames),
            "latent_t": int(latent_t),
            "audio_t": int(audio_t),
            "ref_image_size": ref_image_size,
            "storage": "temp",
            "ref_image_count": len(ref_image_bytes),
            "ref_image_shapes": ref_image_shapes,
            "ref_video_count": len(all_videos),
            "ref_video_audio_count": len(all_video_audios),
            "ref_audio_count": len(all_audios),
            "model": model_info,
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

        self._logz(f"已打包张量文件 -> {temp_path} ({os.path.getsize(temp_path) / (1024 * 1024):.1f} MB)")

        # ---- 8. 备份 ----
        if backup == "permanent":
            perm_path = os.path.join(get_seeds_dir(), f"{safe_name}.pt")
            try:
                import shutil
                shutil.copy2(temp_path, perm_path)
                entry = {
                    "name": safe_name,
                    "file": f"{safe_name}.pt",
                    "seed": int(seed),
                    "prompt": (cleaned_prompt or "")[:200],
                    "width": int(width),
                    "height": int(height),
                    "duration": round(float(duration), 2),
                    "fps": round(float(fps), 2),
                    "size_mb": round(os.path.getsize(perm_path) / (1024 * 1024), 2),
                    "saved_at": now_iso(),
                    "ref_image_count": len(ref_image_bytes),
                    "ref_video_count": len(all_videos),
                    "ref_audio_count": len(all_audios),
                    "plugin_version": PLUGIN_VERSION,
                }
                write_sidecar_meta(f"{safe_name}.pt", entry)
                update_catalog_entry(safe_name, entry)
                self._logz(f"已备份到永久目录: {perm_path}")
            except Exception as exc:  # noqa: BLE001
                self._logz(f"备份失败: {exc}")

        # ---- 9. 清空临时目录其他文件（仅保留新打包的张量）----
        removed = clear_temp_except(f"{safe_name}.pt")
        self._logz(f"已清空临时目录其他文件（移除 {removed} 项）")

        # ---- 10. 卸载显存与内存 ----
        free_memory()
        self._logz("已卸载显存与内存占用")

        # ---- 11. 构建引导器（GUIDER，供自定义采样器 SamplerCustomAdvanced 使用）----
        guider = None
        if model is not None:
            try:
                import comfy.samplers as samplers

                class _Guider_Basic(samplers.CFGGuider):
                    def set_conds(self, positive):
                        self.inner_set_conds({"positive": positive})

                g = _Guider_Basic(model)
                g.set_conds(cond)
                guider = g
                self._logz("已构建引导器 GUIDER")
            except Exception as exc:  # noqa: BLE001
                self._logz(f"构建引导器失败: {exc}")

        self._logz("融合完成")

        logs_text = "\n".join(self._log_lines)
        return {"ui": {"text": [logs_text]}, "result": (cond, guider, latent, logs_text)}
