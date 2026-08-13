"""
节点: ZouyuSeedLoader（融合加载器 / 引导器）— V3 API

融合「加载种子 + 多种子混合 + MiniMax H3 Reference to Video」为一个节点，并采用
**分阶段任务流** 以最小化显存占用、最大化速度、保证零数据丢失：

  阶段 1 — 编码烘焙（仅需 clip + vae + audio_vae，不加载视频模型）
     解析 @引用 → 提取/合并参考媒体 → 参考 VAE 编码 → 卸载 VAE → 文本编码
     → 将 conditioning + latent 序列化为张量种子文件（.pt）备份
  阶段 2 — 卸载全部模型
     卸载 clip / vae / audio_vae，释放全部显存与内存
  阶段 3 — 加载视频模型 + 解包
     仅加载视频模型（DiT）→ 从 .pt 还原 conditioning + latent（移到 GPU）
     → 构建引导器 GUIDER → 输出给自定义采样器

- 参考端口使用官方 io.Autogrow.Input + TemplatePrefix(min=0, max=50) 动态扩展
- 视频端口直接接收 VIDEO（内部提取帧 + 音轨）
- 帧数超训练范围告警/钳制
- 输出「张量输出(conditioning)」+「引导器(GUIDER)」+「Latent」+「日志(logs)」
"""

import os
import re
import gc
import math

import torch
import comfy.model_management as model_management
from comfy_api.latest import io

from ..core import (
    PLUGIN_VERSION,
    FPS,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, MAX_REFERENCE_AUDIOS,
    safe_filename, now_iso,
    get_seeds_dir, get_temp_dir, scan_temp_files, resolve_seed_path, copy_to_temp,
    clear_temp_except,
    convert_to_serializable, convert_from_serializable, move_to_device, extract_media,
    bytes_to_image, image_to_bytes, collect_gpu_info, temporal_shape,
    frames_to_video_bytes, video_bytes_to_frames,
    normalize_choice, update_catalog_entry, write_sidecar_meta,
    make_progress, progress_update, log,
    unload_all_models_thorough,          # 新增导入
)

# 导入新编码器（reference_encoder.py 位于插件根目录）
from ..reference_encoder import encode_references_to_cond


# MiniMax H3 训练帧数范围（官方注释 ~124-362 帧）
MAX_TRAINED_FRAMES = 362
MAX_FRAMES = 3600  # 官方 length 上限


class ZouyuSeedLoader(io.ComfyNode):
    """融合加载器：编码烘焙 → 卸载全部 → 加载视频模型 → 解包 → 采样器。"""

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')
    _REF_PORT_MENTION = re.compile(r'@\s*参考(?:图|视频|音频)\s*\d*')

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuSeedLoader",
            display_name="融合加载器 (Zouyu Loader)",
            category="ZouyuAI/SeedTensor",
            description="融合种子加载 + 混合 + MiniMax H3 Reference to Video，输出张量/引导器/Latent/日志。",
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 视频模型（DiT，阶段3加载，用于构建引导器）"),
                io.Clip.Input("clip", tooltip="MiniMax H3 文本编码器（Qwen3-VL）"),
                io.Vae.Input("vae", tooltip="MiniMax H3 视频 VAE"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 音频 VAE（有参考时必需）"),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="提示词。@文件名 引用永久目录种子张量；@参考图N/@参考视频N/@参考音频N 引用已连接参考端口"),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Float.Input("duration", default=5.0, min=0.1, max=3600.0, step=0.1,
                               tooltip="视频总时长（秒），优先于提示词中的时长。模型训练范围约 5-15 秒，过长会 OOM"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=1.0,
                               tooltip="最终输出视频帧率"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Float.Input("ref_scale", display_name="参考值放大", default=1.0, min=1.0, max=5.0, step=0.1,
                               tooltip="仅 match 模式生效。参考图面积倍率（1.0=官方行为，越大保真度越高越慢）"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input("filename", default="fused_seed", tooltip="重新编码后打包的输出文件名"),
                io.Combo.Input("backup", options=["permanent", "temp", "none"], default="permanent",
                               tooltip="打包前备份位置：permanent=永久目录；temp=临时目录(视频生成结束后清空)；none=不备份"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image", tooltip="参考图（自动扩展，最多 50 张）"),
                                      prefix="ref_image_", min=0, max=MAX_REFERENCE_IMAGES)),
                io.Autogrow.Input("ref_videos", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Video.Input("ref_video", tooltip="参考视频（直连 LoadVideo，内部提取帧与音轨）"),
                                      prefix="ref_video_", min=0, max=MAX_REFERENCE_VIDEOS)),
                io.Autogrow.Input("ref_audios", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("ref_audio", tooltip="参考音频（内部用音频 VAE 编码）"),
                                      prefix="ref_audio_", min=0, max=MAX_REFERENCE_AUDIOS)),
            ],
            outputs=[
                io.Conditioning.Output(display_name="张量输出"),
                io.Guider.Output(display_name="引导器"),
                io.Latent.Output(display_name="Latent"),
                io.String.Output(display_name="日志"),
            ],
        )

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(out):
        args = getattr(out, "args", None)
        if args is None and isinstance(out, (tuple, list)):
            args = out
        if args and len(args) >= 2:
            return args[0], args[1]
        raise RuntimeError(f"MiniMax H3 conditioning 返回异常: {type(out)!r}")

    @staticmethod
    def _video_to_frames(v, logz):
        """把 VIDEO 对象（或 IMAGE 帧序列）转为 (frames, audio)。"""
        if v is None:
            return None, None
        if hasattr(v, "get_components"):
            try:
                comps = v.get_components()
                return comps.images, getattr(comps, "audio", None)
            except Exception as exc:  # noqa: BLE001
                logz(f"提取视频帧失败: {exc}")
                return None, None
        if torch.is_tensor(v):
            return v, None
        return None, None

    # ------------------------------------------------------------------
    # 主流程（分阶段任务流）
    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, clip, vae, audio_vae, model, prompt, width, height, duration, fps,
                ref_image_size="match", ref_scale=1.0, seed=0, filename="fused_seed",
                backup="permanent", language="中文", ref_images=None, ref_videos=None,
                ref_audios=None) -> io.NodeOutput:
        log_lines = []

        def logz(msg):
            log_lines.append(msg)
            log(msg)

        # ---- 0. 彻底卸载所有外部模型（阶段1前） ----
        unload_all_models_thorough("阶段1前：卸载所有模型（包括DiT）")

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

        frame_count = max(5, int(round(duration * fps)))
        logz(f"开始融合：时长={duration}s @ {fps}fps（帧数={frame_count}），画布={width}x{height}，种子={seed}")

        # ---- 帧数守卫 ----
        if frame_count > MAX_TRAINED_FRAMES:
            logz(f"警告：帧数 {frame_count} 超出模型训练范围（约 124-362 帧），可能导致显存不足或效果异常")
        if frame_count > MAX_FRAMES:
            logz(f"帧数 {frame_count} 超过上限，已钳制到 {MAX_FRAMES}")
            frame_count = MAX_FRAMES

        # ---- 0. 剥离参考端口 @ 提及 ----
        ref_port_mentions = cls._REF_PORT_MENTION.findall(prompt)
        if ref_port_mentions:
            logz(f"识别到参考端口 @ 提及: {[m.strip() for m in ref_port_mentions]}")
            prompt = cls._REF_PORT_MENTION.sub('', prompt)

        # ---- 1. 解析 @引用，复制永久 -> 临时 ----
        refs = cls._AT_PATTERN.findall(prompt)
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
                    logz(f"警告: @文件 {r} 不存在，跳过")
                    continue
                if loc == "permanent":
                    dst = copy_to_temp(src)
                    if dst:
                        copied.append(r)
            if copied:
                logz(f"已复制 {len(copied)} 个永久种子到临时目录: {copied}")

        # ---- 2. 从临时目录提取参考媒体 ----
        temp_images, temp_videos, temp_video_audios, temp_audios = cls._collect_from_temp(logz)

        # ---- 3. 加上新参考（Autogrow 按顺序给出，缺失端口自动跳过）----
        new_images = [img for img in (ref_images or {}).values()
                      if img is not None and getattr(img, "shape", None) and img.shape[0] > 0]
        new_videos = []
        new_video_audios = []
        for v in (ref_videos or {}).values():
            if v is None:
                continue
            frames, audio = cls._video_to_frames(v, logz)
            if frames is not None and getattr(frames, "shape", None) and frames.shape[0] > 0:
                new_videos.append(frames)
                new_video_audios.append(audio)
        new_audios = [a for a in (ref_audios or {}).values()
                      if isinstance(a, dict) and a.get("waveform") is not None]

        all_images = temp_images + new_images
        all_videos = temp_videos + new_videos
        all_video_audios = temp_video_audios + new_video_audios
        all_audios = temp_audios + new_audios
        has_ref = bool(all_images or all_videos or all_audios)

        logz(f"参考媒体：图片={len(all_images)}，视频={len(all_videos)}，音频={len(all_audios)}")

        # ---- 4. 清理提示词 ----
        cleaned_prompt = cls._AT_PATTERN.sub('', prompt).strip()
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        # =====================================================================
        # 阶段 1：编码烘焙（仅 clip + vae + audio_vae，不加载视频模型）
        # =====================================================================
        pbar = make_progress(3, label="编码烘焙")
        progress_update(pbar, 1)

        # ---- 编码（使用移植的编码器） ----
        if has_ref:
            # 构造字典供编码器使用
            ref_images_dict = {f"ref_image_{i}": img for i, img in enumerate(all_images)}
            ref_videos_dict = {f"ref_video_{i}": v for i, v in enumerate(all_videos)}
            ref_video_audios_dict = {f"ref_video_audio_{i}": a for i, a in enumerate(all_video_audios)}
            ref_audios_dict = {f"ref_audio_{i}": a for i, a in enumerate(all_audios)}

            logz("阶段1：编码参考媒体（使用移植编码器）…")
            cond, latent = encode_references_to_cond(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=cleaned_prompt,
                width=width,
                height=height,
                length=frame_count,
                ref_image_size=ref_image_size,
                ref_scale=ref_scale,
                ref_images=ref_images_dict,
                ref_videos=ref_videos_dict,
                ref_video_audios=ref_video_audios_dict,
                ref_audios=ref_audios_dict,
            )
            # 注意：latent 已在 GPU 上，但后续我们会将其序列化到 CPU。
        else:
            # 无参考，使用官方 ImageToVideo（纯文本模式）
            logz("阶段1：无参考，走 Image to Video 路径…")
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
            out = MiniMaxH3ImageToVideo.execute(clip, vae, cleaned_prompt, width, height, frame_count)
            cond, latent = cls._unpack(out)

        progress_update(pbar, 1)

        # ---- 序列化 conditioning + latent（CPU）----
        logz("阶段1d：序列化 conditioning + latent → 张量种子文件…")
        cond_cpu = convert_to_serializable(cond)
        latent_cpu = convert_to_serializable(latent)

        safe_name = safe_filename(filename)
        temp_path = os.path.join(get_temp_dir(), f"{safe_name}.pt")

        # ---- 构建媒体元数据（用于 .pt 内的 media 字段） ----
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
            "ref_scale": ref_scale,
            "storage": "temp",
            "ref_image_count": len(ref_image_bytes),
            "ref_image_shapes": ref_image_shapes,
            "ref_video_count": len(all_videos),
            "ref_audio_count": len(all_audios),
            "model": model_info,
            "gpu": collect_gpu_info(),
            "provenance": {
                "plugin": "ComfyUI_ZouyuSeedTensor",
                "plugin_version": PLUGIN_VERSION,
                "saved_at": now_iso(),
                "model": "MiniMax H3",
                "format_version": 3,
                "content_hash": "",
                "fused_sources": copied,
            },
        }

        # 参考视频改用 H.264 压缩存储（大幅减小种子文件体积），记录原始尺寸便于解码还原
        packed_videos = []
        for v in all_videos:
            if v is None or getattr(v, "shape", None) is None or v.shape[0] == 0:
                continue
            b = frames_to_video_bytes(v[..., :3])
            if b is not None:
                packed_videos.append({
                    "bytes": b,
                    "shape": [int(v.shape[1]), int(v.shape[2])],
                })

        media = {
            "ref_images": {"format": "jpeg", "bytes": ref_image_bytes, "shapes": ref_image_shapes},
            "ref_videos": packed_videos,
            "ref_video_audios": [a for a in all_video_audios if isinstance(a, dict) and a.get("waveform") is not None],
            "ref_audios": all_audios,
        }

        wrapper = {
            "conditioning": cond_cpu,
            "latent": latent_cpu,
            "seed": int(seed),
            "metadata": metadata,
            "media": media,
        }
        torch.save(wrapper, temp_path)
        logz(f"已烘焙张量种子文件 -> {temp_path} ({os.path.getsize(temp_path) / (1024 * 1024):.1f} MB)")

        # ---- 备份 ----
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
                logz(f"已备份到永久目录: {perm_path}")
            except Exception as exc:  # noqa: BLE001
                logz(f"备份失败: {exc}")

        # ---- 清空临时目录其他文件（仅保留本次烘焙的 .pt）----
        removed = clear_temp_except(f"{safe_name}.pt")
        logz(f"已清空临时目录其他文件（移除 {removed} 项）")

        # =====================================================================
        # 阶段 2：卸载全部模型（显存 + CPU 内存）
        # =====================================================================
        del cond, latent
        # 丢弃 clip / vae / audio_vae 的本地引用（帮助 gc 回收 CPU 内存）
        del clip, vae, audio_vae
        unload_all_models_thorough("阶段2：卸载全部模型（VAE/CLIP）")

        # =====================================================================
        # 阶段 3：加载视频模型 + 解包 → 采样器数据
        # =====================================================================
        logz("阶段3：加载视频模型（DiT）…")
        if model is not None:
            try:
                model_management.load_models_gpu([model])
            except Exception as exc:  # noqa: BLE001
                logz(f"加载视频模型失败（将交给采样器处理）: {exc}")

        # 解包：从 .pt 还原 conditioning + latent，并移到 GPU
        logz("解包：还原 conditioning + latent 到 GPU…")
        cond = convert_from_serializable(cond_cpu)
        latent = convert_from_serializable(latent_cpu)
        device = model_management.intermediate_device()
        cond = move_to_device(cond, device)
        latent = move_to_device(latent, device)

        # 构建引导器（GUIDER，供自定义采样器 SamplerCustomAdvanced 使用）
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
                logz("已构建引导器 GUIDER")
            except Exception as exc:  # noqa: BLE001
                logz(f"构建引导器失败: {exc}")

        progress_update(pbar, 1)
        logz("融合完成：数据已就绪，可直接供采样器使用")

        logs_text = "\n".join(log_lines)
        return io.NodeOutput(cond, guider, latent, logs_text, ui={"text": [logs_text]})

    # ------------------------------------------------------------------
    # 辅助：从临时目录提取媒体
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_from_temp(logz):
        """从临时目录所有 .pt 提取参考媒体，返回 (images, videos, video_audios, audios)。"""
        images, videos, video_audios, audios = [], [], [], []
        for fname in scan_temp_files():
            path = os.path.join(get_temp_dir(), fname)
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                logz(f"跳过无法读取的临时文件 {fname}: {exc}")
                continue
            media = extract_media(data)
            if not isinstance(media, dict):
                continue
            img_data = media.get("ref_images", {})
            if isinstance(img_data, dict) and img_data.get("bytes"):
                try:
                    imgs = bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))
                    shapes = img_data.get("shapes", [])
                    for i in range(imgs.shape[0]):
                        img = imgs[i:i + 1]
                        # 按原始尺寸裁剪回（bytes_to_image 会 pad 到最大尺寸）
                        if i < len(shapes):
                            h, w = int(shapes[i][0]), int(shapes[i][1])
                            if h > 0 and w > 0:
                                img = img[:, :h, :w, :]
                        images.append(img)
                except Exception as exc:  # noqa: BLE001
                    logz(f"解码参考图失败 {fname}: {exc}")
            for v in media.get("ref_videos", []):
                if isinstance(v, dict) and v.get("bytes") is not None:
                    # 新格式：H.264 字节流，解码回帧
                    frames = video_bytes_to_frames(v["bytes"], original_shape=v.get("shape"))
                    if frames is not None and frames.shape[0] > 0:
                        videos.append(frames)
                        video_audios.append(None)
                elif isinstance(v, torch.Tensor):
                    # 旧格式：直接存张量
                    videos.append(v.float())
                    video_audios.append(None)
            for va in media.get("ref_video_audios", []):
                if isinstance(va, dict) and va.get("waveform") is not None:
                    video_audios.append(va)
            for a in media.get("ref_audios", []):
                if isinstance(a, dict) and a.get("waveform") is not None:
                    audios.append(a)
        return images, videos, video_audios, audios