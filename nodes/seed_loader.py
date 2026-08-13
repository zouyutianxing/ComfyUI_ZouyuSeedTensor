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
    clear_temp_except, free_memory,
    convert_to_serializable, convert_from_serializable, move_to_device, extract_media,
    bytes_to_image, image_to_bytes, collect_gpu_info, temporal_shape,
    normalize_choice, update_catalog_entry, write_sidecar_meta,
    make_progress, progress_update, log,
)


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

    @staticmethod
    def _encode_references(vae, audio_vae, ref_images, ref_videos, ref_video_audios, ref_audios,
                           width, height, frame_count, ref_image_size):
        """镜像官方 MiniMaxH3ReferenceToVideo 的参考编码，返回 (ref_items, ref_blocks)。"""
        from comfy_extras.nodes_minimax_h3 import (
            _resize, adapt_canvas, REF_IMAGE_SHORT_EDGE, CANVAS_MULTIPLE,
            MiniMaxH3ReferenceToVideo,
        )

        ref_items = []
        ref_blocks = []

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = (ref_video_audios or {}).get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 参考视频至少需要 5 帧（约 0.2s @ 24fps）")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = MiniMaxH3ReferenceToVideo._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        return ref_items, ref_blocks

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
                    for i in range(imgs.shape[0]):
                        images.append(imgs[i:i + 1])
                except Exception as exc:  # noqa: BLE001
                    logz(f"解码参考图失败 {fname}: {exc}")
            for v in media.get("ref_videos", []):
                if isinstance(v, torch.Tensor):
                    videos.append(v.float())
                    video_audios.append(None)
            for va in media.get("ref_video_audios", []):
                if isinstance(va, dict) and va.get("waveform") is not None:
                    video_audios.append(va)
            for a in media.get("ref_audios", []):
                if isinstance(a, dict) and a.get("waveform") is not None:
                    audios.append(a)
        return images, videos, video_audios, audios

    # ------------------------------------------------------------------
    # 主流程（分阶段任务流）
    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, clip, vae, audio_vae, model, prompt, width, height, duration, fps,
                ref_image_size="match", seed=0, filename="fused_seed", backup="permanent",
                language="中文", ref_images=None, ref_videos=None, ref_audios=None) -> io.NodeOutput:
        log_lines = []

        def logz(msg):
            log_lines.append(msg)
            log(msg)

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
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent, MiniMaxH3ImageToVideo
        import node_helpers

        pbar = make_progress(3, label="编码烘焙")
        progress_update(pbar, 1)

        if has_ref:
            if audio_vae is None:
                raise ValueError("[ZouyuSeedTensor] 参考模式需要 audio_vae（音频 VAE）")

            latent, frame_count = _empty_av_latent(width, height, frame_count)

            logz("阶段1a：编码参考媒体（视频 VAE + 音频 VAE）…")
            ref_images_dict = {f"ref_image_{i}": img for i, img in enumerate(all_images)}
            ref_videos_dict = {f"ref_video_{i}": v for i, v in enumerate(all_videos)}
            ref_video_audios_dict = {f"ref_video_audio_{i}": a for i, a in enumerate(all_video_audios)}
            ref_audios_dict = {f"ref_audio_{i}": a for i, a in enumerate(all_audios)}
            ref_items, ref_blocks = cls._encode_references(
                vae, audio_vae, ref_images_dict, ref_videos_dict, ref_video_audios_dict, ref_audios_dict,
                width, height, frame_count, ref_image_size)

            # 卸载 VAE，为文本编码腾出显存
            logz("阶段1b：卸载 VAE，释放显存…")
            try:
                model_management.unload_all_models()
            except Exception:
                pass
            model_management.soft_empty_cache(force=True)

            logz("阶段1c：文本编码（Qwen3-VL）…")
            tokens = clip.tokenize(cleaned_prompt, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        else:
            logz("阶段1：无参考，走 Image to Video 路径…")
            out = MiniMaxH3ImageToVideo.execute(clip, vae, cleaned_prompt, width, height, frame_count)
            cond, latent = cls._unpack(out)
            ref_items, ref_blocks = None, None

        progress_update(pbar, 1)

        # ---- 序列化 conditioning + latent（CPU）----
        logz("阶段1d：序列化 conditioning + latent → 张量种子文件…")
        cond_cpu = convert_to_serializable(cond)
        latent_cpu = convert_to_serializable(latent)

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

        media = {
            "ref_images": {"format": "jpeg", "bytes": ref_image_bytes, "shapes": ref_image_shapes},
            "ref_videos": [v[..., :3].detach().to(torch.float16).cpu() for v in all_videos],
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
        # 阶段 2：卸载全部模型，释放显存与内存
        # =====================================================================
        logz("阶段2：卸载全部模型，释放显存…")
        del cond, latent
        if ref_items is not None:
            del ref_items, ref_blocks
        free_memory()
        try:
            model_management.unload_all_models()
        except Exception:
            pass
        model_management.soft_empty_cache(force=True)
        free_memory()
        logz("已卸载 clip / vae / audio_vae，显存已释放")

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
