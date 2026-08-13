"""
节点: ZouyuVAEDecodeAV（音视频联合解码）— V3 API

完全复刻官方 VAEDecode（图像）+ VAEDecodeAudio（音频）功能：
- 输入 latent + vae_image（视频 VAE）+ vae_audio（音频 VAE）
- 输出 image + audio

在 latent 收到数据后，先卸载显存中的视频模型，再将两个 VAE 重新加载到显存中解码。
"""

import torch
import comfy.model_management as model_management
from comfy_api.latest import io

from ..core import free_memory, log


class ZouyuVAEDecodeAV(io.ComfyNode):
    """音视频联合 VAE 解码：latent -> 图像 + 音频。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuVAEDecodeAV",
            display_name="音视频联合解码 (Zouyu VAE Decode)",
            category="ZouyuAI/SeedTensor",
            inputs=[
                io.Latent.Input("samples", tooltip="要解码的 AV latent（来自融合加载器或采样器）"),
                io.Vae.Input("vae_image", tooltip="视频 VAE（图像解码）"),
                io.Vae.Input("vae_audio", tooltip="音频 VAE（音频解码）"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[io.Image.Output(display_name="图像"), io.Audio.Output(display_name="音频")],
        )

    @staticmethod
    def _decode_audio(vae_audio, audio_latent, samples):
        """复刻官方 vae_decode_audio 逻辑。"""
        audio = vae_audio.decode(audio_latent).movedim(-1, 1)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        vae_sample_rate = getattr(vae_audio, "audio_sample_rate_output",
                                  getattr(vae_audio, "audio_sample_rate", 44100))
        return {
            "waveform": audio,
            "sample_rate": vae_sample_rate if "sample_rate" not in samples else samples["sample_rate"],
        }

    @classmethod
    def execute(cls, samples, vae_image, vae_audio, language="中文") -> io.NodeOutput:
        zh = (language != "English")
        latent = samples["samples"]

        # ---- 1. 卸载显存中的视频模型，重新加载两个 VAE ----
        free_memory()
        try:
            model_management.soft_empty_cache(force=True)
        except Exception:
            pass
        try:
            model_management.load_models_gpu([vae_image, vae_audio])
        except Exception:
            pass

        is_nested = bool(getattr(latent, "is_nested", False))

        # ---- 2. 解码视频（图像）----
        video_latent = latent
        if is_nested:
            video_latent = latent.unbind()[0]
        images = vae_image.decode(video_latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        # ---- 3. 解码音频 ----
        audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
        if is_nested:
            audio_latent = latent.unbind()[-1]
            audio = cls._decode_audio(vae_audio, audio_latent, samples)
        else:
            log("警告: latent 不含音频（非嵌套），音频输出为空" if zh else "Warning: latent has no audio")

        log(f"音视频解码完成: 图像 {list(images.shape)}, 音频 {list(audio['waveform'].shape)}" if zh
            else f"AV decode done: images {list(images.shape)}, audio {list(audio['waveform'].shape)}")

        return io.NodeOutput(images, audio)
