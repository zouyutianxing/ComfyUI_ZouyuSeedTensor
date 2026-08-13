"""
节点: ZouyuExtractSeedMedia（提取参考媒体）— V3 API

从已保存的种子文件中提取参考图 / 视频帧 / 音频。
"""

import torch
from comfy_api.latest import io

from ..core import (
    scan_all_seed_files, resolve_seed_path,
    extract_structure, extract_media, bytes_to_image, video_bytes_to_frames, log,
)


class ZouyuExtractSeedMedia(io.ComfyNode):
    """从种子文件中提取参考媒体（参考图 / 视频 / 音频）。"""

    @classmethod
    def define_schema(cls):
        files = scan_all_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return io.Schema(
            node_id="ZouyuExtractSeedMedia",
            display_name="提取参考媒体 (Zouyu Extract)",
            category="ZouyuAI/SeedTensor",
            inputs=[
                io.Combo.Input("file_name", options=files, tooltip="选择种子张量文件"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[
                io.Image.Output(display_name="参考图"),
                io.Image.Output(display_name="参考视频"),
                io.Audio.Output(display_name="参考音频"),
            ],
        )

    @classmethod
    def execute(cls, file_name, language) -> io.NodeOutput:
        zh = (language != "English")
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件")

        path, _ = resolve_seed_path(file_name)
        data = torch.load(path, map_location="cpu", weights_only=False)
        media = extract_media(data)

        ref_images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        img_data = media.get("ref_images", {}) if isinstance(media, dict) else {}
        if isinstance(img_data, dict) and img_data.get("bytes"):
            ref_images = bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))

        ref_videos = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        videos = media.get("ref_videos", []) if isinstance(media, dict) else []
        if videos:
            try:
                decoded = []
                for v in videos:
                    if isinstance(v, dict) and v.get("bytes") is not None:
                        f = video_bytes_to_frames(v["bytes"], original_shape=v.get("shape"))
                        if f is not None and f.shape[0] > 0:
                            decoded.append(f.float())
                    elif isinstance(v, torch.Tensor):
                        decoded.append(v.float())
                if decoded:
                    ref_videos = torch.cat(decoded, dim=0)
            except Exception:
                ref_videos = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

        ref_audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
        audios = media.get("ref_audios", []) if isinstance(media, dict) else []
        if audios and isinstance(audios[0], dict) and "waveform" in audios[0]:
            ref_audio = audios[0]

        log(f"提取种子媒体 <- {file_name}: 参考图={ref_images.shape[0]}, 视频帧={ref_videos.shape[0]}" if zh
            else f"Extracted seed media <- {file_name}: images={ref_images.shape[0]}, frames={ref_videos.shape[0]}")

        return io.NodeOutput(ref_images, ref_videos, ref_audio)
