"""
节点: ZouyuExtractSeedMedia（提取参考媒体）

从已保存的种子文件中提取参考图 / 视频帧 / 音频。
"""

import torch

from ..core import (
    scan_all_seed_files, resolve_seed_path,
    extract_structure, extract_media, bytes_to_image, log,
)


class ZouyuExtractSeedMedia:
    """从种子文件中提取参考媒体（参考图 / 视频 / 音频）。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = scan_all_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return {
            "required": {
                "file_name": (files, {"tooltip": "选择种子张量文件"}),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO")
    RETURN_NAMES = ("ref_images", "ref_videos", "ref_audio")
    FUNCTION = "extract"
    CATEGORY = "ZouyuAI/SeedTensor"

    def extract(self, file_name, language):
        zh = (language != "English")
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件")

        path, _ = resolve_seed_path(file_name)

        data = torch.load(path, map_location="cpu", weights_only=False)
        _, meta, _ = extract_structure(data)
        media = extract_media(data)

        # 参考图
        ref_images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        img_data = media.get("ref_images", {}) if isinstance(media, dict) else {}
        if isinstance(img_data, dict) and img_data.get("bytes"):
            ref_images = bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))

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

        if zh:
            log(f"提取种子媒体 <- {file_name}: 参考图={ref_images.shape[0]}, 视频帧={ref_videos.shape[0]}")
        else:
            log(f"Extracted seed media <- {file_name}: images={ref_images.shape[0]}, frames={ref_videos.shape[0]}")

        return (ref_images, ref_videos, ref_audio)
