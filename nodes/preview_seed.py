"""
节点: ZouyuSeedPreview（.pt 文件预览）— V3 API

读取 .pt 文件，以缩略图 + 元数据摘要的形式预览其效果，
让用户无需真正加载 conditioning 也能大致了解该种子文件的内容。
"""

import torch
from comfy_api.latest import io

from ..core import (
    scan_all_seed_files, resolve_seed_path,
    extract_structure, extract_media, bytes_to_image, log,
)


class ZouyuSeedPreview(io.ComfyNode):
    """预览 .pt 种子文件：显示参考图缩略图 + 元数据摘要。"""

    @classmethod
    def define_schema(cls):
        files = scan_all_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return io.Schema(
            node_id="ZouyuSeedPreview",
            display_name="种子预览 (Zouyu Preview)",
            category="ZouyuAI/SeedTensor",
            is_output_node=True,
            inputs=[
                io.Combo.Input("file_name", options=files, tooltip="选择要预览的种子张量文件"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[io.Image.Output(display_name="参考图")],
        )

    @staticmethod
    def _summary(meta, seed, file_name, location, zh):
        gpu = meta.get("gpu", {}) if isinstance(meta, dict) else {}
        gpu_name = ""
        devices = gpu.get("devices", [])
        if devices and isinstance(devices, list):
            gpu_name = devices[0].get("name", "")
        prov = meta.get("provenance", {}) if isinstance(meta, dict) else {}

        w = meta.get("width", "?")
        h = meta.get("height", "?")
        dur = meta.get("duration", 0)
        img_n = meta.get("ref_image_count", 0)
        vid_n = meta.get("ref_video_count", 0)
        aud_n = meta.get("ref_audio_count", 0)
        loc = "临时" if location == "temp" else "永久"

        if zh:
            return (
                f"文件: {file_name}\n"
                f"位置: {loc}存储\n"
                f"种子: {seed}\n"
                f"分辨率: {w}x{h}\n"
                f"时长: {dur}s\n"
                f"参考图: {img_n}  视频: {vid_n}  音频: {aud_n}\n"
                f"GPU: {gpu_name or 'N/A'}\n"
                f"模型: {prov.get('model', 'MiniMax H3')}  版本: {prov.get('plugin_version', '?')}\n"
                f"保存时间: {prov.get('saved_at', '?')}"
            )
        return (
            f"File: {file_name}\n"
            f"Location: {loc}\n"
            f"Seed: {seed}\n"
            f"Resolution: {w}x{h}\n"
            f"Duration: {dur}s\n"
            f"Ref images: {img_n}  videos: {vid_n}  audios: {aud_n}\n"
            f"GPU: {gpu_name or 'N/A'}\n"
            f"Model: {prov.get('model', 'MiniMax H3')}  ver: {prov.get('plugin_version', '?')}\n"
            f"Saved at: {prov.get('saved_at', '?')}"
        )

    @classmethod
    def execute(cls, file_name, language) -> io.NodeOutput:
        zh = (language != "English")
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件")

        path, location = resolve_seed_path(file_name)
        data = torch.load(path, map_location="cpu", weights_only=False)
        _, meta, seed = extract_structure(data)
        media = extract_media(data)

        images = []
        ref_images_tensor = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        img_data = media.get("ref_images", {}) if isinstance(media, dict) else {}
        if isinstance(img_data, dict) and img_data.get("bytes"):
            ref_images_tensor = bytes_to_image(img_data["bytes"], fmt=img_data.get("format", "jpeg"))
            arr = (ref_images_tensor.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            images = [arr[i] for i in range(arr.shape[0])]

        summary = cls._summary(meta, seed, file_name, location, zh)
        log(f"预览种子 <- {file_name} (参考图={len(images)}, seed={seed})" if zh
            else f"Preview seed <- {file_name} (images={len(images)}, seed={seed})")

        ui = {"text": [summary]}
        if images:
            ui["images"] = images

        return io.NodeOutput(ref_images_tensor, ui=ui)
