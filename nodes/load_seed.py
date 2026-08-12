"""
节点: ZouyuLoadSeedConditioning（加载种子张量）

从永久（seeds/）或临时（temp/）目录加载单个种子张量文件，
输出 conditioning + 种子 + 完整元数据（JSON）。
"""

import os
import json

import torch
import comfy.model_management as model_management

from ..core import (
    scan_all_seed_files, resolve_seed_path,
    convert_from_serializable, move_to_device, extract_structure,
    log,
)


class ZouyuLoadSeedConditioning:
    """加载单个种子张量文件。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = scan_all_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return {
            "required": {
                "file_name": (files, {"tooltip": "选择种子张量文件（含永久与临时目录）"}),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "metadata")
    FUNCTION = "load"
    CATEGORY = "ZouyuAI/SeedTensor"

    def load(self, file_name, language):
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件，请先使用保存节点保存")

        path, location = resolve_seed_path(file_name)

        data = torch.load(path, map_location="cpu", weights_only=False)
        cond_data, meta, seed = extract_structure(data)

        cond = convert_from_serializable(cond_data)
        device = model_management.get_torch_device()
        cond = move_to_device(cond, device)

        mb = os.path.getsize(path) / (1024 * 1024)

        try:
            meta_display = json.dumps(meta, ensure_ascii=False, indent=2, default=str)
        except Exception:
            meta_display = str(meta)

        loc = "temp" if location == "temp" else "permanent"
        if language == "English":
            log(f"Loaded seed tensor <- {file_name} ({mb:.1f} MB, seed={seed}, {loc}) -> {device}")
        else:
            log(f"已加载种子张量 <- {file_name} ({mb:.1f} MB, seed={seed}, {loc}) -> {device}")

        return (cond, int(seed), meta_display)
