"""
节点: ZouyuSeedBlender（多种子混合器）

解析提示词中的 @文件名 引用，加载多个种子张量文件（永久 + 临时目录）并混合。
前端在输入 @ 时自动弹出可用种子文件名下拉菜单。
"""

import re

import torch
import comfy.model_management as model_management

from ..core import (
    resolve_seed_path,
    convert_from_serializable, move_to_device, extract_structure,
    blend_conditionings, make_progress, progress_update, log,
)


class ZouyuSeedBlender:
    """解析提示词中的 @文件名 引用，加载并混合多个种子张量。

    用法：在 prompt 中使用 @文件名 引用种子张量（不含 .pt 扩展名）。
    例如: "@shot_001 @shot_002 一个拿着剑的角色在森林中行走"
    """

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "提示词文本。使用 @文件名 引用种子张量文件（输入 @ 会弹出下拉菜单，支持永久+临时目录）"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
            "optional": {
                "weights": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "可选权重: @name=0.7,@name2=0.3。留空则等权平均。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "source_names", "cleaned_prompt")
    FUNCTION = "blend"
    CATEGORY = "ZouyuAI/SeedTensor"

    def _parse_weight_map(self, weights_str):
        wmap = {}
        if not weights_str or not weights_str.strip():
            return wmap
        wpat = re.compile(r'@([\w\-.\u4e00-\u9fff]+)\s*=\s*([\d.]+)')
        for m in wpat.finditer(weights_str):
            try:
                w = float(m.group(2))
                if w > 0:
                    wmap[m.group(1)] = w
            except ValueError:
                pass
        return wmap

    def blend(self, prompt, language, weights=""):
        prompt = prompt or ""
        zh = (language != "English")

        refs = self._AT_PATTERN.findall(prompt)
        if not refs:
            raise ValueError(
                "[ZouyuSeedTensor] 提示词中未找到任何 @文件名 引用。请使用 @文件名 引用种子张量文件。"
                if zh else
                "[ZouyuSeedTensor] No @filename references found in prompt."
            )

        seen = set()
        unique_refs = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                unique_refs.append(r)

        weight_map = self._parse_weight_map(weights)

        cond_list = []
        seeds = []
        loaded_names = []
        device = model_management.get_torch_device()

        pbar = make_progress(len(unique_refs), label="加载种子")

        for ref_name in unique_refs:
            try:
                path, location = resolve_seed_path(ref_name)
            except FileNotFoundError:
                log(f"警告: 文件 {ref_name} 不存在，跳过" if zh else f"Warning: file {ref_name} not found")
                progress_update(pbar, 1)
                continue

            data = torch.load(path, map_location="cpu", weights_only=False)
            cond_data, meta, seed = extract_structure(data)
            cond = convert_from_serializable(cond_data)
            cond = move_to_device(cond, device)

            cond_list.append(cond)
            seeds.append(seed)
            loaded_names.append(ref_name)
            progress_update(pbar, 1)

        if not cond_list:
            raise FileNotFoundError(
                "[ZouyuSeedTensor] 所有 @引用 的文件都不存在（已搜索永久与临时目录）。"
                if zh else "[ZouyuSeedTensor] None of the @referenced files exist."
            )

        if weight_map and loaded_names:
            blend_weights = [weight_map.get(name, 1.0) for name in loaded_names]
        else:
            blend_weights = [1.0] * len(cond_list)

        blended_cond = blend_conditionings(cond_list, blend_weights)

        total_w = sum(blend_weights)
        if total_w > 0 and seeds:
            blended_seed = int(round(sum(s * w for s, w in zip(seeds, blend_weights)) / total_w))
        else:
            blended_seed = seeds[0] if seeds else 0

        cleaned_prompt = self._AT_PATTERN.sub('', prompt).strip()
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        source_names_str = (
            "源: " + ", ".join(f"@{n}(seed={s})" for n, s in zip(loaded_names, seeds))
            if zh else
            "Sources: " + ", ".join(f"@{n}(seed={s})" for n, s in zip(loaded_names, seeds))
        )

        return (blended_cond, blended_seed, source_names_str, cleaned_prompt)
