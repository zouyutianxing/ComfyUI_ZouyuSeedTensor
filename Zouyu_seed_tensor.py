"""
ZouyuSeedTensor -- 种子张量缓存与混合系统

将 MiniMax H3 视频生成过程中的 conditioning 张量 + 种子打包保存，支持
通过提示词中的 @文件名 引用多个已保存的张量种子文件进行混合生成。

Nodes:
- ZouyuSaveSeedConditioning: 保存 conditioning + 种子到插件 seeds/ 目录
- ZouyuLoadSeedConditioning: 加载单个种子张量文件
- ZouyuSeedBlender: 解析提示词中的 @引用，混合多个种子张量
"""

import os
import re
import torch
import folder_paths
import comfy.model_management as model_management
from pathlib import Path
from comfy.nested_tensor import NestedTensor

# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent
_SEEDS_DIR = _PLUGIN_DIR / "seeds"
_SEEDS_DIR.mkdir(parents=True, exist_ok=True)


def _get_seeds_dir():
    """返回插件内部 seeds/ 目录的绝对路径"""
    return str(_SEEDS_DIR)


def _scan_seed_files():
    """扫描 seeds/ 目录下所有 .pt 文件，返回文件名列表（不含扩展名）"""
    seen = set()
    files = []
    try:
        for name in sorted(os.listdir(_SEEDS_DIR)):
            if name.endswith(".pt") and name not in seen:
                seen.add(name)
                files.append(name)
    except OSError:
        pass
    return files


# Windows 文件名中不允许的字符
_FORBIDDEN_CHARS = set('< > : " / \\ | ? *'.split())


def _safe_filename(name: str) -> str:
    """清理文件名，移除 Windows 不允许的字符，保留中文等 Unicode 字符"""
    safe = "".join(c for c in name if c not in _FORBIDDEN_CHARS).strip()
    # 移除末尾的点号和空格（Windows 不允许）
    safe = safe.rstrip(". ")
    if not safe:
        safe = "seed_tensor"
    return safe


# ---------------------------------------------------------------------------
# 序列化 / 反序列化（处理 NestedTensor）
# ---------------------------------------------------------------------------

def _convert_to_serializable(obj):
    """递归将 NestedTensor 和普通 tensor 转为可 pickle 的纯结构"""
    if isinstance(obj, NestedTensor):
        return {"__nested__": [_convert_to_serializable(t) for t in obj.tensors]}
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _convert_from_serializable(obj):
    """递归将纯结构还原为 NestedTensor / tensor（仍在 CPU 上）"""
    if isinstance(obj, dict):
        if "__nested__" in obj:
            return NestedTensor([_convert_from_serializable(t) for t in obj["__nested__"]])
        return {k: _convert_from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_from_serializable(v) for v in obj]
    return obj


def _move_to_device(obj, device):
    """递归将 tensor / NestedTensor 搬到指定设备"""
    if isinstance(obj, NestedTensor):
        return NestedTensor([_move_to_device(t, device) for t in obj.tensors])
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _extract_structure(data):
    """从加载的 .pt 中提取 conditioning 和元数据"""
    if isinstance(data, dict):
        cond = data.get("conditioning")
        meta = data.get("metadata", {})
        seed = data.get("seed", 0)
        return cond, meta, seed
    return data, {}, 0


# ---------------------------------------------------------------------------
# 混合逻辑
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(f"[JeekSeedTensor] {msg}")


def blend_conditionings(cond_list, weights=None):
    """将多个 conditioning 按权重混合。

    Args:
        cond_list: list of conditioning (每个都是 ComfyUI 标准格式)
        weights: list of float, 权重列表。None 则等权平均。

    Returns:
        blended conditioning
    """
    if not cond_list:
        raise ValueError("[JeekSeedTensor] 没有可混合的 conditioning")

    if len(cond_list) == 1:
        return cond_list[0]

    if weights is None:
        weights = [1.0 / len(cond_list)] * len(cond_list)

    # 确保权重和为 1
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # 取第一个作为基准
    base = cond_list[0]
    result = []

    for batch_idx in range(len(base)):
        # conditioning 格式: list of [tensor, dict] pairs
        base_entry = base[batch_idx]
        base_tensor = base_entry[0]
        base_dict = dict(base_entry[1])

        # 混合 conditioning tensor（加权平均）
        blended_tensor = base_tensor.clone() * weights[0]
        for i in range(1, len(cond_list)):
            other_tensor = cond_list[i][batch_idx][0]
            if other_tensor.shape != blended_tensor.shape:
                _log(f"警告: conditioning[{i}] tensor 形状 {list(other_tensor.shape)} "
                     f"与基准 {list(blended_tensor.shape)} 不一致，将跳过")
                continue
            blended_tensor += other_tensor * weights[i]

        # 收集所有 minimax_refs（参考图/视频潜空间块）
        all_refs = []
        for i, cond in enumerate(cond_list):
            entry = cond[batch_idx]
            entry_dict = entry[1] if isinstance(entry, list) and len(entry) >= 2 else {}
            refs = entry_dict.get("minimax_refs", [])
            if refs:
                all_refs.extend(refs)
                _log(f"  来源[{i}]: 收集 {len(refs)} 个 minimax_refs 块")

        if all_refs:
            base_dict["minimax_refs"] = all_refs
            _log(f"  混合后 minimax_refs 总数: {len(all_refs)}")

        # 混合 pooled_output（如果存在）
        if "pooled_output" in base_dict:
            pooled = base_dict["pooled_output"].clone() * weights[0]
            for i in range(1, len(cond_list)):
                entry = cond_list[i][batch_idx]
                entry_dict = entry[1] if isinstance(entry, list) and len(entry) >= 2 else {}
                if "pooled_output" in entry_dict:
                    pooled += entry_dict["pooled_output"] * weights[i]
            base_dict["pooled_output"] = pooled

        result.append([blended_tensor, base_dict])

    return result


# ---------------------------------------------------------------------------
# 节点: ZouyuSaveSeedConditioning
# ---------------------------------------------------------------------------

class ZouyuSaveSeedConditioning:
    """将 conditioning 张量与种子一起打包保存到插件 seeds/ 目录。

    输入:
    - conditioning: 来自 MiniMaxH3ReferenceToVideo 或 H3 Director Conditioning 的输出
    - seed: 当前使用的随机种子
    - filename: 自定义保存名称（不含扩展名）

    保存格式: {conditioning, seed, metadata} -> seeds/{filename}.pt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "来自 MiniMaxH3ReferenceToVideo 或 Director Conditioning 的 conditioning 输出"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "当前使用的随机种子"
                }),
                "filename": ("STRING", {
                    "default": "my_seed",
                    "tooltip": "保存文件名（不含扩展名），如 shot_001_black_cat"
                }),
            },
            "optional": {
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "可选的提示词文本，写入元数据供参考"
                }),
                "duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "视频时长（秒），写入元数据"
                }),
                "width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "视频宽度"
                }),
                "height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "视频高度"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ZouyuAI/SeedTensor"

    def save(self, conditioning, seed, filename, prompt_text="",
             duration=0.0, width=0, height=0):
        safe_name = _safe_filename(filename)
        path = os.path.join(_get_seeds_dir(), f"{safe_name}.pt")

        # 序列化 conditioning
        cond_data = _convert_to_serializable(conditioning)

        # 构建元数据
        metadata = {
            "seed": int(seed),
            "prompt_text": prompt_text,
            "duration": float(duration) if duration else 0.0,
            "width": int(width) if width else 0,
            "height": int(height) if height else 0,
            "saved_at": str(Path(path).stat().st_ctime if os.path.exists(path) else ""),
        }

        wrapper = {
            "conditioning": cond_data,
            "seed": int(seed),
            "metadata": metadata,
        }

        torch.save(wrapper, path)
        mb = os.path.getsize(path) / (1024 * 1024)
        _log(f"已保存种子张量 -> {path} ({mb:.1f} MB, seed={seed})")

        # 更新前端文件列表
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
        except Exception:
            pass

        return (path,)


# ---------------------------------------------------------------------------
# 节点: ZouyuLoadSeedConditioning
# ---------------------------------------------------------------------------

class ZouyuLoadSeedConditioning:
    """加载单个种子张量文件，输出 conditioning 和种子。

    用于单文件场景，需要混合多个文件请使用 ZouyuSeedBlender。
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _scan_seed_files()
        if not files:
            files = ["(暂无文件)"]
        return {
            "required": {
                "file_name": (files, {
                    "tooltip": "选择 seeds/ 目录下的种子张量文件"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "metadata")
    FUNCTION = "load"
    CATEGORY = "ZouyuAI/SeedTensor"

    def load(self, file_name):
        if file_name == "(暂无文件)" or not file_name:
            raise ValueError("[ZouyuSeedTensor] 没有可用的种子张量文件，请先使用 ZouyuSaveSeedConditioning 保存")

        path = os.path.join(_get_seeds_dir(), file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ZouyuSeedTensor] 文件不存在: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        cond_data, meta, seed = _extract_structure(data)

        # 还原 conditioning 结构
        cond = _convert_from_serializable(cond_data)

        # 搬到计算设备
        device = model_management.get_torch_device()
        cond = _move_to_device(cond, device)

        mb = os.path.getsize(path) / (1024 * 1024)
        meta_str = meta.get("prompt_text", "")[:80] if meta else ""
        _log(f"已加载种子张量 <- {file_name} ({mb:.1f} MB, seed={seed}) -> {device}")

        # 构建可读的元数据字符串
        meta_display = (
            f"文件: {file_name}\n"
            f"种子: {seed}\n"
            f"提示词: {meta_str}\n"
            f"分辨率: {meta.get('width', '?')}x{meta.get('height', '?')}\n"
            f"时长: {meta.get('duration', '?')}s"
        ) if meta else f"文件: {file_name}, 种子: {seed}"

        return (cond, int(seed), meta_display)


# ---------------------------------------------------------------------------
# 节点: ZouyuSeedBlender
# ---------------------------------------------------------------------------

class ZouyuSeedBlender:
    """解析提示词中的 @文件名 引用，加载多个种子张量文件并混合。

    使用方式:
    在 prompt 文本中使用 @文件名 引用已保存的种子张量（不含 .pt 扩展名）。
    例如:
        "@shot_001 @shot_002 一个拿着剑的角色在森林中行走"

    节点会:
    1. 从 seeds/ 目录加载所有被 @ 引用的 .pt 文件
    2. 等权平均混合 conditioning 张量
    3. 收集所有 minimax_refs 参考块
    4. 输出混合后的 conditioning 和综合种子信息
    """

    _AT_PATTERN = re.compile(r'@([\w\-.\u4e00-\u9fff]+)')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "提示词文本。使用 @文件名 引用 seeds/ 目录下的种子张量文件。\n"
                              "例如: @shot_001 @shot_002 一个角色在森林中行走"
                }),
            },
            "optional": {
                "weights": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "可选的权重设置，格式: @name=权重。\n例如: @shot_001=0.7,@shot_002=0.3\n留空则等权平均。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "seed", "source_names", "cleaned_prompt")
    FUNCTION = "blend"
    CATEGORY = "ZouyuAI/SeedTensor"

    def _parse_weight_map(self, weights_str: str) -> dict:
        """解析权重字符串: @name1=0.7,@name2=0.3"""
        wmap = {}
        if not weights_str or not weights_str.strip():
            return wmap
        # 匹配 @name=数字 模式
        wpat = re.compile(r'@([\w\-.\u4e00-\u9fff]+)\s*=\s*([\d.]+)')
        for m in wpat.finditer(weights_str):
            name = m.group(1)
            try:
                w = float(m.group(2))
                if w > 0:
                    wmap[name] = w
            except ValueError:
                pass
        return wmap

    def blend(self, prompt, weights=""):
        prompt = prompt or ""

        # 1. 解析 @引用
        refs = self._AT_PATTERN.findall(prompt)
        if not refs:
            raise ValueError(
                "[ZouyuSeedTensor] 提示词中未找到任何 @文件名 引用。\n"
                "请在提示词中使用 @文件名 格式引用 seeds/ 目录下的种子张量文件。\n"
                "例如: @shot_001 @shot_002 一个角色在森林中行走"
            )

        # 去重保持顺序
        seen = set()
        unique_refs = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                unique_refs.append(r)

        _log(f"提示词中检测到 {len(unique_refs)} 个 @引用: {unique_refs}")

        # 2. 解析权重
        weight_map = self._parse_weight_map(weights)
        if weight_map:
            _log(f"自定义权重: {weight_map}")

        # 3. 加载所有引用的文件
        cond_list = []
        seeds = []
        loaded_names = []
        device = model_management.get_torch_device()

        for ref_name in unique_refs:
            fname = ref_name if ref_name.endswith(".pt") else f"{ref_name}.pt"
            path = os.path.join(_get_seeds_dir(), fname)
            if not os.path.isfile(path):
                _log(f"警告: 文件 {fname} 不存在，跳过")
                continue

            data = torch.load(path, map_location="cpu", weights_only=False)
            cond_data, meta, seed = _extract_structure(data)
            cond = _convert_from_serializable(cond_data)
            cond = _move_to_device(cond, device)

            cond_list.append(cond)
            seeds.append(seed)
            loaded_names.append(ref_name)

            mb = os.path.getsize(path) / (1024 * 1024)
            _log(f"  已加载: {ref_name} ({mb:.1f} MB, seed={seed})")

        if not cond_list:
            raise FileNotFoundError(
                f"[ZouyuSeedTensor] 所有 @引用 的文件都不存在于 {_get_seeds_dir()} 目录下。\n"
                f"请确认已使用 ZouyuSaveSeedConditioning 保存过这些文件。"
            )

        # 4. 计算混合权重
        if weight_map and loaded_names:
            blend_weights = []
            for name in loaded_names:
                w = weight_map.get(name, 1.0)
                blend_weights.append(w)
        else:
            blend_weights = [1.0] * len(cond_list)

        _log(f"开始混合 {len(cond_list)} 个 conditioning, 权重: {blend_weights}")

        # 5. 混合 conditioning
        blended_cond = blend_conditionings(cond_list, blend_weights)

        # 6. 计算综合种子（加权平均取整）
        total_w = sum(blend_weights)
        if total_w > 0 and seeds:
            blended_seed = int(round(sum(s * w for s, w in zip(seeds, blend_weights)) / total_w))
        else:
            blended_seed = seeds[0] if seeds else 0

        # 7. 清理提示词（移除 @引用标记）
        cleaned_prompt = self._AT_PATTERN.sub('', prompt).strip()
        # 清理多余空格
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        source_names_str = ", ".join(
            f"@{n}(seed={s})" for n, s in zip(loaded_names, seeds)
        )

        _log(f"混合完成: {len(loaded_names)} 个源 -> conditioning, "
             f"综合种子={blended_seed}, 源={source_names_str}")

        return (blended_cond, blended_seed, source_names_str, cleaned_prompt)


# ---------------------------------------------------------------------------
# 节点注册
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ZouyuSaveSeedConditioning": ZouyuSaveSeedConditioning,
    "ZouyuLoadSeedConditioning": ZouyuLoadSeedConditioning,
    "ZouyuSeedBlender": ZouyuSeedBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZouyuSaveSeedConditioning": "Zouyu Save Seed+Tensor (保存种子张量)",
    "ZouyuLoadSeedConditioning": "Zouyu Load Seed+Tensor (加载种子张量)",
    "ZouyuSeedBlender": "Zouyu Seed Blender (多种子混合器)",
}
