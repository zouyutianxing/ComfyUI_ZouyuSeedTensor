"""
ZouyuSeedTensor 节点汇总（V3 API）。

每个节点一个文件，均为 io.ComfyNode 子类，通过 comfy_entrypoint 注册。
"""

from .seed_loader import ZouyuSeedLoader
from .vae_decode_av import ZouyuVAEDecodeAV
from .model_loader import ZouyuModelLoader
from .model_guard import ZouyuModelSwitch


# 全部节点（按展示顺序）
ALL_NODES = [
    ZouyuSeedLoader,
    ZouyuVAEDecodeAV,
    ZouyuModelLoader,
    ZouyuModelSwitch,
]
