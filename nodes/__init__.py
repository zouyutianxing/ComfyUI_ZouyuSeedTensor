"""
ZouyuSeedTensor 节点汇总（V3 API）。

每个节点一个文件，均为 io.ComfyNode 子类，通过 comfy_entrypoint 注册。
"""

from .save_seed import ZouyuSaveSeedConditioning
from .seed_loader import ZouyuSeedLoader
from .vae_decode_av import ZouyuVAEDecodeAV
from .extract_media import ZouyuExtractSeedMedia
from .seed_catalog import ZouyuSeedCatalog
from .preview_seed import ZouyuSeedPreview
from .clear_temp import ZouyuClearTemp
from .model_loader import ZouyuModelLoader
from .model_guard import ZouyuModelSwitch


# 全部节点（按展示顺序）
ALL_NODES = [
    ZouyuSaveSeedConditioning,
    ZouyuSeedLoader,
    ZouyuVAEDecodeAV,
    ZouyuExtractSeedMedia,
    ZouyuSeedCatalog,
    ZouyuSeedPreview,
    ZouyuClearTemp,
    ZouyuModelLoader,
    ZouyuModelSwitch,
]
