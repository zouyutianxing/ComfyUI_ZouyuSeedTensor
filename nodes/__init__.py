"""
ZouyuSeedTensor 节点汇总。

每个节点一个文件，均从 core 模块导入共享工具。
"""

from .save_seed import ZouyuSaveSeedConditioning
from .seed_loader import ZouyuSeedLoader
from .vae_decode_av import ZouyuVAEDecodeAV
from .extract_media import ZouyuExtractSeedMedia
from .seed_catalog import ZouyuSeedCatalog
from .preview_seed import ZouyuSeedPreview
from .clear_temp import ZouyuClearTemp


NODE_CLASS_MAPPINGS = {
    "ZouyuSaveSeedConditioning": ZouyuSaveSeedConditioning,
    "ZouyuSeedLoader": ZouyuSeedLoader,
    "ZouyuVAEDecodeAV": ZouyuVAEDecodeAV,
    "ZouyuExtractSeedMedia": ZouyuExtractSeedMedia,
    "ZouyuSeedCatalog": ZouyuSeedCatalog,
    "ZouyuSeedPreview": ZouyuSeedPreview,
    "ZouyuClearTemp": ZouyuClearTemp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZouyuSaveSeedConditioning": "保存种子张量 (Zouyu Save)",
    "ZouyuSeedLoader": "融合加载器 (Zouyu Loader)",
    "ZouyuVAEDecodeAV": "音视频联合解码 (Zouyu VAE Decode)",
    "ZouyuExtractSeedMedia": "提取参考媒体 (Zouyu Extract)",
    "ZouyuSeedCatalog": "种子目录 (Zouyu Catalog)",
    "ZouyuSeedPreview": "种子预览 (Zouyu Preview)",
    "ZouyuClearTemp": "清空临时存储 (Zouyu ClearTemp)",
}
