"""
节点: ZouyuSeedCatalog（种子目录）

查看 / 重建 seeds/ 目录的自动索引 catalog.json，并统计临时目录文件数。
"""

import json

from ..core import (
    load_catalog, rebuild_catalog, scan_temp_files, log,
)


class ZouyuSeedCatalog:
    """查看 / 刷新永久目录索引，并统计临时目录。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rebuild": ("BOOLEAN", {
                    "default": False,
                    "label_on": "重建目录",
                    "label_off": "读取目录",
                    "tooltip": "开启时重新扫描永久目录重建索引"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("catalog_json", "count")
    FUNCTION = "catalog"
    CATEGORY = "ZouyuAI/SeedTensor"

    def catalog(self, rebuild=False, language="中文"):
        zh = (language != "English")
        if rebuild:
            cat = rebuild_catalog()
        else:
            cat = load_catalog()
        files = cat.get("files", [])
        temp_count = len(scan_temp_files())

        cat = dict(cat)
        cat["temp_files"] = temp_count

        cat_json = json.dumps(cat, ensure_ascii=False, indent=2, default=str)
        if zh:
            log(f"目录索引: {len(files)} 个永久种子, {temp_count} 个临时文件")
        else:
            log(f"Catalog: {len(files)} permanent seeds, {temp_count} temp files")
        return (cat_json, len(files))
