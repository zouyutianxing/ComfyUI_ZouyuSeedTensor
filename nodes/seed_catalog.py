"""
节点: ZouyuSeedCatalog（种子目录）— V3 API

查看 / 重建 seeds/ 目录的自动索引 catalog.json，并统计临时目录文件数。
"""

import json
from comfy_api.latest import io

from ..core import (
    load_catalog, rebuild_catalog, scan_temp_files, log,
)


class ZouyuSeedCatalog(io.ComfyNode):
    """查看 / 刷新永久目录索引，并统计临时目录。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuSeedCatalog",
            display_name="种子目录 (Zouyu Catalog)",
            category="ZouyuAI/SeedTensor",
            inputs=[
                io.Boolean.Input("rebuild", default=False, label_on="重建目录", label_off="读取目录",
                                 tooltip="开启时重新扫描永久目录重建索引"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[io.String.Output(display_name="目录 JSON"), io.Int.Output(display_name="数量")],
        )

    @classmethod
    def execute(cls, rebuild=False, language="中文") -> io.NodeOutput:
        zh = (language != "English")
        cat = rebuild_catalog() if rebuild else load_catalog()
        files = cat.get("files", [])
        temp_count = len(scan_temp_files())

        cat = dict(cat)
        cat["temp_files"] = temp_count

        cat_json = json.dumps(cat, ensure_ascii=False, indent=2, default=str)
        log(f"目录索引: {len(files)} 个永久种子, {temp_count} 个临时文件" if zh
            else f"Catalog: {len(files)} permanent seeds, {temp_count} temp files")
        return io.NodeOutput(cat_json, len(files))
