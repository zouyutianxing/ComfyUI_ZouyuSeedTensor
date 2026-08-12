"""
节点: ZouyuClearTemp（清空临时存储）

清空 temp/ 临时存储文件夹。可将任意输入（如视频生成结果）接入 trigger，
在视频彻底生成完成后自动清空临时目录内所有文件。
"""

from ..core import clear_temp_dir, scan_temp_files, get_temp_dir, log


class ZouyuClearTemp:
    """清空临时存储目录。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {
                    "tooltip": "任意输入。建议接入视频生成链路的末尾，视频完成后自动触发清空"
                }),
                "language": (["中文", "English"], {"default": "中文"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleared_info",)
    FUNCTION = "clear"
    OUTPUT_NODE = True
    CATEGORY = "ZouyuAI/SeedTensor"

    def clear(self, trigger=None, language="中文"):
        zh = (language != "English")
        before = len(scan_temp_files())
        removed = clear_temp_dir()

        if zh:
            msg = f"临时存储已清空：移除 {removed} 项（原 {before} 个 .pt 文件）。目录: {get_temp_dir()}"
        else:
            msg = f"Temp storage cleared: removed {removed} items ({before} .pt files). Dir: {get_temp_dir()}"

        log(msg)
        return {"ui": {"text": [msg]}, "result": (msg,)}
