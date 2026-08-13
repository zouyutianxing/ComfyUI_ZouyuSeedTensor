"""
节点: ZouyuClearTemp（清空临时存储）— V3 API

清空 temp/ 临时存储文件夹。可将任意输入（如视频生成结果）接入 trigger，
在视频彻底生成完成后自动清空临时目录内所有文件。
"""

from comfy_api.latest import io

from ..core import clear_temp_dir, scan_temp_files, get_temp_dir, log


class ZouyuClearTemp(io.ComfyNode):
    """清空临时存储目录。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZouyuClearTemp",
            display_name="清空临时存储 (Zouyu ClearTemp)",
            category="ZouyuAI/SeedTensor",
            is_output_node=True,
            inputs=[
                io.AnyType.Input("trigger", tooltip="任意输入。建议接入视频生成链路的末尾，视频完成后自动触发清空"),
                io.Combo.Input("language", options=["中文", "English"], default="中文"),
            ],
            outputs=[io.String.Output(display_name="清理信息")],
        )

    @classmethod
    def execute(cls, trigger=None, language="中文") -> io.NodeOutput:
        zh = (language != "English")
        before = len(scan_temp_files())
        removed = clear_temp_dir()

        msg = (f"临时存储已清空：移除 {removed} 项（原 {before} 个 .pt 文件）。目录: {get_temp_dir()}"
               if zh else
               f"Temp storage cleared: removed {removed} items ({before} .pt files). Dir: {get_temp_dir()}")

        log(msg)
        return io.NodeOutput(msg, ui={"text": [msg]})
