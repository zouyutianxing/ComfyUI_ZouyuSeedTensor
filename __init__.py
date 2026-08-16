"""
ComfyUI_ZouyuSeedTensor - 种子张量缓存与混合系统（V3 API）

将 MiniMax H3 视频生成过程中的 conditioning 张量和种子打包保存，
支持通过提示词 @引用 混合多个种子张量文件进行联合生成。

节点分类: ZouyuAI/SeedTensor
目录：
- seeds/  永久存储（长期保留的张量+种子绑定文件）
- temp/   临时存储（一次性生成任务，完成后自动清理）
"""

from aiohttp import web

from comfy_api.latest import ComfyExtension

from .nodes import ALL_NODES
from .core import (
    get_seeds_dir, get_temp_dir,
    scan_seed_files, scan_temp_files, scan_all_seed_files,
)
from .nodes.model_guard import register_routes as register_model_guard_routes

# 前端扩展目录（中英文切换、@ 下拉、状态灯、模型加载器/开关 UI）
WEB_DIRECTORY = "./web"


class ZouyuSeedTensorExtension(ComfyExtension):
    async def get_node_list(self):
        return ALL_NODES


async def comfy_entrypoint() -> ComfyExtension:
    return ZouyuSeedTensorExtension()


# ---------------------------------------------------------------------------
# HTTP 路由：为前端 @引用 自动补全提供种子文件列表
# ---------------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
        if routes is None:
            return
    except Exception:
        return

    @routes.get("/zouyu_seed_tensor/files")
    async def _files(request):
        return web.json_response({
            "permanent": scan_seed_files(),
            "temp": scan_temp_files(),
            "all": scan_all_seed_files(),
            "seeds_dir": get_seeds_dir(),
            "temp_dir": get_temp_dir(),
        })


_register_routes()
register_model_guard_routes()

