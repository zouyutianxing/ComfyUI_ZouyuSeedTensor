"""
ComfyUI_ZouyuSeedTensor - 种子张量缓存与混合系统

将 MiniMax H3 视频生成过程中的 conditioning 张量和种子打包保存，
支持通过提示词 @引用 混合多个种子张量文件进行联合生成。

节点分类: ZouyuAI/SeedTensor
目录：
- seeds/  永久存储（长期保留的张量+种子绑定文件）
- temp/   临时存储（一次性生成任务，完成后可一键清空）
"""

from aiohttp import web

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .core import (
    get_seeds_dir, get_temp_dir,
    scan_seed_files, scan_temp_files, scan_all_seed_files,
    load_catalog, rebuild_catalog, clear_temp_dir,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 前端扩展目录（自动增减槽位、中英文切换、@ 下拉、预览）
WEB_DIRECTORY = "./web"


# ---------------------------------------------------------------------------
# HTTP 路由：为前端下拉菜单 / 目录 / 临时清理提供数据
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

    @routes.get("/zouyu_seed_tensor/catalog")
    async def _catalog(request):
        cat = load_catalog()
        cat["temp_files"] = len(scan_temp_files())
        return web.json_response(cat)

    @routes.post("/zouyu_seed_tensor/refresh")
    async def _refresh(request):
        cat = rebuild_catalog()
        cat["temp_files"] = len(scan_temp_files())
        try:
            PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
        except Exception:
            pass
        return web.json_response(cat)

    @routes.post("/zouyu_seed_tensor/clear_temp")
    async def _clear_temp(request):
        removed = clear_temp_dir()
        try:
            PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
        except Exception:
            pass
        return web.json_response({"removed": removed, "temp_files": len(scan_temp_files())})


_register_routes()
