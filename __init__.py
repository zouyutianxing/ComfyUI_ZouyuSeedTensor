"""
ComfyUI_ZouyuSeedTensor - 种子张量缓存与混合系统

将 MiniMax H3 视频生成过程中的 conditioning 张量和种子打包保存到插件内部，
支持通过提示词 @引用 混合多个种子张量文件进行联合生成。

节点分类: ZouyuAI/SeedTensor
"""

from aiohttp import web

from .Zouyu_seed_tensor import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    get_seeds_dir,
    scan_seed_files,
    load_catalog,
    rebuild_catalog,
    list_lora_files,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 前端扩展目录（@ 下拉菜单、动态 LoRA 槽、动态参考槽、目录刷新）
WEB_DIRECTORY = "./web"


# ---------------------------------------------------------------------------
# HTTP 路由：为前端下拉菜单 / 目录提供数据
# ---------------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer
    except Exception:
        return

    @PromptServer.instance.routes.get("/zouyu_seed_tensor/files")
    async def _files(request):
        files = scan_seed_files()
        return web.json_response({"files": files, "dir": get_seeds_dir()})

    @PromptServer.instance.routes.get("/zouyu_seed_tensor/catalog")
    async def _catalog(request):
        return web.json_response(load_catalog())

    @PromptServer.instance.routes.post("/zouyu_seed_tensor/refresh")
    async def _refresh(request):
        cat = rebuild_catalog()
        try:
            PromptServer.instance.send_sync("Zouyu-seed-files-refresh", {})
        except Exception:
            pass
        return web.json_response(cat)

    @PromptServer.instance.routes.get("/zouyu_seed_tensor/loras")
    async def _loras(request):
        return web.json_response({"loras": list_lora_files()})


_register_routes()
