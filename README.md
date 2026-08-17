# ComfyUI_ZouyuSeedTensor

面向 **MiniMax H3** 视频生成的 ComfyUI 插件套件：融合「种子张量缓存 + 多参考混合 + 音视频联合解码」，并附带一套 **通用模型加载与管理** 工具（动态多槽位加载器 + 导线式模型加载开关 + 三色状态灯）。

- 基于 ComfyUI **V3 API**（`comfy_api.latest`）
- 全部节点位于 `ZouyuAI/SeedTensor` 分类
- 中英文界面，状态灯实时反映模型位置（工作中 / 闲置 / 已卸载）

---

## 节点总览

| 节点 | 名称 | 作用 |
|---|---|---|
| `ZouyuSeedLoader` | 融合加载器 | 种子加载 + 多参考混合 + MiniMax H3 编码，输出张量/引导器/Latent |
| `ZouyuVAEDecodeAV` | 音视频联合解码 | 一个节点同时解码视频画面与音频 |
| `ZouyuModelLoader` | 通用模型加载器 | 动态多槽位（最多 8 个）加载任意类型模型 |
| `ZouyuModelSwitch` | 模型加载开关 | 导线式信号检测，插入任意连线中按信号触发加载/卸载 |

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zouyutianxing/ComfyUI_ZouyuSeedTensor.git
```

重启 ComfyUI 即可（无额外 Python 依赖，全部复用 ComfyUI 自带环境）。

> 兼容性：ComfyUI **0.33+**（V3 API）。旧版 ComfyUI 未适配。

---

## 节点详解

### 1. ZouyuModelLoader — 通用模型加载器

动态多槽位模型加载：下拉选模型后自动出现下一槽位，最多 8 个模型，**类型完全自动识别**（主模型 / 文本模型 / 视频VAE / 音频VAE / LoRA 等），并按识别结果自动适配官方加载器。

- 每个槽位行尾有**三色状态灯 + 状态文字 + 类型标签 + 文件夹浏览**，输出端口与下拉行同水平线
- 支持**自由文件夹**（任意相对 `models` 目录的路径）+ 拖入文件夹自动导入
- **低显存 / CPU缓存** 模式开关（见下文）
- 配置即推送给后端：模型加载开关无需先运行加载器即可识别槽位

### 2. ZouyuModelSwitch — 模型加载开关

导线式信号检测：左右各一个 `*` 任意类型端口，插入任何插件连线中，信号直接透传；**一旦有数据经过**，按「动作」开关向后端发送加载/卸载任务。

- 信号为 `None`（未接线）时仅透传，不触发任何任务
- 模型下拉自动列出加载器已配置的槽位
- 副标题实时显示当前动作（加载 / 卸载 / 转接）
- 行为由加载器的**低显存模式**开关决定（见下文）

### 3. ZouyuSeedLoader — 融合加载器

融合「加载种子 + 多种子混合 + MiniMax H3 Reference to Video」为一个节点，采用**分阶段任务流**最小化显存占用：

- **阶段 1 — 编码烘焙**：仅需 `clip + vae + audio_vae`（不加载视频模型），解析 `@引用` → 提取/合并参考媒体 → 参考 VAE 编码 → 文本编码 → 备份为 `.pt` 种子文件
- **阶段 2 — 加载视频模型 + 解包**：加载 DiT 模型 → 还原 conditioning + latent → 构建引导器 GUIDER → 输出给自定义采样器

特性：

- 提示词 `@文件名` 引用永久目录种子；`@参考图N / @参考视频N / @参考音频N` 引用已连接参考端口（Autogrow 动态扩展，图最多 50 张）
- 参考视频端口直接接收 `VIDEO`（内部自动提取帧 + 音轨）
- `ref_scale` 参考值放大（1.0~5.0，match 模式生效）
- 输出：张量(conditioning) / 引导器(GUIDER) / Latent / 日志

### 4. ZouyuVAEDecodeAV — 音视频联合解码

输入 `samples`（AV latent）+ `vae_image` + `vae_audio`，一次性输出 `图像` 与 `音频`，供 `CreateVideo` / `SaveVideo` 直接合成音视频。

---

## 三色状态灯语义

| 颜色 | 状态 | 含义 |
|---|---|---|
| 🟢 绿 | 工作中 | 工作流执行中，该模型正被当前节点 `load_models_gpu` 调用 |
| 🟡 黄 | 闲置 | 模型在显存 / CPU 内存中，当前未使用 |
| 🔴 红 | 已卸载 | 权重已释放（硬盘），不在显存也不在内存 |

检测机制（多重冗余，任何形态的模型都能准确检测）：

1. **执行中标志**：类级 hook `PromptExecutor.add_message`（ComfyUI 0.33 执行事件唯一入口），后端独立可靠，不依赖前端、不被其它插件覆盖
2. **工作中主信号**：类级 hook `ModelPatcher.model_patches_models`（`load_models_gpu` 对每个传入模型无条件调用），覆盖原实例 / clone / deepcopy / delegate
3. **官方兜底信号**：只读 `current_loaded_models` 的 `currently_used`——TE-Speed 等深度处理节点产生全新身份的 patcher 也能按**权重规模**关联；开关彻底卸载后自动转红

状态灯由前端 **500ms 轮询** + 节点执行完成事件即时刷新。

---

## 低显存 / CPU缓存 模式

由加载器的 `low_vram_mode` 开关决定（加载器执行 / 前端切换时同步后端）：

| 模式 | 开关卸载信号的行为 | 说明 |
|---|---|---|
| **低显存**（开启） | 正常执行**彻底卸载** | 从显存 + CPU 内存释放权重（DynamicVRAM 模型释放内存），灯变红 |
| **CPU缓存**（默认，关闭） | **完全否决**（开关退化为转接点） | 不主动干预模型，完全交由官方模型管理（显存压力时官方自动卸载，权重保留在内存，灯黄） |

> 卸载会释放该模型**所有形态**（含 TE-Speed 等产生的不同 `clone_base_uuid` patcher），确保真正释放内存。

---

## 种子存储目录

| 目录 | 用途 |
|---|---|
| `seeds/` | 永久存储（长期保留的张量+种子绑定文件，`@文件名` 引用） |
| `temp/` | 临时存储（一次性生成任务，完成后自动清理） |

`catalog.json` 自动维护目录索引；拖入导入的新文件数秒内即可被下拉与校验接受。

---

## 示例工作流

`example_workflows/zouyu_model_loader_guard_example.json`：

加载器 4 槽位（主模型 / CLIP / 视频VAE / 音频VAE）→ MiniMax H3 Reference to Video 编码 → 开关链（编码后卸载 CLIP/VAE、采样后卸载主模型）→ SamplerCustomAdvanced 采样 → 音视频联合解码 → 合成输出。

展示了完整的**低显存调度**流程：编码阶段用 CLIP+VAE，采样阶段主模型独占显存，解码阶段按需重载 VAE。

---

## 目录结构

```
ComfyUI_ZouyuSeedTensor/
├── __init__.py              # 扩展入口 + /zouyu_seed_tensor/files 路由
├── core.py                  # 共享工具（路径/序列化/图像视频编解码/进度/日志）
├── reference_encoder.py     # 参考媒体编码（tokenizer + VAE 编码）
├── nodes/
│   ├── __init__.py          # 节点汇总
│   ├── seed_loader.py       # ZouyuSeedLoader（融合加载器）
│   ├── vae_decode_av.py     # ZouyuVAEDecodeAV（音视频联合解码）
│   ├── model_loader.py      # ZouyuModelLoader（通用模型加载器）
│   └── model_guard.py       # ZouyuModelSwitch + 状态注册表 + HTTP 路由
├── web/zouyu_seed_tensor.js # 前端扩展（UI/状态灯/文件夹浏览器/下拉）
└── example_workflows/       # 示例工作流
```

---

## 安全说明

- 本插件**不包含任何 API key、凭据、配置表格或隐私数据**，可安全开源
- HTTP 端点为本地功能接口（与 ComfyUI 官方/社区插件一致，无鉴权）：仅建议**本地使用**（`127.0.0.1`）；若通过 `--listen` 或云端口转发部署，请注意 `import_folder`（上传写入）、`slot_action`（加载/卸载）、`reveal`（打开资源管理器）等端点可被外部调用，建议限制访问来源

---

## 许可证

本项目代码可自由使用与修改。发布时请自行选择开源许可证（如 MIT / Apache-2.0），并保留作者署名。

---

**Stars 欢迎，问题请提 Issue。**
