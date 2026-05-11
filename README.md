# YOLO训练工具

这是一个面向本地桌面使用的 YOLO 可视化工具，目标是把“标注 -> 数据整理 -> 训练 -> 验证 -> 预测 -> 跟踪 -> 导出”串成一条尽量顺滑的工作流。

当前保留两个入口：
- 原版稳定入口：`app.py`
- V2 并行预览入口：`app_v2.py`

正式发布与默认交付仍以原版为准，V2 继续并行迭代。

## 项目特点

- 原版 UI 以稳定可用为主，适合作为默认发布入口。
- V2 UI 参考新的产品提案图持续重做，保留更强的流程化结构。
- 支持检测、分割、视频标注、自动标注和数据整理。
- 支持训练、验证、预测、跟踪、导出等完整 YOLO 常用流程。
- 已针对 Windows 常见 DPI 与分辨率场景做过适配。
- 支持老显卡 `legacy-cuda` 兼容模式，并持续优化一键配置环境流程。

## 主要功能

### 标注与数据

- 检测框标注
- 实例分割标注
- 视频标注
- 自动标注
- 多来源数据整理
- 自动生成训练所需 `dataset.yaml`

### 训练与推理

- 训练
- 验证
- 预测
- 跟踪
- 导出部署格式

### 工具能力

- 在线一键配置环境
- 模型选择与预设
- 实时日志
- AI 辅助工具
- 独立标注工作台
- V2 并行界面预览

## 分辨率与显示适配

当前重点覆盖场景：

- `720P`
- `1K (1600x900)`
- `1080P`
- `2K`
- `4K`
- Windows 缩放 `100% / 125% / 150% / 200%`

已完成的适配方向：

- 默认窗口大小会根据有效工作区自动调整，避免过大或过小。
- `720P` 下优先保证核心操作区可见。
- V2 顶部导航改为响应式布局，窄屏下会自动换行。
- V2 视频标注页面改为滚动容器，低分辨率下底部内容不再被截断。
- 页面切换后滚轮仍作用于当前页面，减少“滚轮划不动”的情况。
- 启用 Windows DPI 感知与 Tk 缩放同步，降低高 DPI 下控件错位概率。

## 最近完成的关键修复

- 修复大图切片后回传错误目录，切片结果现在可直接进入标注。
- 修复数据整理时类别名长度和标签最大类 ID 不一致的问题，避免生成无效 `dataset.yaml`。
- 修复自动标注回填时覆盖后端返回类别名的问题。
- 修复窗口版 `exe` 调用后台命令时，日志输出触发 `[Errno 22] Invalid argument` 的崩溃。
- 修复 V2 训练页和导出页右侧日志预览不实时刷新、切页后才显示的问题。
- 调整 V2 训练页/导出页日志区域高度，使实时日志更易读。
- 统一 V2 的顶部导航职责，减少页面切换时布局跳动感。

## 打包与发布规制

当前默认发布规制如下：

- 默认完整发布包 `不带 runtime`
- 默认完整发布目录 `不带 presets`
- 只有显式指定 `-IncludeRuntime` 时，才会把运行时一起打包
- 原版与 V2 都统一使用 `build_exe.ps1` 配合对应 `spec` 生成发布目录和 ZIP

常用命令：

```powershell
# 原版完整发布
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Full

# V2 完整发布
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Full -SpecPath .\yolo_local_desktop_v2.spec

# 如需把 runtime 一起打包
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Full -IncludeRuntime
```

## 自检建议

每次发版前至少建议做这些检查：

```powershell
python -m py_compile app.py app_v2.py backend.py `
  ai_platform_panel.py annotation_editor.py annotation_studio.py `
  model_hub_panel.py multitask_dataset_panel.py segmentation_editor.py `
  video_annotation_panel.py
```

另外建议补做：

- 核对原版和 V2 发布目录都不含 `runtime`
- 核对原版和 V2 发布目录都不含 `presets`
- 验证两个 ZIP 可正常解压
- 抽查环境配置、标注、训练、导出等主流程入口是否可打开

## 目录说明

- `app.py`：原版正式入口
- `app_v2.py`：V2 并行预览入口
- `backend.py`：命令行与环境配置后台入口
- `vendor_backend/`：运行时安装、环境配置等后端实现
- `build_exe.ps1`：统一打包脚本
- `delivery/README_FOR_SHARE.txt`：分发包附带说明
- `YOLO训练工具操作说明.txt`：面向最终用户的中文操作说明
- `_local/YOLO_TRAINING_TOOL_SESSION_CONTEXT.md`：本地会话承接文档，不进仓库，不参与打包