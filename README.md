# YOLO训练工具

一个面向本地桌面场景的 YOLO 图形化工具，目标是把“标注 -> 整理 -> 训练 -> 验证 -> 预测 -> 跟踪 -> 导出”尽量收进一套界面里完成。

当前仓库同时保留两套入口：

- 原版稳定入口：`app.py`
- V2 并行设计版：`app_v2.py`

本次仓库上传和正式发布以原版稳定入口为准，V2 继续作为并行预览版本保留在仓库中。

## 当前状态

- 原版 UI 保留，并继续作为正式交付版本。
- V2 UI 已接入仓库，但本次不作为默认发布资产。
- 已补齐标注工作台、数据整理、训练、导出等核心流程。
- 已完成多分辨率和 Windows 高 DPI 适配。

## 核心能力

- 标注工作台
  - 检测框标注
  - 实例分割标注
  - 视频抽帧与标注入口
  - 自动标注结果回填
- 数据整理
  - YOLO TXT
  - LabelMe
  - COCO JSON
  - 自动识别类别并生成 `dataset.yaml`
- 训练与推理
  - 训练
  - 验证
  - 预测
  - 跟踪
  - 导出
- 辅助能力
  - 在线一键配置环境
  - 运行环境自检
  - 模型中心
  - AI 工作流
  - 预设保存与推荐预设
  - 日志和结果目录一键打开

## 分辨率与缩放适配

本轮已针对以下场景做过窗口与布局自检：

- 分辨率：`720P`、`1K(1600x900)`、`1080P`、`2K`、`4K`
- Windows 缩放：`100%`、`125%`、`150%`、`200%`

已完成的适配点：

- 默认窗口大小按分辨率和有效工作区自动收放，不会过大或过小。
- `720P` 下自动进入紧凑布局，避免关键区域被挤出。
- V2 的训练/导出侧栏改为可滚动区域。
- 启用 Windows DPI 感知，减少高缩放场景下的模糊问题。

## 已修复的关键问题

- 修复大图切片后回传错误目录，导致切片结果无法直接进入标注的问题。
- 修复数据整理时类别名数量和标签最大类 ID 不一致，可能生成错误 `dataset.yaml` 的问题。
- 修复自动标注回填时覆盖后端返回类别名的问题。
- 修复多个子页面缺少返回主页面按钮的问题。
- 修复多个低分辨率页面滚轮不可用、内容溢出或切换入口缺失的问题。
- 修复完整发布时后打包的版本会清掉前一个 `release` 产物的问题。

## 已通过的自检

建议至少关注下面这些自测脚本：

```powershell
python -m py_compile app.py app_v2.py backend.py `
  ai_platform_panel.py annotation_editor.py annotation_studio.py `
  model_hub_panel.py multitask_dataset_panel.py segmentation_editor.py `
  video_annotation_panel.py

python _selftest_ui.py
python _selftest_ui_layout.py
python _selftest_interactions.py
python _selftest_annotation_flow.py
python _selftest_segmentation_flow.py
python _selftest_phase2_tools.py
python _selftest_phase3_ai_tools.py
python _selftest_phase3_studio.py
python _selftest_runtime_detection.py
```

本轮交付前，这些脚本均已通过。

## 快速开始

### 给最终用户

1. 完整解压发布包，不要只拿一个 `exe` 单独运行。
2. 双击 `YOLO训练工具.exe`。
3. 第一次打开先点“在线一键配置环境”。
4. 等左侧日志提示完成后，再开始标注、训练或导出。

### 给开发者

```powershell
cd I:\AI\yolo_local_desktop
python app.py
```

如需打开 V2：

```powershell
cd I:\AI\yolo_local_desktop
python app_v2.py
```

## 打包

### 原版稳定版

快速本地打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

完整发布包：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Full
```

如果需要把 `runtime` 一起打进去：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Full -IncludeRuntime
```

### V2 并行版

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe_v2.ps1
```

## 本次正式发布资产

本次正式发布以原版稳定版为准，建议上传：

- `release/yolo_local_desktop_release.zip`

并行预览版保留在本地：

- `release/yolo_local_desktop_v2_release.zip`

## 仓库结构

```text
yolo_local_desktop/
├─ app.py                         # 原版稳定入口
├─ app_v2.py                      # V2 并行入口
├─ annotation_studio.py           # 标注工作台容器
├─ annotation_editor.py           # 检测标注
├─ segmentation_editor.py         # 分割标注
├─ video_annotation_panel.py      # 视频标注辅助
├─ multitask_dataset_panel.py     # 数据整理
├─ ai_platform_panel.py           # AI 工作流
├─ model_hub_panel.py             # 模型中心
├─ backend.py                     # 后端任务入口
├─ build_exe.ps1                  # 原版打包脚本
├─ build_exe_v2.ps1               # V2 打包脚本
├─ delivery/README_FOR_SHARE.txt  # 发给最终用户的简版说明
├─ USER_GUIDE.txt                 # 详细操作说明
└─ CHANGELOG.md                   # 版本记录
```

## 发布建议

### 发布到 GitHub 仓库

- 提交源码、文档和测试脚本。
- 不要把 `dist/`、`release/`、`build/` 提交进仓库。
- 发布页上传打好的 ZIP 资产即可。

### 发给最终用户

- 优先发送完整 ZIP 包，而不是单独的 `exe`
- 提醒对方先完整解压
- 第一次先做环境配置，再开始跑任务

## 文档索引

- 仓库说明：`README.md`
- 操作说明：`USER_GUIDE.txt`
- 分发说明：`delivery/README_FOR_SHARE.txt`
- 版本记录：`CHANGELOG.md`

## 说明

- 原版 UI 仍然是当前正式发布入口。
- V2 继续保留并迭代，但本次不替换原版发布位置。
- `_local/` 下的内容只用于本地会话承接，不参与打包和发布。
