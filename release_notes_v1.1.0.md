# YOLO训练工具 v1.1.0

本次正式发布以原版稳定入口为准，V2 继续作为并行预览版本保留在仓库中。

## 重点更新

- 原版稳定入口接入独立标注工作台，整合检测标注、分割标注、自动标注、数据整理和训练回填。
- 修复大图切片后无法直接进入标注的问题。
- 修复数据整理时类别名与标签类 ID 不一致可能生成错误 `dataset.yaml` 的问题。
- 修复自动标注回填时覆盖后端返回类别名的问题。
- 补齐多个页面的返回主页面入口和自由切换能力。
- 修复低分辨率下的布局溢出、滚轮不可用和页面挤压问题。
- 完成 `720P / 1K / 1080P / 2K / 4K` 与 `100% / 125% / 150% / 200%` DPI 缩放适配。
- 打包脚本修复为可同时保留原版和 V2 的完整发布产物。

## 自检结果

本次发布前已通过：

- `python _selftest_ui.py`
- `python _selftest_ui_layout.py`
- `python _selftest_interactions.py`
- `python _selftest_annotation_flow.py`
- `python _selftest_segmentation_flow.py`
- `python _selftest_phase2_tools.py`
- `python _selftest_phase3_ai_tools.py`
- `python _selftest_phase3_studio.py`
- `python _selftest_runtime_detection.py`

并已重新打包原版完整发布包，验证 `YOLO训练工具.exe` 可正常启动。

## 发布资产

- `yolo_local_desktop_release.zip`

## 说明

- 本次正式发布入口：`YOLO训练工具.exe`
- V2 不作为本次默认发布资产
- 若首次运行无内置运行时，请在软件内点击“在线一键配置环境”
