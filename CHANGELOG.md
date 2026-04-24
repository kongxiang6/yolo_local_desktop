# Changelog

本项目的所有重要变更都会记录在这里。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Changed
- 仓库文档和辅助文件命名继续整理中，后续版本会持续补充发布说明与升级记录。

## [1.0.0] - 2026-04-25

### Added
- 新增训练、验证、预测、跟踪、导出和数据整理的一体化桌面工作流。
- 新增运行环境检测、内置环境优先、在线一键配置环境和环境修复逻辑。
- 新增参数提示、自定义预设、推荐预设、日志查看和结果目录快速打开能力。
- 新增 `USER_GUIDE.txt`、`delivery/README_FOR_SHARE.txt` 和 GitHub 发布包说明，方便普通用户直接上手。

### Changed
- 整理 GitHub 仓库文档结构，补充 `README.md` 中的项目说明、发布建议和文档索引。
- 统一仓库辅助文件命名，避免中文文件名在部分 Git 工具里显示为转义字符串。
- 将 GitHub Release 使用的无运行时压缩包固定为 `YOLO_training_tool_github_no_runtime.zip`。

### Fixed
- 修复参数预设在折叠区域下可能漏保存、漏加载的问题。
- 修复运行环境选择逻辑，避免把主程序 `exe` 误识别为 Python 解释器。
- 修复在线环境配置、滚轮滚动、下拉弹层、日志显示和部分 UI 交互细节问题。

### Docs
- 补充更适合 GitHub 展示的仓库说明。
- 增加标准 `CHANGELOG.md`，便于后续版本持续记录。
