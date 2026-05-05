from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

import app as legacy_app
from annotation_studio import AnnotationStudio, WORKSPACE_DESCRIPTIONS, WORKSPACE_LABELS


APP_NAME = "YOLO训练工具"
APP_VERSION = "v2.1"
V2_DEFAULT_WINDOW_WIDTH = 1600
V2_DEFAULT_WINDOW_HEIGHT = 900
V2_MIN_WINDOW_WIDTH = 1180
V2_MIN_WINDOW_HEIGHT = 680
WINDOW_BG = "#f4f7fb"
CARD_BG = "#ffffff"
CARD_ALT = "#f8fbff"
CARD_SOFT = "#f1f7ff"
PRIMARY = "#2f80ff"
PRIMARY_DARK = "#1e63d7"
PRIMARY_SOFT = "#eaf3ff"
BORDER = "#d9e6fb"
TEXT = "#182338"
TEXT_MUTED = "#6c7a92"
SUCCESS = "#24a36c"
WARNING = "#f0a43b"

WORKSPACE_PAGE_IDS = {
    "detect": "detect",
    "segment": "segment",
    "video": "video",
    "organize": "organize",
    "ai_platform": "ai",
    "model_hub": "model_hub",
}

PAGE_ORDER = ("detect", "segment", "video", "organize", "ai", "model_hub", "train", "export")
IMMERSIVE_WORKSPACE_PAGES = {"detect", "segment", "video", "organize"}

PAGE_TITLES = {
    "home": "总览工作台",
    "detect": "检测框标注",
    "segment": "实例分割",
    "video": "视频标注",
    "organize": "数据集整理",
    "ai": "AI工作流",
    "model_hub": "模型中心",
    "train": "训练工作台",
    "export": "导出工作台",
}

PAGE_SUBTITLES = {
    "home": "把标注、整理、训练和导出串成一条顺滑工作流。",
    "detect": "单独打开检测标注工作区，适合按图逐张修框。",
    "segment": "单独打开分割工作区，专注掩码与轮廓编辑。",
    "video": "从视频抽帧并快速送往检测或分割工作区。",
    "organize": "整理原始标注，生成可直接训练的标准数据集。",
    "ai": "串联大图切片、辅助提示和自动标注能力。",
    "model_hub": "统一管理官方推荐模型与本地模型文件。",
    "train": "围绕数据集、模型、预设和日志的完整训练入口。",
    "export": "将训练结果导出为部署所需格式，并保留验证信息。",
}

NAV_ITEMS = (
    ("home", "总览"),
    ("detect", "检测标注"),
    ("segment", "实例分割"),
    ("video", "视频标注"),
    ("organize", "数据集整理"),
    ("ai", "AI工作流"),
    ("model_hub", "模型中心"),
    ("train", "训练工作台"),
    ("export", "导出工作台"),
)

HOME_CARD_DESCRIPTIONS = {
    "detect": "打开图片目录，修框、改类、调用自动标注。",
    "segment": "针对实例掩码做轮廓编辑与类别整理。",
    "video": "先抽帧，再送入检测或分割工作区继续标注。",
    "organize": "识别标注格式并生成 dataset.yaml 与标准目录。",
    "ai": "把切片、提示词辅助与自动标注串成流程。",
    "model_hub": "查看本地模型、官方推荐，并一键应用。",
    "train": "配置训练、验证、预测、跟踪与预设。",
    "export": "导出 ONNX、TensorRT 等部署格式。",
}

HOME_CARD_BADGES = {
    "detect": "1",
    "segment": "2",
    "video": "3",
    "organize": "4",
    "ai": "5",
    "model_hub": "6",
    "train": "7",
    "export": "8",
}


class AppV2(legacy_app.App):
    def _configure_window_geometry(self) -> None:
        width, height, min_width, min_height = legacy_app.resolve_window_geometry(
            self.root,
            min_width=V2_MIN_WINDOW_WIDTH,
            min_height=V2_MIN_WINDOW_HEIGHT,
            preferred_width=V2_DEFAULT_WINDOW_WIDTH,
            preferred_height=V2_DEFAULT_WINDOW_HEIGHT,
            min_height_floor=620,
        )
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min_width, min_height)

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.root.title(f"{APP_NAME} - {APP_VERSION}")
        self.show_page("home")

    def _build_ui(self) -> None:
        self.root.configure(bg=WINDOW_BG)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._v2_ready = False
        self.active_page_var = tk.StringVar(value="home")
        self.topbar_page_var = tk.StringVar(value=PAGE_TITLES["home"])
        self.workspace_header_title_var = tk.StringVar(value=PAGE_TITLES["detect"])
        self.workspace_header_desc_var = tk.StringVar(value=PAGE_SUBTITLES["detect"])
        self.workspace_header_context_var = tk.StringVar(value="目录：未选择\n内容：未选择")
        self.footer_project_var = tk.StringVar(value="当前路径：未选择")
        self.footer_result_var = tk.StringVar(value="结果位置：未生成")
        self.footer_storage_var = tk.StringVar(value="存储使用：读取中")
        self.home_workspace_var = tk.StringVar(value="未进入工作区")
        self.home_source_var = tk.StringVar(value="未选择")
        self.home_preview_var = tk.StringVar(value="未生成")
        self.home_status_var = tk.StringVar(value="等待开始")
        self.home_recent_var = tk.StringVar(value="最近还没有新的工作记录。")
        self.workspace_metric_vars = {
            "project": tk.StringVar(value="未选择"),
            "focus": tk.StringVar(value="未选择"),
            "count": tk.StringVar(value="0"),
            "next": tk.StringVar(value="未生成"),
        }
        self.train_overview_vars = {
            "action": tk.StringVar(value="训练"),
            "data": tk.StringVar(value="未选择"),
            "model": tk.StringVar(value="未选择"),
            "output": tk.StringVar(value="未生成"),
        }
        self.train_side_status_var = tk.StringVar(value="等待开始")
        self.train_side_detail_var = tk.StringVar(value="还没有开始训练任务。")
        self.train_log_preview_var = tk.StringVar(value="日志预览会在这里更新。")
        self.export_overview_vars = {
            "weights": tk.StringVar(value="未选择"),
            "format": tk.StringVar(value="自动识别"),
            "target": tk.StringVar(value="未设置"),
            "status": tk.StringVar(value="等待开始"),
        }
        self.export_side_detail_var = tk.StringVar(value="还没有开始导出任务。")
        self.export_log_preview_var = tk.StringVar(value="日志预览会在这里更新。")
        self.module_meta_vars = {page_id: tk.StringVar(value="等待开始") for page_id in PAGE_ORDER}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.top_nav_buttons: dict[str, tk.Button] = {}
        self.page_frames: dict[str, tk.Frame] = {}
        self.status_badges: list[tuple[tk.Label, str]] = []
        self.annotation_workspace_page_id = "detect"
        self._responsive_after_id: str | None = None

        self._build_topbar()

        body = tk.Frame(self.root, bg=WINDOW_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self.main_body = body

        self._build_sidebar(body)

        self.page_host = tk.Frame(body, bg=WINDOW_BG)
        self.page_host.grid(row=0, column=1, sticky="nsew")
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)

        self._build_home_page()
        self._build_workspace_page()
        self._build_train_page()
        self._build_export_page()
        self._build_footer()
        self._build_hidden_support_widgets()

        self._v2_ready = True
        self._raise_page("home")
        self._update_nav_state()
        self.root.bind("<Configure>", self._handle_root_configure, add="+")
        self._apply_responsive_layout()

    def _build_topbar(self) -> None:
        topbar = tk.Frame(self.root, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        topbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        topbar.grid_columnconfigure(1, weight=1)
        self.topbar = topbar

        brand = tk.Frame(topbar, bg=CARD_BG)
        brand.grid(row=0, column=0, sticky="w", padx=12, pady=8)
        tk.Label(
            brand,
            text=APP_NAME,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(side="left")
        tk.Label(
            brand,
            text=APP_VERSION,
            bg=PRIMARY_SOFT,
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=3,
        ).pack(side="left", padx=(8, 0))

        center = tk.Frame(topbar, bg=CARD_BG)
        center.grid(row=0, column=1, sticky="ew", padx=8, pady=7)
        center.grid_columnconfigure(0, weight=1)

        tk.Label(
            center,
            textvariable=self.topbar_page_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            center,
            text="保留原版入口，新增一套独立 V2 界面。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 5))
        self.topbar_subtitle_label = center.grid_slaves(row=1, column=0)[0]

        nav = tk.Frame(center, bg=CARD_BG)
        nav.grid(row=2, column=0, sticky="ew")
        self.topbar_nav = nav
        for index, (page_id, label) in enumerate(NAV_ITEMS):
            button = tk.Button(
                nav,
                text=label,
                command=lambda item=page_id: self.show_page(item),
                bg=CARD_BG,
                fg=TEXT_MUTED,
                activebackground=PRIMARY_SOFT,
                activeforeground=PRIMARY_DARK,
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=8,
                pady=4,
                font=("Microsoft YaHei UI", 9, "bold"),
                cursor="hand2",
            )
            button.grid(row=0, column=index, padx=(0, 4))
            self.top_nav_buttons[page_id] = button

        actions = tk.Frame(topbar, bg=CARD_BG)
        actions.grid(row=0, column=2, sticky="e", padx=12, pady=8)
        quick_actions = tk.Frame(actions, bg=CARD_BG)
        quick_actions.pack(side="left")
        self.topbar_quick_actions = quick_actions
        self._secondary_button(quick_actions, "继续当前工作", self._continue_current_work).pack(side="left", padx=(0, 8))
        self._secondary_button(quick_actions, "打开结果", self.open_result_location).pack(side="left", padx=(0, 8))
        self._secondary_button(quick_actions, "打开日志", self.open_log_file).pack(side="left", padx=(0, 8))
        self.topbar_home_button = self._secondary_button(actions, "返回总览", lambda: self.show_page("home"))
        process_badge = tk.Label(
            actions,
            textvariable=self.process_status_var,
            bg=PRIMARY_SOFT,
            fg=PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=12,
            pady=7,
        )
        process_badge.pack(side="left")
        self.topbar_status_badge = process_badge
        self.process_status_labels.append(process_badge)
        self.status_badges.append((process_badge, "process"))

    def _build_sidebar(self, parent: tk.Widget) -> None:
        sidebar = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, width=204)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar = sidebar

        header = tk.Frame(sidebar, bg=CARD_BG)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        tk.Label(
            header,
            text="页面导航",
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            header,
            text="按新的产品结构把工作流拆成独立页面。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            justify="left",
            wraplength=164,
        ).pack(fill="x", pady=(5, 0))

        nav = tk.Frame(sidebar, bg=CARD_BG)
        nav.grid(row=1, column=0, sticky="ew", padx=10)
        nav.grid_columnconfigure(0, weight=1)

        for row, (page_id, label) in enumerate(NAV_ITEMS):
            button = tk.Button(
                nav,
                text=label,
                command=lambda item=page_id: self.show_page(item),
                bg=CARD_BG,
                fg=TEXT,
                activebackground=PRIMARY_SOFT,
                activeforeground=PRIMARY_DARK,
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=PRIMARY,
                font=("Microsoft YaHei UI", 10, "bold"),
                anchor="w",
                padx=12,
                pady=8,
                cursor="hand2",
            )
            button.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            self.nav_buttons[page_id] = button

        status_card = self._section_card(sidebar, title="当前状态")
        status_card.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self._info_value(status_card.content, self.left_result_var, wraplength=156, pady=(0, 10))
        self._info_label(status_card.content, "结果位置", pady=(0, 4))
        self._info_value(status_card.content, self.result_location_var, wraplength=156, fg=TEXT, pady=(0, 10))
        self._info_label(status_card.content, "最近记录", pady=(0, 4))
        self._info_value(status_card.content, self.home_recent_var, wraplength=156, pady=(0, 12))

    def _build_home_page(self) -> None:
        page = tk.Frame(self.page_host, bg=WINDOW_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)
        self.page_frames["home"] = page

        hero = self._page_hero(page, "home")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        stats = tk.Frame(page, bg=WINDOW_BG)
        stats.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for column in range(4):
            stats.grid_columnconfigure(column, weight=1)
        self._metric_card(stats, 0, "当前工作区", self.home_workspace_var)
        self._metric_card(stats, 1, "当前来源", self.home_source_var)
        self._metric_card(stats, 2, "当前预览", self.home_preview_var)
        self._metric_card(stats, 3, "运行状态", self.home_status_var)

        scroll = legacy_app.ScrollableFrame(page, background=WINDOW_BG)
        scroll.grid(row=2, column=0, sticky="nsew")
        self.home_scroll = scroll
        content = scroll.inner
        content.grid_columnconfigure(0, weight=1)

        module_grid = tk.Frame(content, bg=WINDOW_BG)
        module_grid.grid(row=0, column=0, sticky="ew")
        for column in range(4):
            module_grid.grid_columnconfigure(column, weight=1)

        for index, page_id in enumerate(PAGE_ORDER):
            row = index // 4
            column = index % 4
            self._home_module_card(module_grid, row, column, page_id)

        recent = self._section_card(content, title="最近活动")
        recent.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._info_value(recent.content, self.home_recent_var, wraplength=1120)

    def _build_workspace_page(self) -> None:
        page = tk.Frame(self.page_host, bg=WINDOW_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)
        self.page_frames["workspace"] = page
        self.annotation_page = page

        hero = self._page_hero(page, "detect")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.workspace_hero = hero

        metrics = tk.Frame(page, bg=WINDOW_BG)
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.workspace_metrics_row = metrics
        for column in range(4):
            metrics.grid_columnconfigure(column, weight=1)
        self._metric_card(metrics, 0, "当前目录", self.workspace_metric_vars["project"])
        self._metric_card(metrics, 1, "当前内容", self.workspace_metric_vars["focus"])
        self._metric_card(metrics, 2, "样本数量", self.workspace_metric_vars["count"])
        self._metric_card(metrics, 3, "下一步输出", self.workspace_metric_vars["next"])

        studio_shell = tk.Frame(page, bg=WINDOW_BG)
        studio_shell.grid(row=2, column=0, sticky="nsew")
        studio_shell.grid_rowconfigure(1, weight=1)
        studio_shell.grid_columnconfigure(0, weight=1)

        action_row = tk.Frame(studio_shell, bg=WINDOW_BG)
        action_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.workspace_action_row = action_row
        self._secondary_button(action_row, "打开结果", self.open_result_location).pack(side="left", padx=(0, 8))
        self._secondary_button(action_row, "打开日志", self.open_log_file).pack(side="left", padx=(0, 8))
        self._secondary_button(action_row, "去数据整理", lambda: self.show_page("organize")).pack(side="left", padx=(0, 8))
        self._primary_button(action_row, "转到训练", lambda: self.show_page("train")).pack(side="left")

        studio_host = tk.Frame(studio_shell, bg=WINDOW_BG)
        studio_host.grid(row=1, column=0, sticky="nsew")
        studio_host.grid_rowconfigure(0, weight=1)
        studio_host.grid_columnconfigure(0, weight=1)

        self.annotation_editor = AnnotationStudio(
            studio_host,
            detect_session_path=legacy_app.ANNOTATION_SESSION_PATH,
            segment_session_path=legacy_app.SEGMENTATION_SESSION_PATH,
            on_state_change=self._refresh_summary,
            on_notice=self._handle_annotation_notice,
            on_export_request=self.start_prepare_dataset_from_annotation,
            on_auto_label_request=self.start_annotation_auto_label,
            on_dataset_ready=self._handle_annotation_dataset_ready,
            on_switch_to_train=self._open_train_workspace_from_annotation,
            on_workspace_change=self._handle_embedded_workspace_change,
            embedded_mode=True,
        )
        self.annotation_editor.grid(row=0, column=0, sticky="nsew")

    def _build_train_page(self) -> None:
        page = tk.Frame(self.page_host, bg=WINDOW_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=7)
        page.grid_columnconfigure(1, weight=3)
        self.page_frames["train"] = page

        hero = self._page_hero(page, "train")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        stats = tk.Frame(page, bg=WINDOW_BG)
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(4):
            stats.grid_columnconfigure(column, weight=1)
        self._metric_card(stats, 0, "运行模式", self.train_overview_vars["action"])
        self._metric_card(stats, 1, "训练数据", self.train_overview_vars["data"])
        self._metric_card(stats, 2, "模型", self.train_overview_vars["model"])
        self._metric_card(stats, 3, "输出预览", self.train_overview_vars["output"])

        left = tk.Frame(page, bg=WINDOW_BG)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.train_scroll = legacy_app.ScrollableFrame(left, background=WINDOW_BG)
        self.train_scroll.grid(row=0, column=0, sticky="nsew")
        self._build_train_view(self.train_scroll.inner)

        right = tk.Frame(page, bg=WINDOW_BG)
        right.grid(row=2, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.train_side_scroll = legacy_app.ScrollableFrame(right, background=WINDOW_BG)
        self.train_side_scroll.grid(row=0, column=0, sticky="nsew")
        train_side = self.train_side_scroll.inner
        train_side.grid_columnconfigure(0, weight=1)

        run_card = self._section_card(train_side, title="运行状态")
        run_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._info_value(run_card.content, self.train_side_status_var, fg=PRIMARY, font_size=12, bold=True, pady=(0, 8))
        self._info_value(run_card.content, self.train_side_detail_var, wraplength=320)

        action_card = self._section_card(train_side, title="快捷操作")
        action_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self._secondary_button(action_card.content, "打开结果", self.open_result_location).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "打开日志", self.open_log_file).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "转到导出", lambda: self.show_page("export")).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "回到数据整理", lambda: self.show_page("organize")).pack(fill="x")

        log_card = self._section_card(train_side, title="日志预览")
        log_card.grid(row=2, column=0, sticky="nsew")
        self._info_value(log_card.content, self.train_log_preview_var, wraplength=320)

    def _build_export_page(self) -> None:
        page = tk.Frame(self.page_host, bg=WINDOW_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=7)
        page.grid_columnconfigure(1, weight=3)
        self.page_frames["export"] = page

        hero = self._page_hero(page, "export")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        stats = tk.Frame(page, bg=WINDOW_BG)
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(4):
            stats.grid_columnconfigure(column, weight=1)
        self._metric_card(stats, 0, "权重文件", self.export_overview_vars["weights"])
        self._metric_card(stats, 1, "导出格式", self.export_overview_vars["format"])
        self._metric_card(stats, 2, "输出目录", self.export_overview_vars["target"])
        self._metric_card(stats, 3, "任务状态", self.export_overview_vars["status"])

        left = tk.Frame(page, bg=WINDOW_BG)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.export_scroll = legacy_app.ScrollableFrame(left, background=WINDOW_BG)
        self.export_scroll.grid(row=0, column=0, sticky="nsew")
        self._build_export_view(self.export_scroll.inner)

        right = tk.Frame(page, bg=WINDOW_BG)
        right.grid(row=2, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.export_side_scroll = legacy_app.ScrollableFrame(right, background=WINDOW_BG)
        self.export_side_scroll.grid(row=0, column=0, sticky="nsew")
        export_side = self.export_side_scroll.inner
        export_side.grid_columnconfigure(0, weight=1)

        status_card = self._section_card(export_side, title="导出状态")
        status_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._info_value(status_card.content, self.export_overview_vars["status"], fg=PRIMARY, font_size=12, bold=True, pady=(0, 8))
        self._info_value(status_card.content, self.export_side_detail_var, wraplength=320)

        action_card = self._section_card(export_side, title="快捷操作")
        action_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self._secondary_button(action_card.content, "打开结果", self.open_result_location).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "打开日志", self.open_log_file).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "切回训练", lambda: self.show_page("train")).pack(fill="x", pady=(0, 8))
        self._secondary_button(action_card.content, "回到模型中心", lambda: self.show_page("model_hub")).pack(fill="x")

        log_card = self._section_card(export_side, title="日志预览")
        log_card.grid(row=2, column=0, sticky="nsew")
        self._info_value(log_card.content, self.export_log_preview_var, wraplength=320)

    def _build_footer(self) -> None:
        footer = tk.Frame(self.root, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=1)
        footer.grid_columnconfigure(2, weight=1)
        footer.grid_columnconfigure(3, weight=0)
        self.footer = footer

        tk.Label(
            footer,
            textvariable=self.footer_project_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
            padx=12,
            pady=7,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            footer,
            textvariable=self.footer_result_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
            padx=12,
            pady=7,
        ).grid(row=0, column=1, sticky="ew")
        tk.Label(
            footer,
            textvariable=self.footer_storage_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
            padx=12,
            pady=7,
        ).grid(row=0, column=2, sticky="ew")

        log_badge = tk.Label(
            footer,
            textvariable=self.left_log_state_var,
            bg=PRIMARY_SOFT,
            fg=PRIMARY,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=10,
            pady=5,
        )
        log_badge.grid(row=0, column=3, sticky="e", padx=12, pady=5)
        self.status_badges.append((log_badge, "log"))

    def _build_hidden_support_widgets(self) -> None:
        hidden = tk.Frame(self.root, bg=WINDOW_BG)
        self.left_result_entry = tk.Entry(hidden, textvariable=self.left_result_var)
        self.left_log_state_entry = tk.Entry(hidden, textvariable=self.left_log_state_var)
        self.log_text = scrolledtext.ScrolledText(
            hidden,
            wrap="word",
            height=12,
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            bd=0,
        )
        self.log_text.configure(state="disabled")

    def _page_card(self, parent: tk.Widget, *, bg: str = CARD_BG) -> tk.Frame:
        return tk.Frame(parent, bg=bg, highlightbackground=BORDER, highlightthickness=1)

    def _page_hero(self, parent: tk.Widget, page_id: str) -> tk.Frame:
        hero = self._page_card(parent)
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)
        title_var = self.workspace_header_title_var if page_id == "detect" else None
        desc_var = self.workspace_header_desc_var if page_id == "detect" else None

        left = tk.Frame(hero, bg=CARD_BG)
        left.grid(row=0, column=0, sticky="ew", padx=16, pady=14)
        left.grid_columnconfigure(0, weight=1)

        if title_var is None:
            title_widget = tk.Label(
                left,
                text=PAGE_TITLES[page_id],
                bg=CARD_BG,
                fg=TEXT,
                font=("Microsoft YaHei UI", 18, "bold"),
                anchor="w",
            )
        else:
            title_widget = tk.Label(
                left,
                textvariable=title_var,
                bg=CARD_BG,
                fg=TEXT,
                font=("Microsoft YaHei UI", 18, "bold"),
                anchor="w",
            )
        title_widget.grid(row=0, column=0, sticky="w")

        if desc_var is None:
            desc_widget = tk.Label(
                left,
                text=PAGE_SUBTITLES[page_id],
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 10),
                justify="left",
                wraplength=760,
                anchor="w",
            )
        else:
            desc_widget = tk.Label(
                left,
                textvariable=desc_var,
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 10),
                justify="left",
                wraplength=720,
                anchor="w",
            )
        desc_widget.grid(row=1, column=0, sticky="w", pady=(6, 0))

        right = tk.Frame(hero, bg=CARD_BG)
        right.grid(row=0, column=1, sticky="e", padx=16, pady=14)

        if page_id == "detect":
            tk.Label(
                right,
                textvariable=self.workspace_header_context_var,
                bg=PRIMARY_SOFT,
                fg=TEXT,
                font=("Microsoft YaHei UI", 9),
                justify="left",
                wraplength=280,
                padx=12,
                pady=8,
            ).pack(side="right", padx=(12, 0))
        else:
            tk.Label(
                right,
                text=self._hero_badge_text(page_id),
                bg=PRIMARY_SOFT,
                fg=PRIMARY_DARK,
                font=("Microsoft YaHei UI", 9, "bold"),
                padx=10,
                pady=6,
            ).pack(side="right")
        return hero

    def _hero_badge_text(self, page_id: str) -> str:
        if page_id == "train":
            return "训练与验证一体化"
        if page_id == "export":
            return "多格式部署导出"
        if page_id == "home":
            return "全链路总览"
        return "独立工作区"

    def _primary_button(self, parent: tk.Widget, text: str, command: object) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
            cursor="hand2",
        )

    def _secondary_button(self, parent: tk.Widget, text: str, command: object) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=PRIMARY_DARK,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            padx=14,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
            cursor="hand2",
        )

    def _section_card(self, parent: tk.Widget, *, title: str) -> "SectionCard":
        frame = self._page_card(parent)
        frame.grid_columnconfigure(0, weight=1)
        header = tk.Frame(frame, bg=CARD_SOFT, height=38)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        tk.Label(
            header,
            text=title,
            bg=CARD_SOFT,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
            padx=14,
        ).pack(fill="both", expand=True)

        content = tk.Frame(frame, bg=CARD_BG)
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        content.grid_columnconfigure(0, weight=1)
        return SectionCard(frame=frame, content=content)

    def _metric_card(self, parent: tk.Widget, column: int, title: str, value_var: tk.StringVar) -> None:
        card = self._page_card(parent)
        card.grid(row=0, column=column, sticky="ew", padx=(0, 12) if column < 3 else 0)
        tk.Label(
            card,
            text=title,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 5))
        tk.Label(
            card,
            textvariable=value_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            justify="left",
            wraplength=200,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 12))

    def _home_module_card(self, parent: tk.Widget, row: int, column: int, page_id: str) -> None:
        card = self._page_card(parent)
        padx = (0, 12) if column < 3 else 0
        card.grid(row=row, column=column, sticky="nsew", padx=padx, pady=(0, 12))
        parent.grid_rowconfigure(row, weight=1)
        card.grid_columnconfigure(0, weight=1)

        top = tk.Frame(card, bg=CARD_BG)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        top.grid_columnconfigure(1, weight=1)
        tk.Label(
            top,
            text=HOME_CARD_BADGES[page_id],
            bg=PRIMARY_SOFT,
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 11, "bold"),
            width=3,
            pady=4,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            top,
            text=PAGE_TITLES[page_id],
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        tk.Label(
            card,
            text=HOME_CARD_DESCRIPTIONS[page_id],
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            justify="left",
            wraplength=250,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16)
        tk.Label(
            card,
            textvariable=self.module_meta_vars[page_id],
            bg=PRIMARY_SOFT,
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            justify="left",
            wraplength=250,
            anchor="w",
            padx=12,
            pady=10,
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(14, 14))

        actions = tk.Frame(card, bg=CARD_BG)
        actions.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self._primary_button(actions, "进入", lambda item=page_id: self.show_page(item)).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._secondary_button(actions, "结果", self.open_result_location).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _info_label(self, parent: tk.Widget, text: str, *, pady: tuple[int, int] = (0, 6)) -> None:
        tk.Label(
            parent,
            text=text,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=pady)

    def _info_value(
        self,
        parent: tk.Widget,
        variable: tk.StringVar,
        *,
        wraplength: int = 320,
        fg: str = TEXT_MUTED,
        pady: tuple[int, int] = (0, 0),
        font_size: int = 9,
        bold: bool = False,
    ) -> None:
        tk.Label(
            parent,
            textvariable=variable,
            bg=CARD_BG,
            fg=fg,
            justify="left",
            wraplength=wraplength,
            font=("Microsoft YaHei UI", font_size, "bold" if bold else "normal"),
            anchor="w",
        ).pack(fill="x", pady=pady)

    def _raise_page(self, frame_id: str) -> None:
        frame = self.page_frames.get(frame_id)
        if frame is not None:
            frame.tkraise()

    def _update_nav_state(self) -> None:
        active_page = self.active_page_var.get()
        for page_id, button in self.nav_buttons.items():
            is_active = page_id == active_page
            button.configure(
                bg=PRIMARY_SOFT if is_active else CARD_BG,
                fg=PRIMARY_DARK if is_active else TEXT,
                highlightbackground=PRIMARY if is_active else BORDER,
            )
        for page_id, button in self.top_nav_buttons.items():
            is_active = page_id == active_page
            button.configure(
                bg=PRIMARY_SOFT if is_active else CARD_BG,
                fg=PRIMARY_DARK if is_active else TEXT_MUTED,
            )
        self.topbar_page_var.set(PAGE_TITLES.get(active_page, PAGE_TITLES["home"]))

    def _current_window_size(self) -> tuple[int, int]:
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width > 1 and height > 1:
            return width, height
        geometry_token = self.root.winfo_geometry().split("+", 1)[0]
        if "x" in geometry_token:
            width_text, height_text = geometry_token.split("x", 1)
            try:
                return int(width_text), int(height_text)
            except ValueError:
                pass
        return max(self.root.winfo_reqwidth(), 1), max(self.root.winfo_reqheight(), 1)

    def _is_compact_layout(self) -> bool:
        width, height = self._current_window_size()
        return legacy_app.is_compact_window(width, height)

    def _is_low_height_layout(self) -> bool:
        return self._current_window_size()[1] <= legacy_app.COMPACT_LAYOUT_HEIGHT

    def _handle_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root or not self._v2_ready:
            return
        self._schedule_responsive_layout()

    def _schedule_responsive_layout(self) -> None:
        if self._responsive_after_id is not None:
            try:
                self.root.after_cancel(self._responsive_after_id)
            except tk.TclError:
                pass
        self._responsive_after_id = self.root.after_idle(self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._responsive_after_id = None
        if not self._v2_ready:
            return
        self._apply_page_chrome()

    def _apply_page_chrome(self) -> None:
        active_page = self.active_page_var.get()
        immersive = active_page in IMMERSIVE_WORKSPACE_PAGES
        workspace_page = active_page in WORKSPACE_PAGE_IDS.values()
        compact = self._is_compact_layout()
        low_height = self._is_low_height_layout()

        if active_page == "home":
            self.topbar_home_button.pack_forget()
            quick_action_anchor = self.topbar_status_badge
        else:
            self.topbar_home_button.pack(side="left", padx=(0, 8), before=self.topbar_status_badge)
            quick_action_anchor = self.topbar_home_button

        if immersive or compact:
            self.sidebar.grid_remove()
            self.page_host.grid_configure(column=0, columnspan=2)
        else:
            self.sidebar.grid()
            self.page_host.grid_configure(column=1, columnspan=1)

        if immersive or low_height:
            self.footer.grid_remove()
        else:
            self.footer.grid()

        if immersive or compact:
            self.topbar_subtitle_label.grid_remove()
        else:
            self.topbar_subtitle_label.grid()
        self.topbar_nav.grid()
        self.topbar_quick_actions.pack(side="left", before=quick_action_anchor)

        if workspace_page:
            self.workspace_hero.grid_remove()
            self.workspace_metrics_row.grid_remove()
            if immersive:
                self.workspace_action_row.grid_remove()
            else:
                self.workspace_action_row.grid()
        else:
            self.workspace_hero.grid()
            self.workspace_metrics_row.grid()
            self.workspace_action_row.grid_remove()

    def _workspace_current_count(self, workspace_id: str) -> str:
        if self.annotation_editor is None:
            return "0"
        if workspace_id == "detect":
            return str(self.annotation_editor.detect_editor.current_image_count())
        if workspace_id == "segment":
            return str(self.annotation_editor.segment_editor.current_image_count())
        return "0"

    def _refresh_workspace_header(self) -> None:
        if self.annotation_editor is None:
            return
        workspace_id = getattr(self.annotation_editor, "_active_workspace", "detect")
        page_id = WORKSPACE_PAGE_IDS.get(workspace_id, "detect")
        project_dir = self.annotation_editor.project_dir
        current_name = self.annotation_editor.current_image_name()
        preview = self.annotation_editor.export_preview_dir()
        self.workspace_header_title_var.set(PAGE_TITLES.get(page_id, PAGE_TITLES["detect"]))
        self.workspace_header_desc_var.set(PAGE_SUBTITLES.get(page_id, WORKSPACE_DESCRIPTIONS.get(workspace_id, "")))
        project_text = self._short_path(project_dir) if project_dir is not None else "未选择"
        content_text = current_name if current_name != "未选择" else preview
        self.workspace_header_context_var.set(f"目录：{project_text}\n内容：{content_text}")
        self.workspace_metric_vars["project"].set(project_text)
        self.workspace_metric_vars["focus"].set(content_text)
        self.workspace_metric_vars["count"].set(self._workspace_current_count(workspace_id))
        self.workspace_metric_vars["next"].set(preview)

    def _refresh_home_dashboard(self) -> None:
        if self.annotation_editor is not None:
            workspace_label = self.annotation_editor.active_workspace_label()
            project_dir = self.annotation_editor.project_dir
            self.home_workspace_var.set(workspace_label)
            self.home_source_var.set(self._short_path(project_dir) if project_dir is not None else "未选择")
            current_name = self.annotation_editor.current_image_name()
            preview = self.annotation_editor.export_preview_dir()
            self.home_preview_var.set(current_name if current_name != "未选择" else preview)
        else:
            self.home_workspace_var.set("未进入工作区")
            self.home_source_var.set("未选择")
            self.home_preview_var.set("未生成")

        self.home_status_var.set(self.process_status_var.get() or "等待开始")
        self.home_recent_var.set(
            self.left_result_var.get().strip()
            or self.process_status_var.get().strip()
            or "最近还没有新的工作记录。"
        )

        if self.annotation_editor is None:
            return

        detect_editor = self.annotation_editor.detect_editor
        segment_editor = self.annotation_editor.segment_editor
        video_panel = self.annotation_editor.video_panel
        organizer_panel = self.annotation_editor.organizer_panel
        ai_panel = self.annotation_editor.ai_platform_panel
        model_panel = self.annotation_editor.model_hub_panel

        self.module_meta_vars["detect"].set(
            f"{detect_editor.current_image_count()} 张图片 · {detect_editor.current_image_name()}"
        )
        self.module_meta_vars["segment"].set(
            f"{segment_editor.current_image_count()} 张图片 · {segment_editor.current_image_name()}"
        )
        self.module_meta_vars["video"].set(video_panel.output_preview_dir())
        self.module_meta_vars["organize"].set(organizer_panel.output_preview_dir())
        self.module_meta_vars["ai"].set(ai_panel.active_capability_label())
        self.module_meta_vars["model_hub"].set(model_panel.output_preview_dir())
        self.module_meta_vars["train"].set(
            f"{self.selected_train_action_label_var.get()} · {self.process_status_var.get()}"
        )
        export_target = self.export_output_dir_var.get().strip() or self._expected_export_target()
        export_label = self.export_format_label_var.get().strip() or "自动识别"
        self.module_meta_vars["export"].set(f"{export_label} · {export_target}")

    def _refresh_train_export_panels(self) -> None:
        action_label = self.selected_train_action_label_var.get().strip() or "训练"
        self.train_overview_vars["action"].set(action_label)
        if self.train_action_var.get() == "train":
            self.train_overview_vars["data"].set(self._short_path_text(self.train_data_var.get()))
            self.train_overview_vars["model"].set(self._short_path_text(self.train_model_var.get()))
            self.train_overview_vars["output"].set(self._expected_train_output_dir())
        elif self.train_action_var.get() == "val":
            self.train_overview_vars["data"].set(self._short_path_text(self.val_data_var.get()))
            self.train_overview_vars["model"].set(self._short_path_text(self.val_weights_var.get()))
            self.train_overview_vars["output"].set(self._expected_val_output_dir())
        elif self.train_action_var.get() == "predict":
            self.train_overview_vars["data"].set(self._short_path_text(self.predict_source_var.get()))
            self.train_overview_vars["model"].set(self._short_path_text(self.predict_weights_var.get()))
            self.train_overview_vars["output"].set(self._expected_predict_output_dir())
        else:
            self.train_overview_vars["data"].set(self._short_path_text(self.track_source_var.get()))
            self.train_overview_vars["model"].set(self._short_path_text(self.track_weights_var.get()))
            self.train_overview_vars["output"].set(self._expected_track_output_dir())

        self.train_side_status_var.set(self.process_status_var.get() or "等待开始")
        self.train_side_detail_var.set(
            f"日志状态：{self.left_log_state_var.get() or '暂无'}\n"
            f"最近结果：{self._short_path_text(self.result_location_var.get())}"
        )
        self.train_log_preview_var.set(self._current_log_preview())

        self.export_overview_vars["weights"].set(self._short_path_text(self.export_weights_var.get()))
        self.export_overview_vars["format"].set(self.export_format_label_var.get().strip() or "自动识别")
        self.export_overview_vars["target"].set(self._short_path_text(self._expected_export_target()))
        self.export_overview_vars["status"].set(self.process_status_var.get() or "等待开始")
        self.export_side_detail_var.set(
            f"任务类型：{self.export_task_label_var.get() or '自动识别'}\n"
            f"结果位置：{self._short_path_text(self.result_location_var.get())}"
        )
        self.export_log_preview_var.set(self._current_log_preview())

    def _refresh_footer(self) -> None:
        if self.annotation_editor is not None and self.active_tab.get() == "annotation":
            project_dir = self.annotation_editor.project_dir
            self.footer_project_var.set(f"当前路径：{self._short_path(project_dir) if project_dir else '未选择'}")
        else:
            self.footer_project_var.set(f"当前路径：{self._short_path_text(self.summary_value2.get())}")

        result_path = self.result_location_var.get().strip() or self.summary_value3.get() or "未生成"
        self.footer_result_var.set(f"结果位置：{self._short_path_text(result_path)}")
        self.footer_storage_var.set(self._storage_summary())

    def _short_path(self, path: Path | None) -> str:
        if path is None:
            return "未选择"
        return self._short_path_text(str(path))

    def _short_path_text(self, text: str) -> str:
        value = text.strip()
        if not value:
            return "未选择"
        if len(value) <= 48:
            return value
        return f"{value[:20]} ... {value[-22:]}"

    def _storage_summary(self) -> str:
        try:
            usage = shutil.disk_usage(str(legacy_app.WORK_DIR))
        except OSError:
            return "存储使用：读取失败"
        used_gb = (usage.total - usage.free) / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        return f"存储使用：{used_gb:.1f} GB / {total_gb:.1f} GB"

    def _current_log_preview(self) -> str:
        if not hasattr(self, "log_text"):
            return "日志预览会在这里更新。"
        try:
            raw = self.log_text.get("1.0", "end-1c").strip()
        except tk.TclError:
            raw = ""
        if raw:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            return "\n".join(lines[-6:])
        if self.current_log_path and self.current_log_path.exists():
            try:
                lines = self.current_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            cleaned = [line.strip() for line in lines if line.strip()]
            if cleaned:
                return "\n".join(cleaned[-6:])
        return "日志预览会在这里更新。"

    def _continue_current_work(self) -> None:
        if self.active_tab.get() == "export":
            self.show_page("export")
            return
        if self.active_tab.get() == "train":
            self.show_page("train")
            return
        target_page = WORKSPACE_PAGE_IDS.get(getattr(self.annotation_editor, "_active_workspace", "detect"), "detect")
        self.show_page(target_page)

    def _handle_embedded_workspace_change(self, workspace_id: str) -> None:
        self.annotation_workspace_page_id = WORKSPACE_PAGE_IDS.get(workspace_id, "detect")
        if not self._v2_ready:
            return
        self.active_page_var.set(self.annotation_workspace_page_id)
        self._raise_page("workspace")
        self._update_nav_state()
        self._refresh_workspace_header()
        self._refresh_home_dashboard()

    def show_page(self, page_id: str) -> None:
        if page_id == "home":
            self.active_page_var.set("home")
            self._raise_page("home")
            self._update_nav_state()
            self._apply_responsive_layout()
            self._refresh_summary()
            return
        if page_id == "train":
            self._show_tab("train")
            return
        if page_id == "export":
            self._show_tab("export")
            return

        workspace_id = next((key for key, value in WORKSPACE_PAGE_IDS.items() if value == page_id), "detect")
        self.annotation_workspace_page_id = page_id
        self.active_page_var.set(page_id)
        if self.annotation_editor is not None:
            self.annotation_editor.show_workspace(workspace_id)
        self._show_tab("annotation")

    def _show_tab(self, tab: str) -> None:
        legacy_app.ToolTip.hide_all()
        legacy_app.SmartComboBox.close_all()
        self.active_tab.set(tab)

        if tab == "annotation":
            if self.annotation_editor is not None:
                current_workspace = getattr(self.annotation_editor, "_active_workspace", "detect")
                self.annotation_workspace_page_id = WORKSPACE_PAGE_IDS.get(current_workspace, "detect")
            self.active_page_var.set(self.annotation_workspace_page_id)
            self._raise_page("workspace")
            self._refresh_workspace_header()
        elif tab == "export":
            self.active_page_var.set("export")
            self._raise_page("export")
            self._reset_scroll_to_top(self.export_scroll)
        else:
            self.active_page_var.set("train")
            self._raise_page("train")
            self._reset_scroll_to_top(self.train_scroll)

        if not self.process and self.left_log_state_var.get() in {"暂无", "已完成", "已结束", "已取消"}:
            placeholder_mode = "annotate" if tab == "annotation" else ("export" if tab == "export" else self.train_action_var.get())
            self._set_log_placeholder(placeholder_mode)

        self._update_nav_state()
        self._apply_responsive_layout()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        super()._refresh_summary()
        self._refresh_workspace_header()
        self._refresh_home_dashboard()
        self._refresh_train_export_panels()
        self._refresh_footer()

    def _refresh_status_visuals(self) -> None:
        super()._refresh_status_visuals()
        process_color = self._status_color(self.process_status_var.get())
        log_color = self._status_color(self.left_log_state_var.get())
        for badge, badge_kind in self.status_badges:
            color = process_color if badge_kind == "process" else log_color
            try:
                badge.configure(fg=color)
            except tk.TclError:
                pass


class SectionCard:
    def __init__(self, *, frame: tk.Frame, content: tk.Frame) -> None:
        self.frame = frame
        self.content = content

    def grid(self, **kwargs: object) -> None:
        self.frame.grid(**kwargs)


def main() -> None:
    root = legacy_app.create_app_root()
    legacy_app.apply_window_icon(root)
    AppV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
