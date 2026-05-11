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
RESPONSIVE_LAYOUT_DELAY_MS = 180
V2_METRIC_CARD_HEIGHT = 98
V2_METRIC_VALUE_WRAP = 280
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
    "home": "\u603b\u89c8\u5de5\u4f5c\u53f0",
    "detect": "\u68c0\u6d4b\u6846\u6807\u6ce8",
    "segment": "\u5b9e\u4f8b\u5206\u5272",
    "video": "\u89c6\u9891\u6807\u6ce8",
    "organize": "\u6570\u636e\u96c6\u6574\u7406",
    "ai": "AI\u5de5\u4f5c\u6d41",
    "model_hub": "\u6a21\u578b\u4e2d\u5fc3",
    "train": "\u8bad\u7ec3\u5de5\u4f5c\u53f0",
    "export": "\u5bfc\u51fa\u5de5\u4f5c\u53f0",
}

PAGE_SUBTITLES = {
    "home": "\u628a\u6807\u6ce8\u3001\u6574\u7406\u3001\u8bad\u7ec3\u548c\u5bfc\u51fa\u4e32\u6210\u4e00\u6761\u987a\u6ed1\u5de5\u4f5c\u6d41\u3002",
    "detect": "\u5355\u72ec\u6253\u5f00\u68c0\u6d4b\u6807\u6ce8\u5de5\u4f5c\u533a\uff0c\u9002\u5408\u9010\u5f20\u4fee\u6846\u548c\u56de\u586b\u81ea\u52a8\u6807\u6ce8\u7ed3\u679c\u3002",
    "segment": "\u5355\u72ec\u6253\u5f00\u5206\u5272\u5de5\u4f5c\u533a\uff0c\u4e13\u6ce8\u63a9\u7801\u3001\u591a\u8fb9\u5f62\u548c\u7c7b\u522b\u6574\u7406\u3002",
    "video": "\u4ece\u89c6\u9891\u62bd\u5e27\u5e76\u5feb\u901f\u9001\u5f80\u68c0\u6d4b\u6216\u5206\u5272\u5de5\u4f5c\u533a\u7ee7\u7eed\u6807\u6ce8\u3002",
    "organize": "\u6574\u7406\u539f\u59cb\u6807\u6ce8\uff0c\u751f\u6210\u53ef\u76f4\u63a5\u8bad\u7ec3\u7684\u6807\u51c6\u6570\u636e\u96c6\u3002",
    "ai": "\u4e32\u8054\u5927\u56fe\u5207\u7247\u3001\u63d0\u793a\u8f85\u52a9\u548c\u81ea\u52a8\u6807\u6ce8\u80fd\u529b\u3002",
    "model_hub": "\u67e5\u770b\u672c\u5730\u6a21\u578b\u548c\u5b98\u65b9\u63a8\u8350\uff0c\u5e76\u4e00\u952e\u5e94\u7528\u3002",
    "train": "\u56f4\u7ed5\u6570\u636e\u96c6\u3001\u6a21\u578b\u3001\u9884\u8bbe\u548c\u65e5\u5fd7\u7684\u5b8c\u6574\u8bad\u7ec3\u5165\u53e3\u3002",
    "export": "\u5c06\u8bad\u7ec3\u7ed3\u679c\u5bfc\u51fa\u4e3a\u90e8\u7f72\u6240\u9700\u683c\u5f0f\uff0c\u5e76\u4fdd\u7559\u9a8c\u8bc1\u4fe1\u606f\u3002",
}

NAV_ITEMS = (
    ("home", "\u603b\u89c8"),
    ("detect", "\u68c0\u6d4b\u6807\u6ce8"),
    ("segment", "\u5b9e\u4f8b\u5206\u5272"),
    ("video", "\u89c6\u9891\u6807\u6ce8"),
    ("organize", "\u6570\u636e\u6574\u7406"),
    ("ai", "AI\u5de5\u4f5c\u6d41"),
    ("model_hub", "\u6a21\u578b\u4e2d\u5fc3"),
    ("train", "\u8bad\u7ec3\u5de5\u4f5c\u53f0"),
    ("export", "\u5bfc\u51fa\u5de5\u4f5c\u53f0"),
)
NAV_ITEM_LABELS = dict(NAV_ITEMS)
NAV_COMPACT_LABELS = {
    "home": "总览",
    "detect": "检测",
    "segment": "分割",
    "video": "视频",
    "organize": "整理",
    "ai": "AI",
    "model_hub": "模型",
    "train": "训练",
    "export": "导出",
}

HOME_CARD_DESCRIPTIONS = {
    "detect": "\u6253\u5f00\u56fe\u7247\u76ee\u5f55\uff0c\u4fee\u6846\u3001\u6539\u7c7b\u5e76\u8c03\u7528\u81ea\u52a8\u6807\u6ce8\u3002",
    "segment": "\u9488\u5bf9\u5b9e\u4f8b\u63a9\u7801\u505a\u8f6e\u5ed3\u7f16\u8f91\u4e0e\u7c7b\u522b\u6574\u7406\u3002",
    "video": "\u5148\u62bd\u5e27\uff0c\u518d\u9001\u5165\u68c0\u6d4b\u6216\u5206\u5272\u5de5\u4f5c\u533a\u7ee7\u7eed\u6807\u6ce8\u3002",
    "organize": "\u8bc6\u522b\u6807\u6ce8\u683c\u5f0f\u5e76\u751f\u6210 dataset.yaml \u4e0e\u6807\u51c6\u76ee\u5f55\u3002",
    "ai": "\u628a\u5207\u7247\u3001\u63d0\u793a\u8f85\u52a9\u4e0e\u81ea\u52a8\u6807\u6ce8\u4e32\u6210\u6d41\u7a0b\u3002",
    "model_hub": "\u67e5\u770b\u672c\u5730\u6a21\u578b\u548c\u5b98\u65b9\u63a8\u8350\uff0c\u5e76\u4e00\u952e\u5e94\u7528\u3002",
    "train": "\u914d\u7f6e\u8bad\u7ec3\u3001\u9a8c\u8bc1\u3001\u9884\u6d4b\u3001\u8ddf\u8e2a\u4e0e\u9884\u8bbe\u3002",
    "export": "\u5bfc\u51fa ONNX\u3001TensorRT \u7b49\u90e8\u7f72\u683c\u5f0f\u3002",
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
    _SIDE_LOG_CARD_MIN_HEIGHT = 168

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
        self.workspace_header_context_var = tk.StringVar(value="项目目录：未选择\n当前内容：未选择")
        self.footer_project_var = tk.StringVar(value="项目：未选择")
        self.footer_result_var = tk.StringVar(value="结果：暂无")
        self.footer_storage_var = tk.StringVar(value="存储：读取中")
        self.home_workspace_var = tk.StringVar(value="未进入工作区")
        self.home_source_var = tk.StringVar(value="未选择")
        self.home_preview_var = tk.StringVar(value="暂无")
        self.home_status_var = tk.StringVar(value="等待开始")
        self.home_recent_var = tk.StringVar(value="暂无最近结果")
        self.workspace_metric_vars = {
            "project": tk.StringVar(value="未选择"),
            "focus": tk.StringVar(value="未选择"),
            "count": tk.StringVar(value="0"),
            "next": tk.StringVar(value="待处理"),
        }
        self.train_overview_vars = {
            "action": tk.StringVar(value="训练"),
            "data": tk.StringVar(value="未选择"),
            "model": tk.StringVar(value="未选择"),
            "output": tk.StringVar(value="未选择"),
        }
        self.train_side_status_var = tk.StringVar(value="等待开始")
        self.train_side_detail_var = tk.StringVar(value="等待训练任务启动")
        self.train_log_preview_var = tk.StringVar(value="暂无日志内容")
        self.export_overview_vars = {
            "weights": tk.StringVar(value="未选择"),
            "format": tk.StringVar(value="自动识别"),
            "target": tk.StringVar(value="未选择"),
            "status": tk.StringVar(value="等待开始"),
        }
        self.export_side_detail_var = tk.StringVar(value="等待导出任务启动")
        self.export_log_preview_var = tk.StringVar(value="暂无日志内容")
        self.module_meta_vars = {page_id: tk.StringVar(value="待配置") for page_id in PAGE_ORDER}
        self.top_nav_buttons: dict[str, tk.Button] = {}
        self.topbar_quick_buttons: dict[str, tk.Button] = {}
        self.home_module_grid: tk.Frame | None = None
        self.home_module_cards: dict[str, tk.Frame] = {}
        self.operation_pages: dict[str, dict[str, object]] = {}
        self.page_frames: dict[str, tk.Frame] = {}
        self.status_badges: list[tuple[tk.Label, str]] = []
        self.annotation_workspace_page_id = "detect"
        self._responsive_after_id: str | None = None
        self._last_responsive_size: tuple[int, int] | None = None
        self._last_layout_bucket: tuple[object, ...] | None = None
        self._v2_filmstrip_widgets: list[tk.Widget] = []
        self._annotation_workspace_initialized = False
        self.annotation_workspace_host: tk.Frame | None = None

        self._build_topbar()

        body = tk.Frame(self.root, bg=WINDOW_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.main_body = body

        self._build_sidebar(body)

        self.page_host = tk.Frame(body, bg=WINDOW_BG)
        self.page_host.grid(row=0, column=0, sticky="nsew")
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
        brand.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))
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
        center.grid(row=0, column=1, sticky="ew", padx=8, pady=(10, 6))
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
            text="标注、整理、训练、导出一体化 V2 工作台",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 5))
        self.topbar_subtitle_label = center.grid_slaves(row=1, column=0)[0]

        nav = tk.Frame(topbar, bg=CARD_BG)
        nav.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 10))
        self.topbar_nav = nav
        for index, (page_id, label) in enumerate(NAV_ITEMS):
            nav.grid_columnconfigure(index, weight=1)
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
            button.grid(row=0, column=index, sticky="ew", padx=(0, 4) if index < len(NAV_ITEMS) - 1 else 0)
            legacy_app.ToolTip(button, f"{PAGE_TITLES[page_id]}\n{PAGE_SUBTITLES[page_id]}")
            self.top_nav_buttons[page_id] = button

        actions = tk.Frame(topbar, bg=CARD_BG)
        actions.grid(row=0, column=2, sticky="e", padx=12, pady=(10, 6))
        self.topbar_actions = actions
        quick_actions = tk.Frame(actions, bg=CARD_BG)
        quick_actions.pack(side="left")
        self.topbar_quick_actions = quick_actions
        continue_button = self._secondary_button(quick_actions, "继续当前工作", self._continue_current_work)
        continue_button.pack(side="left", padx=(0, 8))
        result_button = self._secondary_button(quick_actions, "打开结果", self.open_result_location)
        result_button.pack(side="left", padx=(0, 8))
        log_button = self._secondary_button(quick_actions, "打开日志", self.open_log_file)
        log_button.pack(side="left", padx=(0, 8))
        self.topbar_quick_buttons.update(
            {
                "continue": continue_button,
                "result": result_button,
                "log": log_button,
            }
        )
        legacy_app.ToolTip(continue_button, "回到当前正在处理的工作区或任务页面。")
        legacy_app.ToolTip(result_button, "打开最近一次任务生成的结果目录。")
        legacy_app.ToolTip(log_button, "打开当前任务对应的日志文件。")
        self.topbar_home_button = self._secondary_button(actions, "返回总览", lambda: self.show_page("home"))
        legacy_app.ToolTip(self.topbar_home_button, "回到 V2 总览工作台。")
        self.topbar_home_spacer = tk.Frame(actions, bg=CARD_BG, width=1, height=1)
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
        # V2 uses the top bar as the only page navigation surface.
        self.sidebar = tk.Frame(parent, bg=WINDOW_BG, width=1, height=1)

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
        self.home_module_grid = module_grid
        for column in range(4):
            module_grid.grid_columnconfigure(column, weight=1)

        for index, page_id in enumerate(PAGE_ORDER):
            row = index // 4
            column = index % 4
            self._home_module_card(module_grid, row, column, page_id)

        recent = self._section_card(content, title="最近动态")
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
        self._metric_card(metrics, 3, "下一步", self.workspace_metric_vars["next"])

        studio_shell = tk.Frame(page, bg=WINDOW_BG)
        studio_shell.grid(row=2, column=0, sticky="nsew")
        studio_shell.grid_rowconfigure(1, weight=1)
        studio_shell.grid_columnconfigure(0, weight=1)

        action_row = tk.Frame(studio_shell, bg=WINDOW_BG)
        action_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.workspace_action_row = action_row
        self._secondary_button(action_row, "打开结果", self.open_result_location).pack(side="left", padx=(0, 8))
        self._secondary_button(action_row, "打开日志", self.open_log_file).pack(side="left", padx=(0, 8))
        self._secondary_button(action_row, "前往整理", lambda: self.show_page("organize")).pack(side="left", padx=(0, 8))
        self._primary_button(action_row, "转到训练", lambda: self.show_page("train")).pack(side="left")

        studio_host = tk.Frame(studio_shell, bg=WINDOW_BG)
        studio_host.grid(row=1, column=0, sticky="nsew")
        studio_host.grid_rowconfigure(0, weight=1)
        studio_host.grid_columnconfigure(0, weight=1)
        self.annotation_workspace_host = studio_host

        self._ensure_annotation_workspace()

    def _ensure_annotation_workspace(self) -> None:
        if self._annotation_workspace_initialized:
            return
        host = self.annotation_workspace_host
        if host is None:
            return
        self.annotation_editor = AnnotationStudio(
            host,
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
        self._annotation_workspace_initialized = True

    def _build_train_page(self) -> None:
        self.page_frames["train"] = self._build_operation_page(
            page_id="train",
            metric_specs=(
                ("\u8fd0\u884c\u6a21\u5f0f", self.train_overview_vars["action"]),
                ("\u8bad\u7ec3\u6570\u636e", self.train_overview_vars["data"]),
                ("\u6a21\u578b", self.train_overview_vars["model"]),
                ("\u8f93\u51fa\u9884\u89c8", self.train_overview_vars["output"]),
            ),
            build_main_view=self._build_train_view,
            status_var=self.train_side_status_var,
            detail_var=self.train_side_detail_var,
            log_var=self.train_log_preview_var,
            action_specs=(
                ("\u6253\u5f00\u7ed3\u679c", self.open_result_location),
                ("\u6253\u5f00\u65e5\u5fd7", self.open_log_file),
                ("\u8f6c\u5230\u5bfc\u51fa", lambda: self.show_page("export")),
                ("\u56de\u5230\u6570\u636e\u6574\u7406", lambda: self.show_page("organize")),
            ),
            scroll_attr_name="train_scroll",
            side_scroll_attr_name="train_side_scroll",
        )

    def _build_export_page(self) -> None:
        self.page_frames["export"] = self._build_operation_page(
            page_id="export",
            metric_specs=(
                ("\u6743\u91cd\u6587\u4ef6", self.export_overview_vars["weights"]),
                ("\u5bfc\u51fa\u683c\u5f0f", self.export_overview_vars["format"]),
                ("\u8f93\u51fa\u76ee\u5f55", self.export_overview_vars["target"]),
                ("\u4efb\u52a1\u72b6\u6001", self.export_overview_vars["status"]),
            ),
            build_main_view=self._build_export_view,
            status_var=self.export_overview_vars["status"],
            detail_var=self.export_side_detail_var,
            log_var=self.export_log_preview_var,
            action_specs=(
                ("\u6253\u5f00\u7ed3\u679c", self.open_result_location),
                ("\u6253\u5f00\u65e5\u5fd7", self.open_log_file),
                ("\u5207\u56de\u8bad\u7ec3", lambda: self.show_page("train")),
                ("\u56de\u5230\u6a21\u578b\u4e2d\u5fc3", lambda: self.show_page("model_hub")),
            ),
            scroll_attr_name="export_scroll",
            side_scroll_attr_name="export_side_scroll",
        )

    def _build_operation_page(
        self,
        *,
        page_id: str,
        metric_specs: tuple[tuple[str, tk.StringVar], ...],
        build_main_view: object,
        status_var: tk.StringVar,
        detail_var: tk.StringVar,
        log_var: tk.StringVar,
        action_specs: tuple[tuple[str, object], ...],
        scroll_attr_name: str,
        side_scroll_attr_name: str,
    ) -> tk.Frame:
        page = tk.Frame(self.page_host, bg=WINDOW_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=7)
        page.grid_columnconfigure(1, weight=3)

        hero = self._page_hero(page, page_id)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        stats = tk.Frame(page, bg=WINDOW_BG)
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(len(metric_specs)):
            stats.grid_columnconfigure(column, weight=1)
        for column, (title, value_var) in enumerate(metric_specs):
            self._metric_card(stats, column, title, value_var)

        left = tk.Frame(page, bg=WINDOW_BG)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        main_scroll = legacy_app.ScrollableFrame(left, background=WINDOW_BG)
        main_scroll.grid(row=0, column=0, sticky="nsew")
        setattr(self, scroll_attr_name, main_scroll)
        build_main_view(main_scroll.inner)

        right = tk.Frame(page, bg=WINDOW_BG)
        right.grid(row=2, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        side_scroll = legacy_app.ScrollableFrame(right, background=WINDOW_BG)
        side_scroll.grid(row=0, column=0, sticky="nsew")
        setattr(self, side_scroll_attr_name, side_scroll)
        side = side_scroll.inner
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(2, weight=1)

        self.operation_pages[page_id] = {
            "page": page,
            "left": left,
            "right": right,
            "main_scroll": main_scroll,
            "side_scroll": side_scroll,
        }

        run_card = self._section_card(side, title="\u8fd0\u884c\u72b6\u6001")
        run_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._info_value(run_card.content, status_var, fg=PRIMARY, font_size=12, bold=True, pady=(0, 8))
        self._info_value(run_card.content, detail_var, wraplength=320)

        action_card = self._section_card(side, title="\u5feb\u6377\u64cd\u4f5c")
        action_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for index, (label, command) in enumerate(action_specs):
            self._secondary_button(action_card.content, label, command).pack(fill="x", pady=(0, 8) if index < len(action_specs) - 1 else (0, 0))

        log_card = self._section_card(side, title="\u65e5\u5fd7\u9884\u89c8")
        log_card.grid(row=2, column=0, sticky="nsew")
        log_card.frame.configure(height=self._SIDE_LOG_CARD_MIN_HEIGHT)
        log_card.frame.grid_propagate(False)
        log_card.content.grid_rowconfigure(0, weight=1)
        self._info_value(log_card.content, log_var, wraplength=320, expand=True)
        return page

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
            return "部署导出工作流"
        if page_id == "home":
            return "全链路总览"
        return "工作流模块"

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
        frame.grid_rowconfigure(1, weight=1)
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
        card.configure(height=V2_METRIC_CARD_HEIGHT)
        card.grid_propagate(False)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 12) if column < 3 else 0)
        parent.grid_rowconfigure(0, weight=1, minsize=V2_METRIC_CARD_HEIGHT)
        metric_cards = getattr(parent, "_v2_metric_cards", None)
        if metric_cards is None:
            metric_cards = []
            setattr(parent, "_v2_metric_cards", metric_cards)
        metric_cards.append(card)
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
            wraplength=V2_METRIC_VALUE_WRAP,
            anchor="w",
        ).pack(fill="both", expand=True, padx=14, pady=(0, 12))

    def _home_module_card(self, parent: tk.Widget, row: int, column: int, page_id: str) -> None:
        card = self._page_card(parent)
        padx = (0, 12) if column < 3 else 0
        card.grid(row=row, column=column, sticky="nsew", padx=padx, pady=(0, 12))
        self.home_module_cards[page_id] = card
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
        expand: bool = False,
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
        ).pack(fill="both" if expand else "x", expand=expand, pady=pady)

    def _sync_live_log_preview(self) -> None:
        if not hasattr(self, "train_log_preview_var"):
            return
        self._refresh_train_export_panels()

    def _append_log_batch(self, lines: list[str]) -> None:
        super()._append_log_batch(lines)
        self._sync_live_log_preview()

    def _clear_log_text(self) -> None:
        super()._clear_log_text()
        self._sync_live_log_preview()

    def _drain_logs(self) -> None:
        super()._drain_logs()
        self._sync_live_log_preview()

    def _raise_page(self, frame_id: str) -> None:
        for item, frame in self.page_frames.items():
            if item == frame_id:
                frame.grid()
                frame.tkraise()
            else:
                frame.grid_remove()

    def _update_nav_state(self) -> None:
        active_page = self.active_page_var.get()
        for page_id, button in self.top_nav_buttons.items():
            is_active = page_id == active_page
            button.configure(
                bg=PRIMARY if is_active else CARD_BG,
                fg="white" if is_active else TEXT_MUTED,
                activebackground=PRIMARY_DARK if is_active else PRIMARY_SOFT,
                activeforeground="white" if is_active else PRIMARY_DARK,
                highlightthickness=1,
                highlightbackground=PRIMARY if is_active else BORDER,
                relief="flat",
            )
        self._set_string_var_if_changed(self.topbar_page_var, PAGE_TITLES.get(active_page, PAGE_TITLES["home"]))

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
        size = (int(getattr(event, "width", 0) or 0), int(getattr(event, "height", 0) or 0))
        if size[0] <= 1 or size[1] <= 1:
            return
        if self._last_responsive_size == size:
            return
        self._last_responsive_size = size
        self._schedule_responsive_layout()

    def _schedule_responsive_layout(self) -> None:
        if self._responsive_after_id is not None:
            try:
                self.root.after_cancel(self._responsive_after_id)
            except tk.TclError:
                pass
        self._responsive_after_id = self.root.after(RESPONSIVE_LAYOUT_DELAY_MS, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._responsive_after_id = None
        if not self._v2_ready:
            return
        self._apply_page_chrome()

    def _forget_geometry(self, widget: tk.Widget) -> None:
        manager = widget.winfo_manager()
        if manager == "pack":
            widget.pack_forget()
        elif manager == "grid":
            widget.grid_forget()

    def _apply_topbar_responsive_layout(self, *, show_home_button: bool, compact: bool) -> None:
        width, _ = self._current_window_size()
        actions_below_nav = compact or width <= 1500
        dense_actions = compact or width <= 1320
        dense_nav = compact or width <= 1280
        short_nav_labels = compact or width <= 1380

        nav_font_size = 8 if dense_nav else 9
        nav_padx = 8 if dense_nav else 10
        nav_pady = 3 if dense_nav else 4
        for page_id, button in self.top_nav_buttons.items():
            button.configure(
                text=NAV_COMPACT_LABELS[page_id] if short_nav_labels else NAV_ITEM_LABELS[page_id],
                font=("Microsoft YaHei UI", nav_font_size, "bold"),
                padx=nav_padx,
                pady=nav_pady,
            )

        action_font_size = 9 if dense_actions else 10
        action_padx = 10 if dense_actions else 14
        action_pady = 7 if dense_actions else 8
        for button in self.topbar_quick_actions.winfo_children():
            if isinstance(button, tk.Button):
                button.configure(font=("Microsoft YaHei UI", action_font_size, "bold"), padx=action_padx, pady=action_pady)
        if dense_actions:
            self.topbar_quick_buttons["continue"].configure(text="继续")
            self.topbar_quick_buttons["result"].configure(text="结果")
            self.topbar_quick_buttons["log"].configure(text="日志")
        else:
            self.topbar_quick_buttons["continue"].configure(text="继续当前工作")
            self.topbar_quick_buttons["result"].configure(text="打开结果")
            self.topbar_quick_buttons["log"].configure(text="打开日志")
        self.topbar_home_button.configure(font=("Microsoft YaHei UI", action_font_size, "bold"), padx=action_padx, pady=action_pady)
        self.topbar_status_badge.configure(
            font=("Microsoft YaHei UI", action_font_size, "bold"),
            padx=10 if dense_actions else 12,
            pady=6 if dense_actions else 7,
        )

        for widget in (self.topbar_quick_actions, self.topbar_home_button, self.topbar_home_spacer, self.topbar_status_badge):
            self._forget_geometry(widget)

        if actions_below_nav:
            self.topbar_actions.grid_configure(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 8 if compact else 10))
            self.topbar_actions.grid_columnconfigure(0, weight=1)
            self.topbar_actions.grid_columnconfigure(1, weight=0)
            self.topbar_actions.grid_columnconfigure(2, weight=0)
            self.topbar_actions.grid_columnconfigure(3, weight=0)
            self.topbar_quick_actions.grid(row=0, column=1, sticky="e")
            if show_home_button:
                self.topbar_home_button.grid(row=0, column=2, sticky="e", padx=(8, 8))
            else:
                self.topbar_home_spacer.configure(
                    width=max(1, self.topbar_home_button.winfo_reqwidth()),
                    height=max(1, self.topbar_home_button.winfo_reqheight()),
                )
                self.topbar_home_spacer.grid(row=0, column=2, sticky="e", padx=(8, 8))
            self.topbar_status_badge.grid(row=0, column=3, sticky="e")
        else:
            self.topbar_actions.grid_configure(row=0, column=2, columnspan=1, sticky="e", padx=12, pady=(10, 6))
            self.topbar_actions.grid_columnconfigure(0, weight=0)
            self.topbar_actions.grid_columnconfigure(1, weight=0)
            self.topbar_actions.grid_columnconfigure(2, weight=0)
            self.topbar_quick_actions.grid(row=0, column=0, sticky="e", padx=(0, 8))
            if show_home_button:
                self.topbar_home_button.grid(row=0, column=1, sticky="e", padx=(0, 8))
                self.topbar_status_badge.grid(row=0, column=2, sticky="e")
            else:
                self.topbar_status_badge.grid(row=0, column=1, sticky="e")

    def _apply_home_responsive_layout(self, *, compact: bool) -> None:
        if self.home_module_grid is None or not self.home_module_cards:
            return
        width, _ = self._current_window_size()
        columns = 2 if compact or width <= 1360 else 4
        for column in range(4):
            self.home_module_grid.grid_columnconfigure(column, weight=1 if column < columns else 0)
        for row in range(4):
            self.home_module_grid.grid_rowconfigure(row, weight=1 if row < ((len(PAGE_ORDER) + columns - 1) // columns) else 0)
        for index, page_id in enumerate(PAGE_ORDER):
            card = self.home_module_cards[page_id]
            row = index // columns
            column = index % columns
            padx = (0, 12) if column < columns - 1 else 0
            card.grid_configure(row=row, column=column, padx=padx, pady=(0, 12))

    def _apply_operation_page_layout(self, *, page_id: str, compact: bool) -> None:
        layout = self.operation_pages.get(page_id)
        if not layout:
            return
        page = layout["page"]
        left = layout["left"]
        right = layout["right"]
        main_scroll = layout["main_scroll"]
        side_scroll = layout["side_scroll"]
        if not isinstance(page, tk.Frame) or not isinstance(left, tk.Frame) or not isinstance(right, tk.Frame):
            return
        if compact:
            page.grid_columnconfigure(0, weight=1)
            page.grid_columnconfigure(1, weight=0)
            page.grid_rowconfigure(2, weight=3)
            page.grid_rowconfigure(3, weight=2)
            left.grid_configure(row=2, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 10))
            right.grid_configure(row=3, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 0))
        else:
            page.grid_columnconfigure(0, weight=7)
            page.grid_columnconfigure(1, weight=3)
            page.grid_rowconfigure(2, weight=1)
            page.grid_rowconfigure(3, weight=0)
            left.grid_configure(row=2, column=0, columnspan=1, sticky="nsew", padx=(0, 12), pady=0)
            right.grid_configure(row=2, column=1, columnspan=1, sticky="nsew", padx=0, pady=0)
        if isinstance(main_scroll, legacy_app.ScrollableFrame):
            main_scroll.grid(row=0, column=0, sticky="nsew")
        if isinstance(side_scroll, legacy_app.ScrollableFrame):
            side_scroll.grid(row=0, column=0, sticky="nsew")

    def _apply_page_chrome(self) -> None:
        active_page = self.active_page_var.get()
        immersive = active_page in IMMERSIVE_WORKSPACE_PAGES
        workspace_page = active_page in WORKSPACE_PAGE_IDS.values()
        width, height = self._current_window_size()
        compact = legacy_app.is_compact_window(width, height)
        low_height = height <= legacy_app.COMPACT_LAYOUT_HEIGHT
        show_home_button = active_page != "home"
        layout_bucket = (
            active_page,
            compact,
            low_height,
            width <= 1500,
            width <= 1380,
            width <= 1360,
            width <= 1320,
            width <= 1280,
        )
        if self._last_layout_bucket == layout_bucket:
            return
        self._last_layout_bucket = layout_bucket

        self.page_host.grid_configure(column=0, columnspan=1)

        if immersive or low_height:
            self.footer.grid_remove()
        else:
            self.footer.grid()

        if compact:
            self.topbar_subtitle_label.grid_remove()
        else:
            self.topbar_subtitle_label.grid()
        self.topbar_nav.grid()
        self._apply_topbar_responsive_layout(show_home_button=show_home_button, compact=compact)
        self._apply_home_responsive_layout(compact=compact)

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
        self._apply_workspace_density(active_page=active_page, low_height=low_height)
        if active_page in {"train", "export"}:
            self._apply_operation_page_layout(page_id=active_page, compact=compact or low_height)

    def _apply_workspace_density(self, *, active_page: str, low_height: bool) -> None:
        if self.annotation_editor is None:
            return
        if not self._v2_filmstrip_widgets:
            for editor in (self.annotation_editor.detect_editor, self.annotation_editor.segment_editor):
                filmstrip = getattr(editor, "v2_filmstrip", None)
                if filmstrip is not None:
                    self._v2_filmstrip_widgets.append(filmstrip)
        hide_filmstrip = low_height and active_page in {"detect", "segment"}
        for filmstrip in self._v2_filmstrip_widgets:
            if hide_filmstrip:
                filmstrip.grid_remove()
            else:
                filmstrip.grid()

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
        self._set_string_var_if_changed(self.workspace_header_title_var, PAGE_TITLES.get(page_id, PAGE_TITLES["detect"]))
        self._set_string_var_if_changed(
            self.workspace_header_desc_var,
            PAGE_SUBTITLES.get(page_id, WORKSPACE_DESCRIPTIONS.get(workspace_id, "")),
        )
        project_text = self._short_path(project_dir) if project_dir is not None else "未选择"
        content_text = current_name if current_name != "未选择" else preview
        self._set_string_var_if_changed(self.workspace_header_context_var, f"项目目录：{project_text}\n当前内容：{content_text}")
        self._set_string_var_if_changed(self.workspace_metric_vars["project"], project_text)
        self._set_string_var_if_changed(self.workspace_metric_vars["focus"], content_text)
        self._set_string_var_if_changed(self.workspace_metric_vars["count"], self._workspace_current_count(workspace_id))
        self._set_string_var_if_changed(self.workspace_metric_vars["next"], preview)

    def _refresh_home_dashboard(self) -> None:
        if self.annotation_editor is not None:
            workspace_label = self.annotation_editor.active_workspace_label()
            project_dir = self.annotation_editor.project_dir
            self._set_string_var_if_changed(self.home_workspace_var, workspace_label)
            self._set_string_var_if_changed(self.home_source_var, self._short_path(project_dir) if project_dir is not None else "未选择")
            current_name = self.annotation_editor.current_image_name()
            preview = self.annotation_editor.export_preview_dir()
            self._set_string_var_if_changed(self.home_preview_var, current_name if current_name != "未选择" else preview)
        else:
            self._set_string_var_if_changed(self.home_workspace_var, "未进入工作区")
            self._set_string_var_if_changed(self.home_source_var, "未选择")
            self._set_string_var_if_changed(self.home_preview_var, "暂无")

        self._set_string_var_if_changed(self.home_status_var, self.process_status_var.get() or "等待开始")
        self._set_string_var_if_changed(
            self.home_recent_var,
            self.left_result_var.get().strip()
            or self.process_status_var.get().strip()
            or "暂无最近结果",
        )

        if self.annotation_editor is None:
            return

        detect_editor = self.annotation_editor.detect_editor
        segment_editor = self.annotation_editor.segment_editor
        video_panel = self.annotation_editor.video_panel
        organizer_panel = self.annotation_editor.organizer_panel
        ai_panel = self.annotation_editor.ai_platform_panel
        model_panel = self.annotation_editor.model_hub_panel

        self._set_string_var_if_changed(
            self.module_meta_vars["detect"],
            f"{detect_editor.current_image_count()} 张图片 · {detect_editor.current_image_name()}",
        )
        self._set_string_var_if_changed(
            self.module_meta_vars["segment"],
            f"{segment_editor.current_image_count()} 张图片 · {segment_editor.current_image_name()}",
        )
        self._set_string_var_if_changed(self.module_meta_vars["video"], video_panel.output_preview_dir())
        self._set_string_var_if_changed(self.module_meta_vars["organize"], organizer_panel.output_preview_dir())
        self._set_string_var_if_changed(self.module_meta_vars["ai"], ai_panel.active_capability_label())
        self._set_string_var_if_changed(self.module_meta_vars["model_hub"], model_panel.output_preview_dir())
        self._set_string_var_if_changed(
            self.module_meta_vars["train"],
            f"{self.selected_train_action_label_var.get()} - {self.process_status_var.get()}",
        )
        export_target = self.export_output_dir_var.get().strip() or self._expected_export_target()
        export_label = self.export_format_label_var.get().strip() or "自动识别"
        self._set_string_var_if_changed(self.module_meta_vars["export"], f"{export_label} - {export_target}")

    def _refresh_train_export_panels(self) -> None:
        action_label = self.selected_train_action_label_var.get().strip() or "训练"
        self._set_string_var_if_changed(self.train_overview_vars["action"], action_label)
        if self.train_action_var.get() == "train":
            self._set_string_var_if_changed(self.train_overview_vars["data"], self._short_path_text(self.train_data_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["model"], self._short_path_text(self.train_model_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["output"], self._expected_train_output_dir())
        elif self.train_action_var.get() == "val":
            self._set_string_var_if_changed(self.train_overview_vars["data"], self._short_path_text(self.val_data_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["model"], self._short_path_text(self.val_weights_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["output"], self._expected_val_output_dir())
        elif self.train_action_var.get() == "predict":
            self._set_string_var_if_changed(self.train_overview_vars["data"], self._short_path_text(self.predict_source_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["model"], self._short_path_text(self.predict_weights_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["output"], self._expected_predict_output_dir())
        else:
            self._set_string_var_if_changed(self.train_overview_vars["data"], self._short_path_text(self.track_source_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["model"], self._short_path_text(self.track_weights_var.get()))
            self._set_string_var_if_changed(self.train_overview_vars["output"], self._expected_track_output_dir())

        self._set_string_var_if_changed(self.train_side_status_var, self.process_status_var.get() or "等待开始")
        self._set_string_var_if_changed(
            self.train_side_detail_var,
            f"日志状态：{self.left_log_state_var.get() or '暂无'}\n最近结果：{self._short_path_text(self.result_location_var.get())}",
        )
        log_preview = self._current_log_preview()
        self._set_string_var_if_changed(self.train_log_preview_var, log_preview)

        self._set_string_var_if_changed(self.export_overview_vars["weights"], self._short_path_text(self.export_weights_var.get()))
        self._set_string_var_if_changed(self.export_overview_vars["format"], self.export_format_label_var.get().strip() or "自动识别")
        self._set_string_var_if_changed(self.export_overview_vars["target"], self._short_path_text(self._expected_export_target()))
        self._set_string_var_if_changed(self.export_overview_vars["status"], self.process_status_var.get() or "等待开始")
        self._set_string_var_if_changed(
            self.export_side_detail_var,
            f"导出任务：{self.export_task_label_var.get() or '未选择'}\n输出位置：{self._short_path_text(self.result_location_var.get())}",
        )
        self._set_string_var_if_changed(self.export_log_preview_var, log_preview)

    def _refresh_footer(self) -> None:
        if self.annotation_editor is not None and self.active_tab.get() == "annotation":
            project_dir = self.annotation_editor.project_dir
            self._set_string_var_if_changed(self.footer_project_var, f"项目：{self._short_path(project_dir) if project_dir else '未选择'}")
        else:
            self._set_string_var_if_changed(self.footer_project_var, f"项目：{self._short_path_text(self.summary_value2.get())}")

        result_path = self.result_location_var.get().strip() or self.summary_value3.get() or "暂无"
        self._set_string_var_if_changed(self.footer_result_var, f"结果：{self._short_path_text(result_path)}")
        self._set_string_var_if_changed(self.footer_storage_var, self._storage_summary())

    def _short_path(self, path: Path | None) -> str:
        if path is None:
            return "未选择"
        return self._short_path_text(str(path))

    def _short_path_text(self, text: str) -> str:
        value = text.strip()
        if not value:
            return "未选择"
        if len(value) <= 42:
            return value
        return f"{value[:16]} ... {value[-18:]}"

    def _storage_summary(self) -> str:
        try:
            usage = shutil.disk_usage(str(legacy_app.WORK_DIR))
        except OSError:
            return "存储：读取失败"
        used_gb = (usage.total - usage.free) / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        return f"存储：已用 {used_gb:.1f} GB / {total_gb:.1f} GB"

    def _current_log_preview(self) -> str:
        if not hasattr(self, "log_text"):
            return "暂无日志内容"
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
        return "暂无日志内容"

    def _continue_current_work(self) -> None:
        if self.active_tab.get() == "export":
            self.show_page("export")
            return
        if self.active_tab.get() == "train":
            self.show_page("train")
            return
        self._ensure_annotation_workspace()
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
        self._ensure_annotation_workspace()
        if self.annotation_editor is not None:
            self.annotation_editor.show_workspace(workspace_id, notify=False)
        self._show_tab("annotation")

    def _show_tab(self, tab: str) -> None:
        legacy_app.ToolTip.hide_all()
        legacy_app.SmartComboBox.close_all()
        self.active_tab.set(tab)

        if tab == "annotation":
            self._ensure_annotation_workspace()
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

    def _on_close(self) -> None:
        if self._responsive_after_id is not None:
            try:
                self.root.after_cancel(self._responsive_after_id)
            except tk.TclError:
                pass
            self._responsive_after_id = None
        super()._on_close()

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
    if len(legacy_app.sys.argv) >= 3 and legacy_app.sys.argv[1] == legacy_app.BACKEND_LAUNCH_FLAG:
        raise SystemExit(legacy_app.run_backend_command_from_launcher(legacy_app.sys.argv[2:]))

    root = legacy_app.create_app_root()
    legacy_app.apply_window_icon(root)
    AppV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
