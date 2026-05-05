from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from annotation_support import (
    AnnotationBox,
    classes_path_for_folder,
    ensure_class_names,
    label_path_for_image,
    list_project_annotation_images,
    load_class_names,
    load_project_session,
    load_yolo_boxes,
    parse_class_names_text,
    save_class_names,
    save_project_session,
    save_yolo_boxes,
)
from annotation_ui_support import VerticalScrolledFrame


CARD_BG = "#ffffff"
CARD_SOFT = "#fbfdff"
PANEL_BG = "#f7fbff"
PRIMARY = "#5b8cff"
PRIMARY_DARK = "#4f7de6"
PRIMARY_SOFT = "#edf3ff"
BORDER = "#dce7fb"
TEXT = "#243348"
TEXT_MUTED = "#6c7c96"
SUCCESS = "#31c48d"
BOX_COLORS = (
    "#ff6b6b",
    "#4ecdc4",
    "#ffd166",
    "#6c5ce7",
    "#00b894",
    "#fd79a8",
    "#0984e3",
    "#e17055",
)


class DetectionAnnotationEditor(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        session_path: Path,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_export_request: Callable[[Path, list[str]], None],
        on_auto_label_request: Callable[[Path, Path | None, str, dict[str, object], list[str]], None],
        on_switch_to_train: Callable[[], None],
        ui_mode: str = "classic",
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.session_path = session_path
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_export_request = on_export_request
        self.on_auto_label_request = on_auto_label_request
        self.on_switch_to_train = on_switch_to_train
        self.ui_mode = ui_mode

        self.project_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index = 0
        self.class_names: list[str] = ["class0"]
        self.boxes: list[AnnotationBox] = []
        self.selected_box_index: int | None = None
        self.current_image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.drag_mode: str | None = None
        self.drag_anchor: tuple[float, float] | None = None
        self.drag_box_snapshot: AnnotationBox | None = None
        self.drag_box_index: int | None = None
        self.drag_handle: str | None = None
        self.preview_box: AnnotationBox | None = None

        self.project_var = tk.StringVar(value="")
        self.current_image_var = tk.StringVar(value="未选择")
        self.image_counter_var = tk.StringVar(value="0 / 0")
        self.selected_class_var = tk.StringVar(value="class0")
        self.auto_model_var = tk.StringVar(value="yolo11n.pt")
        self.auto_conf_var = tk.StringVar(value="0.25")
        self.auto_iou_var = tk.StringVar(value="0.70")
        self.auto_imgsz_var = tk.StringVar(value="640")
        self.auto_device_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="请选择要标注的图片文件夹")
        self.draw_mode_var = tk.BooleanVar(value=True)
        self.mode_hint_var = tk.StringVar(value="绘制框")
        self.selection_title_var = tk.StringVar(value="未选中目标框")
        self.selection_meta_var = tk.StringVar(value="在画布中选择一个框以查看类别与坐标。")
        self.thumbnail_summary_var = tk.StringVar(value="全部图片 0  当前 0 / 0")
        self.thumbnail_refs: list[ImageTk.PhotoImage] = []
        self.mode_hint_var.set("绘制框")
        self.selection_title_var.set("未选中目标框")
        self.selection_meta_var.set("在画布中选择一个框以查看类别与坐标。")
        self.thumbnail_summary_var.set("全部图片 0  当前 0 / 0")

        self._build_ui()

    def _build_ui(self) -> None:
        if self.ui_mode == "v2":
            self._build_v2_ui_modern()
            return
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(1, weight=1)

        tk.Label(header, text="图片目录", bg=PANEL_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        tk.Entry(
            header,
            textvariable=self.project_var,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            readonlybackground=CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(12, 6), ipady=6)
        self._small_button(header, "选择", self.pick_project_dir).grid(row=0, column=2, padx=(0, 8), pady=(12, 6))
        self._small_button(header, "刷新", self.reload_project).grid(row=0, column=3, padx=(0, 12), pady=(12, 6))

        tk.Label(header, text="当前图片", bg=PANEL_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 12))
        image_entry = tk.Entry(
            header,
            textvariable=self.current_image_var,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            readonlybackground=CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        )
        image_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 12), ipady=6)
        self._small_button(header, "上一张", self.prev_image).grid(row=1, column=2, padx=(0, 8), pady=(0, 12))
        self._small_button(header, "下一张", self.next_image).grid(row=1, column=3, padx=(0, 8), pady=(0, 12))
        tk.Label(header, textvariable=self.image_counter_var, bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 10, "bold")).grid(row=1, column=4, sticky="e", padx=(0, 12), pady=(0, 12))

        body = tk.Frame(self, bg=CARD_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        canvas_panel = tk.Frame(body, bg=CARD_SOFT, highlightbackground=BORDER, highlightthickness=1)
        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_panel.grid_columnconfigure(0, weight=1)
        canvas_panel.grid_rowconfigure(1, weight=1)

        canvas_toolbar = tk.Frame(canvas_panel, bg=CARD_SOFT)
        canvas_toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        tk.Checkbutton(
            canvas_toolbar,
            text="绘制新框",
            variable=self.draw_mode_var,
            bg=CARD_SOFT,
            fg=TEXT,
            activebackground=CARD_SOFT,
            activeforeground=TEXT,
            selectcolor=CARD_BG,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        self._small_button(canvas_toolbar, "保存当前标注", self.save_current_annotations).pack(side="left", padx=(10, 0))
        self._small_button(canvas_toolbar, "删除选中框", self.delete_selected_box).pack(side="left", padx=(10, 0))
        tk.Label(canvas_toolbar, textvariable=self.status_var, bg=CARD_SOFT, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="right")

        self.canvas = tk.Canvas(
            canvas_panel,
            bg="#f4f8ff",
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_box())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_box())

        self.sidebar_scroll = VerticalScrolledFrame(body, bg=CARD_BG)
        self.sidebar_scroll.grid(row=0, column=1, sticky="nsew")
        side = self.sidebar_scroll.content
        side.grid_columnconfigure(0, weight=1)

        self.classes_box = self._side_box(side, "类别与项目")
        classes_box = self.classes_box
        classes_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.class_text = tk.Text(
            classes_box,
            height=4,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        self.class_text.pack(fill="x", padx=12, pady=(12, 8))
        self.class_text.insert("1.0", "class0")
        class_actions = tk.Frame(classes_box, bg=CARD_BG)
        class_actions.pack(fill="x", padx=12, pady=(0, 12))
        self._small_button(class_actions, "保存类别", self.apply_class_names).pack(side="left")
        self._small_button(class_actions, "切到训练", self.on_switch_to_train).pack(side="right")

        self.box_box = self._side_box(side, "当前框")
        box_box = self.box_box
        box_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.box_list = tk.Listbox(
            box_box,
            height=6,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            selectbackground=PRIMARY_SOFT,
            selectforeground=TEXT,
            exportselection=False,
            font=("Microsoft YaHei UI", 10),
        )
        self.box_list.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        self.box_list.bind("<<ListboxSelect>>", self.on_box_selected)
        class_pick_row = tk.Frame(box_box, bg=CARD_BG)
        class_pick_row.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(class_pick_row, text="框类别", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.selected_class_combo = ttk.Combobox(
            class_pick_row,
            textvariable=self.selected_class_var,
            state="readonly",
            values=self.class_names,
            font=("Microsoft YaHei UI", 10),
        )
        self.selected_class_combo.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.selected_class_combo.bind("<<ComboboxSelected>>", self.on_selected_class_changed, add="+")

        self.auto_box = self._side_box(side, "自动标注")
        auto_box = self.auto_box
        auto_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        auto_grid = tk.Frame(auto_box, bg=CARD_BG)
        auto_grid.pack(fill="x")
        self._labeled_entry(auto_grid, "模型", self.auto_model_var, row=0, readonly=False)
        browse_row = tk.Frame(auto_grid, bg=CARD_BG)
        browse_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        browse_row.grid_columnconfigure(0, weight=1)
        self._small_button(browse_row, "浏览本地权重", self.pick_auto_model).grid(row=0, column=0, sticky="ew")
        self._small_button(browse_row, "使用官方模型名", self.use_default_auto_model).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._labeled_entry(auto_grid, "conf", self.auto_conf_var, row=2, readonly=False)
        self._labeled_entry(auto_grid, "iou", self.auto_iou_var, row=3, readonly=False)
        self._labeled_entry(auto_grid, "imgsz", self.auto_imgsz_var, row=4, readonly=False)
        self._labeled_entry(auto_grid, "device", self.auto_device_var, row=5, readonly=False)
        auto_actions = tk.Frame(auto_grid, bg=CARD_BG)
        auto_actions.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        auto_actions.grid_columnconfigure(0, weight=1)
        auto_actions.grid_columnconfigure(1, weight=1)
        self._primary_button(auto_actions, "当前图自动标注", self.auto_label_current).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._primary_button(auto_actions, "全部图片自动标注", self.auto_label_all).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.export_box = self._side_box(side, "导出训练集")
        export_box = self.export_box
        export_box.grid(row=3, column=0, sticky="ew")
        tk.Label(
            export_box,
            text="会把当前图片目录里的图片 + 同名 YOLO TXT 标注整理成标准训练集，并自动回填到训练入口。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=280,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=(12, 8))
        self._primary_button(export_box, "整理为训练集", self.export_training_dataset).pack(fill="x", padx=12, pady=(0, 12))

    def _build_v2_ui_modern(self) -> None:
        self.configure(bg="#f5f8fe")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        header.grid_columnconfigure(1, weight=1)

        tk.Label(
            header,
            text="项目路径",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(12, 8), pady=(10, 6))
        tk.Entry(
            header,
            textvariable=self.project_var,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            readonlybackground="#f8fbff",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=1, sticky="ew", pady=(10, 6), ipady=5)
        self._small_button(header, "打开文件夹", self.pick_project_dir).grid(row=0, column=2, padx=8, pady=(10, 6))

        top_actions = tk.Frame(header, bg=CARD_BG)
        top_actions.grid(row=0, column=3, sticky="e", padx=(8, 12), pady=(10, 6))
        self._small_button(top_actions, "刷新", self.reload_project).pack(side="left", padx=(0, 6))
        self._small_button(top_actions, "上一张", self.prev_image).pack(side="left", padx=(0, 6))
        self._small_button(top_actions, "下一张", self.next_image).pack(side="left", padx=(0, 6))
        self._primary_button(top_actions, "保存标注", self.save_current_annotations).pack(side="left", padx=(0, 6))
        self._primary_button(top_actions, "AI辅助标注", self.auto_label_current).pack(side="left")

        tk.Label(
            header,
            text="当前图片",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=(12, 8), pady=(0, 10))
        tk.Entry(
            header,
            textvariable=self.current_image_var,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            readonlybackground="#f8fbff",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=1, sticky="ew", pady=(0, 10), ipady=5)

        image_meta = tk.Frame(header, bg=CARD_BG)
        image_meta.grid(row=1, column=2, columnspan=2, sticky="e", padx=(8, 12), pady=(0, 10))
        tk.Label(
            image_meta,
            textvariable=self.image_counter_var,
            bg="#f3f8ff",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=10,
            pady=5,
        ).pack(side="left", padx=(0, 8))
        tk.Label(
            image_meta,
            textvariable=self.mode_hint_var,
            bg="#edf6ff",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=10,
            pady=5,
        ).pack(side="left", padx=(0, 8))
        tk.Label(
            image_meta,
            text="自动保存已开启",
            bg="#ebfbf4",
            fg=SUCCESS,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=10,
            pady=5,
        ).pack(side="left")

        body = tk.Frame(self, bg="#f5f8fe")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=292)

        toolrail = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, width=118)
        toolrail.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        toolrail.grid_propagate(False)
        tk.Label(
            toolrail,
            text="工具栏",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 8))
        self.v2_draw_button = self._v2_tool_button(toolrail, "绘制框", lambda: self._set_draw_mode(True), selected=True)
        self.v2_draw_button.pack(fill="x", padx=8, pady=(0, 6))
        self.v2_edit_button = self._v2_tool_button(toolrail, "编辑框", lambda: self._set_draw_mode(False))
        self.v2_edit_button.pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "删除选中", self.delete_selected_box).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "上一张", self.prev_image).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "下一张", self.next_image).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "保存标注", self.save_current_annotations).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "导出训练集", self.export_training_dataset, primary=True).pack(fill="x", padx=8, pady=(6, 10))

        center = tk.Frame(body, bg="#f5f8fe")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        canvas_panel = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        canvas_panel.grid(row=0, column=0, sticky="nsew")
        canvas_panel.grid_rowconfigure(1, weight=1)
        canvas_panel.grid_columnconfigure(0, weight=1)

        canvas_toolbar = tk.Frame(canvas_panel, bg=CARD_BG)
        canvas_toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        tk.Label(
            canvas_toolbar,
            text="检测标注画布",
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(side="left")
        tk.Label(
            canvas_toolbar,
            textvariable=self.status_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(side="right")

        self.canvas = tk.Canvas(
            canvas_panel,
            bg="#edf4ff",
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_box())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_box())

        filmstrip = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        filmstrip.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        filmstrip.grid_columnconfigure(1, weight=1)

        tk.Label(
            filmstrip,
            textvariable=self.thumbnail_summary_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        tk.Label(
            filmstrip,
            text="点击缩略图可快速切换当前图片。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=1, sticky="e", padx=10, pady=(8, 4))

        self.thumb_canvas = tk.Canvas(
            filmstrip,
            bg=CARD_BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
            height=104,
        )
        self.thumb_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        thumb_scroll = tk.Scrollbar(
            filmstrip,
            orient="horizontal",
            command=self.thumb_canvas.xview,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        thumb_scroll.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self.thumb_canvas.configure(xscrollcommand=thumb_scroll.set)
        self.thumb_strip = tk.Frame(self.thumb_canvas, bg=CARD_BG)
        self.thumb_window_id = self.thumb_canvas.create_window((0, 0), window=self.thumb_strip, anchor="nw")
        self.thumb_strip.bind("<Configure>", self._sync_thumb_scrollregion)
        self.thumb_canvas.bind("<Configure>", self._sync_thumb_viewport)
        self._bind_thumbnail_mousewheel_target(self.thumb_canvas)
        self._bind_thumbnail_mousewheel_target(self.thumb_strip)

        self.sidebar_scroll = VerticalScrolledFrame(body, bg="#f5f8fe")
        self.sidebar_scroll.grid(row=0, column=2, sticky="nsew")
        side = self.sidebar_scroll.content
        side.configure(bg="#f5f8fe")
        side.grid_columnconfigure(0, weight=1)

        self.classes_box = self._v2_side_box(side, "类别列表", "维护类名，并查看当前图片中的实例统计。")
        self.classes_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.class_summary_frame = tk.Frame(self.classes_box, bg=CARD_BG)
        self.class_summary_frame.pack(fill="x", padx=12, pady=(12, 6))
        self.class_text = tk.Text(
            self.classes_box,
            height=5,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        self.class_text.pack(fill="x", padx=12, pady=(0, 8))
        self.class_text.insert("1.0", "class0")
        class_actions = tk.Frame(self.classes_box, bg=CARD_BG)
        class_actions.pack(fill="x", padx=12, pady=(0, 12))
        self._small_button(class_actions, "保存类别", self.apply_class_names).pack(side="left")
        self._small_button(class_actions, "转到训练", self.on_switch_to_train).pack(side="right")

        self.box_box = self._v2_side_box(side, "框列表", "选中一个框后，可查看类别、坐标和尺寸。")
        self.box_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            self.box_box,
            textvariable=self.selection_title_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(
            self.box_box,
            textvariable=self.selection_meta_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=276,
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 8))
        self.box_list = tk.Listbox(
            self.box_box,
            height=8,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            selectbackground=PRIMARY_SOFT,
            selectforeground=TEXT,
            exportselection=False,
            font=("Microsoft YaHei UI", 10),
        )
        self.box_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.box_list.bind("<<ListboxSelect>>", self.on_box_selected)
        class_pick_row = tk.Frame(self.box_box, bg=CARD_BG)
        class_pick_row.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(class_pick_row, text="类别", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.selected_class_combo = ttk.Combobox(
            class_pick_row,
            textvariable=self.selected_class_var,
            state="readonly",
            values=self.class_names,
            font=("Microsoft YaHei UI", 10),
        )
        self.selected_class_combo.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.selected_class_combo.bind("<<ComboboxSelected>>", self.on_selected_class_changed, add="+")

        self.auto_box = self._v2_side_box(side, "AI 自动标注", "可对当前图片或整个项目批量执行检测自动标注。")
        self.auto_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        auto_grid = tk.Frame(self.auto_box, bg=CARD_BG)
        auto_grid.pack(fill="x")
        self._labeled_entry(auto_grid, "模型", self.auto_model_var, row=0, readonly=False)
        browse_row = tk.Frame(auto_grid, bg=CARD_BG)
        browse_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        browse_row.grid_columnconfigure(0, weight=1)
        browse_row.grid_columnconfigure(1, weight=1)
        self._small_button(browse_row, "浏览权重", self.pick_auto_model).grid(row=0, column=0, sticky="ew")
        self._small_button(browse_row, "默认模型", self.use_default_auto_model).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._labeled_entry(auto_grid, "置信度", self.auto_conf_var, row=2, readonly=False)
        self._labeled_entry(auto_grid, "IoU", self.auto_iou_var, row=3, readonly=False)
        self._labeled_entry(auto_grid, "输入尺寸", self.auto_imgsz_var, row=4, readonly=False)
        self._labeled_entry(auto_grid, "设备", self.auto_device_var, row=5, readonly=False)
        auto_actions = tk.Frame(auto_grid, bg=CARD_BG)
        auto_actions.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        auto_actions.grid_columnconfigure(0, weight=1)
        auto_actions.grid_columnconfigure(1, weight=1)
        self._primary_button(auto_actions, "当前图片", self.auto_label_current).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._primary_button(auto_actions, "整个项目", self.auto_label_all).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.export_box = self._v2_side_box(side, "整理为训练集", "将图片、标签和类名整理为可直接训练的数据集结构。")
        self.export_box.grid(row=3, column=0, sticky="ew")
        tk.Label(
            self.export_box,
            text="会自动输出标准目录与 dataset.yaml，方便直接进入训练页继续训练。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=276,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=(12, 8))
        self._primary_button(self.export_box, "导出训练集", self.export_training_dataset).pack(fill="x", padx=12, pady=(0, 8))
        self._small_button(self.export_box, "切换到训练", self.on_switch_to_train).pack(fill="x", padx=12, pady=(0, 12))

        self._refresh_v2_tool_selection()

    def _set_v2_tool_button_state(self, button: tk.Button | None, *, selected: bool) -> None:
        if button is None:
            return
        button.configure(
            bg=PRIMARY_SOFT if selected else "#f7fbff",
            fg=PRIMARY_DARK if selected else TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=PRIMARY_DARK,
            highlightbackground=PRIMARY if selected else BORDER,
            font=("Microsoft YaHei UI", 10, "bold" if selected else "normal"),
        )

    def _refresh_v2_tool_selection(self) -> None:
        if self.ui_mode != "v2":
            return
        self._set_v2_tool_button_state(getattr(self, "v2_draw_button", None), selected=self.draw_mode_var.get())
        self._set_v2_tool_button_state(getattr(self, "v2_edit_button", None), selected=not self.draw_mode_var.get())

    def _bind_thumbnail_mousewheel_target(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_thumbnail_mousewheel, add="+")
        widget.bind("<Shift-MouseWheel>", self._on_thumbnail_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_thumbnail_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_thumbnail_mousewheel, add="+")

    def _on_thumbnail_mousewheel(self, event: tk.Event) -> str | None:
        if self.ui_mode != "v2" or not hasattr(self, "thumb_canvas"):
            return None
        if getattr(event, "num", None) == 4:
            delta = -3
        elif getattr(event, "num", None) == 5:
            delta = 3
        else:
            raw_delta = getattr(event, "delta", 0)
            if raw_delta == 0:
                return None
            delta = -3 if raw_delta > 0 else 3
        self.thumb_canvas.xview_scroll(delta, "units")
        return "break"

    def _build_v2_ui(self) -> None:
        self.configure(bg="#f5f8fe")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 10))
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(3, weight=1)

        tk.Label(
            header,
            text="椤圭洰璺緞",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(16, 10), pady=(14, 8))
        tk.Entry(
            header,
            textvariable=self.project_var,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            readonlybackground="#f8fbff",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=1, sticky="ew", pady=(14, 8), ipady=8)
        self._small_button(header, "閫夋嫨鏂囦欢澶?", self.pick_project_dir).grid(row=0, column=2, padx=10, pady=(14, 8))

        top_actions = tk.Frame(header, bg=CARD_BG)
        top_actions.grid(row=0, column=3, sticky="e", padx=(10, 16), pady=(14, 8))
        self._small_button(top_actions, "鍒锋柊", self.reload_project).pack(side="left", padx=(0, 8))
        self._small_button(top_actions, "涓婁竴寮?", self.prev_image).pack(side="left", padx=(0, 8))
        self._small_button(top_actions, "涓嬩竴寮?", self.next_image).pack(side="left", padx=(0, 8))
        self._primary_button(top_actions, "淇濆瓨鏍囨敞", self.save_current_annotations).pack(side="left", padx=(0, 8))
        self._primary_button(top_actions, "AI 鑷姩鏍囨敞", self.auto_label_current).pack(side="left")

        tk.Label(
            header,
            text="褰撳墠鍥剧墖",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=(16, 10), pady=(0, 14))
        tk.Entry(
            header,
            textvariable=self.current_image_var,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            readonlybackground="#f8fbff",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            state="readonly",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=1, sticky="ew", pady=(0, 14), ipady=8)

        image_meta = tk.Frame(header, bg=CARD_BG)
        image_meta.grid(row=1, column=2, columnspan=2, sticky="e", padx=(10, 16), pady=(0, 14))
        tk.Label(
            image_meta,
            textvariable=self.image_counter_var,
            bg="#f3f8ff",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=12,
            pady=8,
        ).pack(side="left", padx=(0, 10))
        tk.Label(
            image_meta,
            textvariable=self.mode_hint_var,
            bg="#edf6ff",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=12,
            pady=8,
        ).pack(side="left", padx=(0, 10))
        tk.Label(
            image_meta,
            text="鑷姩淇濆瓨宸插惎鐢?",
            bg="#ebfbf4",
            fg=SUCCESS,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=12,
            pady=8,
        ).pack(side="left")

        body = tk.Frame(self, bg="#f5f8fe")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=316)

        toolrail = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        toolrail.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        tk.Label(
            toolrail,
            text="宸ュ叿",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 12))
        self._v2_tool_button(toolrail, "妗嗛€夋ā寮?", lambda: self._set_draw_mode(True), selected=True).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "缂栬緫妯″紡", lambda: self._set_draw_mode(False)).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "鍒犻櫎閫変腑", self.delete_selected_box).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "涓婁竴寮?", self.prev_image).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "涓嬩竴寮?", self.next_image).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "淇濆瓨", self.save_current_annotations).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "鎵撳寘璁粌闆?", self.export_training_dataset, primary=True).pack(fill="x", padx=10, pady=(8, 14))

        center = tk.Frame(body, bg="#f5f8fe")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        canvas_panel = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        canvas_panel.grid(row=0, column=0, sticky="nsew")
        canvas_panel.grid_rowconfigure(1, weight=1)
        canvas_panel.grid_columnconfigure(0, weight=1)

        canvas_toolbar = tk.Frame(canvas_panel, bg=CARD_BG)
        canvas_toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        tk.Label(
            canvas_toolbar,
            text="妫€娴嬫爣娉ㄧ敾甯?",
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            canvas_toolbar,
            textvariable=self.status_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(side="right")

        self.canvas = tk.Canvas(
            canvas_panel,
            bg="#edf4ff",
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_box())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_box())

        filmstrip = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        filmstrip.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        filmstrip.grid_columnconfigure(1, weight=1)

        tk.Label(
            filmstrip,
            textvariable=self.thumbnail_summary_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(
            filmstrip,
            text="鍙偣鍑婚瑙堝浘蹇€熷垏鎹㈠綋鍓嶅浘鐗?",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=1, sticky="e", padx=14, pady=(12, 6))

        self.thumb_canvas = tk.Canvas(
            filmstrip,
            bg=CARD_BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
            height=138,
        )
        self.thumb_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        thumb_scroll = tk.Scrollbar(
            filmstrip,
            orient="horizontal",
            command=self.thumb_canvas.xview,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        thumb_scroll.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        self.thumb_canvas.configure(xscrollcommand=thumb_scroll.set)
        self.thumb_strip = tk.Frame(self.thumb_canvas, bg=CARD_BG)
        self.thumb_window_id = self.thumb_canvas.create_window((0, 0), window=self.thumb_strip, anchor="nw")
        self.thumb_strip.bind("<Configure>", self._sync_thumb_scrollregion)
        self.thumb_canvas.bind("<Configure>", self._sync_thumb_viewport)

        self.sidebar_scroll = VerticalScrolledFrame(body, bg="#f5f8fe")
        self.sidebar_scroll.grid(row=0, column=2, sticky="nsew")
        side = self.sidebar_scroll.content
        side.configure(bg="#f5f8fe")
        side.grid_columnconfigure(0, weight=1)

        self.classes_box = self._v2_side_box(side, "绫诲埆鍒楄〃", "鍙充晶鍙互鐩存帴缁存姢 classes.txt")
        self.classes_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.class_summary_frame = tk.Frame(self.classes_box, bg=CARD_BG)
        self.class_summary_frame.pack(fill="x", padx=12, pady=(12, 6))
        self.class_text = tk.Text(
            self.classes_box,
            height=5,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        self.class_text.pack(fill="x", padx=12, pady=(0, 8))
        self.class_text.insert("1.0", "class0")
        class_actions = tk.Frame(self.classes_box, bg=CARD_BG)
        class_actions.pack(fill="x", padx=12, pady=(0, 12))
        self._small_button(class_actions, "淇濆瓨绫诲埆", self.apply_class_names).pack(side="left")
        self._small_button(class_actions, "杞埌璁粌", self.on_switch_to_train).pack(side="right")

        self.box_box = self._v2_side_box(side, "Box List", "Select a box to review class and coordinates.")
        self.box_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            self.box_box,
            textvariable=self.selection_title_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(
            self.box_box,
            textvariable=self.selection_meta_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=276,
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 8))
        self.box_list = tk.Listbox(
            self.box_box,
            height=8,
            relief="flat",
            bd=0,
            bg="#f8fbff",
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            selectbackground=PRIMARY_SOFT,
            selectforeground=TEXT,
            exportselection=False,
            font=("Microsoft YaHei UI", 10),
        )
        self.box_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.box_list.bind("<<ListboxSelect>>", self.on_box_selected)
        class_pick_row = tk.Frame(self.box_box, bg=CARD_BG)
        class_pick_row.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(class_pick_row, text="绫诲埆", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.selected_class_combo = ttk.Combobox(
            class_pick_row,
            textvariable=self.selected_class_var,
            state="readonly",
            values=self.class_names,
            font=("Microsoft YaHei UI", 10),
        )
        self.selected_class_combo.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.selected_class_combo.bind("<<ComboboxSelected>>", self.on_selected_class_changed, add="+")

        self.auto_box = self._v2_side_box(side, "AI 鑷姩鏍囨敞", "鏃㈠彲瀵瑰崟寮犲浘锛屼篃鍙鏁翠釜鐩綍鎵归噺杩愯")
        self.auto_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        auto_grid = tk.Frame(self.auto_box, bg=CARD_BG)
        auto_grid.pack(fill="x")
        self._labeled_entry(auto_grid, "妯″瀷", self.auto_model_var, row=0, readonly=False)
        browse_row = tk.Frame(auto_grid, bg=CARD_BG)
        browse_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        browse_row.grid_columnconfigure(0, weight=1)
        browse_row.grid_columnconfigure(1, weight=1)
        self._small_button(browse_row, "娴忚鏈湴鏉冮噸", self.pick_auto_model).grid(row=0, column=0, sticky="ew")
        self._small_button(browse_row, "浣跨敤瀹樻柟妯″瀷", self.use_default_auto_model).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._labeled_entry(auto_grid, "conf", self.auto_conf_var, row=2, readonly=False)
        self._labeled_entry(auto_grid, "iou", self.auto_iou_var, row=3, readonly=False)
        self._labeled_entry(auto_grid, "imgsz", self.auto_imgsz_var, row=4, readonly=False)
        self._labeled_entry(auto_grid, "device", self.auto_device_var, row=5, readonly=False)
        auto_actions = tk.Frame(auto_grid, bg=CARD_BG)
        auto_actions.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        auto_actions.grid_columnconfigure(0, weight=1)
        auto_actions.grid_columnconfigure(1, weight=1)
        self._primary_button(auto_actions, "褰撳墠鍥剧墖", self.auto_label_current).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._primary_button(auto_actions, "鏁翠釜鐩綍", self.auto_label_all).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.export_box = self._v2_side_box(side, "鏁寸悊涓鸿缁冮泦", "浼氫繚鐣欏綋鍓嶇被鍒〃骞剁敓鎴?dataset.yaml")
        self.export_box.grid(row=3, column=0, sticky="ew")
        tk.Label(
            self.export_box,
            text="浠庡綋鍓嶆爣娉ㄩ」鐩洿鎺ョ敓鎴愬彲璁粌鏁版嵁闆嗭紝鍚屾椂鎶婄粨鏋滄帹閫佺粰璁粌椤点€?",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=276,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=(12, 8))
        self._primary_button(self.export_box, "鏁寸悊骞跺鍑?", self.export_training_dataset).pack(fill="x", padx=12, pady=(0, 8))
        self._small_button(self.export_box, "鍒囨崲鍒拌缁冨伐浣滃彴", self.on_switch_to_train).pack(fill="x", padx=12, pady=(0, 12))

        self._apply_v2_text_overrides()

    def _apply_v2_text_overrides(self) -> None:
        header, body = self.winfo_children()[:2]

        for child in header.winfo_children():
            if child.winfo_class() == "Label":
                info = child.grid_info()
                row = int(info.get("row", 0))
                column = int(info.get("column", 0))
                if row == 0 and column == 0:
                    child.configure(text="Project")
                elif row == 1 and column == 0:
                    child.configure(text="Current Image")

        header_children = header.winfo_children()
        open_folder_button = next((child for child in header_children if child.winfo_class() == "Button"), None)
        if open_folder_button is not None:
            open_folder_button.configure(text="Open Folder")

        top_actions = next((child for child in header_children if child.winfo_class() == "Frame"), None)
        if top_actions is not None:
            action_labels = ["Reload", "Prev Image", "Next Image", "Save", "AI Auto Label"]
            for button, label in zip(top_actions.winfo_children(), action_labels):
                button.configure(text=label)

        meta_frames = [child for child in header_children if child.winfo_class() == "Frame"]
        if len(meta_frames) > 1:
            meta_labels = meta_frames[1].winfo_children()
            if len(meta_labels) >= 3:
                meta_labels[2].configure(text="Auto save enabled")

        body_children = body.winfo_children()
        if len(body_children) >= 2:
            toolrail = body_children[0]
            center = body_children[1]
            tool_children = toolrail.winfo_children()
            if tool_children:
                tool_children[0].configure(text="Tools")
                tool_labels = ["Draw Box", "Edit Box", "Delete Selected", "Prev Image", "Next Image", "Save", "Export Dataset"]
                for button, label in zip(tool_children[1:], tool_labels):
                    button.configure(text=label)

            center_children = center.winfo_children()
            if len(center_children) >= 2:
                canvas_panel = center_children[0]
                filmstrip = center_children[1]
                canvas_toolbar = canvas_panel.winfo_children()[0]
                canvas_toolbar.winfo_children()[0].configure(text="Detection Canvas")
                filmstrip_children = filmstrip.winfo_children()
                if len(filmstrip_children) >= 2:
                    filmstrip_children[1].configure(text="Click a preview to switch the current image.")

        side_children = self.sidebar_scroll.content.winfo_children()
        if len(side_children) >= 4:
            classes_box, box_box, auto_box, export_box = side_children[:4]

            classes_children = classes_box.winfo_children()
            classes_children[0].configure(text="Class List")
            classes_children[1].configure(text="Maintain class names and review counts for the current image.")
            class_frames = [child for child in classes_children if child.winfo_class() == "Frame"]
            if len(class_frames) >= 2:
                class_action_buttons = class_frames[1].winfo_children()
                if len(class_action_buttons) >= 2:
                    class_action_buttons[0].configure(text="Save Classes")
                    class_action_buttons[1].configure(text="Go To Train")

            box_children = box_box.winfo_children()
            box_children[0].configure(text="Box List")
            box_children[1].configure(text="Select a box to review class and coordinates.")
            class_pick_row = next((child for child in box_children if child.winfo_class() == "Frame"), None)
            if class_pick_row is not None and class_pick_row.winfo_children():
                class_pick_row.winfo_children()[0].configure(text="Class")

            auto_children = auto_box.winfo_children()
            auto_children[0].configure(text="AI Auto Label")
            auto_children[1].configure(text="Run the detector on one image or the whole project.")
            auto_grid = next((child for child in auto_children if child.winfo_class() == "Frame"), None)
            if auto_grid is not None:
                auto_grid_children = auto_grid.winfo_children()
                if auto_grid_children:
                    auto_grid_children[0].configure(text="Model")
                subframes = [child for child in auto_grid_children if child.winfo_class() == "Frame"]
                if len(subframes) >= 2:
                    browse_buttons = subframes[0].winfo_children()
                    if len(browse_buttons) >= 2:
                        browse_buttons[0].configure(text="Browse Weights")
                        browse_buttons[1].configure(text="Use Default")
                    auto_action_buttons = subframes[1].winfo_children()
                    if len(auto_action_buttons) >= 2:
                        auto_action_buttons[0].configure(text="Current Image")
                        auto_action_buttons[1].configure(text="Whole Project")

            export_children = export_box.winfo_children()
            export_children[0].configure(text="Export Dataset")
            export_children[1].configure(text="Generate dataset.yaml with the current class list.")
            export_children[2].configure(text="Images, labels and class names will be reorganized into a training-ready dataset structure.")
            if len(export_children) >= 5:
                export_children[3].configure(text="Export To Train Set")
                export_children[4].configure(text="Switch To Train")

    def _v2_side_box(self, parent: tk.Widget, title: str, subtitle: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            frame,
            text=title,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(
            frame,
            text=subtitle,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=276,
            justify="left",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(4, 0))
        return frame

    def _v2_tool_button(self, parent: tk.Widget, text: str, command: Callable[[], None], *, primary: bool = False, selected: bool = False) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=PRIMARY if primary else (PRIMARY_SOFT if selected else "#f7fbff"),
            fg="white" if primary else (PRIMARY_DARK if selected else TEXT),
            activebackground=PRIMARY_DARK if primary else PRIMARY_SOFT,
            activeforeground="white" if primary else PRIMARY_DARK,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=PRIMARY if (primary or selected) else BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold" if (primary or selected) else "normal"),
            padx=8,
            pady=7,
            cursor="hand2",
            anchor="w",
        )

    def _set_draw_mode(self, drawing: bool) -> None:
        self.draw_mode_var.set(drawing)
        self.mode_hint_var.set("绘制框" if drawing else "编辑框")
        self._refresh_v2_tool_selection()
        self._refresh_selection_details()
        self.redraw_canvas()

    def _sync_thumb_scrollregion(self, _: tk.Event | None = None) -> None:
        if hasattr(self, "thumb_canvas"):
            bbox = self.thumb_canvas.bbox("all")
            if bbox is not None:
                self.thumb_canvas.configure(scrollregion=bbox)

    def _sync_thumb_viewport(self, event: tk.Event) -> None:
        if hasattr(self, "thumb_canvas"):
            self.thumb_canvas.itemconfigure(self.thumb_window_id, height=max(event.height, 1))

    def _jump_to_image(self, index: int) -> None:
        if not self.image_paths:
            return
        self.save_current_annotations(silent=True)
        self.current_index = max(0, min(index, len(self.image_paths) - 1))
        self.load_current_image()

    def _refresh_thumbnail_strip_modern(self) -> None:
        if self.ui_mode != "v2" or not hasattr(self, "thumb_strip"):
            return
        for child in self.thumb_strip.winfo_children():
            child.destroy()
        self.thumbnail_refs = []

        total = len(self.image_paths)
        current = self.current_index + 1 if total else 0
        self.thumbnail_summary_var.set(f"全部图片 {total}  当前 {current} / {total}")
        if not self.image_paths:
            empty_label = tk.Label(
                self.thumb_strip,
                text="选择图片文件夹后，这里会显示当前项目的缩略图。",
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 10),
                padx=18,
                pady=18,
            )
            empty_label.grid(row=0, column=0, sticky="w")
            self._bind_thumbnail_mousewheel_target(empty_label)
            self._sync_thumb_scrollregion()
            return

        start = max(0, self.current_index - 18)
        end = min(total, start + 36)
        if end - start < 36:
            start = max(0, end - 36)

        for visual_index, image_index in enumerate(range(start, end)):
            image_path = self.image_paths[image_index]
            selected = image_index == self.current_index
            card = tk.Frame(
                self.thumb_strip,
                bg=PRIMARY_SOFT if selected else CARD_BG,
                highlightbackground=PRIMARY if selected else BORDER,
                highlightthickness=1,
                padx=4,
                pady=4,
            )
            card.grid(row=0, column=visual_index, padx=(0, 6), pady=5, sticky="n")
            try:
                thumb_image = Image.open(image_path).convert("RGB")
                thumb_image.thumbnail((92, 58), Image.Resampling.LANCZOS)
                thumb = ImageTk.PhotoImage(thumb_image)
            except OSError:
                thumb = ImageTk.PhotoImage(Image.new("RGB", (92, 58), color="#dfe9fb"))
            self.thumbnail_refs.append(thumb)
            preview = tk.Label(card, image=thumb, bg=card.cget("bg"), cursor="hand2")
            preview.pack()
            label = tk.Label(
                card,
                text=image_path.name,
                bg=card.cget("bg"),
                fg=PRIMARY_DARK if selected else TEXT,
                width=11,
                anchor="w",
                justify="left",
                font=("Microsoft YaHei UI", 9, "bold" if selected else "normal"),
                cursor="hand2",
            )
            label.pack(fill="x", pady=(6, 0))
            meta = tk.Label(
                card,
                text=f"#{image_index + 1}",
                bg=card.cget("bg"),
                fg=TEXT_MUTED,
                anchor="w",
                font=("Microsoft YaHei UI", 8),
                cursor="hand2",
            )
            meta.pack(fill="x")
            for widget in (card, preview, label, meta):
                widget.bind("<Button-1>", lambda _event, idx=image_index: self._jump_to_image(idx), add="+")
                self._bind_thumbnail_mousewheel_target(widget)

        self.thumb_strip.update_idletasks()
        self._sync_thumb_scrollregion()
        if hasattr(self, "thumb_canvas"):
            fraction = 0.0 if total <= 1 else self.current_index / max(total - 1, 1)
            self.thumb_canvas.xview_moveto(max(0.0, min(fraction, 1.0)))

    def _refresh_thumbnail_strip(self) -> None:
        if self.ui_mode != "v2" or not hasattr(self, "thumb_strip"):
            return
        for child in self.thumb_strip.winfo_children():
            child.destroy()
        self.thumbnail_refs = []

        total = len(self.image_paths)
        current = self.current_index + 1 if total else 0
        self.thumbnail_summary_var.set(f"鍏ㄩ儴鍥剧墖 {total} 寮?  褰撳墠 {current} / {total}")
        if not self.image_paths:
            tk.Label(
                self.thumb_strip,
                text="閫夋嫨鍥剧墖鐩綍鍚庯紝杩欓噷浼氭樉绀洪瑙堢缉鐣ュ浘銆?",
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 10),
                padx=18,
                pady=28,
            ).grid(row=0, column=0, sticky="w")
            self._sync_thumb_scrollregion()
            return

        start = max(0, self.current_index - 18)
        end = min(total, start + 36)
        if end - start < 36:
            start = max(0, end - 36)

        for visual_index, image_index in enumerate(range(start, end)):
            image_path = self.image_paths[image_index]
            selected = image_index == self.current_index
            card = tk.Frame(
                self.thumb_strip,
                bg=PRIMARY_SOFT if selected else CARD_BG,
                highlightbackground=PRIMARY if selected else BORDER,
                highlightthickness=1,
                padx=6,
                pady=6,
            )
            card.grid(row=0, column=visual_index, padx=(0, 8), pady=6, sticky="n")
            try:
                thumb_image = Image.open(image_path).convert("RGB")
                thumb_image.thumbnail((112, 72), Image.Resampling.LANCZOS)
                thumb = ImageTk.PhotoImage(thumb_image)
            except OSError:
                thumb = ImageTk.PhotoImage(Image.new("RGB", (112, 72), color="#dfe9fb"))
            self.thumbnail_refs.append(thumb)
            preview = tk.Label(card, image=thumb, bg=card.cget("bg"), cursor="hand2")
            preview.pack()
            label = tk.Label(
                card,
                text=image_path.name,
                bg=card.cget("bg"),
                fg=PRIMARY_DARK if selected else TEXT,
                width=14,
                anchor="w",
                justify="left",
                font=("Microsoft YaHei UI", 9, "bold" if selected else "normal"),
                cursor="hand2",
            )
            label.pack(fill="x", pady=(6, 0))
            meta = tk.Label(
                card,
                text=f"#{image_index + 1}",
                bg=card.cget("bg"),
                fg=TEXT_MUTED,
                anchor="w",
                font=("Microsoft YaHei UI", 8),
                cursor="hand2",
            )
            meta.pack(fill="x")
            for widget in (card, preview, label, meta):
                widget.bind("<Button-1>", lambda _event, idx=image_index: self._jump_to_image(idx), add="+")

        self.thumb_strip.update_idletasks()
        self._sync_thumb_scrollregion()
        if hasattr(self, "thumb_canvas"):
            fraction = 0.0 if total <= 1 else self.current_index / max(total - 1, 1)
            self.thumb_canvas.xview_moveto(max(0.0, min(fraction, 1.0)))

    def _refresh_class_summary(self) -> None:
        if self.ui_mode != "v2" or not hasattr(self, "class_summary_frame"):
            return
        for child in self.class_summary_frame.winfo_children():
            child.destroy()
        if not self.class_names:
            tk.Label(
                self.class_summary_frame,
                text="鏈厤缃被鍒?",
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 9),
            ).pack(anchor="w")
            return

        counts = {index: 0 for index in range(len(self.class_names))}
        for box in self.boxes:
            counts[box.class_id] = counts.get(box.class_id, 0) + 1

        for index, name in enumerate(self.class_names):
            row = tk.Frame(self.class_summary_frame, bg=CARD_BG)
            row.pack(fill="x", pady=(0, 4))
            swatch = tk.Canvas(row, width=12, height=12, bg=CARD_BG, highlightthickness=0, bd=0)
            swatch.create_oval(1, 1, 11, 11, fill=self._box_color(index), outline=self._box_color(index))
            swatch.pack(side="left")
            tk.Label(
                row,
                text=name,
                bg=CARD_BG,
                fg=TEXT,
                font=("Microsoft YaHei UI", 9),
                anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=(8, 0))
            tk.Label(
                row,
                text=f"{counts.get(index, 0)}",
                bg="#f8fbff",
                fg=TEXT_MUTED,
                font=("Microsoft YaHei UI", 8, "bold"),
                padx=8,
                pady=2,
            ).pack(side="right")

    def _refresh_selection_details(self) -> None:
        if self.selected_box_index is None or not (0 <= self.selected_box_index < len(self.boxes)):
            self.selection_title_var.set("未选中目标框")
            self.selection_meta_var.set("在画布中选择一个框以查看类别、坐标和尺寸。")
            return

        box = self.boxes[self.selected_box_index]
        class_name = self.class_names[box.class_id] if box.class_id < len(self.class_names) else f"class{box.class_id}"
        left = int(min(box.x1, box.x2))
        top = int(min(box.y1, box.y2))
        right = int(max(box.x1, box.x2))
        bottom = int(max(box.y1, box.y2))
        width = max(0, right - left)
        height = max(0, bottom - top)
        self.selection_title_var.set(f"#{self.selected_box_index + 1}  {class_name}")
        self.selection_meta_var.set(
            f"坐标: [{left}, {top}] -> [{right}, {bottom}]\n"
            f"尺寸: {width} x {height} px\n"
            f"模式: {'绘制框' if self.draw_mode_var.get() else '编辑框'}"
        )

    def _refresh_v2_surface(self) -> None:
        if self.ui_mode != "v2":
            return
        self.mode_hint_var.set("绘制框" if self.draw_mode_var.get() else "编辑框")
        self._refresh_v2_tool_selection()
        self._refresh_class_summary()
        self._refresh_selection_details()
        self._refresh_thumbnail_strip_modern()

    def _side_box(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(frame, text=title, bg="#edf3ff", fg=TEXT, font=("Microsoft YaHei UI", 11, "bold"), anchor="w", padx=12, pady=10).pack(fill="x")
        return frame

    def _small_button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=CARD_BG,
            fg=TEXT,
            activebackground="#edf3ff",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
            pady=6,
            cursor="hand2",
        )

    def _primary_button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
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
            highlightthickness=1,
            highlightbackground=PRIMARY,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold"),
            pady=8,
            cursor="hand2",
        )

    def _labeled_entry(self, parent: tk.Widget, label: str, variable: tk.StringVar, *, row: int, readonly: bool) -> None:
        parent.grid_columnconfigure(1, weight=1)
        tk.Label(parent, text=label, bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=row, column=0, sticky="w", padx=12, pady=(10 if row == 0 else 0, 8))
        entry = tk.Entry(
            parent,
            textvariable=variable,
            relief="flat",
            bd=0,
            bg=CARD_SOFT if readonly else CARD_BG,
            fg=TEXT,
            readonlybackground=CARD_SOFT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        if readonly:
            entry.configure(state="readonly")
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(10 if row == 0 else 0, 8), ipady=6)

    def pick_project_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择要标注的图片目录")
        if not selected:
            return
        self.load_project(Path(selected))

    def load_project(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            messagebox.showwarning("目录无效", f"找不到目录：\n{folder}")
            return

        try:
            image_paths = list_project_annotation_images(folder)
        except OSError as exc:
            messagebox.showerror("读取失败", str(exc))
            return

        if not image_paths:
            messagebox.showwarning("没有图片", "该目录里没有可标注的图片文件。")
            return

        session = load_project_session(self.session_path, folder)
        session_classes = session.get("classes") if isinstance(session.get("classes"), list) else []
        class_names = load_class_names(folder) or [str(item) for item in session_classes]
        class_names = ensure_class_names(class_names)

        self.project_dir = folder
        self.image_paths = image_paths
        self.class_names = class_names
        self.project_var.set(str(folder))
        self._set_class_text(class_names)
        self._refresh_class_choices()

        current_image_name = str(session.get("current_image") or "")
        if current_image_name:
            for index, path in enumerate(image_paths):
                if path.name == current_image_name:
                    self.current_index = index
                    break
            else:
                self.current_index = 0
        else:
            self.current_index = 0

        self.load_current_image()
        self._notify(f"已加载标注目录：{folder.name}")

    def reload_project(self) -> None:
        if self.project_dir is None:
            self.pick_project_dir()
            return
        self.load_project(self.project_dir)

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()

    def _set_class_text(self, class_names: list[str]) -> None:
        self.class_text.delete("1.0", "end")
        self.class_text.insert("1.0", "\n".join(class_names))

    def _refresh_class_choices(self) -> None:
        self.selected_class_combo.configure(values=self.class_names)
        if self.class_names:
            if self.selected_class_var.get() not in self.class_names:
                self.selected_class_var.set(self.class_names[0])
        self._refresh_class_summary()

    def current_image_count(self) -> int:
        return len(self.image_paths)

    def current_image_name(self) -> str:
        if self.current_image_path is None:
            return "未选择"
        return self.current_image_path.name

    def export_preview_dir(self) -> str:
        if self.project_dir is None:
            return "未选择"
        return str(self.project_dir.parent / f"{self.project_dir.name}_yolo_dataset")

    def load_current_image(self) -> None:
        if not self.image_paths:
            self.current_image_path = None
            self.current_image_var.set("未选择")
            self.image_counter_var.set("0 / 0")
            self.original_image = None
            self.boxes = []
            self.selected_box_index = None
            self.redraw_canvas()
            self.refresh_box_list()
            self._refresh_v2_surface()
            return

        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))
        image_path = self.image_paths[self.current_index]
        self.current_image_path = image_path
        self.current_image_var.set(image_path.name)
        self.image_counter_var.set(f"{self.current_index + 1} / {len(self.image_paths)}")
        self.original_image = Image.open(image_path).convert("RGB")
        self.boxes = load_yolo_boxes(label_path_for_image(image_path), self.original_image.width, self.original_image.height)
        if self.boxes:
            self.class_names = ensure_class_names(self.class_names, max(box.class_id for box in self.boxes))
            self._set_class_text(self.class_names)
            self._refresh_class_choices()
        self.selected_box_index = 0 if self.boxes else None
        self.preview_box = None
        self.drag_mode = None
        self.refresh_box_list()
        self.redraw_canvas()
        self._persist_session()
        self._refresh_v2_surface()
        self.on_state_change()

    def prev_image(self) -> None:
        if not self.image_paths:
            return
        self.save_current_annotations(silent=True)
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self) -> None:
        if not self.image_paths:
            return
        self.save_current_annotations(silent=True)
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_current_image()

    def apply_class_names(self) -> None:
        if self.project_dir is None:
            messagebox.showwarning("未选择目录", "请先选择要标注的图片目录。")
            return
        class_names = ensure_class_names(parse_class_names_text(self.class_text.get("1.0", "end")))
        max_class_id = max((box.class_id for box in self.boxes), default=-1)
        if max_class_id >= len(class_names):
            messagebox.showwarning("类别数量不足", "当前已有框的类别编号超出了新的类别列表，请先调整框类别或补齐类别名称。")
            return
        self.class_names = class_names
        save_class_names(self.project_dir, class_names)
        self._refresh_class_choices()
        self.refresh_box_list()
        self._refresh_selection_details()
        self._persist_session()
        self._notify(f"已保存类别文件：{classes_path_for_folder(self.project_dir).name}")

    def _persist_session(self) -> None:
        if self.project_dir is None:
            return
        save_project_session(
            self.session_path,
            self.project_dir,
            current_image=self.current_image_name() if self.current_image_path else "",
            classes=self.class_names,
        )

    def _ensure_project_ready(self) -> bool:
        if self.project_dir is not None and self.current_image_path is not None and self.original_image is not None:
            return True
        messagebox.showwarning("未准备好", "请先选择要标注的图片目录。")
        return False

    def save_current_annotations(self, *, silent: bool = False) -> None:
        if not self._ensure_project_ready():
            return
        assert self.current_image_path is not None
        assert self.original_image is not None
        save_class_names(self.project_dir, self.class_names)
        save_yolo_boxes(
            label_path_for_image(self.current_image_path),
            self.boxes,
            self.original_image.width,
            self.original_image.height,
        )
        self._persist_session()
        self._refresh_class_summary()
        if not silent:
            self._notify(f"已保存：{self.current_image_path.name}")

    def refresh_box_list(self) -> None:
        self.box_list.delete(0, "end")
        for index, box in enumerate(self.boxes, start=1):
            class_name = self.class_names[box.class_id] if box.class_id < len(self.class_names) else f"class{box.class_id}"
            left = int(min(box.x1, box.x2))
            top = int(min(box.y1, box.y2))
            right = int(max(box.x1, box.x2))
            bottom = int(max(box.y1, box.y2))
            self.box_list.insert("end", f"{index}. {class_name}  [{left},{top}]→[{right},{bottom}]")
        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(self.boxes):
            self.box_list.selection_clear(0, "end")
            self.box_list.selection_set(self.selected_box_index)
            self.box_list.activate(self.selected_box_index)
            class_id = self.boxes[self.selected_box_index].class_id
            if class_id < len(self.class_names):
                self.selected_class_var.set(self.class_names[class_id])
        elif self.class_names:
            self.selected_class_var.set(self.class_names[0])
        self._refresh_selection_details()

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 200),
                max(self.canvas.winfo_height() // 2, 120),
                text="选择图片目录后，这里会显示当前图片和标注框",
                fill=TEXT_MUTED,
                font=("Microsoft YaHei UI", 13),
            )
            return

        canvas_width = max(self.canvas.winfo_width(), 100)
        canvas_height = max(self.canvas.winfo_height(), 100)
        scale = min((canvas_width - 20) / self.original_image.width, (canvas_height - 20) / self.original_image.height)
        scale = max(scale, 0.05)
        self.display_scale = scale
        display_width = max(1, int(self.original_image.width * scale))
        display_height = max(1, int(self.original_image.height * scale))
        self.display_offset_x = (canvas_width - display_width) // 2
        self.display_offset_y = (canvas_height - display_height) // 2

        resized = self.original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, image=self.display_photo, anchor="nw")
        self.canvas.create_rectangle(
            self.display_offset_x - 1,
            self.display_offset_y - 1,
            self.display_offset_x + display_width + 1,
            self.display_offset_y + display_height + 1,
            outline=BORDER,
            width=1,
        )

        for index, box in enumerate(self.boxes):
            self._draw_box(box, selected=index == self.selected_box_index)
        if self.preview_box is not None:
            self._draw_box(self.preview_box, selected=True, dashed=True)

    def _box_color(self, class_id: int) -> str:
        return BOX_COLORS[class_id % len(BOX_COLORS)]

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.display_offset_x + x * self.display_scale,
            self.display_offset_y + y * self.display_scale,
        )

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        if self.original_image is None:
            return 0.0, 0.0
        image_x = (x - self.display_offset_x) / self.display_scale
        image_y = (y - self.display_offset_y) / self.display_scale
        image_x = min(max(image_x, 0.0), float(self.original_image.width))
        image_y = min(max(image_y, 0.0), float(self.original_image.height))
        return image_x, image_y

    def _draw_box(self, box: AnnotationBox, *, selected: bool, dashed: bool = False) -> None:
        x1, y1 = self._image_to_canvas(box.x1, box.y1)
        x2, y2 = self._image_to_canvas(box.x2, box.y2)
        color = self._box_color(box.class_id)
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=color,
            width=3 if selected else 2,
            dash=(6, 3) if dashed else (),
        )
        class_name = self.class_names[box.class_id] if box.class_id < len(self.class_names) else f"class{box.class_id}"
        self.canvas.create_rectangle(x1, y1 - 22, x1 + max(84, len(class_name) * 10), y1, fill=color, outline=color)
        self.canvas.create_text(x1 + 8, y1 - 11, text=class_name, fill="white", anchor="w", font=("Microsoft YaHei UI", 9, "bold"))
        if selected:
            for handle_x, handle_y in self._handle_positions(box).values():
                cx, cy = self._image_to_canvas(handle_x, handle_y)
                self.canvas.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color, width=2)

    def _handle_positions(self, box: AnnotationBox) -> dict[str, tuple[float, float]]:
        left = min(box.x1, box.x2)
        right = max(box.x1, box.x2)
        top = min(box.y1, box.y2)
        bottom = max(box.y1, box.y2)
        return {
            "nw": (left, top),
            "ne": (right, top),
            "sw": (left, bottom),
            "se": (right, bottom),
        }

    def _hit_test(self, canvas_x: float, canvas_y: float) -> tuple[int | None, str | None]:
        if self.original_image is None:
            return None, None

        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(self.boxes):
            for handle_name, (hx, hy) in self._handle_positions(self.boxes[self.selected_box_index]).items():
                cx, cy = self._image_to_canvas(hx, hy)
                if abs(canvas_x - cx) <= 8 and abs(canvas_y - cy) <= 8:
                    return self.selected_box_index, handle_name

        image_x, image_y = self._canvas_to_image(canvas_x, canvas_y)
        for index in reversed(range(len(self.boxes))):
            box = self.boxes[index]
            left = min(box.x1, box.x2)
            right = max(box.x1, box.x2)
            top = min(box.y1, box.y2)
            bottom = max(box.y1, box.y2)
            if left <= image_x <= right and top <= image_y <= bottom:
                return index, "move"
        return None, None

    def _selected_class_id(self) -> int:
        current = self.selected_class_var.get().strip()
        if current and current in self.class_names:
            return self.class_names.index(current)
        return 0

    def on_canvas_press(self, event: tk.Event[tk.Canvas]) -> None:
        if not self._ensure_project_ready():
            return
        self.canvas.focus_set()
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        box_index, hit_kind = self._hit_test(event.x, event.y)

        if self.draw_mode_var.get():
            self.drag_mode = "draw"
            self.drag_anchor = (image_x, image_y)
            self.preview_box = AnnotationBox(self._selected_class_id(), image_x, image_y, image_x, image_y)
            self.selected_box_index = None
            self.refresh_box_list()
            self.redraw_canvas()
            return

        if box_index is None:
            self.selected_box_index = None
            self.drag_mode = None
            self.refresh_box_list()
            self.redraw_canvas()
            return

        self.selected_box_index = box_index
        self.drag_box_index = box_index
        self.drag_box_snapshot = AnnotationBox(**vars(self.boxes[box_index]))
        self.drag_anchor = (image_x, image_y)
        self.drag_handle = hit_kind
        self.drag_mode = "resize" if hit_kind and hit_kind != "move" else "move"
        self.refresh_box_list()
        self.redraw_canvas()

    def on_canvas_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.original_image is None or self.drag_mode is None:
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)

        if self.drag_mode == "draw" and self.preview_box is not None and self.drag_anchor is not None:
            self.preview_box.x2 = image_x
            self.preview_box.y2 = image_y
            self.redraw_canvas()
            return

        if self.drag_box_index is None or self.drag_box_snapshot is None or self.drag_anchor is None:
            return

        dx = image_x - self.drag_anchor[0]
        dy = image_y - self.drag_anchor[1]
        snapshot = self.drag_box_snapshot
        box = self.boxes[self.drag_box_index]

        if self.drag_mode == "move":
            box.x1 = snapshot.x1 + dx
            box.x2 = snapshot.x2 + dx
            box.y1 = snapshot.y1 + dy
            box.y2 = snapshot.y2 + dy
        elif self.drag_mode == "resize":
            box.x1 = snapshot.x1
            box.x2 = snapshot.x2
            box.y1 = snapshot.y1
            box.y2 = snapshot.y2
            if self.drag_handle == "nw":
                box.x1 = image_x
                box.y1 = image_y
            elif self.drag_handle == "ne":
                box.x2 = image_x
                box.y1 = image_y
            elif self.drag_handle == "sw":
                box.x1 = image_x
                box.y2 = image_y
            elif self.drag_handle == "se":
                box.x2 = image_x
                box.y2 = image_y
        self._clamp_box(box)
        self.refresh_box_list()
        self.redraw_canvas()

    def on_canvas_release(self, _event: tk.Event[tk.Canvas]) -> None:
        if self.original_image is None:
            return
        refresh_counts = False
        if self.drag_mode == "draw" and self.preview_box is not None:
            box = self.preview_box
            if abs(box.x2 - box.x1) >= 3 and abs(box.y2 - box.y1) >= 3:
                self._clamp_box(box)
                self.boxes.append(box)
                self.selected_box_index = len(self.boxes) - 1
                self.save_current_annotations(silent=True)
                refresh_counts = True
                self._notify(f"已新增框：{self.current_image_name()}")
            self.preview_box = None
        elif self.drag_mode in {"move", "resize"}:
            self.save_current_annotations(silent=True)
            self._notify(f"已更新框：{self.current_image_name()}")
        self.drag_mode = None
        self.drag_anchor = None
        self.drag_box_index = None
        self.drag_box_snapshot = None
        self.drag_handle = None
        self.refresh_box_list()
        if refresh_counts:
            self._refresh_class_summary()
        self.redraw_canvas()

    def _clamp_box(self, box: AnnotationBox) -> None:
        if self.original_image is None:
            return
        width = float(self.original_image.width)
        height = float(self.original_image.height)
        box.x1 = min(max(box.x1, 0.0), width)
        box.x2 = min(max(box.x2, 0.0), width)
        box.y1 = min(max(box.y1, 0.0), height)
        box.y2 = min(max(box.y2, 0.0), height)

    def on_box_selected(self, _event: tk.Event[tk.Listbox]) -> None:
        selection = self.box_list.curselection()
        self.selected_box_index = selection[0] if selection else None
        self.refresh_box_list()
        self.redraw_canvas()
        self._refresh_selection_details()
        self.on_state_change()

    def on_selected_class_changed(self, _event: tk.Event[ttk.Combobox] | None = None) -> None:
        if self.selected_box_index is None or self.selected_box_index >= len(self.boxes):
            return
        class_name = self.selected_class_var.get().strip()
        if class_name not in self.class_names:
            return
        self.boxes[self.selected_box_index].class_id = self.class_names.index(class_name)
        self.refresh_box_list()
        self.redraw_canvas()
        self.save_current_annotations(silent=True)
        self._refresh_class_summary()
        self._notify(f"已修改类别：{class_name}")

    def delete_selected_box(self) -> None:
        if self.selected_box_index is None or self.selected_box_index >= len(self.boxes):
            return
        del self.boxes[self.selected_box_index]
        if self.selected_box_index >= len(self.boxes):
            self.selected_box_index = len(self.boxes) - 1 if self.boxes else None
        self.refresh_box_list()
        self.redraw_canvas()
        self.save_current_annotations(silent=True)
        self._notify("已删除选中框")

    def pick_auto_model(self) -> None:
        path = filedialog.askopenfilename(title="选择自动标注模型", filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")])
        if path:
            self.auto_model_var.set(path)

    def use_default_auto_model(self) -> None:
        self.auto_model_var.set("yolo11n.pt")

    def _auto_label_config(self) -> dict[str, object]:
        config: dict[str, object] = {}
        try:
            config["conf"] = float(self.auto_conf_var.get().strip() or "0.25")
            config["iou"] = float(self.auto_iou_var.get().strip() or "0.70")
            config["imgsz"] = int(self.auto_imgsz_var.get().strip() or "640")
        except ValueError:
            raise ValueError("自动标注参数格式不正确，请检查 conf / iou / imgsz。")
        device = self.auto_device_var.get().strip()
        if device:
            config["device"] = device
        config["save"] = False
        config["verbose"] = False
        config["max_det"] = 300
        return config

    def auto_label_current(self) -> None:
        if not self._ensure_project_ready():
            return
        try:
            config = self._auto_label_config()
        except ValueError as exc:
            messagebox.showwarning("参数无效", str(exc))
            return
        assert self.project_dir is not None
        assert self.current_image_path is not None
        self.save_current_annotations(silent=True)
        self.on_auto_label_request(self.project_dir, self.current_image_path, self.auto_model_var.get().strip(), config, list(self.class_names))

    def auto_label_all(self) -> None:
        if not self._ensure_project_ready():
            return
        try:
            config = self._auto_label_config()
        except ValueError as exc:
            messagebox.showwarning("参数无效", str(exc))
            return
        assert self.project_dir is not None
        self.save_current_annotations(silent=True)
        self.on_auto_label_request(self.project_dir, None, self.auto_model_var.get().strip(), config, list(self.class_names))

    def export_training_dataset(self) -> None:
        if not self._ensure_project_ready():
            return
        assert self.project_dir is not None
        self.save_current_annotations(silent=True)
        class_names = ensure_class_names(parse_class_names_text(self.class_text.get("1.0", "end")))
        self.class_names = class_names
        save_class_names(self.project_dir, class_names)
        self._refresh_class_choices()
        self.on_export_request(self.project_dir, class_names)

    def apply_auto_label_result(self, payload: dict[str, object]) -> None:
        if self.project_dir is None:
            return
        if str(payload.get("image_dir") or "") != str(self.project_dir):
            return

        model_class_names = payload.get("model_class_names")
        if isinstance(model_class_names, list):
            cleaned_model_classes = [str(item).strip() for item in model_class_names if str(item).strip()]
            if cleaned_model_classes:
                generic = all(name == f"class{index}" for index, name in enumerate(self.class_names))
                if generic or not self.class_names:
                    self.class_names = cleaned_model_classes
                    self._set_class_text(self.class_names)
                    self._refresh_class_choices()
                    save_class_names(self.project_dir, self.class_names)
                elif len(self.class_names) < len(cleaned_model_classes):
                    final_class_names = list(self.class_names)
                    while len(final_class_names) < len(cleaned_model_classes):
                        index = len(final_class_names)
                        fallback_name = cleaned_model_classes[index] if index < len(cleaned_model_classes) else f"class{index}"
                        final_class_names.append(fallback_name or f"class{index}")
                    self.class_names = final_class_names
                    self._set_class_text(self.class_names)
                    self._refresh_class_choices()
                    save_class_names(self.project_dir, self.class_names)

        updated_files = payload.get("updated_files") or []
        if self.current_image_path is not None and str(self.current_image_path) in [str(item) for item in updated_files]:
            self.load_current_image()
        else:
            self._persist_session()
            self._refresh_v2_surface()
            self.on_state_change()
