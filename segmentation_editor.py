from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

from PIL import Image, ImageTk

from annotation_support import (
    AnnotationPolygon,
    classes_path_for_folder,
    ensure_class_names,
    label_path_for_image,
    list_project_annotation_images,
    load_class_names,
    load_project_session,
    load_yolo_polygons,
    parse_class_names_text,
    save_class_names,
    save_project_session,
    save_yolo_polygons,
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
MASK_COLORS = (
    "#ff6b6b",
    "#4ecdc4",
    "#ffd166",
    "#6c5ce7",
    "#00b894",
    "#fd79a8",
    "#0984e3",
    "#e17055",
)


class SegmentationAnnotationEditor(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        session_path: Path,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_export_request: Callable[[Path, list[str]], None],
        on_switch_to_train: Callable[[], None],
        ui_mode: str = "classic",
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.session_path = session_path
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_export_request = on_export_request
        self.on_switch_to_train = on_switch_to_train
        self.ui_mode = ui_mode

        self.project_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index = 0
        self.class_names: list[str] = ["class0"]
        self.polygons: list[AnnotationPolygon] = []
        self.current_image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.selected_polygon_index: int | None = None
        self.dragging_vertex_index: int | None = None
        self.current_points: list[tuple[float, float]] = []

        self.project_var = tk.StringVar(value="")
        self.current_image_var = tk.StringVar(value="未选择")
        self.image_counter_var = tk.StringVar(value="0 / 0")
        self.status_var = tk.StringVar(value="请选择图片目录，开始手动画轮廓。")
        self.draw_mode_var = tk.BooleanVar(value=True)
        self.mode_hint_var = tk.StringVar(value="添加顶点")
        self.selection_title_var = tk.StringVar(value="未选中轮廓")
        self.selection_meta_var = tk.StringVar(value="在画布中选择一个轮廓以查看详情。")
        self.thumbnail_summary_var = tk.StringVar(value="全部图片 0  当前 0 / 0")
        self.thumbnail_refs: list[ImageTk.PhotoImage] = []

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
        tk.Entry(
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
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 12), ipady=6)
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

        toolbar = tk.Frame(canvas_panel, bg=CARD_SOFT)
        toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        tk.Checkbutton(
            toolbar,
            text="点击加点",
            variable=self.draw_mode_var,
            bg=CARD_SOFT,
            fg=TEXT,
            activebackground=CARD_SOFT,
            activeforeground=TEXT,
            selectcolor=CARD_BG,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        self._small_button(toolbar, "完成轮廓", self.finish_current_polygon).pack(side="left", padx=(10, 0))
        self._small_button(toolbar, "撤销一点", self.undo_last_point).pack(side="left", padx=(10, 0))
        self._small_button(toolbar, "保存当前标注", self.save_current_annotations).pack(side="left", padx=(10, 0))
        self._small_button(toolbar, "删除选中轮廓", self.delete_selected_polygon).pack(side="left", padx=(10, 0))
        tk.Label(toolbar, textvariable=self.status_var, bg=CARD_SOFT, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="right")

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
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Return>", lambda _event: self.finish_current_polygon())
        self.canvas.bind("<Escape>", lambda _event: self.clear_current_points())
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_polygon())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_polygon())

        self.sidebar_scroll = VerticalScrolledFrame(body, bg=CARD_BG)
        self.sidebar_scroll.grid(row=0, column=1, sticky="nsew")
        side = self.sidebar_scroll.content
        side.grid_columnconfigure(0, weight=1)

        self.guide_box = self._side_box(side, "交互式分割说明")
        guide_box = self.guide_box
        guide_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            guide_box,
            text="1. 勾选“点击加点”后，在图片上依次点击轮廓点。\n2. 双击左键或点“完成轮廓”结束当前掩码。\n3. 想拖动已有点时，先取消“点击加点”，再拖动小圆点。\n4. 当前版本先提供离线手动交互，后续第三期再接模型辅助。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=290,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=12)

        self.class_box = self._side_box(side, "类别与当前画笔")
        class_box = self.class_box
        class_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.class_text = tk.Text(
            class_box,
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
        self._small_button(class_box, "保存类别列表", self.apply_class_names).pack(fill="x", padx=12, pady=(0, 8))
        self.class_list = tk.Listbox(
            class_box,
            height=4,
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
        self.class_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.class_list.bind("<<ListboxSelect>>", lambda _event: self._refresh_polygon_list_selection())
        self._small_button(class_box, "把当前类别应用到选中轮廓", self.apply_selected_class_to_polygon).pack(fill="x", padx=12, pady=(0, 12))

        self.polygon_box = self._side_box(side, "当前轮廓")
        polygon_box = self.polygon_box
        polygon_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.polygon_list = tk.Listbox(
            polygon_box,
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
        self.polygon_list.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        self.polygon_list.bind("<<ListboxSelect>>", self.on_polygon_selected)
        tk.Label(
            polygon_box,
            text="提示：选中轮廓后可切换到“非点击加点”模式，直接拖动轮廓点修边。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=290,
            justify="left",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

        self.export_box = self._side_box(side, "整理成训练集")
        export_box = self.export_box
        export_box.grid(row=3, column=0, sticky="ew")
        tk.Label(
            export_box,
            text="会把当前图片目录里的图片 + 同名 YOLO 分割 TXT 整理到多任务数据整理面板，生成可直接训练的分割数据集。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=290,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=(12, 8))
        self._primary_button(export_box, "送去整理分割训练集", self.export_training_dataset).pack(fill="x", padx=12, pady=(0, 8))
        self._small_button(export_box, "切到训练页", self.on_switch_to_train).pack(fill="x", padx=12, pady=(0, 12))

    def _build_v2_ui_modern(self) -> None:
        self.configure(bg="#f5f8fe")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        header.grid_columnconfigure(1, weight=1)

        tk.Label(header, text="项目路径", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(12, 8), pady=(10, 6))
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
        self._small_button(top_actions, "撤销一点", self.undo_last_point).pack(side="left", padx=(0, 6))
        self._primary_button(top_actions, "完成轮廓", self.finish_current_polygon).pack(side="left", padx=(0, 6))
        self._primary_button(top_actions, "保存标注", self.save_current_annotations).pack(side="left")

        tk.Label(header, text="当前图片", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(12, 8), pady=(0, 10))
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
        tk.Label(image_meta, textvariable=self.image_counter_var, bg="#f3f8ff", fg=PRIMARY_DARK, font=("Microsoft YaHei UI", 10, "bold"), padx=10, pady=5).pack(side="left", padx=(0, 8))
        tk.Label(image_meta, textvariable=self.mode_hint_var, bg="#edf6ff", fg=PRIMARY_DARK, font=("Microsoft YaHei UI", 10, "bold"), padx=10, pady=5).pack(side="left", padx=(0, 8))
        tk.Label(image_meta, text="自动保存已开启", bg="#ebfbf4", fg="#31c48d", font=("Microsoft YaHei UI", 10, "bold"), padx=10, pady=5).pack(side="left")

        body = tk.Frame(self, bg="#f5f8fe")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=292)

        toolrail = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, width=118)
        toolrail.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        toolrail.grid_propagate(False)
        tk.Label(toolrail, text="工具栏", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 8))
        self.v2_draw_button = self._v2_tool_button(toolrail, "添加顶点", lambda: self._set_draw_mode(True), selected=True)
        self.v2_draw_button.pack(fill="x", padx=8, pady=(0, 6))
        self.v2_edit_button = self._v2_tool_button(toolrail, "编辑轮廓", lambda: self._set_draw_mode(False))
        self.v2_edit_button.pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "撤销一点", self.undo_last_point).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "完成轮廓", self.finish_current_polygon).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "删除选中", self.delete_selected_polygon).pack(fill="x", padx=8, pady=(0, 6))
        self._v2_tool_button(toolrail, "导出训练集", self.export_training_dataset, primary=True).pack(fill="x", padx=8, pady=(6, 10))

        center = tk.Frame(body, bg="#f5f8fe")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        canvas_panel = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        canvas_panel.grid(row=0, column=0, sticky="nsew")
        canvas_panel.grid_rowconfigure(1, weight=1)
        canvas_panel.grid_columnconfigure(0, weight=1)

        toolbar = tk.Frame(canvas_panel, bg=CARD_BG)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        tk.Label(toolbar, text="实例分割画布", bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 13, "bold")).pack(side="left")
        tk.Label(toolbar, textvariable=self.status_var, bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="right")

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
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Return>", lambda _event: self.finish_current_polygon())
        self.canvas.bind("<Escape>", lambda _event: self.clear_current_points())
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_polygon())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_polygon())

        filmstrip = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        filmstrip.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        filmstrip.grid_columnconfigure(1, weight=1)
        tk.Label(filmstrip, textvariable=self.thumbnail_summary_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        tk.Label(filmstrip, text="点击缩略图可快速切换当前图片。", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9)).grid(row=0, column=1, sticky="e", padx=10, pady=(8, 4))

        self.thumb_canvas = tk.Canvas(filmstrip, bg=CARD_BG, highlightthickness=0, bd=0, relief="flat", height=104)
        self.thumb_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        thumb_scroll = tk.Scrollbar(filmstrip, orient="horizontal", command=self.thumb_canvas.xview, relief="flat", bd=0, highlightthickness=0)
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

        self.class_box = self._v2_side_box(side, "类别列表", "维护类名，并把类别应用到当前选中的轮廓。")
        self.class_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.class_summary_frame = tk.Frame(self.class_box, bg=CARD_BG)
        self.class_summary_frame.pack(fill="x", padx=12, pady=(12, 6))
        self.class_text = tk.Text(
            self.class_box,
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
        self._small_button(self.class_box, "保存类别", self.apply_class_names).pack(fill="x", padx=12, pady=(0, 8))
        self.class_list = tk.Listbox(
            self.class_box,
            height=5,
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
        self.class_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.class_list.bind("<<ListboxSelect>>", lambda _event: self._refresh_polygon_list_selection())
        self._small_button(self.class_box, "应用到选中轮廓", self.apply_selected_class_to_polygon).pack(fill="x", padx=12, pady=(0, 12))

        self.polygon_box = self._v2_side_box(side, "轮廓列表", "选中轮廓后，可查看顶点数量、类别和编辑状态。")
        self.polygon_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tk.Label(self.polygon_box, textvariable=self.selection_title_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(self.polygon_box, textvariable=self.selection_meta_var, bg=CARD_BG, fg=TEXT_MUTED, justify="left", wraplength=276, font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=12, pady=(0, 8))
        self.polygon_list = tk.Listbox(
            self.polygon_box,
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
        self.polygon_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.polygon_list.bind("<<ListboxSelect>>", self.on_polygon_selected)
        tk.Label(
            self.polygon_box,
            text="提示：双击完成轮廓，右键可撤销最近一个顶点。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=276,
            justify="left",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

        self.export_box = self._v2_side_box(side, "整理为训练集", "生成可直接训练的实例分割数据集结构。")
        self.export_box.grid(row=2, column=0, sticky="ew")
        self._primary_button(self.export_box, "导出训练集", self.export_training_dataset).pack(fill="x", padx=12, pady=(12, 8))
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

        tk.Label(header, text="Project", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(16, 10), pady=(14, 8))
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
        self._small_button(header, "Open Folder", self.pick_project_dir).grid(row=0, column=2, padx=10, pady=(14, 8))

        top_actions = tk.Frame(header, bg=CARD_BG)
        top_actions.grid(row=0, column=3, sticky="e", padx=(10, 16), pady=(14, 8))
        self._small_button(top_actions, "Reload", self.reload_project).pack(side="left", padx=(0, 8))
        self._small_button(top_actions, "Prev", self.prev_image).pack(side="left", padx=(0, 8))
        self._small_button(top_actions, "Next", self.next_image).pack(side="left", padx=(0, 8))
        self._small_button(top_actions, "Undo", self.undo_last_point).pack(side="left", padx=(0, 8))
        self._primary_button(top_actions, "Finish", self.finish_current_polygon).pack(side="left", padx=(0, 8))
        self._primary_button(top_actions, "Save", self.save_current_annotations).pack(side="left")

        tk.Label(header, text="Current Image", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(16, 10), pady=(0, 14))
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
        tk.Label(image_meta, textvariable=self.image_counter_var, bg="#f3f8ff", fg=PRIMARY_DARK, font=("Microsoft YaHei UI", 10, "bold"), padx=12, pady=8).pack(side="left", padx=(0, 10))
        tk.Label(image_meta, textvariable=self.mode_hint_var, bg="#edf6ff", fg=PRIMARY_DARK, font=("Microsoft YaHei UI", 10, "bold"), padx=12, pady=8).pack(side="left")

        body = tk.Frame(self, bg="#f5f8fe")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=316)

        toolrail = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        toolrail.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        tk.Label(toolrail, text="Tools", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=14, pady=(14, 12))
        self._v2_tool_button(toolrail, "Add Points", lambda: self._set_draw_mode(True), selected=True).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "Edit Polygon", lambda: self._set_draw_mode(False)).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "Undo Point", self.undo_last_point).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "Finish", self.finish_current_polygon).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "Delete", self.delete_selected_polygon).pack(fill="x", padx=10, pady=(0, 8))
        self._v2_tool_button(toolrail, "Export", self.export_training_dataset, primary=True).pack(fill="x", padx=10, pady=(8, 14))

        center = tk.Frame(body, bg="#f5f8fe")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        canvas_panel = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        canvas_panel.grid(row=0, column=0, sticky="nsew")
        canvas_panel.grid_rowconfigure(1, weight=1)
        canvas_panel.grid_columnconfigure(0, weight=1)

        toolbar = tk.Frame(canvas_panel, bg=CARD_BG)
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        tk.Label(toolbar, text="Segmentation Canvas", bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        tk.Label(toolbar, textvariable=self.status_var, bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).pack(side="right")

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
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Return>", lambda _event: self.finish_current_polygon())
        self.canvas.bind("<Escape>", lambda _event: self.clear_current_points())
        self.canvas.bind("<Delete>", lambda _event: self.delete_selected_polygon())
        self.canvas.bind("<BackSpace>", lambda _event: self.delete_selected_polygon())

        filmstrip = tk.Frame(center, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        filmstrip.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        filmstrip.grid_columnconfigure(1, weight=1)
        tk.Label(filmstrip, textvariable=self.thumbnail_summary_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(filmstrip, text="Click a preview to switch the current image.", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9)).grid(row=0, column=1, sticky="e", padx=14, pady=(12, 6))

        self.thumb_canvas = tk.Canvas(filmstrip, bg=CARD_BG, highlightthickness=0, bd=0, relief="flat", height=138)
        self.thumb_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        thumb_scroll = tk.Scrollbar(filmstrip, orient="horizontal", command=self.thumb_canvas.xview, relief="flat", bd=0, highlightthickness=0)
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

        self.class_box = self._v2_side_box(side, "Class List", "Maintain class names and remap selected polygons.")
        self.class_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.class_summary_frame = tk.Frame(self.class_box, bg=CARD_BG)
        self.class_summary_frame.pack(fill="x", padx=12, pady=(12, 6))
        self.class_text = tk.Text(
            self.class_box,
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
        self._small_button(self.class_box, "Save Classes", self.apply_class_names).pack(fill="x", padx=12, pady=(0, 8))
        self.class_list = tk.Listbox(
            self.class_box,
            height=5,
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
        self.class_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.class_list.bind("<<ListboxSelect>>", lambda _event: self._refresh_polygon_list_selection())
        self._small_button(self.class_box, "Apply To Selected", self.apply_selected_class_to_polygon).pack(fill="x", padx=12, pady=(0, 12))

        self.polygon_box = self._v2_side_box(side, "Polygon List", "Select a polygon to inspect points and class.")
        self.polygon_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tk.Label(self.polygon_box, textvariable=self.selection_title_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(self.polygon_box, textvariable=self.selection_meta_var, bg=CARD_BG, fg=TEXT_MUTED, justify="left", wraplength=276, font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=12, pady=(0, 8))
        self.polygon_list = tk.Listbox(
            self.polygon_box,
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
        self.polygon_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.polygon_list.bind("<<ListboxSelect>>", self.on_polygon_selected)
        tk.Label(
            self.polygon_box,
            text="Tips: double click to finish; right click to undo the latest point.",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=276,
            justify="left",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

        self.export_box = self._v2_side_box(side, "Export Dataset", "Generate a training-ready segmentation dataset.")
        self.export_box.grid(row=2, column=0, sticky="ew")
        self._primary_button(self.export_box, "Export To Train Set", self.export_training_dataset).pack(fill="x", padx=12, pady=(12, 8))
        self._small_button(self.export_box, "Switch To Train", self.on_switch_to_train).pack(fill="x", padx=12, pady=(0, 12))

    def _v2_side_box(self, parent: tk.Widget, title: str, subtitle: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(frame, text=title, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold"), anchor="w").pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(frame, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED, wraplength=276, justify="left", font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=12, pady=(4, 0))
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
        self.mode_hint_var.set("添加顶点" if drawing else "编辑轮廓")
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
        self.thumbnail_summary_var.set(f"All images {total}  Current {current} / {total}")
        if not self.image_paths:
            tk.Label(self.thumb_strip, text="Choose an image folder to populate the thumbnail rail.", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10), padx=18, pady=28).grid(row=0, column=0, sticky="w")
            self._sync_thumb_scrollregion()
            return

        start = max(0, self.current_index - 18)
        end = min(total, start + 36)
        if end - start < 36:
            start = max(0, end - 36)

        for visual_index, image_index in enumerate(range(start, end)):
            image_path = self.image_paths[image_index]
            selected = image_index == self.current_index
            card = tk.Frame(self.thumb_strip, bg=PRIMARY_SOFT if selected else CARD_BG, highlightbackground=PRIMARY if selected else BORDER, highlightthickness=1, padx=6, pady=6)
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
            label = tk.Label(card, text=image_path.name, bg=card.cget("bg"), fg=PRIMARY_DARK if selected else TEXT, width=14, anchor="w", justify="left", font=("Microsoft YaHei UI", 9, "bold" if selected else "normal"), cursor="hand2")
            label.pack(fill="x", pady=(6, 0))
            meta = tk.Label(card, text=f"#{image_index + 1}", bg=card.cget("bg"), fg=TEXT_MUTED, anchor="w", font=("Microsoft YaHei UI", 8), cursor="hand2")
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
            tk.Label(self.class_summary_frame, text="No classes configured.", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w")
            return

        counts = {index: 0 for index in range(len(self.class_names))}
        for polygon in self.polygons:
            counts[polygon.class_id] = counts.get(polygon.class_id, 0) + 1

        for index, name in enumerate(self.class_names):
            row = tk.Frame(self.class_summary_frame, bg=CARD_BG)
            row.pack(fill="x", pady=(0, 4))
            swatch = tk.Canvas(row, width=12, height=12, bg=CARD_BG, highlightthickness=0, bd=0)
            swatch.create_oval(1, 1, 11, 11, fill=self._mask_color(index), outline=self._mask_color(index))
            swatch.pack(side="left")
            tk.Label(row, text=name, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei UI", 9), anchor="w").pack(side="left", fill="x", expand=True, padx=(8, 0))
            tk.Label(row, text=f"{counts.get(index, 0)}", bg="#f8fbff", fg=TEXT_MUTED, font=("Microsoft YaHei UI", 8, "bold"), padx=8, pady=2).pack(side="right")

    def _refresh_selection_details(self) -> None:
        if self.selected_polygon_index is None or not (0 <= self.selected_polygon_index < len(self.polygons)):
            self.selection_title_var.set("未选中轮廓")
            self.selection_meta_var.set("在画布中选择一个轮廓以查看顶点数量、类别和编辑状态。")
            return

        polygon = self.polygons[self.selected_polygon_index]
        class_name = self.class_names[polygon.class_id] if polygon.class_id < len(self.class_names) else f"class{polygon.class_id}"
        self.selection_title_var.set(f"#{self.selected_polygon_index + 1}  {class_name}")
        self.selection_meta_var.set(
            f"顶点数: {len(polygon.points)}\n"
            f"模式: {'添加顶点' if self.draw_mode_var.get() else '编辑轮廓'}\n"
            f"类别 ID: {polygon.class_id}"
        )

    def _refresh_v2_surface(self) -> None:
        if self.ui_mode != "v2":
            return
        self.mode_hint_var.set("添加顶点" if self.draw_mode_var.get() else "编辑轮廓")
        self._refresh_v2_tool_selection()
        self._refresh_class_summary()
        self._refresh_selection_details()
        self._refresh_thumbnail_strip_modern()

    def _side_box(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            frame,
            text=title,
            bg=PRIMARY_SOFT,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x")
        return frame

    def _small_button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
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

    def pick_project_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择要做分割标注的图片目录")
        if selected:
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
            messagebox.showwarning("没有图片", "这个目录里没有可标注的图片文件。")
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
        self._refresh_class_list()

        current_image_name = str(session.get("current_image") or "")
        self.current_index = 0
        if current_image_name:
            for index, path in enumerate(image_paths):
                if path.name == current_image_name:
                    self.current_index = index
                    break
        self.load_current_image()
        self._notify(f"已加载分割项目：{folder.name}")

    def reload_project(self) -> None:
        if self.project_dir is None:
            self.pick_project_dir()
            return
        self.load_project(self.project_dir)

    def _set_class_text(self, class_names: list[str]) -> None:
        self.class_text.delete("1.0", "end")
        self.class_text.insert("1.0", "\n".join(class_names))

    def _refresh_class_list(self) -> None:
        current_selection = self._selected_class_id()
        self.class_list.delete(0, "end")
        for class_name in self.class_names:
            self.class_list.insert("end", class_name)
        if self.class_names:
            target_index = min(max(current_selection, 0), len(self.class_names) - 1)
            self.class_list.selection_clear(0, "end")
            self.class_list.selection_set(target_index)
            self.class_list.activate(target_index)
        self._refresh_class_summary()

    def _selected_class_id(self) -> int:
        selection = self.class_list.curselection()
        if selection:
            return int(selection[0])
        return 0

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()

    def current_image_name(self) -> str:
        if self.current_image_path is None:
            return "未选择"
        return self.current_image_path.name

    def export_preview_dir(self) -> str:
        if self.project_dir is None:
            return "未选择"
        return str(self.project_dir.parent / f"{self.project_dir.name}_segment_dataset")

    def current_image_count(self) -> int:
        return len(self.image_paths)

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
        messagebox.showwarning("还没准备好", "请先选择要做分割标注的图片目录。")
        return False

    def load_current_image(self) -> None:
        if not self.image_paths:
            self.current_image_path = None
            self.current_image_var.set("未选择")
            self.image_counter_var.set("0 / 0")
            self.original_image = None
            self.polygons = []
            self.selected_polygon_index = None
            self.current_points = []
            self.redraw_canvas()
            self.refresh_polygon_list()
            self._refresh_v2_surface()
            return

        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))
        image_path = self.image_paths[self.current_index]
        self.current_image_path = image_path
        self.current_image_var.set(image_path.name)
        self.image_counter_var.set(f"{self.current_index + 1} / {len(self.image_paths)}")
        self.original_image = Image.open(image_path).convert("RGB")
        self.polygons = load_yolo_polygons(label_path_for_image(image_path), self.original_image.width, self.original_image.height)
        max_class_id = max((polygon.class_id for polygon in self.polygons), default=-1)
        if max_class_id >= 0:
            self.class_names = ensure_class_names(self.class_names, max_class_id=max_class_id)
            self._set_class_text(self.class_names)
            self._refresh_class_list()
        self.selected_polygon_index = 0 if self.polygons else None
        self.current_points = []
        self.dragging_vertex_index = None
        self.refresh_polygon_list()
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
            messagebox.showwarning("未选择目录", "请先选择图片目录。")
            return
        class_names = ensure_class_names(parse_class_names_text(self.class_text.get("1.0", "end")))
        max_class_id = max((polygon.class_id for polygon in self.polygons), default=-1)
        if max_class_id >= len(class_names):
            messagebox.showwarning("类别数量不够", "当前已有轮廓的类别编号超出了新的类别列表，请先改轮廓类别或补齐类别名。")
            return
        self.class_names = class_names
        save_class_names(self.project_dir, class_names)
        self._refresh_class_list()
        self.refresh_polygon_list()
        self._persist_session()
        self._notify(f"已保存类别文件：{classes_path_for_folder(self.project_dir).name}")

    def apply_selected_class_to_polygon(self) -> None:
        if self.selected_polygon_index is None or not (0 <= self.selected_polygon_index < len(self.polygons)):
            messagebox.showwarning("未选中轮廓", "请先在右侧选中一个轮廓。")
            return
        class_id = self._selected_class_id()
        self.polygons[self.selected_polygon_index].class_id = class_id
        self.refresh_polygon_list()
        self.redraw_canvas()
        self.save_current_annotations(silent=True)
        self._notify(f"已更新轮廓类别：{self.current_image_name()}")

    def refresh_polygon_list(self) -> None:
        self.polygon_list.delete(0, "end")
        for index, polygon in enumerate(self.polygons, start=1):
            class_name = self.class_names[polygon.class_id] if polygon.class_id < len(self.class_names) else f"class{polygon.class_id}"
            self.polygon_list.insert("end", f"{index}. {class_name}  ·  {len(polygon.points)} 点")
        self._refresh_polygon_list_selection()

    def _refresh_polygon_list_selection(self) -> None:
        if self.selected_polygon_index is not None and 0 <= self.selected_polygon_index < len(self.polygons):
            self.polygon_list.selection_clear(0, "end")
            self.polygon_list.selection_set(self.selected_polygon_index)
            self.polygon_list.activate(self.selected_polygon_index)
            class_id = self.polygons[self.selected_polygon_index].class_id
            if class_id < len(self.class_names):
                self.class_list.selection_clear(0, "end")
                self.class_list.selection_set(class_id)
                self.class_list.activate(class_id)
        self._refresh_selection_details()

    def on_polygon_selected(self, _event: tk.Event[tk.Listbox]) -> None:
        selection = self.polygon_list.curselection()
        if not selection:
            self.selected_polygon_index = None
        else:
            self.selected_polygon_index = int(selection[0])
        self._refresh_polygon_list_selection()
        self.redraw_canvas()
        self._refresh_selection_details()

    def save_current_annotations(self, *, silent: bool = False) -> None:
        if not self._ensure_project_ready():
            return
        assert self.current_image_path is not None
        assert self.original_image is not None
        save_class_names(self.project_dir, self.class_names)
        save_yolo_polygons(
            label_path_for_image(self.current_image_path),
            self.polygons,
            self.original_image.width,
            self.original_image.height,
        )
        self._persist_session()
        self._refresh_class_summary()
        if not silent:
            self._notify(f"已保存：{self.current_image_path.name}")

    def clear_current_points(self) -> None:
        if not self.current_points:
            return
        self.current_points = []
        self.redraw_canvas()
        self._notify("已清空未完成的轮廓。")

    def undo_last_point(self) -> None:
        if not self.current_points:
            return
        self.current_points.pop()
        self.redraw_canvas()
        self._notify("已撤销最后一个点。")

    def finish_current_polygon(self) -> None:
        if not self._ensure_project_ready():
            return
        if len(self.current_points) < 3:
            messagebox.showwarning("点数不够", "至少要 3 个点才能组成一个分割轮廓。")
            return
        polygon = AnnotationPolygon(class_id=self._selected_class_id(), points=list(self.current_points))
        self.polygons.append(polygon)
        self.current_points = []
        self.selected_polygon_index = len(self.polygons) - 1
        self.refresh_polygon_list()
        self.redraw_canvas()
        self.save_current_annotations(silent=True)
        self._notify(f"已新增轮廓：{self.current_image_name()}")

    def delete_selected_polygon(self) -> None:
        if self.selected_polygon_index is None or not (0 <= self.selected_polygon_index < len(self.polygons)):
            return
        del self.polygons[self.selected_polygon_index]
        if self.polygons:
            self.selected_polygon_index = min(self.selected_polygon_index, len(self.polygons) - 1)
        else:
            self.selected_polygon_index = None
        self.refresh_polygon_list()
        self.redraw_canvas()
        self.save_current_annotations(silent=True)
        self._notify(f"已删除轮廓：{self.current_image_name()}")

    def export_training_dataset(self) -> None:
        if not self._ensure_project_ready():
            return
        self.save_current_annotations(silent=True)
        assert self.project_dir is not None
        self.on_export_request(self.project_dir, list(self.class_names))
        self._notify("已把分割项目送去多任务数据整理。")

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 200),
                max(self.canvas.winfo_height() // 2, 120),
                text="选择图片目录后，这里会显示当前图片和分割轮廓。",
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

        for index, polygon in enumerate(self.polygons):
            self._draw_polygon(polygon, selected=index == self.selected_polygon_index)
        if self.current_points:
            self._draw_temp_polygon()

    def _mask_color(self, class_id: int) -> str:
        return MASK_COLORS[class_id % len(MASK_COLORS)]

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

    def _draw_polygon(self, polygon: AnnotationPolygon, *, selected: bool) -> None:
        if len(polygon.points) < 2:
            return
        flat_points: list[float] = []
        for x, y in polygon.points:
            cx, cy = self._image_to_canvas(x, y)
            flat_points.extend([cx, cy])
        color = self._mask_color(polygon.class_id)
        self.canvas.create_polygon(
            *flat_points,
            outline=color,
            fill=PRIMARY_SOFT if selected else "",
            width=3 if selected else 2,
        )
        anchor_x, anchor_y = self._image_to_canvas(*polygon.points[0])
        class_name = self.class_names[polygon.class_id] if polygon.class_id < len(self.class_names) else f"class{polygon.class_id}"
        self.canvas.create_rectangle(anchor_x, anchor_y - 22, anchor_x + max(94, len(class_name) * 10), anchor_y, fill=color, outline=color)
        self.canvas.create_text(anchor_x + 8, anchor_y - 11, text=class_name, fill="white", anchor="w", font=("Microsoft YaHei UI", 9, "bold"))
        if selected:
            for x, y in polygon.points:
                cx, cy = self._image_to_canvas(x, y)
                self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color, width=2)

    def _draw_temp_polygon(self) -> None:
        flat_points: list[float] = []
        color = self._mask_color(self._selected_class_id())
        for x, y in self.current_points:
            cx, cy = self._image_to_canvas(x, y)
            flat_points.extend([cx, cy])
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color, width=2)
        if len(flat_points) >= 4:
            self.canvas.create_line(*flat_points, fill=color, width=2, dash=(5, 3))

    def _hit_selected_vertex(self, canvas_x: float, canvas_y: float) -> int | None:
        if self.selected_polygon_index is None or not (0 <= self.selected_polygon_index < len(self.polygons)):
            return None
        points = self.polygons[self.selected_polygon_index].points
        for index, (x, y) in enumerate(points):
            cx, cy = self._image_to_canvas(x, y)
            if abs(canvas_x - cx) <= 8 and abs(canvas_y - cy) <= 8:
                return index
        return None

    def _find_polygon_at(self, image_x: float, image_y: float) -> int | None:
        for index in reversed(range(len(self.polygons))):
            if self._point_in_polygon((image_x, image_y), self.polygons[index].points):
                return index
        return None

    def _point_in_polygon(self, point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        if len(polygon) < 3:
            return False
        x_value, y_value = point
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            if (current_y > y_value) != (previous_y > y_value):
                denominator = previous_y - current_y
                if abs(denominator) > 1e-6:
                    cross_x = (previous_x - current_x) * (y_value - current_y) / denominator + current_x
                    if x_value < cross_x:
                        inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside

    def on_canvas_press(self, event: tk.Event[tk.Canvas]) -> None:
        if not self._ensure_project_ready():
            return
        self.canvas.focus_set()
        image_x, image_y = self._canvas_to_image(event.x, event.y)

        if not self.draw_mode_var.get():
            hit_vertex = self._hit_selected_vertex(event.x, event.y)
            if hit_vertex is not None:
                self.dragging_vertex_index = hit_vertex
                return
            polygon_index = self._find_polygon_at(image_x, image_y)
            self.selected_polygon_index = polygon_index
            self.refresh_polygon_list()
            self.redraw_canvas()
            return

        self.current_points.append((image_x, image_y))
        self.redraw_canvas()
        self._notify(f"当前轮廓已添加 {len(self.current_points)} 个点。")

    def on_canvas_double_click(self, _event: tk.Event[tk.Canvas]) -> None:
        if self.draw_mode_var.get() and len(self.current_points) >= 3:
            self.finish_current_polygon()

    def on_canvas_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.original_image is None:
            return
        if self.dragging_vertex_index is None or self.selected_polygon_index is None:
            return
        if not (0 <= self.selected_polygon_index < len(self.polygons)):
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        polygon = self.polygons[self.selected_polygon_index]
        points = list(polygon.points)
        points[self.dragging_vertex_index] = (image_x, image_y)
        polygon.points = points
        self.redraw_canvas()

    def on_canvas_release(self, _event: tk.Event[tk.Canvas]) -> None:
        if self.dragging_vertex_index is None:
            return
        self.dragging_vertex_index = None
        self.save_current_annotations(silent=True)
        self._notify(f"已更新轮廓边缘：{self.current_image_name()}")

    def on_canvas_right_click(self, _event: tk.Event[tk.Canvas]) -> None:
        if self.draw_mode_var.get():
            self.undo_last_point()
