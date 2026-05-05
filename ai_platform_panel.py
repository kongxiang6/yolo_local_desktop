from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

from ai_platform_support import AI_CAPABILITIES, AiCapability, TileReport, slice_large_images
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
WARNING = "#f59e0b"


class AiPlatformPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_detect_auto_label_request: Callable[[Path, str, dict[str, object]], None],
        on_open_workspace: Callable[[str, Path], None],
        on_open_model_hub: Callable[[], None],
        get_detect_project: Callable[[], Path | None],
        get_segment_project: Callable[[], Path | None],
        local_state_dir: Path,
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_detect_auto_label_request = on_detect_auto_label_request
        self.on_open_workspace = on_open_workspace
        self.on_open_model_hub = on_open_model_hub
        self.get_detect_project = get_detect_project
        self.get_segment_project = get_segment_project
        self.local_state_dir = local_state_dir

        self.capability_id = "detect_auto"
        self.source_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="yolo11n.pt")
        self.prompt_var = tk.StringVar(value="")
        self.detect_conf_var = tk.StringVar(value="0.25")
        self.detect_iou_var = tk.StringVar(value="0.70")
        self.detect_imgsz_var = tk.StringVar(value="640")
        self.detect_device_var = tk.StringVar(value="")
        self.tile_size_var = tk.StringVar(value="1280")
        self.tile_overlap_var = tk.StringVar(value="160")
        self.tile_output_var = tk.StringVar(value="")
        self.tile_target_workspace_var = tk.StringVar(value="detect")
        self.status_var = tk.StringVar(value="这里是第三期的 AI 平台入口，既能继续跑现有能力，也给后续文本提示/OCR扩展预留了统一界面。")
        self.detail_var = tk.StringVar(value="")
        self.capability_summary_var = tk.StringVar(value="")
        self.pending_output_dir: Path | None = None

        self._worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker_running = False

        self._build_ui()
        self.set_capability("detect_auto")

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = tk.Frame(self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="综合 AI 标注平台", bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text="这里集中放“模型辅助”“文本提示”“大图切片”“OCR 准备”这类第三期能力。当前先把能立即落地的能力做实，再为后续复杂模型接入保留统一入口。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=820,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

        capability_bar = tk.Frame(self, bg=CARD_BG)
        capability_bar.grid(row=1, column=0, sticky="ew", padx=10)
        self.capability_buttons: dict[str, tk.Button] = {}
        for index, item in enumerate(AI_CAPABILITIES):
            capability_bar.grid_columnconfigure(index, weight=1)
            button = tk.Button(
                capability_bar,
                text=item.label,
                command=lambda capability_id=item.capability_id: self.set_capability(capability_id),
                bg=CARD_BG,
                fg=TEXT,
                activebackground=PRIMARY_SOFT,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=PRIMARY,
                font=("Microsoft YaHei UI", 10, "bold"),
                pady=8,
                cursor="hand2",
            )
            button.grid(row=0, column=index, sticky="ew", padx=(0, 6) if index < len(AI_CAPABILITIES) - 1 else 0)
            self.capability_buttons[item.capability_id] = button

        self.body_scroll = VerticalScrolledFrame(self, bg=CARD_BG)
        self.body_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 10))
        body = self.body_scroll.content
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=CARD_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        source_box = self._side_box(left, "数据来源")
        source_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_source_box(source_box)

        config_box = self._side_box(left, "当前能力参数")
        config_box.grid(row=1, column=0, sticky="ew")
        self._build_config_box(config_box)

        right = tk.Frame(body, bg=CARD_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        info_box = self._side_box(right, "能力说明")
        info_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(info_box, textvariable=self.capability_summary_var, bg=CARD_BG, fg=TEXT, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(info_box, textvariable=self.detail_var, bg=CARD_BG, fg=TEXT_MUTED, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=12, pady=(0, 12))

        prompt_box = self._side_box(right, "提示词 / 任务说明")
        prompt_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.prompt_text = tk.Text(
            prompt_box,
            height=8,
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
        self.prompt_text.pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(
            prompt_box,
            text="检测自动标注和大图切片当前不强依赖这里；文本提示 / OCR 预留能力会把这里的说明保存成任务简报，方便后续继续接模型。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

        action_box = self._side_box(right, "执行")
        action_box.grid(row=2, column=0, sticky="ew")
        tk.Label(action_box, textvariable=self.status_var, bg=CARD_BG, fg=SUCCESS, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=12, pady=(12, 8))
        action_row = tk.Frame(action_box, bg=CARD_BG)
        action_row.pack(fill="x", padx=12, pady=(0, 8))
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=1)
        self.primary_action_button = self._primary_button(action_row, "执行当前 AI 能力", self.run_primary_action)
        self.primary_action_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(action_row, "去模型中心", self.on_open_model_hub).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.secondary_hint_label = tk.Label(action_box, text="", bg=CARD_BG, fg=WARNING, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 9))
        self.secondary_hint_label.pack(fill="x", padx=12, pady=(0, 12))

    def _build_source_box(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=12)
        grid.grid_columnconfigure(0, weight=1)

        tk.Label(grid, text="源目录", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Entry(
            grid,
            textvariable=self.source_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", pady=(6, 8), ipady=6)

        source_row = tk.Frame(grid, bg=CARD_BG)
        source_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        source_row.grid_columnconfigure(0, weight=1)
        source_row.grid_columnconfigure(1, weight=1)
        source_row.grid_columnconfigure(2, weight=1)
        self._small_button(source_row, "选择目录", self.pick_source_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(source_row, "用检测工作区", self.use_detect_source).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self._small_button(source_row, "用分割工作区", self.use_segment_source).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        tk.Label(grid, text="模型 / 权重", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=3, column=0, sticky="w")
        tk.Entry(
            grid,
            textvariable=self.model_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=4, column=0, sticky="ew", pady=(6, 8), ipady=6)

        model_row = tk.Frame(grid, bg=CARD_BG)
        model_row.grid(row=5, column=0, sticky="ew")
        model_row.grid_columnconfigure(0, weight=1)
        model_row.grid_columnconfigure(1, weight=1)
        self._small_button(model_row, "选择本地模型", self.pick_model_file).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(model_row, "去模型中心挑选", self.on_open_model_hub).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_config_box(self, parent: tk.Frame) -> None:
        self.config_body = tk.Frame(parent, bg=CARD_BG)
        self.config_body.pack(fill="x", padx=12, pady=12)
        self.config_body.grid_columnconfigure(1, weight=1)

        self.detect_widgets: list[tk.Widget] = []
        self.tile_widgets: list[tk.Widget] = []

        self.detect_widgets.extend(
            self._labeled_entry(self.config_body, "conf", self.detect_conf_var, row=0, hint="越高越严格，误检少一点。")
        )
        self.detect_widgets.extend(
            self._labeled_entry(self.config_body, "iou", self.detect_iou_var, row=1, hint="重复框合并阈值。")
        )
        self.detect_widgets.extend(
            self._labeled_entry(self.config_body, "imgsz", self.detect_imgsz_var, row=2, hint="输入尺寸，常用 640。")
        )
        self.detect_widgets.extend(
            self._labeled_entry(self.config_body, "device", self.detect_device_var, row=3, hint="留空自动；也可填 cpu / 0。")
        )

        start_row = 4
        self.tile_widgets.extend(self._labeled_entry(self.config_body, "切片尺寸", self.tile_size_var, row=start_row, hint="大图切成多大的小图块。"))
        self.tile_widgets.extend(self._labeled_entry(self.config_body, "重叠尺寸", self.tile_overlap_var, row=start_row + 1, hint="相邻小图交叠区域，避免边缘漏目标。"))

        tile_output_label = tk.Label(self.config_body, text="切片输出目录", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10))
        tile_output_label.grid(row=start_row + 2, column=0, sticky="w", pady=(0, 10))
        output_row = tk.Frame(self.config_body, bg=CARD_BG)
        output_row.grid(row=start_row + 2, column=1, sticky="ew", pady=(0, 10))
        output_row.grid_columnconfigure(0, weight=1)
        output_entry = tk.Entry(
            output_row,
            textvariable=self.tile_output_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        output_entry.grid(row=0, column=0, sticky="ew", ipady=6)
        output_button = self._small_button(output_row, "选择", self.pick_tile_output_dir)
        output_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.tile_widgets.extend([tile_output_label, output_row])

        tile_target_label = tk.Label(self.config_body, text="切完后打开到", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10))
        tile_target_label.grid(row=start_row + 3, column=0, sticky="w")
        target_row = tk.Frame(self.config_body, bg=CARD_BG)
        target_row.grid(row=start_row + 3, column=1, sticky="w")
        target_detect = tk.Radiobutton(target_row, text="检测工作区", value="detect", variable=self.tile_target_workspace_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10))
        target_detect.pack(side="left")
        target_segment = tk.Radiobutton(target_row, text="分割工作区", value="segment", variable=self.tile_target_workspace_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10))
        target_segment.pack(side="left", padx=(12, 0))
        self.tile_widgets.extend([tile_target_label, target_row])

    def _labeled_entry(self, parent: tk.Widget, label_text: str, variable: tk.StringVar, *, row: int, hint: str) -> list[tk.Widget]:
        label = tk.Label(parent, text=label_text, bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10))
        label.grid(row=row, column=0, sticky="w", pady=(0, 10))
        wrapper = tk.Frame(parent, bg=CARD_BG)
        wrapper.grid(row=row, column=1, sticky="ew", pady=(0, 10))
        wrapper.grid_columnconfigure(0, weight=1)
        entry = tk.Entry(
            wrapper,
            textvariable=variable,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        )
        entry.grid(row=0, column=0, sticky="ew", ipady=6)
        hint_label = tk.Label(wrapper, text=hint, bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 9))
        hint_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        return [label, wrapper]

    def _side_box(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(frame, text=title, bg=PRIMARY_SOFT, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold"), anchor="w", padx=12, pady=10).pack(fill="x")
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

    def set_capability(self, capability_id: str) -> None:
        capability = next(item for item in AI_CAPABILITIES if item.capability_id == capability_id)
        self.capability_id = capability_id
        self.capability_summary_var.set(f"当前能力：{capability.label}\n状态：{'可直接使用' if capability.status == 'ready' else '已做界面预留'}\n\n{capability.summary}")
        self.detail_var.set(capability.detail)
        self.secondary_hint_label.configure(
            text="说明：带“预留”的能力本轮先提供统一入口和任务简报，后面继续接模型。" if capability.status != "ready" else "说明：当前能力已经可以直接使用。"
        )
        self.primary_action_button.configure(text="执行当前 AI 能力" if capability.status == "ready" else "生成任务简报")
        for item, button in self.capability_buttons.items():
            is_active = item == capability_id
            button.configure(
                bg=PRIMARY if is_active else CARD_BG,
                fg="white" if is_active else TEXT,
                highlightbackground=PRIMARY if is_active else BORDER,
            )
        for widget in self.detect_widgets:
            if capability_id == "detect_auto":
                widget.grid() if hasattr(widget, "grid") else None
            else:
                widget.grid_remove() if hasattr(widget, "grid_remove") else None
        for widget in self.tile_widgets:
            if capability_id == "large_image_tiling":
                widget.grid() if hasattr(widget, "grid") else None
            else:
                widget.grid_remove() if hasattr(widget, "grid_remove") else None
        self.status_var.set(f"已切换到：{capability.label}")
        self.on_state_change()

    def pick_source_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择要处理的图片目录")
        if selected:
            self.source_var.set(selected)
            self._sync_default_tile_output()
            self._notify("已选择 AI 处理源目录。")

    def use_detect_source(self) -> None:
        folder = self.get_detect_project()
        if folder is None:
            messagebox.showwarning("没有检测目录", "检测工作区当前还没有打开目录。")
            return
        self.source_var.set(str(folder))
        self._sync_default_tile_output()
        self._notify("已读取检测工作区目录。")

    def use_segment_source(self) -> None:
        folder = self.get_segment_project()
        if folder is None:
            messagebox.showwarning("没有分割目录", "分割工作区当前还没有打开目录。")
            return
        self.source_var.set(str(folder))
        self._sync_default_tile_output()
        self._notify("已读取分割工作区目录。")

    def pick_model_file(self) -> None:
        path = filedialog.askopenfilename(title="选择模型文件", filetypes=[("模型文件", "*.pt *.onnx *.engine *.torchscript *.bin"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)
            self.on_state_change()

    def pick_tile_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择切片输出目录")
        if selected:
            self.tile_output_var.set(selected)
            self.on_state_change()

    def _sync_default_tile_output(self) -> None:
        if self.capability_id != "large_image_tiling":
            return
        source_text = self.source_var.get().strip()
        if not source_text:
            return
        source_dir = Path(source_text).expanduser()
        if not self.tile_output_var.get().strip():
            self.tile_output_var.set(str(source_dir.parent / f"{source_dir.name}_tiles"))

    def set_selected_model(self, model_ref: str) -> None:
        self.model_var.set(model_ref)
        self._notify(f"已从模型中心选中：{Path(model_ref).name if model_ref else model_ref}")

    def run_primary_action(self) -> None:
        if self.capability_id == "detect_auto":
            self.run_detect_auto_label()
        elif self.capability_id == "large_image_tiling":
            self.run_large_image_tiling()
        else:
            self.save_task_brief()

    def _require_source_dir(self) -> Path | None:
        source_text = self.source_var.get().strip()
        if not source_text:
            messagebox.showwarning("缺少目录", "请先选择一个图片目录。")
            return None
        source_dir = Path(source_text).expanduser().resolve()
        if not source_dir.exists():
            messagebox.showwarning("目录不存在", f"找不到目录：\n{source_dir}")
            return None
        return source_dir

    def run_detect_auto_label(self) -> None:
        source_dir = self._require_source_dir()
        if source_dir is None:
            return
        model_ref = self.model_var.get().strip()
        if not model_ref:
            messagebox.showwarning("缺少模型", "请先填写模型或到模型中心选择。")
            return
        try:
            config: dict[str, object] = {
                "conf": float(self.detect_conf_var.get().strip() or "0.25"),
                "iou": float(self.detect_iou_var.get().strip() or "0.70"),
                "imgsz": int(self.detect_imgsz_var.get().strip() or "640"),
                "save": False,
                "verbose": False,
                "max_det": 300,
            }
        except ValueError:
            messagebox.showwarning("参数格式错误", "请检查 conf / iou / imgsz。")
            return
        device_text = self.detect_device_var.get().strip()
        if device_text:
            config["device"] = device_text
        self.status_var.set("正在把目录送去检测自动标注……")
        self.on_detect_auto_label_request(source_dir, model_ref, config)
        self.on_open_workspace("detect", source_dir)
        self._notify("已切到检测工作区，并触发批量自动标注。")

    def run_large_image_tiling(self) -> None:
        if self._worker_running:
            return
        source_dir = self._require_source_dir()
        if source_dir is None:
            return
        output_text = self.tile_output_var.get().strip()
        if not output_text:
            self._sync_default_tile_output()
            output_text = self.tile_output_var.get().strip()
        if not output_text:
            messagebox.showwarning("缺少输出目录", "请先设置切片输出目录。")
            return
        output_dir = Path(output_text).expanduser().resolve()
        try:
            tile_size = int(self.tile_size_var.get().strip() or "1280")
            overlap = int(self.tile_overlap_var.get().strip() or "160")
        except ValueError:
            messagebox.showwarning("参数格式错误", "切片尺寸和重叠尺寸都要填整数。")
            return
        self.pending_output_dir = output_dir
        self._worker_running = True
        self.status_var.set("正在切片大图，请稍等……")
        worker = threading.Thread(
            target=self._run_tile_worker,
            args=(source_dir, output_dir, tile_size, overlap),
            daemon=True,
        )
        worker.start()
        self.after(80, self._poll_worker_queue)

    def _run_tile_worker(self, source_dir: Path, output_dir: Path, tile_size: int, overlap: int) -> None:
        try:
            report = slice_large_images(
                source_dir=source_dir,
                output_dir=output_dir,
                tile_size=tile_size,
                overlap=overlap,
                progress=lambda message: self._worker_queue.put(("progress", message)),
            )
            self._worker_queue.put(("result", report))
        except Exception as exc:
            self._worker_queue.put(("error", str(exc)))

    def _poll_worker_queue(self) -> None:
        if not self._worker_running:
            return
        try:
            kind, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_worker_queue)
            return

        if kind == "progress":
            self.status_var.set(str(payload))
            self.after(80, self._poll_worker_queue)
            return

        self._worker_running = False
        if kind == "error":
            self.status_var.set("大图切片失败。")
            messagebox.showerror("大图切片失败", str(payload))
            return

        report = payload
        assert isinstance(report, TileReport)
        open_dir = report.output_dir / "images"
        if not open_dir.exists():
            open_dir = report.output_dir
        self.pending_output_dir = open_dir
        self.status_var.set(f"切片完成，共生成 {report.tile_count} 张小图。")
        self.on_open_workspace(self.tile_target_workspace_var.get(), open_dir)
        self._notify("大图切片完成，已自动切到对应标注工作区。")

    def save_task_brief(self) -> None:
        source_dir = self._require_source_dir()
        if source_dir is None:
            return
        prompt_text = self.prompt_text.get("1.0", "end").strip()
        if not prompt_text:
            messagebox.showwarning("缺少说明", "请先写一下这次想要的文本提示 / OCR 目标说明。")
            return
        capability = next(item for item in AI_CAPABILITIES if item.capability_id == self.capability_id)
        task_dir = self.local_state_dir / "ai_task_briefs"
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{self.capability_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "capability_id": capability.capability_id,
            "capability_label": capability.label,
            "source_dir": str(source_dir),
            "prompt_text": prompt_text,
            "model_ref": self.model_var.get().strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已生成任务简报：{output_path.name}")
        self.pending_output_dir = output_path
        messagebox.showinfo("任务简报已生成", f"已保存到：\n{output_path}\n\n后续接入对应模型时，可以直接复用这个任务说明。")
        self.on_state_change()

    def current_project_dir(self) -> Path | None:
        source_text = self.source_var.get().strip()
        if source_text:
            return Path(source_text).expanduser().resolve()
        return None

    def output_preview_dir(self) -> str:
        if self.pending_output_dir is not None:
            return str(self.pending_output_dir)
        if self.capability_id == "large_image_tiling":
            return self.tile_output_var.get().strip() or "未选择"
        return self.source_var.get().strip() or "未选择"

    def active_capability_label(self) -> str:
        capability = next(item for item in AI_CAPABILITIES if item.capability_id == self.capability_id)
        return capability.label

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()
