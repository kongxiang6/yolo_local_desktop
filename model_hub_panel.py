from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

from ai_platform_support import (
    OFFICIAL_MODEL_SUGGESTIONS,
    ModelRecord,
    default_model_scan_roots,
    load_model_hub_state,
    normalize_model_roots,
    save_model_hub_state,
    scan_model_files,
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


class ModelHubPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        project_root: Path,
        state_path: Path,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_apply_model: Callable[[str, str], None],
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.project_root = project_root
        self.state_path = state_path
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_apply_model = on_apply_model

        self.status_var = tk.StringVar(value="这里集中管理模型来源，方便后面第三期继续扩多模型 AI 标注。")
        self.current_root_var = tk.StringVar(value="")
        self.local_models: list[ModelRecord] = []
        self.scan_roots: list[str] = []
        self._scan_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._scan_running = False

        self._build_ui()
        self._load_state()
        if not self.scan_roots:
            self.scan_roots = [str(self.project_root)]
            self._save_state()
        self._refresh_root_list()
        self.refresh_official_models()
        self._show_idle_model_message()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="模型中心", bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text="这里统一看官方推荐模型和本机已有模型，后续第三期继续接更多后端时，就从这里挑模型、管理来源、分发到 AI 工作流。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=820,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

        self.body_scroll = VerticalScrolledFrame(self, bg=CARD_BG)
        self.body_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body = self.body_scroll.content
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=CARD_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        root_box = self._side_box(left, "扫描目录")
        root_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_root_box(root_box)

        local_box = self._side_box(left, "本机已发现模型")
        local_box.grid(row=1, column=0, sticky="ew")
        self._build_local_box(local_box)

        right = tk.Frame(body, bg=CARD_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        official_box = self._side_box(right, "官方推荐模型")
        official_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_official_box(official_box)

        action_box = self._side_box(right, "操作")
        action_box.grid(row=1, column=0, sticky="ew")
        tk.Label(action_box, textvariable=self.status_var, bg=CARD_BG, fg=SUCCESS, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=12, pady=(12, 8))
        action_row = tk.Frame(action_box, bg=CARD_BG)
        action_row.pack(fill="x", padx=12, pady=(0, 8))
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=1)
        self._primary_button(action_row, "应用选中本地模型", self.apply_selected_local_model).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(action_row, "应用选中官方模型", self.apply_selected_official_model).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        tk.Label(
            action_box,
            text="提示：应用后会直接回填到 AI 平台模型输入框；如果是检测模型，也会顺手同步到检测自动标注区。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

    def _build_root_box(self, parent: tk.Frame) -> None:
        wrapper = tk.Frame(parent, bg=CARD_BG)
        wrapper.pack(fill="x", padx=12, pady=12)
        wrapper.grid_columnconfigure(0, weight=1)

        tk.Entry(
            wrapper,
            textvariable=self.current_root_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=0, sticky="ew", ipady=6)
        self._small_button(wrapper, "选择目录", self.pick_root_dir).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row = tk.Frame(wrapper, bg=CARD_BG)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, weight=1)
        self._small_button(row, "加入扫描列表", self.add_scan_root).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(row, "加入常用目录", self.add_default_roots).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self._small_button(row, "重新扫描", self.scan_models).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.root_list = tk.Listbox(
            wrapper,
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
        self.root_list.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._small_button(wrapper, "删除选中目录", self.remove_selected_root).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_local_box(self, parent: tk.Frame) -> None:
        wrapper = tk.Frame(parent, bg=CARD_BG)
        wrapper.pack(fill="both", expand=True, padx=12, pady=12)
        self.local_model_list = tk.Listbox(
            wrapper,
            height=16,
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
        self.local_model_list.pack(fill="both", expand=True)

    def _build_official_box(self, parent: tk.Frame) -> None:
        wrapper = tk.Frame(parent, bg=CARD_BG)
        wrapper.pack(fill="both", expand=True, padx=12, pady=12)
        self.official_model_list = tk.Listbox(
            wrapper,
            height=16,
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
        self.official_model_list.pack(fill="both", expand=True)

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

    def _load_state(self) -> None:
        payload = load_model_hub_state(self.state_path)
        roots = payload.get("roots")
        self.scan_roots = normalize_model_roots(roots if isinstance(roots, list) else [])

    def _save_state(self) -> None:
        save_model_hub_state(self.state_path, {"roots": normalize_model_roots(self.scan_roots)})

    def pick_root_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择模型搜索目录")
        if selected:
            self.current_root_var.set(selected)

    def add_scan_root(self) -> None:
        root_text = self.current_root_var.get().strip()
        if not root_text:
            return
        self.scan_roots = normalize_model_roots(self.scan_roots + [root_text])
        self._refresh_root_list()
        self._save_state()
        self._notify("已加入模型扫描目录。")

    def add_default_roots(self) -> None:
        extra = [str(path) for path in default_model_scan_roots(self.project_root)]
        self.scan_roots = normalize_model_roots(self.scan_roots + extra)
        self._refresh_root_list()
        self._save_state()
        self._notify("已加入常用目录。")

    def remove_selected_root(self) -> None:
        selection = self.root_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        del self.scan_roots[index]
        self._refresh_root_list()
        self._save_state()
        self._notify("已移除选中的扫描目录。")

    def _refresh_root_list(self) -> None:
        self.root_list.delete(0, "end")
        for item in self.scan_roots:
            self.root_list.insert("end", item)
        self.on_state_change()

    def refresh_official_models(self) -> None:
        self.official_model_list.delete(0, "end")
        for task_id, items in OFFICIAL_MODEL_SUGGESTIONS.items():
            self.official_model_list.insert("end", f"[{task_id}]")
            for model_name, summary in items:
                self.official_model_list.insert("end", f"  {model_name}  ·  {summary}")

    def _show_idle_model_message(self) -> None:
        self.local_model_list.delete(0, "end")
        self.local_model_list.insert("end", "点击“重新扫描”后再读取本机模型，避免启动时卡住界面。")

    def scan_models(self) -> None:
        if self._scan_running:
            return
        self._scan_running = True
        self.local_model_list.delete(0, "end")
        self.local_model_list.insert("end", "正在扫描模型文件，请稍等……")
        self.status_var.set("正在后台扫描模型文件……")
        worker = threading.Thread(target=self._scan_models_worker, daemon=True)
        worker.start()
        self.after(80, self._poll_scan_queue)

    def _scan_models_worker(self) -> None:
        try:
            result = scan_model_files([Path(item) for item in self.scan_roots], limit=300)
            self._scan_queue.put(("result", result))
        except Exception as exc:
            self._scan_queue.put(("error", str(exc)))

    def _poll_scan_queue(self) -> None:
        if not self._scan_running:
            return
        try:
            kind, payload = self._scan_queue.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_scan_queue)
            return

        self._scan_running = False
        if kind == "error":
            self.local_models = []
            self._show_idle_model_message()
            messagebox.showerror("模型扫描失败", str(payload))
            return

        self.local_models = list(payload)
        self.local_model_list.delete(0, "end")
        if not self.local_models:
            self.local_model_list.insert("end", "没有扫描到模型文件，可以加目录后重新扫描。")
        else:
            for item in self.local_models:
                self.local_model_list.insert("end", item.display_name)
        self._notify(f"模型扫描完成，共发现 {len(self.local_models)} 个模型文件。")

    def _selected_local_model(self) -> ModelRecord | None:
        selection = self.local_model_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self.local_models):
            return None
        return self.local_models[index]

    def apply_selected_local_model(self) -> None:
        record = self._selected_local_model()
        if record is None:
            messagebox.showwarning("未选中模型", "请先在本机模型列表里选中一个模型。")
            return
        self.on_apply_model(str(record.path), record.task_hint)
        self._notify(f"已应用本地模型：{record.path.name}")

    def apply_selected_official_model(self) -> None:
        selection = self.official_model_list.curselection()
        if not selection:
            messagebox.showwarning("未选中模型", "请先在官方推荐列表里选中一个模型。")
            return
        text = self.official_model_list.get(selection[0]).strip()
        if not text or text.startswith("["):
            messagebox.showwarning("请选择具体模型", "当前选中的是任务标题，不是具体模型。")
            return
        model_name = text.split("·", 1)[0].strip()
        task_hint = "detect"
        for task_id, items in OFFICIAL_MODEL_SUGGESTIONS.items():
            if any(model_name.startswith(item[0]) for item in items):
                task_hint = task_id
                break
        self.on_apply_model(model_name, task_hint)
        self._notify(f"已应用官方模型：{model_name}")

    def current_project_dir(self) -> Path | None:
        return self.project_root

    def output_preview_dir(self) -> str:
        model = self._selected_local_model()
        if model is not None:
            return str(model.path)
        return "模型中心"

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()
