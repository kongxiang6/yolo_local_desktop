from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable


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

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class ExtractReport:
    output_dir: Path
    extracted_count: int
    frame_step: int
    fps: float
    frame_count: int


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("当前环境缺少 OpenCV，暂时无法做视频抽帧。请先完成运行环境配置。") from exc
    return cv2


def inspect_video(video_path: Path) -> VideoInfo:
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    duration = float(frame_count / fps) if fps > 0 else 0.0
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count, duration_seconds=duration)


def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    frame_step: int,
    progress: Callable[[str], None] | None = None,
) -> ExtractReport:
    if frame_step <= 0:
        raise ValueError("抽帧间隔必须大于 0。")

    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    extracted_count = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            extracted_count += 1
            target_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(target_path), frame)
            if progress is not None and (extracted_count == 1 or extracted_count % 20 == 0):
                progress(f"已抽出 {extracted_count} 张，当前源帧位置 {frame_index + 1}/{frame_count or '?'}")
        frame_index += 1

    capture.release()
    meta_path = output_dir / "video_annotation_project.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_video": str(video_path),
                "frame_step": frame_step,
                "fps": fps,
                "frame_count": frame_count,
                "extracted_count": extracted_count,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ExtractReport(
        output_dir=output_dir,
        extracted_count=extracted_count,
        frame_step=frame_step,
        fps=fps,
        frame_count=frame_count,
    )


class VideoAnnotationPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_open_project: Callable[[str, Path], None],
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_open_project = on_open_project

        self.video_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="")
        self.frame_step_var = tk.StringVar(value="10")
        self.target_workspace_var = tk.StringVar(value="detect")
        self.status_var = tk.StringVar(value="先选择视频，再抽帧进入检测或分割标注。")
        self.video_info_var = tk.StringVar(value="还没有读取视频信息。")

        self._worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker_running = False
        self.last_output_dir: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="视频标注准备", bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text="第二期先做稳定的“抽帧 → 标注”闭环。后面第三期会继续在这里接关键帧传播、跟踪沿用和更强的自动辅助。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=760,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

        body = tk.Frame(self, bg=CARD_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=CARD_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        source_box = self._side_box(left, "视频和抽帧目录")
        source_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_source_box(source_box)

        options_box = self._side_box(left, "抽帧设置")
        options_box.grid(row=1, column=0, sticky="ew")
        self._build_options_box(options_box)

        right = tk.Frame(body, bg=CARD_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        info_box = self._side_box(right, "视频信息")
        info_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(info_box, textvariable=self.video_info_var, bg=CARD_BG, fg=TEXT, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=12, pady=12)

        guide_box = self._side_box(right, "使用建议")
        guide_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            guide_box,
            text="如果视频变化不快，可以先用“每 10 帧取 1 张”；动作很快时，再改成每 3~5 帧取 1 张。抽帧后会直接打开到你选择的标注工作区。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=12)

        action_box = self._side_box(right, "开始处理")
        action_box.grid(row=2, column=0, sticky="ew")
        tk.Label(action_box, textvariable=self.status_var, bg=CARD_BG, fg=SUCCESS, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=12, pady=(12, 8))
        self._primary_button(action_box, "开始抽帧并进入标注", self.start_extract).pack(fill="x", padx=12, pady=(0, 12))

    def _build_source_box(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=12)
        grid.grid_columnconfigure(0, weight=1)

        tk.Label(grid, text="视频文件", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Entry(
            grid,
            textvariable=self.video_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", pady=(6, 8), ipady=6)
        row = tk.Frame(grid, bg=CARD_BG)
        row.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        self._small_button(row, "选择视频", self.pick_video_file).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(row, "读取视频信息", self.refresh_video_info).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        tk.Label(grid, text="抽帧输出目录", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=3, column=0, sticky="w")
        tk.Entry(
            grid,
            textvariable=self.output_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=4, column=0, sticky="ew", pady=(6, 8), ipady=6)
        output_row = tk.Frame(grid, bg=CARD_BG)
        output_row.grid(row=5, column=0, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        output_row.grid_columnconfigure(1, weight=1)
        self._small_button(output_row, "选择输出目录", self.pick_output_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(output_row, "使用默认输出目录", self.fill_default_output).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_options_box(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=12)
        grid.grid_columnconfigure(1, weight=1)

        tk.Label(grid, text="每隔多少帧取 1 张", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w", pady=(0, 10))
        tk.Entry(
            grid,
            textvariable=self.frame_step_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=1, sticky="ew", pady=(0, 10), ipady=6)

        tk.Label(grid, text="抽帧后打开到", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=1, column=0, sticky="w")
        target_row = tk.Frame(grid, bg=CARD_BG)
        target_row.grid(row=1, column=1, sticky="w")
        tk.Radiobutton(target_row, text="检测框标注", value="detect", variable=self.target_workspace_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10)).pack(side="left")
        tk.Radiobutton(target_row, text="交互式分割", value="segment", variable=self.target_workspace_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(12, 0))

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

    def pick_video_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v *.webm"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        self.video_var.set(selected)
        if not self.output_var.get().strip():
            self.fill_default_output()
        self.refresh_video_info()

    def pick_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择抽帧输出目录")
        if selected:
            self.output_var.set(selected)
            self.on_state_change()

    def fill_default_output(self) -> None:
        video_text = self.video_var.get().strip()
        if not video_text:
            return
        video_path = Path(video_text).expanduser()
        self.output_var.set(str(video_path.parent / f"{video_path.stem}_frames"))
        self.on_state_change()

    def refresh_video_info(self) -> None:
        video_path = self._get_video_path()
        if video_path is None:
            return
        try:
            info = inspect_video(video_path)
        except Exception as exc:
            messagebox.showerror("读取视频失败", str(exc))
            return
        duration_text = f"{info.duration_seconds:.1f} 秒" if info.duration_seconds else "未知"
        self.video_info_var.set(
            f"分辨率：{info.width} x {info.height}\n"
            f"帧率：{info.fps:.2f}\n"
            f"总帧数：{info.frame_count}\n"
            f"时长：{duration_text}"
        )
        self._notify("视频信息已读取。")

    def _get_video_path(self) -> Path | None:
        video_text = self.video_var.get().strip()
        if not video_text:
            messagebox.showwarning("缺少视频", "请先选择一个视频文件。")
            return None
        video_path = Path(video_text).expanduser().resolve()
        if not video_path.exists():
            messagebox.showwarning("视频不存在", f"找不到文件：\n{video_path}")
            return None
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            if not messagebox.askyesno("文件后缀不常见", "这个文件后缀不是常见视频格式，仍然继续尝试打开吗？"):
                return None
        return video_path

    def start_extract(self) -> None:
        if self._worker_running:
            return
        video_path = self._get_video_path()
        if video_path is None:
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            self.fill_default_output()
            output_text = self.output_var.get().strip()
        try:
            frame_step = int(self.frame_step_var.get().strip() or "10")
        except ValueError:
            messagebox.showwarning("参数错误", "抽帧间隔要填整数。")
            return
        output_dir = Path(output_text).expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            if not messagebox.askyesno("输出目录非空", "输出目录里已经有文件，是否继续抽帧并覆盖同名图片？"):
                return

        self._worker_running = True
        self.status_var.set("正在抽帧，请稍等……")
        worker = threading.Thread(
            target=self._run_extract_worker,
            args=(video_path, output_dir, frame_step),
            daemon=True,
        )
        worker.start()
        self.after(80, self._poll_worker_queue)

    def _run_extract_worker(self, video_path: Path, output_dir: Path, frame_step: int) -> None:
        try:
            report = extract_video_frames(
                video_path=video_path,
                output_dir=output_dir,
                frame_step=frame_step,
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
            self.status_var.set("抽帧失败。")
            messagebox.showerror("抽帧失败", str(payload))
            return

        report = payload
        assert isinstance(report, ExtractReport)
        self.last_output_dir = report.output_dir
        self.status_var.set(f"抽帧完成，共生成 {report.extracted_count} 张图片。")
        self.video_info_var.set(
            f"{self.video_info_var.get()}\n\n本次抽帧：每 {report.frame_step} 帧取 1 张\n输出图片：{report.extracted_count} 张\n输出目录：{report.output_dir}"
        )
        self.on_open_project(self.target_workspace_var.get(), report.output_dir)
        self._notify("抽帧完成，已自动切到标注工作区。")

    def current_project_dir(self) -> Path | None:
        if self.last_output_dir is not None:
            return self.last_output_dir
        output_text = self.output_var.get().strip()
        if output_text:
            return Path(output_text).expanduser().resolve()
        return None

    def output_preview_dir(self) -> str:
        if self.last_output_dir is not None:
            return str(self.last_output_dir)
        return self.output_var.get().strip() or "未选择"

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()
