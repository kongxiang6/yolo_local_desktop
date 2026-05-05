from __future__ import annotations

import queue
import random
import shutil
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import yaml

from annotation_support import (
    IMAGE_SUFFIXES,
    ensure_class_names,
    infer_max_class_id_from_label_file,
    load_class_names,
    parse_class_names_text,
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

TASK_LABELS = {
    "detect": "目标检测",
    "segment": "实例分割",
    "classify": "图像分类",
    "pose": "姿态估计",
    "obb": "旋转框检测",
}

YOLO_LIKE_TASKS = ("detect", "segment", "pose", "obb")


@dataclass
class OrganizeReport:
    task_id: str
    output_path: Path
    sample_count: int
    train_count: int
    val_count: int
    class_names: list[str]
    warnings: list[str]


def guess_output_path(source_dir: Path, task_id: str) -> Path:
    suffix = {
        "detect": "_detect_dataset",
        "segment": "_segment_dataset",
        "classify": "_classify_dataset",
        "pose": "_pose_dataset",
        "obb": "_obb_dataset",
    }.get(task_id, "_dataset")
    return source_dir.parent / f"{source_dir.name}{suffix}"


def list_images_recursive(root: Path, *, skip_dir: Path | None = None) -> list[Path]:
    skip_dir = skip_dir.resolve() if skip_dir is not None else None
    results: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if skip_dir is not None:
            try:
                path.resolve().relative_to(skip_dir)
                continue
            except ValueError:
                pass
        results.append(path.resolve())
    return sorted(results)


def resolve_yolo_label_path(source_dir: Path, image_path: Path) -> Path | None:
    sibling = image_path.with_suffix(".txt")
    if sibling.exists():
        return sibling

    source_dir = source_dir.resolve()
    image_path = image_path.resolve()
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if images_dir.exists() and labels_dir.exists():
        try:
            relative = image_path.relative_to(images_dir)
        except ValueError:
            pass
        else:
            candidate = labels_dir / relative.with_suffix(".txt")
            if candidate.exists():
                return candidate
    return None


def split_items(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if len(items) <= 1 or val_ratio <= 0:
        return list(items), []
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _link_or_copy_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode == "hardlink":
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _read_explicit_class_names(raw_text: str) -> list[str]:
    return parse_class_names_text(raw_text)


def _resolve_class_names(source_dir: Path, explicit_names: list[str], label_paths: list[Path]) -> list[str]:
    max_class_id = max((infer_max_class_id_from_label_file(path) for path in label_paths), default=-1)
    if explicit_names:
        return ensure_class_names(explicit_names, max_class_id=max_class_id)
    auto_names = load_class_names(source_dir)
    if auto_names:
        return ensure_class_names(auto_names, max_class_id=max_class_id)
    return ensure_class_names([], max_class_id=max_class_id)


def prepare_yolo_task_dataset(
    *,
    task_id: str,
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    class_names: list[str],
    copy_mode: str,
    strict: bool,
) -> OrganizeReport:
    output_dir = output_dir.resolve()
    images = list_images_recursive(source_dir, skip_dir=output_dir)
    if not images:
        raise ValueError("源目录里没有找到图片。")

    warnings: list[str] = []
    valid_samples: list[tuple[Path, Path]] = []
    label_paths: list[Path] = []
    for image_path in images:
        label_path = resolve_yolo_label_path(source_dir, image_path)
        if label_path is None:
            warnings.append(f"缺少同名标签：{image_path.name}")
            if strict:
                raise ValueError(f"找不到图片对应的标签：{image_path}")
            continue
        valid_samples.append((image_path, label_path))
        label_paths.append(label_path)

    if not valid_samples:
        raise ValueError("没有找到可整理的图片+标签配对。")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_dir in (output_dir / "images", output_dir / "labels"):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
    train_items, val_items = split_items([item[0] for item in valid_samples], val_ratio, seed)
    item_map = {image_path: label_path for image_path, label_path in valid_samples}
    resolved_names = _resolve_class_names(source_dir, class_names, label_paths)

    image_index = {path: index for index, (path, _label_path) in enumerate(valid_samples)}

    def write_split(split_name: str, split_items_list: list[Path]) -> None:
        for image_path in split_items_list:
            label_path = item_map[image_path]
            sample_index = image_index[image_path] + 1
            target_image = output_dir / "images" / split_name / f"{sample_index:06d}{image_path.suffix.lower()}"
            target_label = output_dir / "labels" / split_name / f"{sample_index:06d}.txt"
            _link_or_copy_file(image_path, target_image, copy_mode)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_split("train", train_items)
    write_split("val", val_items)

    dataset_yaml = output_dir / "dataset.yaml"
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": resolved_names,
        "nc": len(resolved_names),
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return OrganizeReport(
        task_id=task_id,
        output_path=dataset_yaml,
        sample_count=len(valid_samples),
        train_count=len(train_items),
        val_count=len(val_items),
        class_names=resolved_names,
        warnings=warnings,
    )


def prepare_classification_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    copy_mode: str,
) -> OrganizeReport:
    output_dir = output_dir.resolve()
    class_dirs = [path for path in sorted(source_dir.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError("分类源目录下没有发现类别子文件夹。")

    warnings: list[str] = []
    class_names: list[str] = []
    train_count = 0
    val_count = 0
    sample_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_dir in (output_dir / "train", output_dir / "val"):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)

    for class_dir in class_dirs:
        images = list_images_recursive(class_dir)
        if not images:
            warnings.append(f"空类别目录：{class_dir.name}")
            continue
        class_names.append(class_dir.name)
        train_items, val_items = split_items(images, val_ratio, seed)
        sample_count += len(images)
        train_count += len(train_items)
        val_count += len(val_items)

        for split_name, split_items_list in (("train", train_items), ("val", val_items)):
            for index, image_path in enumerate(split_items_list, start=1):
                target = output_dir / split_name / class_dir.name / f"{index:06d}{image_path.suffix.lower()}"
                _link_or_copy_file(image_path, target, copy_mode)

    if not class_names:
        raise ValueError("没有找到有效的分类图片。")

    return OrganizeReport(
        task_id="classify",
        output_path=output_dir,
        sample_count=sample_count,
        train_count=train_count,
        val_count=val_count,
        class_names=class_names,
        warnings=warnings,
    )


class MultiTaskDatasetOrganizerPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_dataset_ready: Callable[[str, Path, OrganizeReport], None],
        on_switch_to_train: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=CARD_BG)
        self.on_state_change = on_state_change
        self.on_notice = on_notice
        self.on_dataset_ready = on_dataset_ready
        self.on_switch_to_train = on_switch_to_train

        self.task_id = "segment"
        self.source_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="")
        self.val_ratio_var = tk.StringVar(value="0.2")
        self.seed_var = tk.StringVar(value="42")
        self.copy_mode_var = tk.StringVar(value="copy")
        self.strict_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="这里可以把检测 / 分割 / 分类 / 姿态 / 旋转框数据整理成训练集。")
        self.structure_var = tk.StringVar(value="")
        self.result_summary_var = tk.StringVar(value="还没有开始整理。")

        self._worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker_running = False

        self._build_ui()
        self._refresh_task_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = tk.Frame(self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)

        tk.Label(header, text="多任务数据整理", bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text="把现有图片/标签目录整理成可直接训练的标准结构。后续第三期会在这里继续扩更多自动质检和模型辅助能力。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=760,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

        task_bar = tk.Frame(self, bg=CARD_BG)
        task_bar.grid(row=1, column=0, sticky="ew", padx=10)
        self.task_buttons: dict[str, tk.Button] = {}
        for index, task_id in enumerate(("detect", "segment", "classify", "pose", "obb")):
            task_bar.grid_columnconfigure(index, weight=1)
            button = tk.Button(
                task_bar,
                text=TASK_LABELS[task_id],
                command=lambda item=task_id: self.set_task(item),
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
            button.grid(row=0, column=index, sticky="ew", padx=(0, 6) if index < 4 else 0)
            self.task_buttons[task_id] = button

        self.body_scroll = VerticalScrolledFrame(self, bg=CARD_BG)
        self.body_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 10))
        body = self.body_scroll.content
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=CARD_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        source_box = self._side_box(left, "源目录和输出目录")
        source_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_path_controls(source_box)

        options_box = self._side_box(left, "整理参数")
        options_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._build_options_box(options_box)

        right = tk.Frame(body, bg=CARD_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        guide_box = self._side_box(right, "当前任务说明")
        guide_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            guide_box,
            textvariable=self.structure_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=12, pady=12)

        class_box = self._side_box(right, "类别名（可选）")
        class_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.class_names_text = tk.Text(
            class_box,
            height=7,
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
        self.class_names_text.pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(
            class_box,
            text="留空时会优先读取 `classes.txt` / `classes.names`；再不行就按标签里的类别 ID 自动补 `class0/class1...`。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=12, pady=(0, 12))

        result_box = self._side_box(right, "整理结果")
        result_box.grid(row=2, column=0, sticky="ew")
        tk.Label(result_box, textvariable=self.result_summary_var, bg=CARD_BG, fg=TEXT, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(result_box, textvariable=self.status_var, bg=CARD_BG, fg=SUCCESS, justify="left", wraplength=360, anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=12, pady=(0, 8))
        action_row = tk.Frame(result_box, bg=CARD_BG)
        action_row.pack(fill="x", padx=12, pady=(0, 12))
        self._primary_button(action_row, "开始整理并回填训练", self.start_prepare).pack(fill="x")

    def _build_path_controls(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=12)
        grid.grid_columnconfigure(0, weight=1)

        tk.Label(grid, text="源目录", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w")
        source_entry = tk.Entry(
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
        )
        source_entry.grid(row=1, column=0, sticky="ew", pady=(6, 8), ipady=6)
        source_row = tk.Frame(grid, bg=CARD_BG)
        source_row.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        source_row.grid_columnconfigure(0, weight=1)
        source_row.grid_columnconfigure(1, weight=1)
        self._small_button(source_row, "选择源目录", self.pick_source_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(source_row, "使用默认输出路径", self.fill_default_output).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        tk.Label(grid, text="输出目录", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=3, column=0, sticky="w")
        output_entry = tk.Entry(
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
        )
        output_entry.grid(row=4, column=0, sticky="ew", pady=(6, 8), ipady=6)
        output_row = tk.Frame(grid, bg=CARD_BG)
        output_row.grid(row=5, column=0, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        output_row.grid_columnconfigure(1, weight=1)
        self._small_button(output_row, "选择输出目录", self.pick_output_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(output_row, "切到训练页", self.on_switch_to_train).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_options_box(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=CARD_BG)
        grid.pack(fill="x", padx=12, pady=12)
        grid.grid_columnconfigure(1, weight=1)

        tk.Label(grid, text="验证集比例", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w", pady=(0, 10))
        tk.Entry(
            grid,
            textvariable=self.val_ratio_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=0, column=1, sticky="ew", pady=(0, 10), ipady=6)

        tk.Label(grid, text="随机种子", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=1, column=0, sticky="w", pady=(0, 10))
        tk.Entry(
            grid,
            textvariable=self.seed_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=1, sticky="ew", pady=(0, 10), ipady=6)

        tk.Label(grid, text="图片处理方式", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=2, column=0, sticky="w", pady=(0, 10))
        mode_row = tk.Frame(grid, bg=CARD_BG)
        mode_row.grid(row=2, column=1, sticky="ew", pady=(0, 10))
        tk.Radiobutton(mode_row, text="复制（更稳）", value="copy", variable=self.copy_mode_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10)).pack(side="left")
        tk.Radiobutton(mode_row, text="硬链接（更省空间）", value="hardlink", variable=self.copy_mode_var, bg=CARD_BG, fg=TEXT, selectcolor=CARD_BG, activebackground=CARD_BG, font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(12, 0))

        tk.Label(grid, text="严格模式", bg=CARD_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(
            grid,
            text="遇到缺标签直接报错",
            variable=self.strict_var,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=CARD_BG,
            activeforeground=TEXT,
            selectcolor=CARD_BG,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=3, column=1, sticky="w")

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

    def set_task(self, task_id: str) -> None:
        self.task_id = task_id
        self._refresh_task_ui()
        if self.source_var.get().strip() and not self.output_var.get().strip():
            self.fill_default_output()
        self.on_state_change()

    def _refresh_task_ui(self) -> None:
        for task_id, button in self.task_buttons.items():
            is_active = task_id == self.task_id
            button.configure(
                bg=PRIMARY if is_active else CARD_BG,
                fg="white" if is_active else TEXT,
                highlightbackground=PRIMARY if is_active else BORDER,
            )

        task_guides = {
            "detect": "适合已经有图片 + 同名检测 TXT 标签的目录。\n支持两种常见结构：\n- 图片和 TXT 放在同一层\n- `images/` 与 `labels/` 平行目录\n整理后会输出标准 detect `dataset.yaml`。",
            "segment": "适合已经有图片 + 同名分割 TXT 标签的目录。\n每一行使用 YOLO 分割格式：`类别 x1 y1 x2 y2 ...`。\n整理后会输出标准 segment `dataset.yaml`。",
            "classify": "源目录里每个子文件夹就是一个类别，例如：\n`猫/`、`狗/`、`车/`\n每个子文件夹里直接放图片即可。\n整理后会输出 `train/类名` 和 `val/类名` 结构。",
            "pose": "适合已做好 YOLO Pose 标签的目录。\n工具会按图片+同名 TXT 配对，并整理为可训练的 pose 数据集结构。",
            "obb": "适合已做好 YOLO OBB 标签的目录。\n工具会按图片+同名 TXT 配对，并整理为可训练的 obb 数据集结构。",
        }
        self.structure_var.set(task_guides[self.task_id])

    def pick_source_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择要整理的数据源目录")
        if not selected:
            return
        self.source_var.set(selected)
        if not self.output_var.get().strip():
            self.fill_default_output()
        self._sync_class_names_from_source()
        self._notify("已选择源目录，接下来可以直接开始整理。")

    def pick_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择整理后的输出目录")
        if selected:
            self.output_var.set(selected)
            self.on_state_change()

    def fill_default_output(self) -> None:
        source_text = self.source_var.get().strip()
        if not source_text:
            return
        source_dir = Path(source_text).expanduser()
        self.output_var.set(str(guess_output_path(source_dir, self.task_id)))
        self.on_state_change()

    def _sync_class_names_from_source(self) -> None:
        source_text = self.source_var.get().strip()
        if not source_text:
            return
        source_dir = Path(source_text).expanduser()
        auto_names = load_class_names(source_dir)
        if not auto_names:
            return
        self.class_names_text.delete("1.0", "end")
        self.class_names_text.insert("1.0", "\n".join(auto_names))

    def prefill_from_annotation(self, task_id: str, source_dir: Path, class_names: list[str]) -> None:
        self.set_task(task_id)
        self.source_var.set(str(source_dir))
        self.output_var.set(str(guess_output_path(source_dir, task_id)))
        self.class_names_text.delete("1.0", "end")
        if class_names:
            self.class_names_text.insert("1.0", "\n".join(class_names))
        self.result_summary_var.set("已经从标注工作台带入当前目录，可以直接整理。")
        self._notify("已从标注工作台带入目录。")

    def current_project_dir(self) -> Path | None:
        source_text = self.source_var.get().strip()
        if not source_text:
            return None
        return Path(source_text).expanduser().resolve()

    def output_preview_dir(self) -> str:
        return self.output_var.get().strip() or "未选择"

    def _notify(self, message: str) -> None:
        self.status_var.set(message)
        self.on_notice(message)
        self.on_state_change()

    def start_prepare(self) -> None:
        if self._worker_running:
            return
        source_text = self.source_var.get().strip()
        output_text = self.output_var.get().strip()
        if not source_text or not output_text:
            messagebox.showwarning("参数不完整", "请先选择源目录和输出目录。")
            return

        try:
            val_ratio = float(self.val_ratio_var.get().strip() or "0.2")
            seed = int(self.seed_var.get().strip() or "42")
        except ValueError:
            messagebox.showwarning("参数格式错误", "验证集比例要填小数，随机种子要填整数。")
            return

        source_dir = Path(source_text).expanduser().resolve()
        output_dir = Path(output_text).expanduser().resolve()
        if not source_dir.exists():
            messagebox.showwarning("源目录不存在", f"找不到目录：\n{source_dir}")
            return
        if output_dir.exists() and any(output_dir.iterdir()):
            if not messagebox.askyesno("输出目录非空", "输出目录里已经有内容，是否继续整理并覆盖同名文件？"):
                return

        explicit_names = _read_explicit_class_names(self.class_names_text.get("1.0", "end"))
        self._worker_running = True
        self.result_summary_var.set("正在整理，请稍等……")
        self._notify("开始整理数据集。")
        worker = threading.Thread(
            target=self._run_prepare_worker,
            args=(source_dir, output_dir, val_ratio, seed, explicit_names, self.copy_mode_var.get(), bool(self.strict_var.get())),
            daemon=True,
        )
        worker.start()
        self.after(80, self._poll_worker_queue)

    def _run_prepare_worker(
        self,
        source_dir: Path,
        output_dir: Path,
        val_ratio: float,
        seed: int,
        explicit_names: list[str],
        copy_mode: str,
        strict: bool,
    ) -> None:
        try:
            if self.task_id in YOLO_LIKE_TASKS:
                report = prepare_yolo_task_dataset(
                    task_id=self.task_id,
                    source_dir=source_dir,
                    output_dir=output_dir,
                    val_ratio=val_ratio,
                    seed=seed,
                    class_names=explicit_names,
                    copy_mode=copy_mode,
                    strict=strict,
                )
            else:
                report = prepare_classification_dataset(
                    source_dir=source_dir,
                    output_dir=output_dir,
                    val_ratio=val_ratio,
                    seed=seed,
                    copy_mode=copy_mode,
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

        self._worker_running = False
        if kind == "error":
            self.result_summary_var.set("整理失败。")
            self._notify("整理失败，请检查目录结构和标签格式。")
            messagebox.showerror("整理失败", str(payload))
            return

        report = payload
        assert isinstance(report, OrganizeReport)
        warning_text = "\n".join(f"- {item}" for item in report.warnings[:6])
        if len(report.warnings) > 6:
            warning_text += f"\n- 还有 {len(report.warnings) - 6} 条未展开"
        summary_lines = [
            f"任务类型：{TASK_LABELS.get(report.task_id, report.task_id)}",
            f"样本数量：{report.sample_count}",
            f"训练集：{report.train_count}",
            f"验证集：{report.val_count}",
            f"类别数量：{len(report.class_names)}",
            f"输出位置：{report.output_path}",
        ]
        if warning_text:
            summary_lines.append("提醒：")
            summary_lines.append(warning_text)
        self.result_summary_var.set("\n".join(summary_lines))
        self._notify("整理完成，已经可以直接回填训练。")
        self.on_dataset_ready(report.task_id, report.output_path, report)
