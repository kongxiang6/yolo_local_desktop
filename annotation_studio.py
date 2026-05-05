from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Callable

from ai_platform_panel import AiPlatformPanel
from annotation_editor import DetectionAnnotationEditor
from model_hub_panel import ModelHubPanel
from multitask_dataset_panel import MultiTaskDatasetOrganizerPanel, OrganizeReport
from segmentation_editor import SegmentationAnnotationEditor
from video_annotation_panel import VideoAnnotationPanel


CARD_BG = "#ffffff"
WINDOW_BG = "#eef4ff"
PANEL_BG = "#f7fbff"
PRIMARY = "#5b8cff"
PRIMARY_DARK = "#4f7de6"
PRIMARY_SOFT = "#edf3ff"
BORDER = "#dce7fb"
TEXT = "#243348"
TEXT_MUTED = "#6c7c96"

WORKSPACE_LABELS = {
    "detect": "检测框标注",
    "segment": "交互式分割",
    "video": "视频标注",
    "organize": "多任务整理",
    "ai_platform": "AI 工作流",
    "model_hub": "模型中心",
}

WORKSPACE_DESCRIPTIONS = {
    "detect": "适合先用检测模型打一轮框，再回到画布里人工修边。",
    "segment": "适合手动画掩码、多边形修边，后续会继续接 SAM/SAM2 辅助。",
    "video": "适合先把视频抽成关键帧，再送去检测或分割工作区做标注。",
    "organize": "适合把 detect / segment / classify / pose / obb 目录整理成正式训练集。",
    "ai_platform": "面向第三期的综合 AI 标注入口，负责模型辅助、文本提示和大图切片工作流。",
    "model_hub": "统一管理官方推荐模型和本机已有模型，后续扩模型后端时都从这里接。",
}

WORKSPACE_GROUPS = (
    ("基础标注", ("detect", "segment", "video", "organize")),
    ("第三期 AI 平台", ("ai_platform", "model_hub")),
)


class AnnotationStudio(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        detect_session_path: Path,
        segment_session_path: Path,
        on_state_change: Callable[[], None],
        on_notice: Callable[[str], None],
        on_export_request: Callable[[Path, list[str]], None],
        on_auto_label_request: Callable[[Path, Path | None, str, dict[str, object], list[str]], None],
        on_dataset_ready: Callable[[str, Path, OrganizeReport], None],
        on_switch_to_train: Callable[[], None],
        on_workspace_change: Callable[[str], None] | None = None,
        embedded_mode: bool = False,
    ) -> None:
        super().__init__(parent, bg=WINDOW_BG)
        self.project_root = Path(__file__).resolve().parent
        self.local_state_dir = detect_session_path.parent
        self.external_on_state_change = on_state_change
        self.external_on_notice = on_notice
        self.external_on_auto_label_request = on_auto_label_request
        self.external_on_workspace_change = on_workspace_change
        self.embedded_mode = embedded_mode
        self._active_workspace = "detect"

        self.workspace_title_var = tk.StringVar(value=WORKSPACE_LABELS["detect"])
        self.workspace_desc_var = tk.StringVar(value=WORKSPACE_DESCRIPTIONS["detect"])
        self.workspace_context_var = tk.StringVar(value="当前目录\n未选择\n\n当前内容\n未选择")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.workspace_buttons: dict[str, tk.Button] = {}

        self._build_shell()

        self.detect_editor = DetectionAnnotationEditor(
            self.workspace_host,
            session_path=detect_session_path,
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_export_request=on_export_request,
            on_auto_label_request=on_auto_label_request,
            on_switch_to_train=on_switch_to_train,
            ui_mode="v2" if embedded_mode else "classic",
        )
        self.segment_editor = SegmentationAnnotationEditor(
            self.workspace_host,
            session_path=segment_session_path,
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_export_request=self._prefill_segment_organizer,
            on_switch_to_train=on_switch_to_train,
            ui_mode="v2" if embedded_mode else "classic",
        )
        self.video_panel = VideoAnnotationPanel(
            self.workspace_host,
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_open_project=self._open_project_from_video,
        )
        self.organizer_panel = MultiTaskDatasetOrganizerPanel(
            self.workspace_host,
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_dataset_ready=on_dataset_ready,
            on_switch_to_train=on_switch_to_train,
        )
        self.ai_platform_panel = AiPlatformPanel(
            self.workspace_host,
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_detect_auto_label_request=self._launch_detect_auto_label_from_ai,
            on_open_workspace=self._open_workspace_from_ai,
            on_open_model_hub=lambda: self.show_workspace("model_hub"),
            get_detect_project=lambda: self.detect_editor.project_dir,
            get_segment_project=lambda: self.segment_editor.project_dir,
            local_state_dir=self.local_state_dir,
        )
        self.model_hub_panel = ModelHubPanel(
            self.workspace_host,
            project_root=self.project_root,
            state_path=self.local_state_dir / "model_hub_state.json",
            on_state_change=self._handle_child_state_change,
            on_notice=self._handle_child_notice,
            on_apply_model=self._apply_selected_model_from_hub,
        )

        self.workspace_views = {
            "detect": self.detect_editor,
            "segment": self.segment_editor,
            "video": self.video_panel,
            "organize": self.organizer_panel,
            "ai_platform": self.ai_platform_panel,
            "model_hub": self.model_hub_panel,
        }
        for view in self.workspace_views.values():
            view.grid(row=0, column=0, sticky="nsew")

        self.show_workspace("detect")

    def _build_shell(self) -> None:
        shell = tk.Frame(self, bg=WINDOW_BG)
        shell.grid(row=0, column=0, sticky="nsew", padx=0 if self.embedded_mode else 10, pady=0 if self.embedded_mode else 10)
        shell.grid_columnconfigure(0, weight=1)
        host_row = 0 if self.embedded_mode else 2
        shell.grid_rowconfigure(host_row, weight=1)

        if not self.embedded_mode:
            self._build_workspace_topbar(shell)
            self._build_workspace_switcher(shell)

        self.workspace_host = tk.Frame(shell, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        self.workspace_host.grid(row=host_row, column=0, sticky="nsew")
        self.workspace_host.grid_columnconfigure(0, weight=1)
        self.workspace_host.grid_rowconfigure(0, weight=1)

    def _build_workspace_switcher(self, parent: tk.Widget) -> None:
        switcher = tk.Frame(parent, bg=WINDOW_BG)
        switcher.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        workspace_order = ("detect", "segment", "video", "organize", "ai_platform", "model_hub")
        for index in range(len(workspace_order)):
            switcher.grid_columnconfigure(index, weight=1)

        tk.Label(
            switcher,
            text="工作区切换",
            bg=WINDOW_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).grid(row=0, column=0, columnspan=len(workspace_order), sticky="w", pady=(0, 6))

        for column, workspace_id in enumerate(workspace_order):
            button = tk.Button(
                switcher,
                text=WORKSPACE_LABELS[workspace_id],
                command=lambda item=workspace_id: self.show_workspace(item),
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
                pady=10,
                cursor="hand2",
            )
            button.grid(row=1, column=column, sticky="ew", padx=(0, 8) if column < len(workspace_order) - 1 else 0)
            self.workspace_buttons[workspace_id] = button

    def _build_workspace_topbar(self, parent: tk.Widget) -> None:
        topbar = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        topbar.grid_columnconfigure(0, weight=1)

        title_block = tk.Frame(topbar, bg=CARD_BG)
        title_block.grid(row=0, column=0, sticky="ew", padx=14, pady=12)
        title_block.grid_columnconfigure(0, weight=1)

        tk.Label(
            title_block,
            text="标注工作台",
            bg=PRIMARY_SOFT,
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=10,
            pady=5,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            title_block,
            textvariable=self.workspace_title_var,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 17, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(10, 4))
        tk.Label(
            title_block,
            textvariable=self.workspace_desc_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify="left",
            wraplength=860,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=2, column=0, sticky="ew")
        tk.Label(
            title_block,
            textvariable=self.workspace_context_var,
            bg=PANEL_BG,
            fg=TEXT,
            justify="left",
            wraplength=320,
            font=("Microsoft YaHei UI", 9),
            padx=12,
            pady=10,
        ).grid(row=0, column=1, rowspan=3, sticky="ne", padx=(18, 0))

    def _handle_child_state_change(self) -> None:
        self._refresh_dashboard()
        self.external_on_state_change()

    def _handle_child_notice(self, message: str) -> None:
        self._refresh_dashboard()
        self.external_on_notice(message)

    def _refresh_dashboard(self) -> None:
        workspace_label = WORKSPACE_LABELS.get(self._active_workspace, self._active_workspace)
        self.workspace_title_var.set(workspace_label)
        self.workspace_desc_var.set(WORKSPACE_DESCRIPTIONS.get(self._active_workspace, ""))
        project_dir = self.project_dir
        project_text = str(project_dir) if project_dir is not None else "未选择"
        current_text = self.current_image_name()
        if current_text == "未选择":
            current_text = self.export_preview_dir()
        self.workspace_context_var.set(f"当前目录\n{project_text}\n\n当前内容\n{current_text}")

    def show_workspace(self, workspace_id: str) -> None:
        self._active_workspace = workspace_id
        self.workspace_views[workspace_id].tkraise()
        for item, button in self.workspace_buttons.items():
            is_active = item == workspace_id
            button.configure(
                bg=PRIMARY if is_active else CARD_BG,
                fg="white" if is_active else TEXT,
                activebackground=PRIMARY_DARK if is_active else PRIMARY_SOFT,
                highlightbackground=PRIMARY if is_active else BORDER,
            )
        self._refresh_dashboard()
        self.external_on_state_change()
        if self.external_on_workspace_change is not None:
            self.external_on_workspace_change(workspace_id)

    def _prefill_segment_organizer(self, source_dir: Path, class_names: list[str]) -> None:
        self.organizer_panel.prefill_from_annotation("segment", source_dir, class_names)
        self.show_workspace("organize")

    def _open_project_from_video(self, workspace_id: str, folder: Path) -> None:
        self._open_workspace_from_ai(workspace_id, folder)

    def _open_workspace_from_ai(self, workspace_id: str, folder: Path) -> None:
        if workspace_id == "segment":
            self.segment_editor.load_project(folder)
            self.show_workspace("segment")
        elif workspace_id == "detect":
            self.detect_editor.load_project(folder)
            self.show_workspace("detect")
        elif workspace_id == "organize":
            self.organizer_panel.prefill_from_annotation("detect", folder, [])
            self.show_workspace("organize")
        else:
            self.show_workspace(workspace_id)

    def _launch_detect_auto_label_from_ai(self, folder: Path, model_ref: str, config: dict[str, object]) -> None:
        self.detect_editor.load_project(folder)
        if self.detect_editor.project_dir is None or self.detect_editor.project_dir != folder.resolve():
            return
        self.detect_editor.auto_model_var.set(model_ref)
        self.external_on_auto_label_request(folder, None, model_ref, dict(config), list(self.detect_editor.class_names))

    def _apply_selected_model_from_hub(self, model_ref: str, task_hint: str) -> None:
        self.ai_platform_panel.set_selected_model(model_ref)
        if task_hint == "detect":
            self.detect_editor.auto_model_var.set(model_ref)
        self.show_workspace("ai_platform")

    @property
    def project_dir(self) -> Path | None:
        if self._active_workspace == "detect":
            return self.detect_editor.project_dir
        if self._active_workspace == "segment":
            return self.segment_editor.project_dir
        if self._active_workspace == "video":
            return self.video_panel.current_project_dir()
        if self._active_workspace == "organize":
            return self.organizer_panel.current_project_dir()
        if self._active_workspace == "ai_platform":
            return self.ai_platform_panel.current_project_dir()
        if self._active_workspace == "model_hub":
            return self.model_hub_panel.current_project_dir()
        return self.detect_editor.project_dir or self.segment_editor.project_dir

    def current_image_name(self) -> str:
        if self._active_workspace == "detect":
            return self.detect_editor.current_image_name()
        if self._active_workspace == "segment":
            return self.segment_editor.current_image_name()
        if self._active_workspace == "video":
            return self.video_panel.output_preview_dir()
        if self._active_workspace == "organize":
            return self.organizer_panel.output_preview_dir()
        if self._active_workspace == "ai_platform":
            return self.ai_platform_panel.active_capability_label()
        if self._active_workspace == "model_hub":
            return "模型中心"
        return "未选择"

    def export_preview_dir(self) -> str:
        if self._active_workspace == "detect":
            return self.detect_editor.export_preview_dir()
        if self._active_workspace == "segment":
            return self.segment_editor.export_preview_dir()
        if self._active_workspace == "video":
            return self.video_panel.output_preview_dir()
        if self._active_workspace == "organize":
            return self.organizer_panel.output_preview_dir()
        if self._active_workspace == "ai_platform":
            return self.ai_platform_panel.output_preview_dir()
        if self._active_workspace == "model_hub":
            return self.model_hub_panel.output_preview_dir()
        return "未选择"

    def active_workspace_label(self) -> str:
        return WORKSPACE_LABELS.get(self._active_workspace, self._active_workspace)

    def apply_auto_label_result(self, payload: dict[str, object]) -> None:
        self.detect_editor.apply_auto_label_result(payload)
        self._refresh_dashboard()
