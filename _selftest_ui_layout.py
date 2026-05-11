import sys
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import ai_platform_panel
import annotation_editor
import annotation_ui_support
import app
import app_v2
import model_hub_panel
import multitask_dataset_panel
import segmentation_editor
import video_annotation_panel


for module in (
    ai_platform_panel,
    annotation_editor,
    app,
    model_hub_panel,
    multitask_dataset_panel,
    segmentation_editor,
    video_annotation_panel,
):
    if hasattr(module, "messagebox"):
        module.messagebox.showinfo = lambda *a, **k: None
        module.messagebox.showwarning = lambda *a, **k: None
        module.messagebox.showerror = lambda *a, **k: None
        if hasattr(module.messagebox, "askyesno"):
            module.messagebox.askyesno = lambda *a, **k: True


INTERACTIVE_TYPES = (tk.Button, tk.Entry, tk.Text, tk.Listbox, tk.Canvas, tk.Radiobutton, tk.Checkbutton)
TEXT_TYPES = (tk.Label, tk.Button, tk.Radiobutton, tk.Checkbutton)
MOJIBAKE_MARKERS = {
    "???",
    "\u93cd\u950b\u6e70",
    "\u93c1\u4f34\u567a",
    "\u7481\ue160\u7c8c",
    "\u95ab\u590b\u5ae8",
    "\u9477\ue044\u59e9",
    "\u9365\u5267\u5896",
    "\u93c2\u56e6\u6b22",
}


class GeometryRootStub:
    def __init__(self, screen_width: int, screen_height: int, geometry: str = "1x1+0+0", dpi: int = 96) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._geometry = geometry
        self._dpi = dpi

    def update_idletasks(self) -> None:
        return None

    def winfo_screenwidth(self) -> int:
        return self._screen_width

    def winfo_screenheight(self) -> int:
        return self._screen_height

    def winfo_geometry(self) -> str:
        return self._geometry

    def winfo_fpixels(self, value: str) -> float:
        _ = value
        return float(self._dpi)


def _is_under_scrollable(widget: tk.Widget) -> bool:
    current = widget
    while current is not None:
        class_name = current.__class__.__name__
        if class_name in {"VerticalScrolledFrame", "ScrollableFrame"}:
            return True
        current = getattr(current, "master", None)
    return False


def _iter_children(widget: tk.Widget):
    for child in widget.winfo_children():
        yield child
        yield from _iter_children(child)


def _collect_overflow(panel: tk.Widget) -> list[tuple[str, int, int, int, int]]:
    panel_x = panel.winfo_rootx()
    panel_y = panel.winfo_rooty()
    panel_w = panel.winfo_width()
    panel_h = panel.winfo_height()
    issues: list[tuple[str, int, int, int, int]] = []
    for child in _iter_children(panel):
        if not isinstance(child, INTERACTIVE_TYPES):
            continue
        if not child.winfo_ismapped():
            continue
        if _is_under_scrollable(child):
            continue
        width = child.winfo_width()
        height = child.winfo_height()
        if width <= 1 or height <= 1:
            continue
        x = child.winfo_rootx() - panel_x
        y = child.winfo_rooty() - panel_y
        if x < 0 or y < 0 or x + width > panel_w or y + height > panel_h:
            issues.append((child.__class__.__name__, x, y, width, height))
    return issues


def _collect_visible_text_artifacts(panel: tk.Widget) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    for child in _iter_children(panel):
        if not isinstance(child, TEXT_TYPES):
            continue
        if not child.winfo_ismapped():
            continue
        values = []
        text = child.cget("text")
        if text:
            values.append(str(text))
        textvariable = child.cget("textvariable")
        if textvariable:
            try:
                value = child.getvar(textvariable)
            except tk.TclError:
                value = ""
            if value:
                values.append(str(value))
        for value in values:
            if any(marker in value for marker in MOJIBAKE_MARKERS):
                issues.append((child.__class__.__name__, value))
    return issues


def _assert_no_visible_text_artifacts(panel: tk.Widget, *, context: str) -> None:
    issues = _collect_visible_text_artifacts(panel)
    assert issues == [], f"{context} visible text artifacts: {issues[:3]}"


def _assert_buttons_fit(parent: tk.Widget, buttons: dict[str, tk.Button], *, context: str) -> None:
    assert _collect_overflow(parent) == [], f"{context} overflow"
    for button_id, button in buttons.items():
        if not button.winfo_ismapped():
            continue
        actual_width = button.winfo_width()
        required_width = button.winfo_reqwidth()
        assert actual_width >= required_width - 4, (
            f"{context} button clipped: {button_id} ({actual_width} < {required_width})"
        )


def _assert_operation_layout(ui: app_v2.AppV2, page_id: str, *, stacked: bool) -> None:
    layout = ui.operation_pages[page_id]
    left_info = layout["left"].grid_info()
    right_info = layout["right"].grid_info()
    if stacked:
        assert int(left_info["row"]) == 2 and int(right_info["row"]) == 3, f"{page_id} should stack side panel below main panel"
        assert int(left_info["columnspan"]) == 2 and int(right_info["columnspan"]) == 2, f"{page_id} stacked panels should use full width"
    else:
        assert int(left_info["row"]) == 2 and int(right_info["row"]) == 2, f"{page_id} should use side-by-side panels"
        assert int(left_info["column"]) == 0 and int(right_info["column"]) == 1, f"{page_id} side panel should stay on the right"


def _assert_v2_metric_cards_uniform(parent: tk.Widget, *, context: str) -> None:
    cards = getattr(parent, "_v2_metric_cards", [])
    assert len(cards) == 4, f"{context} should have four metric cards"
    heights = [card.winfo_height() for card in cards if card.winfo_ismapped()]
    assert len(set(heights)) == 1, f"{context} metric card heights differ: {heights}"
    assert heights and heights[0] == app_v2.V2_METRIC_CARD_HEIGHT, f"{context} metric card height should be fixed: {heights}"


def _assert_scroller_moves(scroller: annotation_ui_support.VerticalScrolledFrame, event_widget: tk.Widget, root: tk.Tk) -> None:
    bbox = scroller.canvas.bbox("all")
    assert bbox is not None, "scroll region missing"
    assert bbox[3] > scroller.canvas.winfo_height(), "scroll region should exceed viewport for this layout test"
    yview_before = scroller.canvas.yview()
    result = annotation_ui_support.VerticalScrolledFrame._dispatch_mousewheel(
        SimpleNamespace(widget=event_widget, delta=-120)
    )
    root.update_idletasks()
    root.update()
    yview_after = scroller.canvas.yview()
    assert result == "break", "mousewheel dispatch should be handled by VerticalScrolledFrame"
    assert yview_after != yview_before, "mousewheel should move the scroller"


def _assert_app_scrollable_moves(scroller: app.ScrollableFrame, event_widget: tk.Widget, root: tk.Misc) -> None:
    bbox = scroller.canvas.bbox("all")
    assert bbox is not None, "app scroll region missing"
    assert bbox[3] > scroller.canvas.winfo_height(), "app scroll region should exceed viewport for this layout test"
    yview_before = scroller.canvas.yview()
    result = app.ScrollableFrame._dispatch_mousewheel(SimpleNamespace(widget=event_widget, delta=-120))
    root.update_idletasks()
    root.update()
    yview_after = scroller.canvas.yview()
    assert result == "break", "mousewheel dispatch should be handled by ScrollableFrame"
    assert yview_after != yview_before, "mousewheel should move the app scroller"


def _assert_default_window_profiles() -> None:
    classic_profiles = {
        "720p": (1280, 720, (1208, 660, 1180, 660)),
        "1k": (1600, 900, (1396, 828, 1180, 700)),
        "1080p": (1920, 1080, (1600, 980, 1180, 700)),
        "2k": (2560, 1440, (1985, 1216, 1180, 700)),
        "4k": (3840, 2160, (2272, 1392, 1180, 700)),
    }
    for size_name, (screen_width, screen_height, expected) in classic_profiles.items():
        stub = GeometryRootStub(screen_width, screen_height)
        actual = app.resolve_window_geometry(
            stub,
            min_width=app.COMPACT_MIN_WINDOW_WIDTH,
            min_height=app.COMPACT_MIN_WINDOW_HEIGHT,
            preferred_width=app.DEFAULT_WINDOW_WIDTH,
            preferred_height=app.DEFAULT_WINDOW_HEIGHT,
        )
        assert actual == expected, f"classic default geometry mismatch at {size_name}: {actual} != {expected}"

    v2_profiles = {
        "720p": (1280, 720, (1208, 648, 1180, 624)),
        "1k": (1600, 900, (1396, 785, 1180, 680)),
        "1080p": (1920, 1080, (1600, 900, 1180, 680)),
        "2k": (2560, 1440, (1985, 1117, 1180, 680)),
        "4k": (3840, 2160, (2272, 1278, 1180, 680)),
    }
    for size_name, (screen_width, screen_height, expected) in v2_profiles.items():
        stub = GeometryRootStub(screen_width, screen_height)
        actual = app.resolve_window_geometry(
            stub,
            min_width=app_v2.V2_MIN_WINDOW_WIDTH,
            min_height=app_v2.V2_MIN_WINDOW_HEIGHT,
            preferred_width=app_v2.V2_DEFAULT_WINDOW_WIDTH,
            preferred_height=app_v2.V2_DEFAULT_WINDOW_HEIGHT,
            min_height_floor=620,
        )
        assert actual == expected, f"v2 default geometry mismatch at {size_name}: {actual} != {expected}"

    classic_scaled_profiles = {
        "1080p-125%": (1920, 1080, 120, (1700, 990, 1475, 875)),
        "1080p-150%": (1920, 1080, 144, (1812, 990, 1770, 990)),
        "4k-125%": (3840, 2160, 120, (2840, 1740, 1475, 875)),
        "4k-150%": (3840, 2160, 144, (2978, 1824, 1770, 1050)),
        "4k-200%": (3840, 2160, 192, (3200, 1960, 2360, 1400)),
    }
    for size_name, (screen_width, screen_height, dpi, expected) in classic_scaled_profiles.items():
        stub = GeometryRootStub(screen_width, screen_height, dpi=dpi)
        actual = app.resolve_window_geometry(
            stub,
            min_width=app.COMPACT_MIN_WINDOW_WIDTH,
            min_height=app.COMPACT_MIN_WINDOW_HEIGHT,
            preferred_width=app.DEFAULT_WINDOW_WIDTH,
            preferred_height=app.DEFAULT_WINDOW_HEIGHT,
        )
        assert actual == expected, f"classic scaled geometry mismatch at {size_name}: {actual} != {expected}"

    v2_scaled_profiles = {
        "1080p-125%": (1920, 1080, 120, (1700, 956, 1475, 850)),
        "1080p-150%": (1920, 1080, 144, (1812, 972, 1770, 936)),
        "4k-125%": (3840, 2160, 120, (2840, 1598, 1475, 850)),
        "4k-150%": (3840, 2160, 144, (2978, 1676, 1770, 1020)),
        "4k-200%": (3840, 2160, 192, (3200, 1800, 2360, 1360)),
    }
    for size_name, (screen_width, screen_height, dpi, expected) in v2_scaled_profiles.items():
        stub = GeometryRootStub(screen_width, screen_height, dpi=dpi)
        actual = app.resolve_window_geometry(
            stub,
            min_width=app_v2.V2_MIN_WINDOW_WIDTH,
            min_height=app_v2.V2_MIN_WINDOW_HEIGHT,
            preferred_width=app_v2.V2_DEFAULT_WINDOW_WIDTH,
            preferred_height=app_v2.V2_DEFAULT_WINDOW_HEIGHT,
            min_height_floor=620,
        )
        assert actual == expected, f"v2 scaled geometry mismatch at {size_name}: {actual} != {expected}"


_assert_default_window_profiles()


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    sample_dir = tmp_path / "images"
    sample_dir.mkdir()
    Image.new("RGB", (120, 90), color=(255, 255, 255)).save(sample_dir / "sample.png")
    (sample_dir / "classes.txt").write_text("class0\nclass1", encoding="utf-8")

    root = tk.Tk()
    root.geometry("1180x700+40+40")

    ai_panel = ai_platform_panel.AiPlatformPanel(
        root,
        on_state_change=lambda: None,
        on_notice=lambda *_: None,
        on_detect_auto_label_request=lambda *_: None,
        on_open_workspace=lambda *_: None,
        on_open_model_hub=lambda: None,
        get_detect_project=lambda: None,
        get_segment_project=lambda: None,
        local_state_dir=tmp_path,
    )
    ai_panel.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()
    assert _collect_overflow(ai_panel) == []
    _assert_scroller_moves(ai_panel.body_scroll, ai_panel.primary_action_button, root)
    ai_panel.pack_forget()

    model_panel = model_hub_panel.ModelHubPanel(
        root,
        project_root=PROJECT_ROOT,
        state_path=tmp_path / "hub.json",
        on_state_change=lambda: None,
        on_notice=lambda *_: None,
        on_apply_model=lambda *_: None,
    )
    model_panel.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()
    assert _collect_overflow(model_panel) == []
    _assert_scroller_moves(model_panel.body_scroll, model_panel.local_model_list, root)
    model_panel.pack_forget()

    organize_panel = multitask_dataset_panel.MultiTaskDatasetOrganizerPanel(
        root,
        on_state_change=lambda: None,
        on_notice=lambda *_: None,
        on_dataset_ready=lambda *_: None,
        on_switch_to_train=lambda: None,
    )
    organize_panel.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()
    assert _collect_overflow(organize_panel) == []
    _assert_scroller_moves(organize_panel.body_scroll, organize_panel.class_names_text, root)
    organize_panel.pack_forget()

    video_panel = video_annotation_panel.VideoAnnotationPanel(
        root,
        on_state_change=lambda: None,
        on_notice=lambda *_: None,
        on_open_project=lambda *_: None,
    )
    video_panel.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()
    assert _collect_overflow(video_panel) == []
    video_panel.pack_forget()

    window_sizes = (
        ("720p", "1280x720+60+60"),
        ("1k", "1600x900+60+60"),
        ("1080p", "1920x1080+60+60"),
        ("2k", "2560x1440+60+60"),
        ("4k", "3840x2160+60+60"),
    )

    for size_name, geometry in window_sizes:
        app_root = tk.Toplevel(root)
        app_root.geometry(geometry)
        ui = app.App(app_root)
        app_root.geometry(geometry)
        app_root.update_idletasks()
        app_root.update()
        assert app_root.winfo_width() <= app_root.winfo_screenwidth() + 24
        assert app_root.winfo_height() <= app_root.winfo_screenheight() + 24

        ui._show_tab("train")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.left_panel) == [], f"classic train left overflow at {size_name}"
        assert _collect_overflow(ui.right_panel) == [], f"classic train right overflow at {size_name}"
        if size_name == "720p":
            _assert_app_scrollable_moves(ui.train_scroll, ui.train_action_buttons["train"], app_root)

        ui._show_tab("export")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.left_panel) == [], f"classic export left overflow at {size_name}"
        assert _collect_overflow(ui.right_panel) == [], f"classic export right overflow at {size_name}"
        if size_name == "720p":
            _assert_app_scrollable_moves(ui.export_scroll, ui.export_weights_entry, app_root)

        ui._show_tab("annotation")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.detect_editor) == [], f"classic annotation overflow at {size_name}"

        ui.annotation_editor.show_workspace("segment")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.segment_editor) == [], f"classic segment overflow at {size_name}"

        ui.annotation_editor.show_workspace("video")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.video_panel) == [], f"classic video overflow at {size_name}"

        ui.annotation_editor.show_workspace("ai_platform")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.ai_platform_panel) == [], f"classic ai overflow at {size_name}"
        if size_name == "720p":
            _assert_scroller_moves(ui.annotation_editor.ai_platform_panel.body_scroll, ui.annotation_editor.ai_platform_panel.primary_action_button, app_root)

        ui.annotation_editor.show_workspace("organize")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.organizer_panel) == [], f"classic organize overflow at {size_name}"
        if size_name == "720p":
            _assert_scroller_moves(ui.annotation_editor.organizer_panel.body_scroll, ui.annotation_editor.organizer_panel.class_names_text, app_root)

        ui.annotation_editor.show_workspace("model_hub")
        app_root.update_idletasks()
        app_root.update()
        assert _collect_overflow(ui.annotation_editor.model_hub_panel) == [], f"classic model hub overflow at {size_name}"
        if size_name == "720p":
            _assert_scroller_moves(ui.annotation_editor.model_hub_panel.body_scroll, ui.annotation_editor.model_hub_panel.local_model_list, app_root)
        app_root.destroy()

    v2_breakpoint_root = tk.Toplevel(root)
    v2_breakpoint_root.geometry("1600x900+60+60")
    v2_breakpoint = app_v2.AppV2(v2_breakpoint_root)
    v2_breakpoint_root.update_idletasks()
    v2_breakpoint_root.update()
    assert int(v2_breakpoint.topbar_actions.grid_info()["row"]) == 0, "wide v2 actions should sit on the first topbar row"
    v2_breakpoint_root.geometry("1450x900+60+60")
    v2_breakpoint_root.update_idletasks()
    v2_breakpoint._apply_responsive_layout()
    v2_breakpoint_root.update()
    assert int(v2_breakpoint.topbar_actions.grid_info()["row"]) == 2, "mid-width v2 actions should move below nav after resize"
    v2_breakpoint_root.destroy()

    for size_name, geometry in window_sizes:
        v2_root = tk.Toplevel(root)
        v2_root.geometry(geometry)
        v2 = app_v2.AppV2(v2_root)
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights = [v2.topbar.winfo_height()]
        assert not v2.topbar_home_button.winfo_ismapped(), "home page should hide the return button"
        assert _collect_overflow(v2.page_frames["home"]) == [], f"v2 home overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 home at {size_name}")
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 home topbar nav at {size_name}")
        home_stats = v2.page_frames["home"].grid_slaves(row=1, column=0)[0]
        _assert_v2_metric_cards_uniform(home_stats, context=f"v2 home stats at {size_name}")
        if size_name == "720p":
            assert v2.top_nav_buttons["detect"].cget("text") == "检测", "compact nav should use short labels"
            assert v2.topbar_quick_buttons["continue"].cget("text") == "继续", "compact actions should use short labels"
            assert v2.topbar.winfo_height() <= 150, f"compact topbar should leave enough workspace height: {v2.topbar.winfo_height()}"
            home_columns = {int(card.grid_info()["column"]) for card in v2.home_module_cards.values()}
            assert home_columns == {0, 1}, f"compact home cards should use a readable two-column grid: {home_columns}"
        else:
            assert v2.top_nav_buttons["detect"].cget("text") == "检测标注", "wide nav should use full labels"
            assert v2.topbar_quick_buttons["continue"].cget("text") == "继续当前工作", "wide actions should use full labels"

        low_height_layout = v2._is_low_height_layout()
        if low_height_layout:
            assert not v2.footer.winfo_ismapped(), "compact home should hide the footer"
        else:
            assert v2.footer.winfo_ismapped(), f"{size_name} home should keep the footer"

        v2.show_page("detect")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.detect_editor) == [], f"v2 detect overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 detect at {size_name}")
        assert v2.topbar_home_button.winfo_ismapped(), "immersive detect page should show the return button"
        assert v2.topbar_nav.winfo_ismapped(), "immersive detect page should keep the top navigation visible"
        assert v2.topbar_quick_actions.winfo_ismapped(), "immersive detect page should keep the quick actions visible"
        assert v2.top_nav_buttons["detect"].cget("bg") == app_v2.PRIMARY, "active nav item should be visually prominent"
        if size_name == "720p":
            assert not v2.annotation_editor.detect_editor.v2_filmstrip.winfo_ismapped(), "compact detect should hide the filmstrip"
            assert v2.annotation_editor.detect_editor.canvas.winfo_height() >= 240, "compact detect canvas should keep usable height"
            assert v2.annotation_editor.detect_editor.canvas.winfo_height() >= 300, "compact detect canvas should not be cramped after topbar compression"
        else:
            assert v2.annotation_editor.detect_editor.v2_filmstrip.winfo_ismapped(), f"{size_name} detect should show the filmstrip"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 detect topbar nav at {size_name}")

        v2.top_nav_buttons["segment"].invoke()
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.segment_editor) == [], f"v2 segment overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 segment at {size_name}")
        assert v2.active_page_var.get() == "segment", "top navigation should allow switching from detect to segment"
        assert v2.topbar_nav.winfo_ismapped(), "immersive segment page should keep the top navigation visible"
        if size_name == "720p":
            assert not v2.annotation_editor.segment_editor.v2_filmstrip.winfo_ismapped(), "compact segment should hide the filmstrip"
            assert v2.annotation_editor.segment_editor.canvas.winfo_height() >= 240, "compact segment canvas should keep usable height"
            assert v2.annotation_editor.segment_editor.canvas.winfo_height() >= 300, "compact segment canvas should not be cramped after topbar compression"
        else:
            assert v2.annotation_editor.segment_editor.v2_filmstrip.winfo_ismapped(), f"{size_name} segment should show the filmstrip"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 segment topbar nav at {size_name}")

        v2.show_page("video")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.video_panel) == [], f"v2 video overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 video at {size_name}")
        assert v2.active_page_var.get() == "video", "video page should stay in sync with active nav state"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 video topbar nav at {size_name}")

        v2.show_page("organize")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.organizer_panel) == [], f"v2 organize overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 organize at {size_name}")
        assert v2.active_page_var.get() == "organize", "organize page should stay in sync with active nav state"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 organize topbar nav at {size_name}")
        if size_name == "720p":
            _assert_scroller_moves(v2.annotation_editor.organizer_panel.body_scroll, v2.annotation_editor.organizer_panel.class_names_text, v2_root)

        v2.show_page("ai")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.ai_platform_panel) == [], f"v2 ai overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 ai at {size_name}")
        assert v2.active_page_var.get() == "ai", "ai page should stay in sync with active nav state"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 ai topbar nav at {size_name}")
        if size_name == "720p":
            _assert_scroller_moves(v2.annotation_editor.ai_platform_panel.body_scroll, v2.annotation_editor.ai_platform_panel.primary_action_button, v2_root)

        v2.show_page("model_hub")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.annotation_editor.model_hub_panel) == [], f"v2 model hub overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 model hub at {size_name}")
        assert v2.active_page_var.get() == "model_hub", "model hub page should stay in sync with active nav state"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 model hub topbar nav at {size_name}")
        if size_name == "720p":
            _assert_scroller_moves(v2.annotation_editor.model_hub_panel.body_scroll, v2.annotation_editor.model_hub_panel.local_model_list, v2_root)

        v2.topbar_home_button.invoke()
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert v2.active_page_var.get() == "home", "return button should navigate back to home"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 returned home at {size_name}")
        assert not v2.topbar_home_button.winfo_ismapped(), "home page should hide the return button after navigating back"

        v2.show_page("train")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.page_frames["train"]) == [], f"v2 train overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 train at {size_name}")
        assert v2.topbar_home_button.winfo_ismapped(), "train page should show the return button"
        assert v2.topbar_nav.winfo_ismapped(), "train page should keep the top navigation visible"
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 train topbar nav at {size_name}")
        train_stats = v2.page_frames["train"].grid_slaves(row=1, column=0)[0]
        _assert_v2_metric_cards_uniform(train_stats, context=f"v2 train stats at {size_name}")
        if low_height_layout:
            assert not v2.footer.winfo_ismapped(), "compact train should hide the footer"
        _assert_operation_layout(v2, "train", stacked=low_height_layout)

        v2.show_page("export")
        v2_root.update_idletasks()
        v2_root.update()
        v2_topbar_heights.append(v2.topbar.winfo_height())
        assert _collect_overflow(v2.page_frames["export"]) == [], f"v2 export overflow at {size_name}"
        _assert_no_visible_text_artifacts(v2_root, context=f"v2 export at {size_name}")
        _assert_buttons_fit(v2.topbar_nav, v2.top_nav_buttons, context=f"v2 export topbar nav at {size_name}")
        export_stats = v2.page_frames["export"].grid_slaves(row=1, column=0)[0]
        _assert_v2_metric_cards_uniform(export_stats, context=f"v2 export stats at {size_name}")
        if low_height_layout:
            assert not v2.footer.winfo_ismapped(), "compact export should hide the footer"
        _assert_operation_layout(v2, "export", stacked=low_height_layout)
        assert len(set(v2_topbar_heights)) == 1, f"v2 topbar height should stay fixed at {size_name}: {v2_topbar_heights}"
        v2_root.destroy()
    root.destroy()

print("UI_LAYOUT_SELFTEST_OK")
