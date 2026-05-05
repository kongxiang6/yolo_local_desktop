import sys
import tempfile
import tkinter as tk
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import ai_platform_panel
import annotation_editor
import annotation_studio
import model_hub_panel
import segmentation_editor
from ai_platform_support import slice_large_images


for module in (annotation_editor, segmentation_editor, ai_platform_panel, model_hub_panel):
    module.messagebox.showinfo = lambda *a, **k: None
    module.messagebox.showwarning = lambda *a, **k: None
    module.messagebox.showerror = lambda *a, **k: None


captured = {"auto": []}


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    Image.new("RGB", (100, 80), color=(255, 255, 255)).save(project_dir / "sample.jpg")
    (project_dir / "classes.txt").write_text("cat\ndog", encoding="utf-8")

    root = tk.Tk()
    root.withdraw()

    studio = annotation_studio.AnnotationStudio(
        root,
        detect_session_path=tmp_path / "detect_sessions.json",
        segment_session_path=tmp_path / "segment_sessions.json",
        on_state_change=lambda: None,
        on_notice=lambda _message: None,
        on_export_request=lambda *_args, **_kwargs: None,
        on_auto_label_request=lambda image_dir, image_path_or_none, model_ref, config, class_names: captured["auto"].append(
            (Path(image_dir), image_path_or_none, model_ref, dict(config), list(class_names))
        ),
        on_dataset_ready=lambda *_args, **_kwargs: None,
        on_switch_to_train=lambda: None,
    )
    root.update_idletasks()
    root.update()

    studio.show_workspace("ai_platform")
    studio.ai_platform_panel.source_var.set(str(project_dir))
    studio.ai_platform_panel.model_var.set("yolo11n.pt")
    studio.ai_platform_panel.run_detect_auto_label()
    assert len(captured["auto"]) == 1
    auto_call = captured["auto"][0]
    assert auto_call[0] == project_dir.resolve()
    assert auto_call[2] == "yolo11n.pt"
    assert auto_call[4] == ["cat", "dog"]
    assert studio.active_workspace_label() == "检测框标注"

    studio._apply_selected_model_from_hub("custom-seg.pt", "segment")
    assert studio.ai_platform_panel.model_var.get() == "custom-seg.pt"
    assert studio.active_workspace_label() == "AI 工作流"

    tile_source = tmp_path / "tile_source"
    tile_source.mkdir()
    Image.new("RGB", (1800, 1200), color=(245, 245, 245)).save(tile_source / "big.jpg")
    (tile_source / "classes.txt").write_text("cat\ndog", encoding="utf-8")
    tile_report = slice_large_images(
        source_dir=tile_source,
        output_dir=tmp_path / "tiles",
        tile_size=800,
        overlap=100,
    )
    studio._open_workspace_from_ai("detect", tile_report.output_dir)
    assert studio.detect_editor.project_dir == tile_report.output_dir.resolve()
    assert studio.detect_editor.current_image_count() == tile_report.tile_count
    assert studio.active_workspace_label() == "检测框标注"

    root.destroy()

print("PHASE3_STUDIO_SELFTEST_OK")
