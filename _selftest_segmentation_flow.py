import sys
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import annotation_support
import annotation_ui_support
import segmentation_editor


segmentation_editor.messagebox.showinfo = lambda *a, **k: None
segmentation_editor.messagebox.showwarning = lambda *a, **k: None
segmentation_editor.messagebox.showerror = lambda *a, **k: None


captured = {
    "notice": [],
    "export": [],
    "switch": 0,
}


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 90), color=(255, 255, 255)).save(image_path)
    session_path = tmp_path / "seg_sessions.json"

    root = tk.Tk()
    root.geometry("1180x680+40+40")

    editor = segmentation_editor.SegmentationAnnotationEditor(
        root,
        session_path=session_path,
        on_state_change=lambda: None,
        on_notice=lambda message: captured["notice"].append(message),
        on_export_request=lambda image_dir, class_names: captured["export"].append((Path(image_dir), list(class_names))),
        on_switch_to_train=lambda: captured.__setitem__("switch", captured["switch"] + 1),
    )
    editor.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()

    sidebar_bbox = editor.sidebar_scroll.canvas.bbox("all")
    assert sidebar_bbox is not None
    assert sidebar_bbox[3] > editor.sidebar_scroll.canvas.winfo_height()
    assert editor.polygon_box.winfo_exists()
    assert editor.export_box.winfo_exists()
    yview_before = editor.sidebar_scroll.canvas.yview()
    result = annotation_ui_support.VerticalScrolledFrame._dispatch_mousewheel(
        SimpleNamespace(widget=editor.class_text, delta=-120)
    )
    root.update_idletasks()
    root.update()
    yview_after = editor.sidebar_scroll.canvas.yview()
    assert result == "break"
    assert yview_after != yview_before

    editor.load_project(tmp_path)
    assert editor.project_dir == tmp_path.resolve()
    assert editor.current_image_count() == 1
    assert editor.current_image_name() == "sample.png"

    editor.class_text.delete("1.0", "end")
    editor.class_text.insert("1.0", "car\nbus")
    editor.apply_class_names()
    assert editor.class_names == ["car", "bus"]

    editor.class_list.selection_clear(0, "end")
    editor.class_list.selection_set(1)
    editor.class_list.activate(1)
    editor.current_points = [(10, 12), (70, 14), (68, 60), (12, 62)]
    editor.finish_current_polygon()
    assert len(editor.polygons) == 1
    assert editor.polygons[0].class_id == 1

    label_path = tmp_path / "sample.txt"
    assert label_path.exists()
    polygons = annotation_support.load_yolo_polygons(label_path, 120, 90)
    assert len(polygons) == 1
    assert polygons[0].class_id == 1

    editor.export_training_dataset()
    assert len(captured["export"]) == 1
    assert captured["export"][0][0] == tmp_path.resolve()
    assert captured["export"][0][1] == ["car", "bus"]

    root.destroy()

print("SEGMENTATION_FLOW_SELFTEST_OK")
