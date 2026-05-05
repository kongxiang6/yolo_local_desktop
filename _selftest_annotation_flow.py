import sys
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import annotation_editor
import annotation_support
import annotation_ui_support


annotation_editor.messagebox.showinfo = lambda *a, **k: None
annotation_editor.messagebox.showwarning = lambda *a, **k: None
annotation_editor.messagebox.showerror = lambda *a, **k: None


captured = {
    "notice": [],
    "export": [],
    "auto": [],
    "switch": 0,
}


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
    session_path = tmp_path / "sessions.json"

    root = tk.Tk()
    root.geometry("1180x680+40+40")

    editor = annotation_editor.DetectionAnnotationEditor(
        root,
        session_path=session_path,
        on_state_change=lambda: None,
        on_notice=lambda message: captured["notice"].append(message),
        on_export_request=lambda image_dir, class_names: captured["export"].append((Path(image_dir), list(class_names))),
        on_auto_label_request=lambda image_dir, image_path_or_none, model_ref, config, class_names: captured["auto"].append(
            (
                Path(image_dir),
                None if image_path_or_none is None else Path(image_path_or_none),
                model_ref,
                dict(config),
                list(class_names),
            )
        ),
        on_switch_to_train=lambda: captured.__setitem__("switch", captured["switch"] + 1),
    )
    editor.pack(fill="both", expand=True)
    root.update_idletasks()
    root.update()

    sidebar_bbox = editor.sidebar_scroll.canvas.bbox("all")
    assert sidebar_bbox is not None
    assert sidebar_bbox[3] > editor.sidebar_scroll.canvas.winfo_height()
    assert editor.auto_box.winfo_exists()
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
    editor.class_text.insert("1.0", "cat\ndog")
    editor.apply_class_names()
    assert editor.class_names == ["cat", "dog"]
    assert (tmp_path / "classes.txt").exists()

    editor.boxes = [annotation_support.AnnotationBox(class_id=0, x1=10, y1=12, x2=60, y2=50)]
    editor.save_current_annotations(silent=True)
    label_path = tmp_path / "sample.txt"
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8").strip().startswith("0 ")

    editor.auto_model_var.set("yolo11n.pt")
    editor.auto_conf_var.set("0.25")
    editor.auto_iou_var.set("0.70")
    editor.auto_imgsz_var.set("640")
    editor.auto_device_var.set("cpu")
    editor.auto_label_current()
    assert len(captured["auto"]) == 1
    auto_call = captured["auto"][0]
    assert auto_call[0] == tmp_path.resolve()
    assert auto_call[1] == image_path.resolve()
    assert auto_call[2] == "yolo11n.pt"
    assert auto_call[3]["device"] == "cpu"
    assert auto_call[4] == ["cat", "dog"]

    editor.export_training_dataset()
    assert len(captured["export"]) == 1
    assert captured["export"][0][0] == tmp_path.resolve()
    assert captured["export"][0][1] == ["cat", "dog"]

    editor.apply_auto_label_result(
        {
            "image_dir": str(tmp_path.resolve()),
            "updated_files": [str(image_path.resolve())],
            "model_class_names": ["cat", "dog"],
        }
    )
    assert editor.current_image_name() == "sample.png"

    editor.class_text.delete("1.0", "end")
    editor.class_text.insert("1.0", "custom0")
    editor.apply_class_names()
    editor.apply_auto_label_result(
        {
            "image_dir": str(tmp_path.resolve()),
            "updated_files": [str(image_path.resolve())],
            "model_class_names": ["custom0", "car", "bus"],
        }
    )
    assert editor.class_names == ["custom0", "car", "bus"]
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8").splitlines() == ["custom0", "car", "bus"]

    root.destroy()

print("ANNOTATION_FLOW_SELFTEST_OK")
