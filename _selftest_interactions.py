import sys
from pathlib import Path

sys.path.insert(0, r"I:\AI\yolo_local_desktop")

import tkinter as tk

import app


messages: list[tuple[str, tuple, dict]] = []


def _record(kind: str):
    return lambda *args, **kwargs: messages.append((kind, args, kwargs)) or None


app.messagebox.showinfo = _record("info")
app.messagebox.showwarning = _record("warning")
app.messagebox.showerror = _record("error")
app.messagebox.askyesno = lambda *args, **kwargs: True


def walk(widget: tk.Widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)


root = tk.Tk()
root.geometry("1500x980+40+40")
ui = app.App(root)
root.update_idletasks()
root.update()

# 启动状态
assert ui.active_tab.get() == "train"
assert ui.train_scroll.winfo_ismapped()
assert not ui.export_scroll.winfo_ismapped()

# 页签切换 + 滚动复位
ui.train_scroll.canvas.yview_moveto(1.0)
ui._show_tab("export")
root.update_idletasks()
root.update()
assert ui.export_scroll.winfo_ismapped()
assert not ui.train_scroll.winfo_ismapped()
assert ui.export_scroll.canvas.yview()[0] == 0.0

ui.export_scroll.canvas.yview_moveto(1.0)
ui._show_tab("train")
root.update_idletasks()
root.update()
assert ui.train_scroll.canvas.yview()[0] == 0.0

# 训练四种子模式对应区块显示
expected_sections = {
    "train": {"env", "train_entry", "prep", "basic", "save_val", "optimization", "augmentation", "loss_task", "run"},
    "val": {"env", "val_entry", "val_params", "run"},
    "predict": {"env", "predict_entry", "predict_params", "run"},
    "track": {"env", "track_entry", "track_params", "run"},
}
for action, expected in expected_sections.items():
    ui.train_scroll.canvas.yview_moveto(1.0)
    ui._show_train_action(action)
    root.update_idletasks()
    root.update()
    assert ui.train_scroll.canvas.yview()[0] == 0.0
    for section_id, section in ui.train_sections.items():
        assert section.winfo_ismapped() == (section_id in expected), (action, section_id)

# 折叠区展开/收起
env_section = ui.train_sections["env"]
start_state = env_section.expanded
env_section._toggle()
root.update_idletasks()
root.update()
assert env_section.expanded != start_state
env_section._toggle()
root.update_idletasks()
root.update()
assert env_section.expanded == start_state

# 导出参数显隐
ui._show_tab("export")
ui.export_sections["params"].set_expanded(True)
root.update_idletasks()
root.update()

ui.export_format_label_var.set(ui.export_label_map["onnx"])
ui._refresh_export_visibility()
root.update_idletasks()
root.update()
assert ui.export_fields["opset"]["container"].winfo_ismapped()
assert ui.export_fields["simplify"]["container"].winfo_ismapped()
assert not ui.export_fields["keras"]["container"].winfo_ismapped()

ui.export_format_label_var.set(ui.export_label_map["saved_model"])
ui._refresh_export_visibility()
root.update_idletasks()
root.update()
assert ui.export_fields["keras"]["container"].winfo_ismapped()
assert not ui.export_fields["opset"]["container"].winfo_ismapped()

# 下拉框交互
ui._show_tab("train")
ui._show_train_action("train")
root.update_idletasks()
root.update()
visible_combo = next(widget for widget in walk(root) if isinstance(widget, app.SmartComboBox) and widget.winfo_ismapped())
owner_scroll = visible_combo._find_scrollable_parent()
assert owner_scroll is not None
visible_combo.open_popup()
root.update_idletasks()
root.update()
assert visible_combo.popup is not None
geo1 = visible_combo.popup.geometry()
owner_scroll.canvas.yview_scroll(20, "units")
root.update_idletasks()
root.update()
popup = visible_combo.popup
if popup is not None and popup.winfo_exists():
    geo2 = popup.geometry()
    assert geo1 != geo2
    visible_combo.close_popup()
else:
    assert visible_combo.popup is None

# 预设保存 / 加载 / 删除
for context, name_var, recommended_var in (
    ("train", ui.train_preset_var, ui.train_recommended_preset_var),
    ("export", ui.export_preset_var, ui.export_recommended_preset_var),
):
    preset_name = f"selftest_{context}_preset"
    name_var.set(preset_name)
    if context == "train":
        ui.train_fields["epochs"]["var"].set("77")
    else:
        ui.export_sections["params"].set_expanded(True)
        ui.export_fields["batch"]["var"].set("3")
    ui._save_preset(context)
    preset_path = ui._find_preset_path(ui._preset_scope_from_context(context), preset_name)
    assert preset_path.exists(), preset_path
    if context == "train":
        ui.train_fields["epochs"]["var"].set("11")
    else:
        ui.export_fields["batch"]["var"].set("1")
    ui._load_preset(context, notify=False)
    if context == "train":
        assert ui.train_fields["epochs"]["var"].get() == "77"
    else:
        assert ui.export_fields["batch"]["var"].get() == "3"
    assert recommended_var.get() == ""
    ui._delete_preset(context)
    assert not preset_path.exists(), preset_path

# 缺少参数时的提示
messages.clear()
ui.train_data_var.set("暂无")
ui.start_train()
assert any(kind == "warning" for kind, *_ in messages)

messages.clear()
ui.val_weights_var.set("未选择")
ui.start_val()
assert any(kind == "warning" for kind, *_ in messages)

messages.clear()
ui.predict_weights_var.set("未选择")
ui.predict_source_var.set("")
ui.start_predict()
assert any(kind == "warning" for kind, *_ in messages)

messages.clear()
ui.track_weights_var.set("未选择")
ui.track_source_var.set("")
ui.start_track()
assert any(kind == "warning" for kind, *_ in messages)

messages.clear()
ui.export_weights_var.set("未选择")
ui.start_export()
assert any(kind == "warning" for kind, *_ in messages)

# 打开结果 / 日志的无结果提示
messages.clear()
ui.result_location_var.set("")
ui.open_result_location()
assert any(kind == "info" for kind, *_ in messages)

messages.clear()
ui.current_log_path = Path(r"I:\AI\yolo_local_desktop\logs\__missing_selftest__.log")
ui.open_log_file()
assert any(kind == "info" for kind, *_ in messages)

root.destroy()
print("UI_INTERACTION_SELFTEST_OK")
