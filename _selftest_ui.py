import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk

import app


app.messagebox.showinfo = lambda *a, **k: None
app.messagebox.showwarning = lambda *a, **k: None
app.messagebox.askyesno = lambda *a, **k: True


root = app.create_app_root()
root.geometry("1400x980+40+40")
ui = app.App(root)
root.update_idletasks()
root.update()

assert ui.train_sections["env"].expanded is True
assert ui.export_sections["env"].expanded is True

assert ui.train_preset_var.get() == ""
assert ui.train_recommended_preset_var.get() == ""
assert ui.export_preset_var.get() == ""
assert ui.export_recommended_preset_var.get() == ""

ui._show_tab("annotation")
root.update_idletasks()
root.update()
assert not ui.left_panel.winfo_ismapped(), "annotation tab should hide the outer summary panel"
assert int(ui.right_panel.grid_info().get("columnspan", 1)) == 2, "annotation tab should expand to the full content width"

ui._show_tab("train")
root.update_idletasks()
root.update()
assert ui.left_panel.winfo_ismapped(), "train tab should restore the outer summary panel"
assert int(ui.right_panel.grid_info().get("columnspan", 1)) == 1, "train tab should restore the split layout"


def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)


# 推荐预设选择即时生效
train_choices = list(ui.train_preset_combo.values)
assert train_choices, "no train recommended presets"
ui.train_recommended_preset_var.set(train_choices[0])
ui._load_recommended_preset("train")
assert ui.train_recommended_preset_var.get() == train_choices[0]

# 自定义预设按输入框名称保存/加载/删除
preset_name = "selftest_custom_preset"
ui.train_preset_var.set(preset_name)
ui._save_preset("train")
preset_path = ui._find_preset_path("train", preset_name)
assert preset_path.exists(), f"custom preset not saved: {preset_path}"
ui._load_preset("train", notify=False)
assert ui.train_recommended_preset_var.get() == "", "custom preset load should clear recommended selection"
ui._delete_preset("train")
assert not preset_path.exists(), "custom preset not deleted"

# 找一个当前可见的下拉框，验证展开后滚轮滚动时能跟着界面移动
visible_combo = next(w for w in walk(root) if isinstance(w, app.SmartComboBox) and w.winfo_ismapped())
owner_scroll = visible_combo._find_scrollable_parent()
assert owner_scroll is not None, "visible combo has no scrollable parent"
visible_combo.open_popup()
root.update_idletasks()
root.update()
assert visible_combo.popup is not None, "popup did not open"
geo1 = visible_combo.popup.geometry()
owner_scroll.canvas.yview_moveto(0.2)
scroll_widget = next(
    widget
    for widget in walk(visible_combo)
    if widget is not visible_combo and app.ScrollableFrame._resolve_from_widget(widget) is owner_scroll
)
result = app.ScrollableFrame._dispatch_mousewheel(SimpleNamespace(widget=scroll_widget, delta=-120))
root.update_idletasks()
root.update()
popup = visible_combo.popup
assert result == "break", "mousewheel dispatch did not scroll owner"
if popup is None or not popup.winfo_exists():
    pass
else:
    geo2 = popup.geometry()
    assert geo1 != geo2, f"popup did not move with scroll: {geo1} == {geo2}"
    visible_combo.close_popup()

root.destroy()
print("APP_SELFTEST_OK")
