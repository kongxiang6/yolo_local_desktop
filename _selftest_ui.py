import sys
from pathlib import Path
sys.path.insert(0, r"I:\AI\yolo_local_desktop")
import tkinter as tk
import app

app.messagebox.showinfo = lambda *a, **k: None
app.messagebox.showwarning = lambda *a, **k: None
app.messagebox.askyesno = lambda *a, **k: True

root = tk.Tk()
root.geometry('1400x980+40+40')
ui = app.App(root)
root.update_idletasks()
root.update()

assert ui.train_preset_var.get() == ''
assert ui.train_recommended_preset_var.get() == ''
assert ui.export_preset_var.get() == ''
assert ui.export_recommended_preset_var.get() == ''

# 推荐预设选择即时生效
train_choices = list(ui.train_preset_combo.values)
assert train_choices, 'no train recommended presets'
ui.train_recommended_preset_var.set(train_choices[0])
ui._load_recommended_preset('train')
assert ui.train_recommended_preset_var.get() == train_choices[0]

# 自定义预设按输入框名称保存/加载/删除
preset_name = 'selftest_custom_preset'
ui.train_preset_var.set(preset_name)
ui._save_preset('train')
preset_path = ui._find_preset_path('train', preset_name)
assert preset_path.exists(), f'custom preset not saved: {preset_path}'
ui._load_preset('train', notify=False)
assert ui.train_recommended_preset_var.get() == '', 'custom preset load should clear recommended selection'
ui._delete_preset('train')
assert not preset_path.exists(), 'custom preset not deleted'

# 找一个当前可见的下拉框，验证展开和重定位

def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)

visible_combo = next(w for w in walk(root) if isinstance(w, app.SmartComboBox) and w.winfo_ismapped())
owner_scroll = visible_combo._find_scrollable_parent()
assert owner_scroll is not None, 'visible combo has no scrollable parent'
visible_combo.open_popup()
root.update_idletasks()
root.update()
assert visible_combo.popup is not None, 'popup did not open'
geo1 = visible_combo.popup.geometry()
owner_scroll.canvas.yview_scroll(8, 'units')
root.update_idletasks()
root.update()
popup = visible_combo.popup
if popup is None or not popup.winfo_exists():
    pass
else:
    geo2 = popup.geometry()
    assert geo1 != geo2, f'popup did not move with scroll: {geo1} == {geo2}'
    visible_combo.close_popup()

root.destroy()
print('APP_SELFTEST_OK')
