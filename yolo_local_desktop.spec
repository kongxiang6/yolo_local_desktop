# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


project_dir = Path.cwd()
app_name = "YOLO训练工具"
fast_build = (os.environ.get("YOLO_TOOL_BUILD_FAST", "0") == "1")

datas = [
    (str(project_dir / "backend.py"), "."),
    (str(project_dir / "assets" / "yolotool_icon.png"), "assets"),
    (str(project_dir / "assets" / "yolotool_icon.ico"), "assets"),
    (str(project_dir / "contracts" / "train_capabilities.json"), "contracts"),
    (str(project_dir / "contracts" / "export_capabilities.json"), "contracts"),
    (str(project_dir / "vendor_backend" / "export_capabilities.json"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "export_capabilities.py"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "prepare_detection_dataset.py"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "runtime_preflight.py"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "runtime_installer.py"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "train_capabilities.json"), "vendor_backend"),
    (str(project_dir / "vendor_backend" / "yolo_runner.py"), "vendor_backend"),
]


a = Analysis(
    ["app.py"],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not fast_build,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_dir / "assets" / "yolotool_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=not fast_build,
    upx_exclude=[],
    name=app_name,
)
