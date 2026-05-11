from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def list_related_processes() -> list[str]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Process | Where-Object { $_.ProcessName -like '*YOLO*' -or $_.ProcessName -like '*python*' } | "
        "Select-Object ProcessName,Id,MainWindowTitle,Path | ConvertTo-Json -Compress",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (result.stdout or "").strip()
    return [output] if output else []


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python _selftest_process_cleanup.py <exe路径或python脚本路径>")
        return 1

    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.exists():
        print(f"目标不存在: {target}")
        return 1

    if target.suffix.lower() == ".py":
        launch = [sys.executable, str(target)]
    else:
        launch = [str(target)]

    before = list_related_processes()
    print("=== 启动前相关进程 ===")
    print(before[0] if before else "无")

    process = subprocess.Popen(launch)
    print(f"已启动: pid={process.pid}")
    print("请手动关闭软件窗口，脚本会在关闭后检查是否有残留进程。")

    process.wait()
    print(f"主进程退出码: {process.returncode}")
    time.sleep(2.5)

    after = list_related_processes()
    print("=== 关闭后相关进程 ===")
    print(after[0] if after else "无")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())