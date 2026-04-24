from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


MIN_PYTHON = (3, 9)
MAX_PYTHON_WINDOWS = (3, 13)
MAX_PYTHON_DEFAULT = (3, 14)
PIP_BOOTSTRAP_URL = "https://bootstrap.pypa.io/get-pip.py"
EMBEDDED_PYTHON_VERSION = "3.13.13"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0


def python_version_supported() -> bool:
    current = sys.version_info[:2]
    max_python = MAX_PYTHON_WINDOWS if sys.platform.startswith("win") else MAX_PYTHON_DEFAULT
    return MIN_PYTHON <= current <= max_python


def detect_nvidia_environment() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, object] = {"available": False, "gpu_name": "", "driver_version": "", "cuda_version": ""}
    if not executable:
        return result

    try:
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
        summary = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
    except OSError:
        return result

    if query.returncode != 0 and summary.returncode != 0:
        return result

    first_line = (query.stdout or "").splitlines()[0].strip() if (query.stdout or "").splitlines() else ""
    if first_line:
        parts = [item.strip() for item in first_line.split(",")]
        if parts:
            result["gpu_name"] = parts[0]
        if len(parts) > 1:
            result["driver_version"] = parts[1]

    summary_text = (summary.stdout or "") + "\n" + (summary.stderr or "")
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", summary_text)
    if match:
        result["cuda_version"] = match.group(1)

    result["available"] = bool(result["gpu_name"] or result["cuda_version"])
    return result


def choose_torch_index(cuda_version: str) -> tuple[str, str]:
    if not cuda_version:
        return "cpu", ""
    try:
        major, minor = [int(item) for item in cuda_version.split(".")[:2]]
    except ValueError:
        return "cpu", ""
    if (major, minor) >= (12, 8):
        return "cu128", "https://download.pytorch.org/whl/cu128"
    if (major, minor) >= (12, 6):
        return "cu126", "https://download.pytorch.org/whl/cu126"
    if (major, minor) >= (11, 8):
        return "cu118", "https://download.pytorch.org/whl/cu118"
    return "cpu", ""


def _process_env() -> dict[str, str]:
    process_env = os.environ.copy()
    process_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    return process_env


def _module_available(name: str) -> bool:
    importlib.invalidate_caches()
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def has_pip() -> bool:
    return _module_available("pip")


def has_ensurepip() -> bool:
    return _module_available("ensurepip")


def detect_pip_bootstrap_strategy() -> str:
    if has_pip():
        return "已检测到 pip，可直接安装依赖"
    if has_ensurepip():
        return "当前 Python 支持 ensurepip，会先本地初始化 pip"
    return "当前 Python 不带 ensurepip，将自动在线下载 get-pip.py 安装 pip"


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path.cwd()),
        env=_process_env(),
        creationflags=NO_WINDOW_FLAGS,
        check=False,
    )


def choose_embedded_python_url() -> str:
    if not sys.platform.startswith("win"):
        raise RuntimeError("当前只支持在 Windows 上自动创建内置 runtime。")

    machine = platform.machine().lower()
    if any(token in machine for token in ("amd64", "x86_64", "x64", "arm64")):
        arch = "amd64"
    else:
        arch = "win32"
    return f"https://www.python.org/ftp/python/{EMBEDDED_PYTHON_VERSION}/python-{EMBEDDED_PYTHON_VERSION}-embed-{arch}.zip"


def download_file(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if not payload:
            raise RuntimeError("下载内容为空。")
        target_path.write_bytes(payload)
        return target_path
    except Exception as exc:
        if not sys.platform.startswith("win"):
            raise RuntimeError(f"下载文件失败：{exc}") from exc

        escaped_url = url.replace("'", "''")
        escaped_path = str(target_path).replace("'", "''")
        powershell_command = [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
                f"Invoke-WebRequest -Uri '{escaped_url}' -OutFile '{escaped_path}'"
            ),
        ]
        fallback = _run_capture(powershell_command)
        if fallback.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
            return target_path
        detail = (fallback.stderr or fallback.stdout or "").strip()
        raise RuntimeError(f"下载文件失败：{exc}；PowerShell 回退也失败：{detail}") from exc


def configure_embedded_python(target_dir: Path) -> None:
    (target_dir / "DLLs").mkdir(parents=True, exist_ok=True)
    (target_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (target_dir / "Scripts").mkdir(parents=True, exist_ok=True)
    version_parts = EMBEDDED_PYTHON_VERSION.split(".")
    zip_name = f"python{version_parts[0]}{version_parts[1]}.zip"
    pth_filename = f"python{version_parts[0]}{version_parts[1]}._pth"
    pth_content = f"{zip_name}\nDLLs\nLib\nLib\\site-packages\n.\nimport site\n"
    for filename in (pth_filename, "python._pth"):
        (target_dir / filename).write_text(pth_content, encoding="utf-8")


def install_embedded_runtime(target_python: Path, log: Callable[[str], None] | None = None) -> dict[str, str]:
    logger = log or (lambda _message: None)
    target_python = target_python.expanduser().resolve(strict=False)
    target_dir = target_python.parent
    runtime_root = target_dir.parent
    archive_path = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / f"python-{EMBEDDED_PYTHON_VERSION}-embed.zip"
    download_url = choose_embedded_python_url()

    logger(f"正在下载内置 Python 运行时：{download_url}")
    download_file(download_url, archive_path)
    logger(f"已下载内置 Python 压缩包：{archive_path}")

    if runtime_root.exists():
        shutil.rmtree(runtime_root, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"内置 Python 压缩包损坏：{archive_path}") from exc

    configure_embedded_python(target_dir)
    if not target_python.exists():
        raise RuntimeError(f"内置 Python 创建失败，未找到解释器：{target_python}")

    logger(f"内置 Python 已创建：{target_python}")
    return {
        "target_python": str(target_python),
        "runtime_root": str(runtime_root),
        "python_version": EMBEDDED_PYTHON_VERSION,
        "download_url": download_url,
    }


def download_get_pip(target_dir: Path | None = None) -> Path:
    cache_dir = target_dir or (Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "get-pip.py"

    try:
        with urllib.request.urlopen(PIP_BOOTSTRAP_URL, timeout=60) as response:
            payload = response.read()
        if not payload:
            raise RuntimeError("下载内容为空。")
        target_path.write_bytes(payload)
        return target_path
    except Exception as exc:
        if not sys.platform.startswith("win"):
            raise RuntimeError(f"下载 get-pip.py 失败：{exc}") from exc

        escaped_url = PIP_BOOTSTRAP_URL.replace("'", "''")
        escaped_path = str(target_path).replace("'", "''")
        powershell_command = [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
                f"Invoke-WebRequest -Uri '{escaped_url}' -OutFile '{escaped_path}'"
            ),
        ]
        fallback = _run_capture(powershell_command)
        if fallback.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
            return target_path
        detail = (fallback.stderr or fallback.stdout or "").strip()
        raise RuntimeError(f"下载 get-pip.py 失败：{exc}；PowerShell 回退也失败：{detail}") from exc


def bootstrap_pip(log: Callable[[str], None] | None = None) -> dict[str, str]:
    logger = log or (lambda _message: None)

    if has_pip():
        logger("已检测到 pip，跳过引导步骤。")
        return {"method": "existing", "message": "已检测到 pip，直接继续安装依赖。"}

    ensurepip_command = [sys.executable, "-m", "ensurepip", "--upgrade"]
    if has_ensurepip():
        logger("检测到 ensurepip，正在本地初始化 pip。")
        return_code = stream_command(ensurepip_command)
        if return_code == 0 and has_pip():
            logger("ensurepip 初始化成功。")
            return {"method": "ensurepip", "message": "已通过 ensurepip 初始化 pip。"}
        logger("ensurepip 初始化失败，改为在线下载 get-pip.py 安装 pip。")
    else:
        logger("当前 Python 不包含 ensurepip，改为在线下载 get-pip.py 安装 pip。")

    logger(f"正在下载 pip 引导脚本：{PIP_BOOTSTRAP_URL}")
    script_path = download_get_pip()
    logger(f"已下载 pip 引导脚本：{script_path}")

    install_command = [sys.executable, str(script_path), "--disable-pip-version-check"]
    return_code = stream_command(install_command)
    if return_code != 0:
        raise RuntimeError(f"在线安装 pip 失败，退出码为 {return_code}。")
    if not has_pip():
        raise RuntimeError("在线安装 pip 已执行完成，但当前解释器仍未检测到 pip。")

    logger("在线安装 pip 成功。")
    return {"method": "get-pip", "message": "已通过 get-pip.py 在线安装 pip。", "script": str(script_path)}


def build_install_plan() -> dict[str, object]:
    if not python_version_supported():
        max_python = MAX_PYTHON_WINDOWS if sys.platform.startswith("win") else MAX_PYTHON_DEFAULT
        raise RuntimeError(
            f"当前 Python 版本为 {platform.python_version()}，超出推荐自动安装范围。"
            f"请优先切换到工具内置 runtime，或使用 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
            f"~ {max_python[0]}.{max_python[1]}。"
        )

    gpu = detect_nvidia_environment()
    accelerator, index_url = choose_torch_index(str(gpu.get("cuda_version") or ""))
    package_step = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"]
    if index_url:
        package_step.extend(["--index-url", index_url])

    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        package_step,
        [sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics", "pillow", "pyyaml"],
    ]

    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "accelerator": accelerator,
        "torch_index": index_url or "默认 PyPI / CPU",
        "gpu": gpu,
        "pip_bootstrap": detect_pip_bootstrap_strategy(),
        "commands": commands,
        "notes": [
            "安装方案会根据 NVIDIA 环境自动判断。",
            "CUDA 版本到 PyTorch 下载源的映射策略：12.8+ -> cu128，12.6+ -> cu126，11.8+ -> cu118，否则回退 CPU。",
            "如果当前 Python 没有 pip，会自动尝试 ensurepip；若不可用，则在线下载 get-pip.py 安装。",
        ],
    }


def stream_command(command: list[str]) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path.cwd()),
        env=_process_env(),
        creationflags=NO_WINDOW_FLAGS,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return process.wait()
