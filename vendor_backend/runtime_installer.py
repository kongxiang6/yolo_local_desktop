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
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


MIN_PYTHON = (3, 9)
MAX_PYTHON_WINDOWS = (3, 13)
MAX_PYTHON_DEFAULT = (3, 14)
PIP_BOOTSTRAP_URL = "https://bootstrap.pypa.io/get-pip.py"
EMBEDDED_PYTHON_VERSION = "3.13.13"
DEFAULT_PIP_INDEX_URL = os.environ.get("YOLO_TOOL_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
DEFAULT_PIP_EXTRA_INDEX_URL = os.environ.get("YOLO_TOOL_PIP_EXTRA_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple/")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0
PROBE_TIMEOUT_SECONDS = 4.0
FALLBACK_HOST_KEYWORDS = (
    "python.org",
    "bootstrap.pypa.io",
    "pypi.org",
    "pythonhosted.org",
    "download.pytorch.org",
)


def python_version_supported() -> bool:
    current = sys.version_info[:2]
    max_python = MAX_PYTHON_WINDOWS if sys.platform.startswith("win") else MAX_PYTHON_DEFAULT
    return MIN_PYTHON <= current <= max_python


def detect_nvidia_environment() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, object] = {
        "available": False,
        "gpu_name": "",
        "gpu_architecture": "",
        "driver_version": "",
        "cuda_version": "",
    }
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
            [executable, "-q"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
        table = subprocess.run(
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

    if query.returncode != 0 and summary.returncode != 0 and table.returncode != 0:
        return result

    first_line = (query.stdout or "").splitlines()[0].strip() if (query.stdout or "").splitlines() else ""
    if first_line:
        parts = [item.strip() for item in first_line.split(",")]
        if parts:
            result["gpu_name"] = parts[0]
        if len(parts) > 1:
            result["driver_version"] = parts[1]

    summary_text = (summary.stdout or "") + "\n" + (summary.stderr or "") + "\n" + (table.stdout or "") + "\n" + (table.stderr or "")
    match = re.search(r"CUDA\s+Version\s*:?\s*([0-9]+(?:\.[0-9]+)?)", summary_text, flags=re.IGNORECASE)
    if match:
        result["cuda_version"] = match.group(1)
    architecture_match = re.search(r"Product Architecture\s*:\s*(.+)", summary_text)
    if architecture_match:
        result["gpu_architecture"] = architecture_match.group(1).strip()

    if not result["gpu_architecture"]:
        result["gpu_architecture"] = infer_gpu_architecture(str(result["gpu_name"] or ""))

    result["available"] = bool(result["gpu_name"] or result["cuda_version"])
    return result


def infer_gpu_architecture(gpu_name: str) -> str:
    normalized = gpu_name.lower()
    if re.search(r"\brtx\s*50(?:50|60|70|80|90)\w*\b", normalized) or re.search(r"\b(?:b100|b200|gb200)\b", normalized) or "blackwell" in normalized:
        return "Blackwell"
    if (
        re.search(r"\brtx\s*40(?:50|60|70|80|90)\w*\b", normalized)
        or re.search(r"\brtx\s*(?:6000|5000|4500|4000|3500|3000|2000)\s+ada\b", normalized)
        or re.search(r"\b(?:l4|l20|l40|l40s)\b", normalized)
        or "ada" in normalized
    ):
        return "Ada"
    if (
        re.search(r"\brtx\s*30(?:50|60|70|80|90)\w*\b", normalized)
        or re.search(r"\brtx\s*a[24658]000\b", normalized)
        or re.search(r"\brtx\s*a(?:2000|4000|4500|5000|5500|6000)\b", normalized)
        or re.search(r"\b(?:a2|a10|a16|a30|a40|a100|a800)\b", normalized)
        or "ampere" in normalized
    ):
        return "Ampere"
    if re.search(r"\b(?:h100|h200|gh200)\b", normalized) or "hopper" in normalized:
        return "Hopper"
    if (
        re.search(r"\brtx\s*20(?:50|60|70|80)\w*\b", normalized)
        or re.search(r"\bgtx\s*16", normalized)
        or re.search(r"\b(?:t4|titan\s+rtx)\b", normalized)
        or re.search(r"\bquadro\s+rtx\b", normalized)
        or "turing" in normalized
    ):
        return "Turing"
    if re.search(r"\b(?:v100|titan\s+v)\b", normalized) or "volta" in normalized:
        return "Volta"
    if (
        re.search(r"\bgtx\s*10", normalized)
        or re.search(r"\b(?:p4|p40|p100)\b", normalized)
        or re.search(r"\bquadro\s+p\d+", normalized)
        or "pascal" in normalized
    ):
        return "Pascal"
    if (
        re.search(r"\bgtx\s*9", normalized)
        or re.search(r"\bgtx\s*75[0-9]", normalized)
        or re.search(r"\b(?:m4|m40|m60)\b", normalized)
        or re.search(r"\bquadro\s+m\d+", normalized)
        or "maxwell" in normalized
    ):
        return "Maxwell"
    if (
        re.search(r"\bgtx\s*(?:6|7)", normalized)
        or re.search(r"\b(?:k20|k40|k80)\b", normalized)
        or re.search(r"\bquadro\s+k\d+", normalized)
        or "kepler" in normalized
    ):
        return "Kepler"
    return ""


def choose_torch_index(cuda_version: str, gpu_architecture: str = "") -> tuple[str, str]:
    legacy_architectures = {"maxwell", "pascal", "volta"}
    modern_architectures = {"blackwell", "ada", "lovelace", "hopper", "ampere", "turing"}
    unsupported_architectures = {"kepler", "fermi"}
    normalized_architecture = gpu_architecture.strip().lower()
    if normalized_architecture in unsupported_architectures:
        return "cpu", ""
    if normalized_architecture in legacy_architectures:
        return "cu126", "https://download.pytorch.org/whl/cu126"
    if not cuda_version:
        if normalized_architecture in modern_architectures:
            return "cu128", "https://download.pytorch.org/whl/cu128"
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


def detect_installed_torch_accelerator() -> str:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return ""
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return ""

    cuda_version = str(getattr(getattr(torch, "version", object()), "cuda", None) or "").strip()
    if not cuda_version:
        return "cpu"
    try:
        major, minor = [int(item) for item in cuda_version.split(".")[:2]]
    except ValueError:
        return f"cuda-{cuda_version}"
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (11, 8):
        return "cu118"
    return f"cuda-{cuda_version}"


def build_accelerator_summary(accelerator: str, gpu: dict[str, object] | None = None) -> dict[str, str]:
    gpu_info = gpu or {}
    gpu_available = bool(gpu_info.get("available"))
    gpu_name = str(gpu_info.get("gpu_name") or "").strip()
    gpu_architecture = str(gpu_info.get("gpu_architecture") or "").strip()
    cuda_version = str(gpu_info.get("cuda_version") or "").strip()

    if accelerator != "cpu":
        gpu_text = gpu_name or "NVIDIA 显卡"
        detail_parts = []
        if gpu_architecture:
            detail_parts.append(gpu_architecture)
        if cuda_version:
            detail_parts.append(f"CUDA {cuda_version}")
        cuda_text = f"，{'，'.join(detail_parts)}" if detail_parts else ""
        return {
            "accelerator_label": f"NVIDIA 显卡版（{accelerator.upper()}）",
            "hardware_label": f"已检测到 NVIDIA 显卡：{gpu_text}{cuda_text}",
        }

    if gpu_available:
        gpu_text = gpu_name or "NVIDIA 显卡"
        detail_parts = []
        if gpu_architecture:
            detail_parts.append(gpu_architecture)
        if cuda_version:
            detail_parts.append(f"CUDA {cuda_version}")
        cuda_text = f"，{'，'.join(detail_parts)}" if detail_parts else ""
        return {
            "accelerator_label": "CPU 版",
            "hardware_label": f"已检测到 NVIDIA 显卡：{gpu_text}{cuda_text}，但当前回退为 CPU 方案",
        }

    return {
        "accelerator_label": "CPU 版",
        "hardware_label": "未检测到可用 NVIDIA 显卡，当前按 CPU 方案处理",
    }


def build_torch_index_candidates(accelerator: str, official_index: str) -> list[str]:
    if accelerator == "cpu":
        return build_generic_pip_index_candidates()

    candidates: list[str] = []
    env_override = os.environ.get("YOLO_TOOL_TORCH_INDEX_URL", "").strip()
    if env_override:
        candidates.append(env_override)

    candidates.extend(
        [
            f"https://mirror.sjtu.edu.cn/pytorch-wheels/{accelerator}",
            f"https://mirrors.aliyun.com/pytorch-wheels/{accelerator}",
            f"https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/{accelerator}",
            official_index,
        ]
    )

    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.rstrip("/")
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def build_generic_pip_index_candidates() -> list[str]:
    env_override = os.environ.get("YOLO_TOOL_PIP_INDEX_URL", "").strip()
    candidates = [
        env_override,
        DEFAULT_PIP_INDEX_URL,
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://mirrors.huaweicloud.com/repository/pypi/simple",
        "https://mirrors.bfsu.edu.cn/pypi/web/simple",
    ]
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().rstrip("/")
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _process_env() -> dict[str, str]:
    process_env = os.environ.copy()
    process_env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_DEFAULT_TIMEOUT": "180",
            "PIP_RETRIES": "5",
            "PIP_PREFER_BINARY": "1",
        }
    )
    if DEFAULT_PIP_EXTRA_INDEX_URL:
        process_env.setdefault("PIP_EXTRA_INDEX_URL", DEFAULT_PIP_EXTRA_INDEX_URL)
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
    if any(token in machine for token in ("arm64", "aarch64")):
        arch = "arm64"
    elif any(token in machine for token in ("amd64", "x86_64", "x64")):
        arch = "amd64"
    else:
        arch = "win32"
    return f"https://www.python.org/ftp/python/{EMBEDDED_PYTHON_VERSION}/python-{EMBEDDED_PYTHON_VERSION}-embed-{arch}.zip"


def choose_embedded_python_urls() -> list[str]:
    official_url = choose_embedded_python_url()
    filename = Path(official_url).name
    env_override = os.environ.get("YOLO_TOOL_PYTHON_EMBED_URL", "").strip()
    candidates = [
        env_override,
        f"https://mirrors.aliyun.com/python-release/windows/{filename}",
        f"https://mirrors.huaweicloud.com/python/{EMBEDDED_PYTHON_VERSION}/{filename}",
        f"https://mirrors.tuna.tsinghua.edu.cn/python/{EMBEDDED_PYTHON_VERSION}/{filename}",
        official_url,
    ]
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().rstrip("/")
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def choose_get_pip_urls() -> list[str]:
    env_override = os.environ.get("YOLO_TOOL_GET_PIP_URL", "").strip()
    candidates = [
        env_override,
        "https://mirrors.aliyun.com/pypi/get-pip.py",
        PIP_BOOTSTRAP_URL,
    ]
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().rstrip("/")
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _download_cache_ready(target_path: Path) -> bool:
    if not target_path.exists() or target_path.stat().st_size <= 0:
        return False
    if target_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(target_path) as zip_file:
                return zip_file.testzip() is None
        except zipfile.BadZipFile:
            return False
    return True


def rank_candidate_urls(urls: list[str], log: Callable[[str], None] | None = None) -> list[str]:
    logger = log or (lambda _message: None)
    if len(urls) <= 1:
        return list(urls)

    def candidate_tier(url: str) -> int:
        host = (urlparse(url).netloc or "").lower()
        if any(keyword in host for keyword in FALLBACK_HOST_KEYWORDS):
            return 1
        return 0

    scored: list[tuple[int, float, str]] = []
    fallback: list[str] = []
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
                response.read(1)
            latency = time.perf_counter() - started
            scored.append((candidate_tier(url), latency, url))
            logger(f"测速成功：{url}（约 {latency:.2f} 秒）")
        except Exception as exc:
            fallback.append(url)
            logger(f"测速失败，保留回退：{url}（{exc}）")

    ordered = [url for _, _, url in sorted(scored, key=lambda item: (item[0], item[1]))]
    for url in fallback:
        if url not in ordered:
            ordered.append(url)
    if ordered:
        logger("下载源排序：" + " -> ".join(ordered))
    return ordered or list(urls)


def download_file(urls: str | list[str], target_path: Path, log: Callable[[str], None] | None = None) -> tuple[Path, str]:
    logger = log or (lambda _message: None)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _download_cache_ready(target_path):
        logger(f"复用已下载缓存：{target_path}")
        return target_path, "cache"

    candidates = [urls] if isinstance(urls, str) else list(urls)
    candidates = rank_candidate_urls(candidates, logger)
    errors: list[str] = []
    for index, url in enumerate(candidates, start=1):
        try:
            logger(f"尝试下载源 {index}/{len(candidates)}：{url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("下载内容为空。")
            target_path.write_bytes(payload)
            return target_path, url
        except Exception as exc:
            if sys.platform.startswith("win"):
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
                if fallback.returncode == 0 and _download_cache_ready(target_path):
                    return target_path, url
                detail = (fallback.stderr or fallback.stdout or "").strip()
                errors.append(f"{url} -> {exc}；PowerShell 回退失败：{detail}")
            else:
                errors.append(f"{url} -> {exc}")
            if target_path.exists() and not _download_cache_ready(target_path):
                target_path.unlink(missing_ok=True)
    raise RuntimeError("下载文件失败，已尝试以下地址：\n" + "\n".join(errors))


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
    download_urls = choose_embedded_python_urls()

    logger("正在下载内置 Python 运行时（优先尝试国内镜像）")
    _, download_url = download_file(download_urls, archive_path, logger)
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
    download_file(choose_get_pip_urls(), target_path)
    return target_path


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

    logger("正在下载 pip 引导脚本（优先尝试国内镜像）")
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


def build_install_plan(log: Callable[[str], None] | None = None) -> dict[str, object]:
    logger = log or (lambda _message: None)
    if not python_version_supported():
        max_python = MAX_PYTHON_WINDOWS if sys.platform.startswith("win") else MAX_PYTHON_DEFAULT
        raise RuntimeError(
            f"当前 Python 版本为 {platform.python_version()}，超出推荐自动安装范围。"
            f"请优先切换到工具内置 runtime，或使用 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
            f"~ {max_python[0]}.{max_python[1]}。"
        )

    logger("正在检测 NVIDIA 显卡和当前 Torch 环境。")
    gpu = detect_nvidia_environment()
    accelerator, index_url = choose_torch_index(
        str(gpu.get("cuda_version") or ""),
        str(gpu.get("gpu_architecture") or ""),
    )
    logger("正在测速常规 Python 依赖下载源。")
    generic_indexes = rank_candidate_urls(build_generic_pip_index_candidates(), logger)
    if accelerator == "cpu":
        torch_indexes = list(generic_indexes)
        logger("当前推荐 CPU 方案，PyTorch 将复用常规 pip 下载源。")
    else:
        logger("正在测速 PyTorch 下载源。")
        torch_indexes = rank_candidate_urls(build_torch_index_candidates(accelerator, index_url), logger)

    installed_torch_accelerator = detect_installed_torch_accelerator()
    force_torch_reinstall = bool(installed_torch_accelerator and installed_torch_accelerator != accelerator)

    def build_pip_command(index_url_value: str, *packages: str, force_reinstall: bool = False) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--timeout",
            "180",
            "--retries",
            "5",
            "-i",
            index_url_value,
        ]
        if force_reinstall:
            command.append("--force-reinstall")
        command.extend(packages)
        return command

    tool_steps = [build_pip_command(index_url_value, "pip", "setuptools", "wheel") for index_url_value in generic_indexes]
    app_steps = [build_pip_command(index_url_value, "ultralytics", "pillow", "pyyaml") for index_url_value in generic_indexes]

    torch_steps: list[list[str]] = []
    if accelerator == "cpu":
        torch_steps = [
            build_pip_command(index_url_value, "torch", "torchvision", "torchaudio", force_reinstall=force_torch_reinstall)
            for index_url_value in generic_indexes
        ]
    else:
        for torch_index in torch_indexes:
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--timeout",
                "180",
                "--retries",
                "5",
                "--index-url",
                torch_index,
            ]
            if force_torch_reinstall:
                command.append("--force-reinstall")
            command.extend(["torch", "torchvision", "torchaudio"])
            torch_steps.append(command)

    commands = [
        tool_steps,
        torch_steps,
        app_steps,
    ]

    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "accelerator": accelerator,
        "torch_index": " -> ".join(torch_indexes),
        "pip_indexes": " -> ".join(generic_indexes),
        "gpu": gpu,
        "installed_torch_accelerator": installed_torch_accelerator,
        "force_torch_reinstall": force_torch_reinstall,
        "pip_bootstrap": detect_pip_bootstrap_strategy(),
        "commands": commands,
        "notes": [
            "安装方案会根据 NVIDIA 环境自动判断。",
            "CUDA 版本到 PyTorch 下载源的映射策略：12.8+ -> cu128，12.6+ -> cu126，11.8+ -> cu118，否则回退 CPU。",
            "如果检测到 Maxwell / Pascal / Volta 这类老架构显卡，会优先走兼容性更好的 cu126 方案，避免误装到过新的轮子。",
            "如果检测到 Blackwell / Ada / Hopper / Ampere / Turing 等显卡，但 nvidia-smi 没有返回 CUDA 版本，会优先按 cu128 显卡方案处理。",
            "如果现有 Torch 类型和推荐方案不一致，会自动重装 Torch，避免 CPU 版残留导致显卡不可用。",
            "如果当前 Python 没有 pip，会自动尝试 ensurepip；若不可用，则在线下载 get-pip.py 安装。",
            "内置 Python、get-pip 和 pip 镜像会先测速，再优先走响应更快的国内源；不可用时再自动回退。",
            "PyTorch CUDA 轮子会按测速结果优先尝试更快的镜像源，再回退官方源。",
            "所有 pip 安装步骤都启用更长超时与重试，尽量减少弱网环境下的失败概率。",
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
