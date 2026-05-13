from __future__ import annotations

import html
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import quote, unquote, urljoin, urlparse


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.environ.get(name, "").strip()
    try:
        value = float(raw_value) if raw_value else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


MIN_PYTHON = (3, 9)
MAX_PYTHON_WINDOWS = (3, 13)
MAX_PYTHON_DEFAULT = (3, 14)
PIP_BOOTSTRAP_URL = "https://bootstrap.pypa.io/get-pip.py"
DEFAULT_EMBEDDED_PYTHON_VERSION = "3.13.13"
LEGACY_CUDA_EMBEDDED_PYTHON_VERSION = "3.12.10"
EMBEDDED_PYTHON_VERSION = DEFAULT_EMBEDDED_PYTHON_VERSION
DEFAULT_PIP_INDEX_URL = os.environ.get("YOLO_TOOL_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
DEFAULT_PIP_EXTRA_INDEX_URL = os.environ.get("YOLO_TOOL_PIP_EXTRA_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple/")
DEFAULT_TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
PINNED_CUDA_TORCH_PACKAGES = {
    # CU118 is the compatibility lane used by legacy NVIDIA GPUs. Keep this
    # trio pinned so pip does not silently prefer a newer CPU wheel elsewhere.
    "cu118": ("torch==2.7.1+cu118", "torchvision==0.22.1+cu118", "torchaudio==2.7.1+cu118"),
}
TORCH_WHEEL_CACHE_DIRNAME = "torch_wheels"
PYTHON_WHEEL_CACHE_DIRNAME = "python_wheels"
APP_WHEEL_CACHE_DIRNAME = "app_wheels"
TOOL_WHEEL_PACKAGES = (
    "pip",
    "setuptools",
    "wheel",
    "packaging",
)
TORCH_RUNTIME_WHEEL_PACKAGES = (
    "filelock",
    "typing-extensions",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "numpy",
    "pillow",
    "mpmath",
    "markupsafe",
    "packaging",
    "setuptools",
)
APP_WHEEL_PACKAGES = (
    "ultralytics",
    "pillow",
    "pyyaml",
    "matplotlib",
    "opencv-python",
    "requests",
    "psutil",
    "polars",
    "ultralytics-thop",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pyparsing",
    "python-dateutil",
    "polars-runtime-32",
    "six",
    "charset-normalizer",
    "idna",
    "urllib3",
    "certifi",
    "scipy",
)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0
PROBE_TIMEOUT_SECONDS = _env_float("YOLO_TOOL_PROBE_TIMEOUT_SECONDS", 4.0, 1.0, 30.0)
DOWNLOAD_TIMEOUT_SECONDS = _env_float("YOLO_TOOL_DOWNLOAD_TIMEOUT_SECONDS", 20.0, 10.0, 180.0)
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = _env_float("YOLO_TOOL_DOWNLOAD_PROGRESS_SECONDS", 2.0, 0.5, 15.0)
DOWNLOAD_STALL_TIMEOUT_SECONDS = _env_float("YOLO_TOOL_DOWNLOAD_STALL_SECONDS", 30.0, 10.0, 300.0)
DOWNLOAD_READ_TIMEOUT_SECONDS = _env_float("YOLO_TOOL_DOWNLOAD_READ_TIMEOUT_SECONDS", 12.0, 5.0, 60.0)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SEGMENTED_DOWNLOAD_MIN_BYTES = 8 * 1024 * 1024
DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS = 16
MAX_SEGMENTED_DOWNLOAD_CONNECTIONS = 32
SEGMENTED_DOWNLOAD_SEGMENT_RETRIES = 4
FALLBACK_HOST_KEYWORDS = (
    "python.org",
    "bootstrap.pypa.io",
    "pypi.org",
    "pythonhosted.org",
    "download.pytorch.org",
)


def dedupe_candidates(candidates: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        key = normalized.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


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
        "compute_capability": "",
        "driver_version": "",
        "cuda_version": "",
    }
    if not executable:
        return result

    try:
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
        if query.returncode != 0:
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
        if len(parts) > 2:
            result["compute_capability"] = parts[2]

    summary_text = (summary.stdout or "") + "\n" + (summary.stderr or "") + "\n" + (table.stdout or "") + "\n" + (table.stderr or "")
    match = re.search(r"CUDA\s+Version\s*:?\s*([0-9]+(?:\.[0-9]+)?)", summary_text, flags=re.IGNORECASE)
    if match:
        result["cuda_version"] = match.group(1)
    architecture_match = re.search(r"Product Architecture\s*:\s*(.+)", summary_text)
    if architecture_match:
        result["gpu_architecture"] = architecture_match.group(1).strip()

    if not result["gpu_architecture"]:
        result["gpu_architecture"] = infer_gpu_architecture(str(result["gpu_name"] or ""))
    if not result["compute_capability"]:
        result["compute_capability"] = infer_compute_capability(
            str(result["gpu_name"] or ""),
            str(result["gpu_architecture"] or ""),
        )

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
        or re.search(r"\bgt\s*1030\b", normalized)
        or re.search(r"\bmx(?:150|250|350)\b", normalized)
        or re.search(r"\b(?:p4|p6|p40|p100)\b", normalized)
        or re.search(r"\btitan\s+xp\b", normalized)
        or re.search(r"\btitan\s+x\s*\(pascal\)\b", normalized)
        or re.search(r"\bquadro\s+p\d+", normalized)
        or "pascal" in normalized
    ):
        return "Pascal"
    if (
        re.search(r"\bgtx\s*9", normalized)
        or re.search(r"\bgtx\s*75[0-9]", normalized)
        or re.search(r"\bgtx\s+titan\s+x\b", normalized)
        or re.search(r"\b(?:m4|m6|m10|m40|m60)\b", normalized)
        or re.search(r"\bquadro\s+m\d+", normalized)
        or "maxwell" in normalized
    ):
        return "Maxwell"
    if (
        re.search(r"\bgtx\s*(?:6|7)", normalized)
        or re.search(r"\b(?:k20|k40|k80)\b", normalized)
        or re.search(r"\btitan(?:\s+black|\s+z)\b", normalized)
        or re.search(r"\bgtx\s+titan(?!\s+x)\b", normalized)
        or re.search(r"\bquadro\s+k\d+", normalized)
        or "kepler" in normalized
    ):
        return "Kepler"
    return ""


def infer_compute_capability(gpu_name: str, gpu_architecture: str = "") -> str:
    normalized = gpu_name.lower()
    architecture = gpu_architecture.strip().lower()
    if (
        re.search(r"\b(?:gtx\s*4|gtx\s*5)\d{2}\b", normalized)
        or re.search(r"\bquadro\s+(?:[2456]000|fx)\b", normalized)
        or "fermi" in normalized
        or architecture == "fermi"
    ):
        return "2.0"
    if re.search(r"\bk80\b", normalized):
        return "3.7"
    if (
        re.search(r"\b(?:k20|k40)\b", normalized)
        or re.search(r"\bgtx\s*(?:6|7)(?:60|70|80|90)\b", normalized)
        or re.search(r"\btitan(?:\s+black|\s+z)\b", normalized)
        or re.search(r"\bgtx\s+titan(?!\s+x)\b", normalized)
        or re.search(r"\bquadro\s+k\d+", normalized)
        or "kepler" in normalized
        or architecture == "kepler"
    ):
        return "3.5"
    if (
        re.search(r"\bgtx\s*75[0-9]\b", normalized)
        or re.search(r"\bgtx\s*9\d{2}\b", normalized)
        or re.search(r"\bgtx\s+titan\s+x\b", normalized)
        or re.search(r"\b(?:m4|m6|m10|m40|m60)\b", normalized)
        or re.search(r"\bquadro\s+m\d+", normalized)
        or "maxwell" in normalized
        or architecture == "maxwell"
    ):
        return "5.2"
    if re.search(r"\b(?:p100)\b", normalized):
        return "6.0"
    if (
        re.search(r"\bgtx\s*10\d{2}\b", normalized)
        or re.search(r"\bgt\s*1030\b", normalized)
        or re.search(r"\bmx(?:150|250|350)\b", normalized)
        or re.search(r"\b(?:p4|p6|p40)\b", normalized)
        or re.search(r"\btitan\s+xp\b", normalized)
        or re.search(r"\btitan\s+x\s*\(pascal\)\b", normalized)
        or re.search(r"\bquadro\s+p\d+", normalized)
        or "pascal" in normalized
        or architecture == "pascal"
    ):
        return "6.1"
    if (
        re.search(r"\b(?:v100|titan\s+v|gv100)\b", normalized)
        or "volta" in normalized
        or architecture == "volta"
    ):
        return "7.0"
    if (
        re.search(r"\brtx\s*20(?:50|60|70|80)\w*\b", normalized)
        or re.search(r"\bgtx\s*16", normalized)
        or re.search(r"\b(?:t4|titan\s+rtx)\b", normalized)
        or re.search(r"\bquadro\s+rtx\b", normalized)
        or "turing" in normalized
        or architecture == "turing"
    ):
        return "7.5"
    return ""


def normalize_accelerator_mode(value: str | None) -> str:
    normalized = (value or "auto").strip().lower().replace("_", "-")
    if normalized in {"auto", "stable-cpu", "legacy-cuda"}:
        return normalized
    return "auto"


def resolve_embedded_python_version(accelerator_mode: str = "auto") -> str:
    if normalize_accelerator_mode(accelerator_mode) == "legacy-cuda":
        return LEGACY_CUDA_EMBEDDED_PYTHON_VERSION
    return DEFAULT_EMBEDDED_PYTHON_VERSION


def parse_compute_capability(value: object) -> tuple[int, int] | None:
    match = re.search(r"(\d+)(?:\.(\d+))?", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def is_legacy_cuda_architecture(gpu_architecture: str) -> bool:
    return gpu_architecture.strip().lower() in {"maxwell", "pascal", "volta"}


def legacy_cuda_compatibility(
    gpu_name: str = "",
    gpu_architecture: str = "",
    compute_capability: str = "",
) -> dict[str, object]:
    architecture = gpu_architecture.strip().lower()
    capability_text = str(compute_capability or "").strip()
    if not capability_text:
        capability_text = infer_compute_capability(gpu_name, gpu_architecture)
    capability = parse_compute_capability(capability_text)

    if capability is None:
        if architecture in {"fermi", "kepler"}:
            return {
                "status": "unsupported",
                "legacy_cuda_available": False,
                "compute_capability": capability_text,
                "message": "Fermi / Kepler 架构过旧，不提供老显卡 CUDA 兼容模式。",
            }
        if architecture in {"maxwell", "pascal", "volta"}:
            return {
                "status": "legacy-compatible",
                "legacy_cuda_available": True,
                "compute_capability": capability_text,
                "message": "检测到 Maxwell / Pascal / Volta 老架构，可尝试 CUDA 11.8 兼容模式，但默认仍推荐 CPU。",
            }
        return {
            "status": "unknown",
            "legacy_cuda_available": False,
            "compute_capability": capability_text,
            "message": "未能确认显卡算力，不自动提供老显卡 CUDA 兼容模式。",
        }

    major, minor = capability
    normalized_capability = f"{major}.{minor}"
    if major < 5:
        return {
            "status": "unsupported",
            "legacy_cuda_available": False,
            "compute_capability": normalized_capability,
            "message": "显卡算力低于 5.0，当前 PyTorch CUDA 轮子不建议尝试，保持 CPU 更稳。",
        }
    if major < 7 or (major == 7 and minor < 5):
        risk = "high-risk" if major == 5 else "legacy-compatible"
        return {
            "status": risk,
            "legacy_cuda_available": True,
            "compute_capability": normalized_capability,
            "message": "显卡属于 CUDA 11.8 兼容尝试范围；该模式依赖当前 Python 是否有可用 PyTorch cu118 轮子。",
        }
    return {
        "status": "modern",
        "legacy_cuda_available": False,
        "compute_capability": normalized_capability,
        "message": "显卡算力不属于老显卡范围，应优先使用自动 CUDA 方案。",
    }


def choose_torch_index(
    cuda_version: str,
    gpu_architecture: str = "",
    accelerator_mode: str = "auto",
    gpu_name: str = "",
    compute_capability: str = "",
) -> tuple[str, str]:
    legacy_architectures = {"maxwell", "pascal", "volta"}
    modern_architectures = {"blackwell", "ada", "lovelace", "hopper", "ampere", "turing"}
    unsupported_architectures = {"kepler", "fermi"}
    normalized_architecture = gpu_architecture.strip().lower()
    normalized_mode = normalize_accelerator_mode(accelerator_mode)
    compatibility = legacy_cuda_compatibility(gpu_name, gpu_architecture, compute_capability)
    if normalized_mode == "stable-cpu":
        return "cpu", ""
    if normalized_mode == "legacy-cuda":
        if not compatibility.get("legacy_cuda_available"):
            return "cpu", ""
        return "cu118", "https://download.pytorch.org/whl/cu118"
    if str(compatibility.get("status") or "") in {"unsupported", "high-risk", "legacy-compatible"}:
        return "cpu", ""
    if normalized_architecture in unsupported_architectures or normalized_architecture in legacy_architectures:
        return "cpu", ""
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


def classify_torch_accelerator(cuda_version: str) -> str:
    cuda_version = str(cuda_version or "").strip()
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


def detect_installed_torch_accelerator() -> str:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return ""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(getattr(getattr(torch, 'version', object()), 'cuda', None) or '')",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path.cwd()),
        env=_process_env(),
        creationflags=NO_WINDOW_FLAGS,
        check=False,
    )
    if probe.returncode != 0:
        return ""
    output = (probe.stdout or "").strip()
    cuda_version = output.splitlines()[-1].strip() if output else ""
    return classify_torch_accelerator(cuda_version)


def torch_packages_for_accelerator(accelerator: str) -> tuple[str, ...]:
    return PINNED_CUDA_TORCH_PACKAGES.get(accelerator, DEFAULT_TORCH_PACKAGES)


def _python_wheel_tags() -> tuple[str, str, str]:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi_tag = py_tag
    if sys.platform.startswith("win"):
        machine = platform.machine().lower()
        platform_tag = "win_amd64" if any(token in machine for token in ("amd64", "x86_64", "x64")) else "win32"
    else:
        platform_tag = "any"
    return py_tag, abi_tag, platform_tag


def build_torch_wheel_filename(requirement: str) -> str | None:
    if "==" not in requirement:
        return None
    package_name, version = requirement.split("==", 1)
    package_name = package_name.strip()
    version = version.strip()
    if not package_name or not version:
        return None
    py_tag, abi_tag, platform_tag = _python_wheel_tags()
    normalized_name = package_name.replace("-", "_")
    return f"{normalized_name}-{version}-{py_tag}-{abi_tag}-{platform_tag}.whl"


def build_torch_wheel_downloads(torch_indexes: list[str], requirements: tuple[str, ...]) -> list[dict[str, object]]:
    cache_dir = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / TORCH_WHEEL_CACHE_DIRNAME
    downloads: list[dict[str, object]] = []
    for requirement in requirements:
        filename = build_torch_wheel_filename(requirement)
        if not filename:
            return []
        encoded_filename = quote(filename, safe="-_.")
        urls = []
        for index_url in torch_indexes:
            candidate_url = f"{index_url.rstrip('/')}/{encoded_filename}"
            if unquote(Path(urlparse(candidate_url).path).name) == filename:
                urls.append(candidate_url)
        urls = dedupe_candidates(urls)
        if not urls:
            return []
        downloads.append(
            {
                "requirement": requirement,
                "filename": filename,
                "target_path": str(cache_dir / filename),
                "urls": urls,
            }
        )
    return downloads


def _normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _iter_simple_index_links(index_url: str, package: str) -> list[str]:
    base_url = index_url.rstrip("/") + "/" + package.replace("_", "-") + "/"
    request = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
    with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        page = response.read().decode(charset, errors="ignore")
    links: list[str] = []
    for match in re.finditer(r"""href\s*=\s*['"]([^'"]+)['"]""", page, flags=re.IGNORECASE):
        href = html.unescape(match.group(1)).split("#", 1)[0].strip()
        if href:
            links.append(urljoin(base_url, href))
    return links


def _cpython_tag_version(tag: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"cp(\d)(\d{1,2})", tag)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _wheel_python_tag_compatible(wheel_py_tags: set[str], wheel_abi_tags: set[str], py_tags: tuple[str, ...]) -> bool:
    if wheel_py_tags.intersection(py_tags) or "py3" in wheel_py_tags:
        return True
    if "abi3" not in wheel_abi_tags:
        return False
    current = (sys.version_info.major, sys.version_info.minor)
    for tag in wheel_py_tags:
        tag_version = _cpython_tag_version(tag)
        if tag_version and tag_version[0] == current[0] and tag_version[1] <= current[1]:
            return True
    return False


def _wheel_compatible(filename: str, package: str, py_tags: tuple[str, ...], abi_tags: tuple[str, ...]) -> bool:
    decoded = unquote(Path(filename).name)
    if not decoded.endswith(".whl"):
        return False
    normalized_package = _normalize_project_name(package)
    parts = decoded[:-4].rsplit("-", 3)
    if len(parts) != 4:
        return False
    prefix, py_tag_text, abi_tag_text, platform_tag_text = parts
    normalized_prefix = _normalize_project_name(prefix)
    package_prefix = normalized_package + "-"
    if not normalized_prefix.startswith(package_prefix):
        return False
    version_text = normalized_prefix[len(package_prefix) :]
    if not version_text or not version_text[0].isdigit():
        return False
    wheel_py_tags = set(py_tag_text.split("."))
    wheel_abi_tags = set(abi_tag_text.split("."))
    wheel_platform_tags = set(platform_tag_text.split("."))
    if not _wheel_python_tag_compatible(wheel_py_tags, wheel_abi_tags, py_tags):
        return False
    if not wheel_abi_tags.intersection(abi_tags):
        return False
    return "any" in wheel_platform_tags or "win_amd64" in wheel_platform_tags


def _package_wheel_candidates(package: str, indexes: list[str]) -> list[tuple[str, str]]:
    py_tag, abi_tag, _platform_tag = _python_wheel_tags()
    py_major_minor = py_tag[2:]
    py_tags = (py_tag, "py3", f"py{py_major_minor[0]}")
    abi_tags = (abi_tag, "abi3", "none")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def collect(index_url: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        try:
            for link in _iter_simple_index_links(index_url, package):
                filename = unquote(Path(urlparse(link).path).name)
                if not _wheel_compatible(filename, package, py_tags, abi_tags):
                    continue
                found.append((link, filename))
        except Exception:
            return []
        return found

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(indexes)))) as pool:
        future_map = {pool.submit(collect, index_url): index_url for index_url in indexes}
        for future in as_completed(future_map):
            for link, filename in future.result():
                if link in seen:
                    continue
                seen.add(link)
                candidates.append((link, filename))
    return candidates


def _wheel_version_key(filename: str, package: str) -> tuple:
    decoded = unquote(Path(filename).name)
    parts = decoded[:-4].rsplit("-", 3)
    prefix = parts[0] if parts else decoded
    normalized_prefix = _normalize_project_name(prefix)
    normalized_package = _normalize_project_name(package)
    version_text = normalized_prefix
    if normalized_prefix.startswith(normalized_package + "-"):
        version_text = normalized_prefix[len(normalized_package) + 1 :]
    is_stable = not re.search(r"(?:a|alpha|b|beta|rc|dev|pre|preview)\d*", version_text, flags=re.IGNORECASE)
    tokens = re.split(r"([0-9]+)", version_text)
    key: list[tuple[int, object]] = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            key.append((1, int(token)))
        else:
            key.append((0, token))
    return (1 if is_stable else 0, tuple(key))


def _wheel_version_text(filename: str, package: str) -> str:
    decoded = unquote(Path(filename).name)
    parts = decoded[:-4].rsplit("-", 3)
    prefix = parts[0] if parts else decoded
    if _normalize_project_name(package) == "mpmath" and prefix.lower().startswith("mpmath-"):
        return prefix[len("mpmath-") :]
    normalized_prefix = _normalize_project_name(prefix)
    normalized_package = _normalize_project_name(package)
    if normalized_prefix.startswith(normalized_package + "-"):
        return normalized_prefix[len(normalized_package) + 1 :]
    return normalized_prefix


def _wheel_allowed_for_package(package: str, filename: str) -> bool:
    normalized_package = _normalize_project_name(package)
    if normalized_package != "mpmath":
        return True
    version_text = _wheel_version_text(filename, package)
    match = re.match(r"(\d+)\.(\d+)", version_text)
    if not match:
        return True
    # SymPy currently requires mpmath<1.4. Avoid selecting the newest mpmath
    # wheel just because it exists on mirrors; otherwise offline Torch install
    # can fail dependency resolution.
    return (int(match.group(1)), int(match.group(2))) < (1, 4)


def _select_best_wheel_urls(package: str, candidates: list[tuple[str, str]]) -> tuple[str, list[str]] | None:
    grouped: dict[str, list[str]] = {}
    for url, filename in candidates:
        if not _wheel_allowed_for_package(package, filename):
            continue
        grouped.setdefault(filename, []).append(url)
    if not grouped:
        return None
    best_filename = max(grouped, key=lambda filename: _wheel_version_key(filename, package))
    return best_filename, dedupe_candidates(grouped[best_filename])


def _sort_urls_by_index_order(urls: list[str], indexes: list[str]) -> list[str]:
    normalized_indexes = [
        (index_url.rstrip("/"), (urlparse(index_url).netloc or "").lower())
        for index_url in indexes
    ]

    def order_key(url: str) -> tuple[int, int, str]:
        normalized_url = url.rstrip("/")
        url_host = (urlparse(url).netloc or "").lower()
        for index, (base, host) in enumerate(normalized_indexes):
            if normalized_url.startswith(base):
                return index, 0, normalized_url
            # Mirror simple pages often point to wheel files under another path
            # on the same host, e.g. `/pypi/packages/...` instead of `/simple/...`.
            # Keep those URLs ahead of third-party hosts such as pythonhosted.org.
            if host and url_host == host:
                return index, 1, normalized_url
        return len(normalized_indexes), 2, normalized_url

    return sorted(dedupe_candidates(urls), key=order_key)


def build_python_wheel_downloads(
    packages: tuple[str, ...] | list[str],
    indexes: list[str],
    *,
    cache_dirname: str,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    logger = log or (lambda _message: None)
    cache_dir = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / cache_dirname
    downloads: list[dict[str, object]] = []
    package_list = list(packages)

    def resolve_package(package: str) -> dict[str, object] | None:
        candidates = _package_wheel_candidates(package, indexes)
        selected = _select_best_wheel_urls(package, candidates)
        if not selected:
            return None
        filename, urls = selected
        return {
            "package": package,
            "filename": filename,
            "target_path": str(cache_dir / filename),
            "urls": _sort_urls_by_index_order(urls, indexes),
        }

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(package_list)))) as pool:
        future_map = {pool.submit(resolve_package, package): package for package in package_list}
        resolved_by_package: dict[str, dict[str, object]] = {}
        for future in as_completed(future_map):
            package = future_map[future]
            item = future.result()
            if not item:
                logger(f"未找到可直接下载的 wheel，稍后会交给 pip 回退处理：{package}")
                continue
            logger(f"已解析 wheel 下载地址：{package} -> {item['filename']}（{len(item.get('urls') or [])} 个源）")
            resolved_by_package[package] = item

    for package in package_list:
        item = resolved_by_package.get(package)
        if item:
            downloads.append(item)
    return downloads


def build_app_wheel_downloads(
    indexes: list[str],
    log: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    return build_python_wheel_downloads(APP_WHEEL_PACKAGES, indexes, cache_dirname=APP_WHEEL_CACHE_DIRNAME, log=log)


def build_accelerator_summary(accelerator: str, gpu: dict[str, object] | None = None) -> dict[str, str]:
    gpu_info = gpu or {}
    gpu_available = bool(gpu_info.get("available"))
    gpu_name = str(gpu_info.get("gpu_name") or "").strip()
    gpu_architecture = str(gpu_info.get("gpu_architecture") or "").strip()
    compute_capability = str(gpu_info.get("compute_capability") or "").strip()
    cuda_version = str(gpu_info.get("cuda_version") or "").strip()

    if accelerator != "cpu":
        gpu_text = gpu_name or "NVIDIA 显卡"
        detail_parts = []
        if gpu_architecture:
            detail_parts.append(gpu_architecture)
        if compute_capability:
            detail_parts.append(f"算力 {compute_capability}")
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
        if compute_capability:
            detail_parts.append(f"算力 {compute_capability}")
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
            f"https://repo.huaweicloud.com/pytorch-wheels/{accelerator}",
            official_index,
        ]
    )

    return dedupe_candidates(candidates)


def build_generic_pip_index_candidates() -> list[str]:
    env_override = os.environ.get("YOLO_TOOL_PIP_INDEX_URL", "").strip()
    candidates = [
        env_override,
        DEFAULT_PIP_INDEX_URL,
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://repo.huaweicloud.com/pypi/simple/",
        "https://mirrors.huaweicloud.com/repository/pypi/simple/",
        "https://mirror.sjtu.edu.cn/pypi/web/simple/",
        "https://mirrors.ustc.edu.cn/pypi/simple/",
        "https://mirrors.bfsu.edu.cn/pypi/web/simple/",
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://mirrors.cloud.tencent.com/pypi/simple/",
        "https://pypi.doubanio.com/simple/",
        "https://pypi.org/simple/",
    ]
    return dedupe_candidates(candidates)


def _process_env(*, include_extra_index: bool = True) -> dict[str, str]:
    process_env = os.environ.copy()
    process_env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_DEFAULT_TIMEOUT": "180",
            "PIP_NO_INPUT": "1",
            "PIP_RETRIES": "5",
            "PIP_PREFER_BINARY": "1",
        }
    )
    if include_extra_index and DEFAULT_PIP_EXTRA_INDEX_URL:
        process_env.setdefault("PIP_EXTRA_INDEX_URL", DEFAULT_PIP_EXTRA_INDEX_URL)
    elif not include_extra_index:
        process_env.pop("PIP_EXTRA_INDEX_URL", None)
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
    if has_pip() and pip_is_usable():
        return "已检测到 pip，可直接安装依赖"
    if has_ensurepip():
        return "当前 Python 支持 ensurepip，会先本地初始化 pip"
    return "当前 Python 不带 ensurepip，将优先用 pip/setuptools/wheel 的本地 wheel 引导 pip，失败后再回退 get-pip.py"


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


def choose_embedded_python_url(embedded_python_version: str | None = None) -> str:
    if not sys.platform.startswith("win"):
        raise RuntimeError("当前只支持在 Windows 上自动创建内置 runtime。")

    version = (embedded_python_version or DEFAULT_EMBEDDED_PYTHON_VERSION).strip() or DEFAULT_EMBEDDED_PYTHON_VERSION
    machine = platform.machine().lower()
    if any(token in machine for token in ("arm64", "aarch64")):
        arch = "arm64"
    elif any(token in machine for token in ("amd64", "x86_64", "x64")):
        arch = "amd64"
    else:
        arch = "win32"
    return f"https://www.python.org/ftp/python/{version}/python-{version}-embed-{arch}.zip"


def choose_embedded_python_urls(embedded_python_version: str | None = None) -> list[str]:
    version = (embedded_python_version or DEFAULT_EMBEDDED_PYTHON_VERSION).strip() or DEFAULT_EMBEDDED_PYTHON_VERSION
    official_url = choose_embedded_python_url(version)
    filename = Path(official_url).name
    env_override = os.environ.get("YOLO_TOOL_PYTHON_EMBED_URL", "").strip()
    candidates = [
        env_override,
        f"https://mirrors.aliyun.com/python-release/windows/{filename}",
        f"https://repo.huaweicloud.com/python/{version}/{filename}",
        f"https://repo.huaweicloud.com/repository/toolkit/python/{version}/{filename}",
        f"https://mirrors.huaweicloud.com/python/{version}/{filename}",
        f"https://mirrors.bfsu.edu.cn/python/{version}/{filename}",
        f"https://mirror.sjtu.edu.cn/python/{version}/{filename}",
        f"https://mirrors.tuna.tsinghua.edu.cn/python/{version}/{filename}",
        official_url,
    ]
    return dedupe_candidates(candidates)


def choose_get_pip_urls() -> list[str]:
    env_override = os.environ.get("YOLO_TOOL_GET_PIP_URL", "").strip()
    candidates = [
        env_override,
        "https://mirrors.aliyun.com/pypi/get-pip.py",
        "https://repo.huaweicloud.com/pypi/get-pip.py",
        PIP_BOOTSTRAP_URL,
    ]
    return dedupe_candidates(candidates)


def _download_cache_ready(target_path: Path) -> bool:
    if not target_path.exists():
        return False
    size = target_path.stat().st_size
    if size <= 0:
        return False
    if target_path.suffix.lower() in {".zip", ".whl"}:
        try:
            with zipfile.ZipFile(target_path) as zip_file:
                return zip_file.testzip() is None
        except zipfile.BadZipFile:
            return False
    if target_path.name == "get-pip.py":
        if size < 100_000:
            return False
        try:
            content = target_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "def main" in content and "__main__" in content
    return True


def _download_part_cache_ready(part_path: Path, final_path: Path) -> bool:
    if not part_path.exists():
        return False
    suffix = final_path.suffix.lower()
    if suffix not in {".zip", ".whl"} and final_path.name != "get-pip.py":
        return False
    probe_path = part_path.with_name(part_path.name + ".readycheck" + suffix)
    try:
        if probe_path.exists():
            probe_path.unlink(missing_ok=True)
        shutil.copy2(part_path, probe_path)
        return _download_cache_ready(probe_path)
    except OSError:
        return False
    finally:
        probe_path.unlink(missing_ok=True)


def cleanup_download_cache(
    log: Callable[[str], None] | None = None,
    *,
    remove_ready_wheels: bool = False,
    max_age_hours: float = 24.0,
) -> None:
    logger = log or (lambda _message: None)
    cache_root = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap"
    if not cache_root.exists():
        return
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    removed_count = 0
    released_bytes = 0

    def remove_path(path: Path) -> None:
        nonlocal removed_count, released_bytes
        try:
            if path.is_dir():
                released_bytes += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
                shutil.rmtree(path, ignore_errors=True)
            else:
                released_bytes += path.stat().st_size
                path.unlink(missing_ok=True)
            removed_count += 1
        except OSError:
            pass

    for candidate in cache_root.rglob("*"):
        if not candidate.exists():
            continue
        name = candidate.name.lower()
        is_temp_download = name.endswith(".part") or ".readycheck" in name or name.endswith(".tmp")
        is_part_dir = candidate.is_dir() and name.endswith(".parts")
        if is_temp_download or is_part_dir:
            try:
                age = now - candidate.stat().st_mtime
            except OSError:
                age = max_age_seconds + 1
            if age >= max_age_seconds:
                remove_path(candidate)

    if remove_ready_wheels:
        for subdir_name in (TORCH_WHEEL_CACHE_DIRNAME, PYTHON_WHEEL_CACHE_DIRNAME, APP_WHEEL_CACHE_DIRNAME):
            subdir = cache_root / subdir_name
            if subdir.exists():
                remove_path(subdir)

    if removed_count:
        logger(f"已清理下载缓存：{removed_count} 项，释放约 {_format_mb(released_bytes)}。")


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

    def probe(url: str) -> tuple[str, int, float | None, str]:
        try:
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
                response.status
            latency = time.perf_counter() - started
            return url, candidate_tier(url), latency, ""
        except Exception as exc:
            return url, candidate_tier(url), None, str(exc)

    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        future_map = {pool.submit(probe, url): url for url in urls}
        for future in as_completed(future_map):
            url, tier, latency, error = future.result()
            if error:
                fallback.append(url)
                logger(f"测速失败，保留回退：{url}（{error}）")
                continue
            assert latency is not None
            scored.append((tier, latency, url))
            logger(f"测速成功：{url}（约 {latency:.2f} 秒）")

    ordered = [url for _, _, url in sorted(scored, key=lambda item: (item[0], item[1]))]
    for url in urls:
        if url not in fallback:
            continue
        if url not in ordered:
            ordered.append(url)
    if ordered:
        logger("下载源排序：" + " -> ".join(ordered))
    return ordered or list(urls)


def _probe_content_length(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length", "").strip()
        if not content_length:
            return None
        return int(content_length)
    except Exception:
        return None


def _probe_download_info(url: str) -> tuple[int | None, bool]:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length", "").strip()
            accept_ranges = response.headers.get("Accept-Ranges", "").lower()
        size = int(content_length) if content_length else None
        supports_range = "bytes" in accept_ranges
        if size is not None and not supports_range:
            probe_request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Connection": "close",
                    "Range": "bytes=0-0",
                },
            )
            with urllib.request.urlopen(probe_request, timeout=PROBE_TIMEOUT_SECONDS) as response:
                if getattr(response, "status", None) == 206:
                    supports_range = True
                if not size:
                    content_range = response.headers.get("Content-Range", "").strip()
                    match = re.search(r"/(\d+)$", content_range)
                    if match:
                        size = int(match.group(1))
        return size, supports_range
    except Exception:
        return None, False


def _segmented_download_connections() -> int:
    raw_value = os.environ.get("YOLO_TOOL_DOWNLOAD_CONNECTIONS", "").strip()
    try:
        value = int(raw_value) if raw_value else DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS
    except ValueError:
        value = DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS
    return max(1, min(MAX_SEGMENTED_DOWNLOAD_CONNECTIONS, value))


def _format_mb(byte_count: int | float | None) -> str:
    if byte_count is None:
        return "未知大小"
    return f"{float(byte_count) / 1024 / 1024:.1f} MB"


def _download_range_to_path(url: str, start: int, end: int, segment_path: Path) -> None:
    downloaded = segment_path.stat().st_size if segment_path.exists() else 0
    expected_size = end - start + 1
    if downloaded > expected_size:
        segment_path.unlink(missing_ok=True)
        downloaded = 0
    if downloaded == expected_size:
        return
    range_start = start + downloaded
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Connection": "close",
        "Range": f"bytes={range_start}-{end}",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_READ_TIMEOUT_SECONDS) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise RuntimeError("服务器未按分段请求返回数据。")
        with segment_path.open("ab") as output:
            while True:
                remaining = expected_size - downloaded
                if remaining <= 0:
                    break
                try:
                    chunk = response.read(min(DOWNLOAD_CHUNK_SIZE, remaining))
                except (TimeoutError, socket.timeout) as exc:
                    raise TimeoutError(f"Segment download stalled for {DOWNLOAD_READ_TIMEOUT_SECONDS:.0f}s: {segment_path.name}") from exc
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
    final_size = segment_path.stat().st_size
    if final_size != expected_size:
        raise RuntimeError(f"分段下载未完成：{segment_path.name}，当前 {final_size} 字节，预期 {expected_size} 字节。")


def _download_url_to_path_segmented_legacy_unused(
    url: str,
    target_path: Path,
    *,
    content_length: int,
    connections: int,
    logger: Callable[[str], None],
) -> None:
    part_dir = target_path.with_name(target_path.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_name(target_path.name + ".part")
    segment_count = min(connections, max(1, content_length // SEGMENTED_DOWNLOAD_MIN_BYTES))
    segment_size = (content_length + segment_count - 1) // segment_count
    ranges: list[tuple[int, int, Path]] = []
    for index in range(segment_count):
        start = index * segment_size
        if start >= content_length:
            break
        end = min(content_length - 1, start + segment_size - 1)
        ranges.append((start, end, part_dir / f"{index:03d}.part"))

    logger(f"启用多线程分段下载：{len(ranges)} 连接，预计大小 {content_length / 1024 / 1024:.0f} MB。")
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        future_map = {
            pool.submit(_download_range_to_path, url, start, end, segment_path): segment_path
            for start, end, segment_path in ranges
        }
        completed_segments = 0
        for future in as_completed(future_map):
            future.result()
            completed_segments += 1
            downloaded_bytes = sum(segment_path.stat().st_size for _start, _end, segment_path in ranges if segment_path.exists())
            logger(
                "分段下载进度："
                f"{completed_segments}/{len(ranges)} 段，"
                f"{downloaded_bytes / 1024 / 1024:.0f}/{content_length / 1024 / 1024:.0f} MB"
            )

    with part_path.open("wb") as output:
        for _start, _end, segment_path in ranges:
            with segment_path.open("rb") as segment_file:
                shutil.copyfileobj(segment_file, output, DOWNLOAD_CHUNK_SIZE)
    if part_path.stat().st_size != content_length:
        raise RuntimeError(f"合并后的文件大小不正确，当前 {part_path.stat().st_size} 字节，预期 {content_length} 字节。")
    if target_path.exists():
        target_path.unlink(missing_ok=True)
    part_path.replace(target_path)
    shutil.rmtree(part_dir, ignore_errors=True)


def _download_url_to_path_legacy_unused(url: str, target_path: Path) -> None:
    part_path = target_path.with_name(target_path.name + ".part")
    existing_size = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        status = getattr(response, "status", None)
        if existing_size > 0 and status != 206:
            existing_size = 0
            part_path.unlink(missing_ok=True)
        total_expected = None
        content_range = response.headers.get("Content-Range", "").strip()
        if content_range:
            match = re.search(r"/(\d+)$", content_range)
            if match:
                total_expected = int(match.group(1))
        if total_expected is None:
            content_length = response.headers.get("Content-Length", "").strip()
            if content_length:
                length_value = int(content_length)
                total_expected = existing_size + length_value if existing_size > 0 and status == 206 else length_value
        mode = "ab" if existing_size > 0 and status == 206 else "wb"
        with part_path.open(mode) as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
    if part_path.stat().st_size <= 0:
        part_path.unlink(missing_ok=True)
        raise RuntimeError("下载内容为空。")
    if total_expected is not None and part_path.stat().st_size < total_expected:
        raise RuntimeError(f"下载未完成，当前 {part_path.stat().st_size} 字节，预期 {total_expected} 字节。")
    if target_path.exists():
        target_path.unlink(missing_ok=True)
    part_path.replace(target_path)


def _download_url_to_path_segmented(
    url: str,
    target_path: Path,
    *,
    content_length: int,
    connections: int,
    logger: Callable[[str], None],
) -> None:
    part_dir = target_path.with_name(target_path.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_name(target_path.name + ".part")
    segment_count = min(connections, max(1, content_length // SEGMENTED_DOWNLOAD_MIN_BYTES))
    segment_size = (content_length + segment_count - 1) // segment_count
    ranges: list[tuple[int, int, Path]] = []
    for index in range(segment_count):
        start = index * segment_size
        if start >= content_length:
            break
        end = min(content_length - 1, start + segment_size - 1)
        ranges.append((start, end, part_dir / f"{index:03d}.part"))

    if any(segment_path.exists() and segment_path.stat().st_size > 0 for _start, _end, segment_path in ranges):
        logger("检测到未完成的分段下载缓存，将继续断点续传。")

    logger(f"启用多线程分段下载：{len(ranges)} 连接，预计大小 {_format_mb(content_length)}。")
    stop_event = threading.Event()
    progress_state = {
        "bytes": sum(segment_path.stat().st_size for _start, _end, segment_path in ranges if segment_path.exists()),
        "time": time.perf_counter(),
        "last_growth_time": time.perf_counter(),
        "stall_reported": False,
        "completed": 0,
    }

    def emit_progress(force: bool = False) -> None:
        downloaded_bytes = sum(segment_path.stat().st_size for _start, _end, segment_path in ranges if segment_path.exists())
        now = time.perf_counter()
        delta_bytes = downloaded_bytes - int(progress_state["bytes"])
        elapsed = max(now - float(progress_state["time"]), 1e-6)
        if not force and delta_bytes <= 0:
            if now - float(progress_state["last_growth_time"]) >= DOWNLOAD_STALL_TIMEOUT_SECONDS and not bool(progress_state["stall_reported"]):
                logger(f"当前下载源超过 {DOWNLOAD_STALL_TIMEOUT_SECONDS:.0f} 秒没有新增数据，连接超时后会自动切换下一个源。")
                progress_state["stall_reported"] = True
                return
            logger(
                f"分段下载进度：{progress_state['completed']}/{len(ranges)} 段，"
                f"{_format_mb(downloaded_bytes)}/{_format_mb(content_length)}，仍在等待网络响应"
            )
            return
        speed = delta_bytes / elapsed / 1024 / 1024 if delta_bytes > 0 else 0.0
        percent = downloaded_bytes / content_length * 100.0 if content_length else 0.0
        logger(
            f"分段下载进度：{progress_state['completed']}/{len(ranges)} 段，"
            f"{_format_mb(downloaded_bytes)}/{_format_mb(content_length)}（{percent:.1f}%），{speed:.1f} MB/s"
        )
        progress_state["bytes"] = downloaded_bytes
        progress_state["time"] = now
        progress_state["last_growth_time"] = now
        progress_state["stall_reported"] = False

    def monitor_progress() -> None:
        while not stop_event.wait(DOWNLOAD_PROGRESS_INTERVAL_SECONDS):
            emit_progress()

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            def download_range_with_retries(start: int, end: int, segment_path: Path) -> None:
                last_error: Exception | None = None
                for attempt in range(1, SEGMENTED_DOWNLOAD_SEGMENT_RETRIES + 1):
                    try:
                        _download_range_to_path(url, start, end, segment_path)
                        return
                    except Exception as exc:
                        last_error = exc
                        if attempt >= SEGMENTED_DOWNLOAD_SEGMENT_RETRIES:
                            break
                        logger(
                            f"分段下载短暂卡住，正在重试当前分段 "
                            f"{segment_path.name}（{attempt + 1}/{SEGMENTED_DOWNLOAD_SEGMENT_RETRIES}）：{exc}"
                        )
                        time.sleep(min(2.0 * attempt, 8.0))
                assert last_error is not None
                raise last_error

            future_map = {
                pool.submit(download_range_with_retries, start, end, segment_path): segment_path
                for start, end, segment_path in ranges
            }
            for future in as_completed(future_map):
                future.result()
                progress_state["completed"] = int(progress_state["completed"]) + 1
                emit_progress(force=True)

        with part_path.open("wb") as output:
            for _start, _end, segment_path in ranges:
                with segment_path.open("rb") as segment_file:
                    shutil.copyfileobj(segment_file, output, DOWNLOAD_CHUNK_SIZE)
        if part_path.stat().st_size != content_length:
            raise RuntimeError(f"合并后的文件大小不正确，当前 {part_path.stat().st_size} 字节，预期 {content_length} 字节。")
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        part_path.replace(target_path)
        shutil.rmtree(part_dir, ignore_errors=True)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)


def _download_url_to_path(
    url: str,
    target_path: Path,
    *,
    logger: Callable[[str], None],
    content_length: int | None = None,
) -> None:
    part_path = target_path.with_name(target_path.name + ".part")
    existing_size = part_path.stat().st_size if part_path.exists() else 0
    if existing_size > 0:
        logger(f"检测到未完成的下载缓存，将继续断点续传：{target_path.name}（已下载 {_format_mb(existing_size)}）")
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response_context = urllib.request.urlopen(request, timeout=DOWNLOAD_READ_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if existing_size > 0 and exc.code == 416:
            part_path.unlink(missing_ok=True)
            logger("断点缓存与当前下载源不匹配，已清理并从头重新下载。")
            return _download_url_to_path(url, target_path, logger=logger, content_length=content_length)
        raise
    with response_context as response:
        status = getattr(response, "status", None)
        if existing_size > 0 and status != 206:
            existing_size = 0
            part_path.unlink(missing_ok=True)
            logger("当前下载源不支持断点续传，已自动从头重新下载。")
        total_expected = content_length
        content_range = response.headers.get("Content-Range", "").strip()
        if content_range:
            match = re.search(r"/(\d+)$", content_range)
            if match:
                total_expected = int(match.group(1))
        if total_expected is None:
            response_length = response.headers.get("Content-Length", "").strip()
            if response_length:
                length_value = int(response_length)
                total_expected = existing_size + length_value if existing_size > 0 and status == 206 else length_value

        mode = "ab" if existing_size > 0 and status == 206 else "wb"
        downloaded = existing_size
        last_report_bytes = downloaded
        last_report_time = time.perf_counter()
        with part_path.open(mode) as output:
            while True:
                remaining = total_expected - downloaded if total_expected is not None else DOWNLOAD_CHUNK_SIZE
                if total_expected is not None and remaining <= 0:
                    break
                try:
                    chunk = response.read(min(DOWNLOAD_CHUNK_SIZE, remaining))
                except (TimeoutError, socket.timeout) as exc:
                    raise TimeoutError(f"Download stalled for {DOWNLOAD_READ_TIMEOUT_SECONDS:.0f}s: {target_path.name}") from exc
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_report_time < DOWNLOAD_PROGRESS_INTERVAL_SECONDS and downloaded - last_report_bytes < 8 * 1024 * 1024:
                    continue
                elapsed = max(now - last_report_time, 1e-6)
                speed = (downloaded - last_report_bytes) / elapsed / 1024 / 1024
                if total_expected:
                    percent = downloaded / total_expected * 100.0
                    logger(f"Download progress: {_format_mb(downloaded)}/{_format_mb(total_expected)} ({percent:.1f}%), {speed:.1f} MB/s")
                else:
                    logger(f"Download progress: {_format_mb(downloaded)}, {speed:.1f} MB/s")
                last_report_bytes = downloaded
                last_report_time = now
    if part_path.stat().st_size <= 0:
        part_path.unlink(missing_ok=True)
        raise RuntimeError("下载内容为空。")
    if total_expected is not None and part_path.stat().st_size < total_expected:
        raise RuntimeError(f"下载未完成，当前 {part_path.stat().st_size} 字节，预期 {total_expected} 字节。")
    if target_path.exists():
        target_path.unlink(missing_ok=True)
    part_path.replace(target_path)


def _download_url_to_path_auto(
    url: str,
    target_path: Path,
    *,
    logger: Callable[[str], None],
    prefer_segmented: bool = False,
) -> None:
    content_length, supports_range = _probe_download_info(url)
    connections = _segmented_download_connections()
    if (
        prefer_segmented
        and content_length is not None
        and content_length >= SEGMENTED_DOWNLOAD_MIN_BYTES
        and supports_range
        and connections > 1
    ):
        _download_url_to_path_segmented(
            url,
            target_path,
            content_length=content_length,
            connections=connections,
            logger=logger,
        )
        return
    if prefer_segmented and content_length and not supports_range:
        logger("当前下载源不支持分段下载，自动改用普通断点续传。")
    _download_url_to_path(url, target_path, logger=logger, content_length=content_length)


def download_torch_wheels(
    downloads: list[dict[str, object]],
    log: Callable[[str], None] | None = None,
) -> tuple[list[Path], list[str]]:
    logger = log or (lambda _message: None)
    wheel_paths: list[Path] = []
    chosen_urls: list[str] = []

    torch_downloads: list[dict[str, object]] = []
    runtime_downloads: list[dict[str, object]] = []
    for item in downloads:
        category = str(item.get("category") or "")
        if category == "torch-runtime":
            runtime_downloads.append(item)
        else:
            torch_downloads.append(item)

    for item in torch_downloads:
        requirement = str(item.get("requirement") or "")
        filename = str(item.get("filename") or "")
        target_path = Path(item["target_path"])
        urls = [str(url) for url in item.get("urls") or []]
        logger(f"正在下载 Torch 依赖包：{requirement}（{filename}）")
        content_length = _probe_content_length(urls[0]) if urls else None
        if content_length:
            logger(f"预计大小：{_format_mb(content_length)}，将启用本地缓存、多线程分段下载和卡顿自动切源。")
        wheel_path, chosen_url = download_file(urls, target_path, logger, prefer_segmented=True, rank_sources=False)
        try:
            size_mb = wheel_path.stat().st_size / 1024 / 1024
            logger(f"Torch wheel 已就绪：{filename}（{size_mb:.1f} MB）")
        except OSError:
            logger(f"Torch wheel 已就绪：{filename}")
        wheel_paths.append(wheel_path)
        chosen_urls.append(chosen_url)

    if runtime_downloads:
        logger(f"正在并行预下载 Torch 小依赖 wheel：{len(runtime_downloads)} 个包，避免 pip 安装阶段逐个联网等待。")

        def worker(item: dict[str, object]) -> tuple[Path, str, str]:
            package = str(item.get("package") or "")
            filename = str(item.get("filename") or "")
            target_path = Path(item["target_path"])
            urls = [str(url) for url in item.get("urls") or []]
            wheel_path, chosen_url = download_file(urls, target_path, logger, prefer_segmented=True, rank_sources=False)
            return wheel_path, chosen_url, filename or package

        max_workers = min(8, max(1, len(runtime_downloads)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(worker, item): item for item in runtime_downloads}
            for future in as_completed(future_map):
                wheel_path, chosen_url, label = future.result()
                wheel_paths.append(wheel_path)
                chosen_urls.append(chosen_url)
                try:
                    logger(f"Torch 小依赖 wheel 已就绪：{label}（{_format_mb(wheel_path.stat().st_size)}）")
                except OSError:
                    logger(f"Torch 小依赖 wheel 已就绪：{label}")

    return wheel_paths, chosen_urls


def download_app_wheels(
    downloads: list[dict[str, object]],
    log: Callable[[str], None] | None = None,
    *,
    label: str = "常用 Python 依赖",
) -> tuple[list[Path], list[str]]:
    logger = log or (lambda _message: None)
    wheel_paths: list[Path] = []
    chosen_urls: list[str] = []
    if not downloads:
        return wheel_paths, chosen_urls
    logger(f"正在并行预下载{label} wheel：{len(downloads)} 个包，完成后将从本地缓存安装。")

    def worker(item: dict[str, object]) -> tuple[Path, str, str]:
        package = str(item.get("package") or "")
        filename = str(item.get("filename") or "")
        target_path = Path(item["target_path"])
        urls = [str(url) for url in item.get("urls") or []]
        wheel_path, chosen_url = download_file(urls, target_path, logger, prefer_segmented=True, rank_sources=False)
        return wheel_path, chosen_url, filename or package

    max_workers = min(8, max(1, len(downloads)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(worker, item): item for item in downloads}
        for future in as_completed(future_map):
            wheel_path, chosen_url, label = future.result()
            wheel_paths.append(wheel_path)
            chosen_urls.append(chosen_url)
            try:
                logger(f"依赖 wheel 已就绪：{label}（{_format_mb(wheel_path.stat().st_size)}）")
            except OSError:
                logger(f"依赖 wheel 已就绪：{label}")
    return sorted(wheel_paths), chosen_urls


def install_app_wheels(
    wheel_paths: list[Path],
    *,
    dependency_index: str = "",
    dependency_indexes: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--prefer-binary",
        "--no-compile",
        "--timeout",
        "180",
        "--retries",
        "5",
        "--no-index",
        "--find-links",
        str((Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / APP_WHEEL_CACHE_DIRNAME).resolve()),
        "ultralytics",
        "pillow",
        "pyyaml",
    ]
    # If a transitive dependency has no compatible wheel on the mirrors we parsed,
    # pip may still need a normal index as a fallback. Keep it disabled by default
    # through --no-index, but include the URLs in the command metadata for clearer logs.
    _ = (wheel_paths, dependency_index, dependency_indexes)
    return command


def install_tool_wheels(wheel_paths: list[Path]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--prefer-binary",
        "--no-compile",
        "--timeout",
        "180",
        "--retries",
        "5",
        "--no-index",
        "--find-links",
        str((Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / PYTHON_WHEEL_CACHE_DIRNAME).resolve()),
        "pip",
        "setuptools",
        "wheel",
    ]
    _ = wheel_paths
    return command


def install_torch_wheels(
    wheel_paths: list[Path],
    *,
    force_reinstall: bool = False,
    dependency_index: str = "",
    dependency_indexes: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--upgrade",
        "--prefer-binary",
        "--no-compile",
        "--timeout",
        "180",
        "--retries",
        "5",
    ]
    indexes: list[str] = []
    if dependency_index:
        indexes.append(dependency_index)
    if dependency_indexes:
        indexes.extend(str(index) for index in dependency_indexes)
    deduped_indexes: list[str] = []
    for index_url in indexes:
        index_url = str(index_url).strip()
        if index_url and index_url not in deduped_indexes:
            deduped_indexes.append(index_url)
    local_dependency_dir = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / PYTHON_WHEEL_CACHE_DIRNAME
    local_dependency_paths = [
        path
        for path in wheel_paths
        if path.parent.resolve(strict=False) == local_dependency_dir.resolve(strict=False)
    ]
    if len(local_dependency_paths) >= len(TORCH_RUNTIME_WHEEL_PACKAGES):
        command.extend(["--no-index", "--find-links", str(local_dependency_dir.resolve())])
    elif deduped_indexes:
        command.extend(["-i", deduped_indexes[0]])
        for extra_index_url in deduped_indexes[1:]:
            command.extend(["--extra-index-url", extra_index_url])
    if force_reinstall:
        command.append("--force-reinstall")
    command.extend(str(path) for path in wheel_paths)
    return command


def download_file(
    urls: str | list[str],
    target_path: Path,
    log: Callable[[str], None] | None = None,
    *,
    prefer_segmented: bool = False,
    rank_sources: bool = True,
) -> tuple[Path, str]:
    logger = log or (lambda _message: None)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_name(target_path.name + ".part")
    part_dir = target_path.with_name(target_path.name + ".parts")
    if _download_cache_ready(target_path):
        part_path.unlink(missing_ok=True)
        shutil.rmtree(part_dir, ignore_errors=True)
        logger(f"复用已下载缓存：{target_path}")
        return target_path, "cache"
    if _download_part_cache_ready(part_path, target_path):
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        part_path.replace(target_path)
        shutil.rmtree(part_dir, ignore_errors=True)
        logger(f"检测到已完成的临时下载文件，已恢复为正式缓存：{target_path.name}")
        return target_path, "cache"
    if target_path.exists():
        logger(f"检测到损坏缓存，已清理并重新下载：{target_path.name}")
        target_path.unlink(missing_ok=True)
    if part_path.exists() or part_dir.exists():
        logger(f"检测到未完成的下载缓存，将继续断点续传：{target_path.name}")

    candidates = [urls] if isinstance(urls, str) else list(urls)
    candidates = dedupe_candidates(candidates)
    if rank_sources:
        candidates = rank_candidate_urls(candidates, logger)
    if not candidates:
        raise RuntimeError(f"没有可用的下载源：{target_path.name}")
    errors: list[str] = []
    for index, url in enumerate(candidates, start=1):
        try:
            logger(f"尝试下载源 {index}/{len(candidates)}：{url}")
            _download_url_to_path_auto(url, target_path, logger=logger, prefer_segmented=prefer_segmented)
            if not _download_cache_ready(target_path):
                raise RuntimeError(f"下载完成后校验失败：{target_path}")
            part_path.unlink(missing_ok=True)
            shutil.rmtree(part_dir, ignore_errors=True)
            logger(f"下载完成并通过校验：{target_path.name}")
            return target_path, url
        except Exception as exc:
            logger(f"当前下载源失败，准备尝试下一个源：{url}（{exc}）")
            if sys.platform.startswith("win") and not prefer_segmented:
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
                        f"Invoke-WebRequest -Uri '{escaped_url}' -OutFile '{escaped_path}' -TimeoutSec 90"
                    ),
                ]
                fallback = _run_capture(powershell_command)
                if fallback.returncode == 0 and _download_cache_ready(target_path):
                    part_path.unlink(missing_ok=True)
                    shutil.rmtree(part_dir, ignore_errors=True)
                    logger(f"PowerShell 回退下载完成并通过校验：{target_path.name}")
                    return target_path, url
                detail = (fallback.stderr or fallback.stdout or "").strip()
                errors.append(f"{url} -> {exc}；PowerShell 回退失败：{detail}")
            else:
                errors.append(f"{url} -> {exc}")
            if part_path.exists() or part_dir.exists():
                part_path.unlink(missing_ok=True)
                shutil.rmtree(part_dir, ignore_errors=True)
            if target_path.exists() and not _download_cache_ready(target_path):
                target_path.unlink(missing_ok=True)
                logger(f"Downloaded file failed validation, cleaned cache and will try next source: {target_path.name}")
    if errors:
        logger("下载文件失败详情：\n" + "\n".join(errors))
    raise RuntimeError(f"下载文件失败：{target_path.name}。已尝试 {len(errors)} 个下载源，请查看实时日志。")


def configure_embedded_python(target_dir: Path, embedded_python_version: str = DEFAULT_EMBEDDED_PYTHON_VERSION) -> None:
    (target_dir / "DLLs").mkdir(parents=True, exist_ok=True)
    (target_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (target_dir / "Scripts").mkdir(parents=True, exist_ok=True)
    version_parts = embedded_python_version.split(".")
    zip_name = f"python{version_parts[0]}{version_parts[1]}.zip"
    pth_filename = f"python{version_parts[0]}{version_parts[1]}._pth"
    pth_content = f"{zip_name}\nDLLs\nLib\nLib\\site-packages\n.\nimport site\n"
    for filename in (pth_filename, "python._pth"):
        (target_dir / filename).write_text(pth_content, encoding="utf-8")


def install_embedded_runtime(
    target_python: Path,
    log: Callable[[str], None] | None = None,
    accelerator_mode: str = "auto",
) -> dict[str, str]:
    logger = log or (lambda _message: None)
    target_python = target_python.expanduser().resolve(strict=False)
    target_dir = target_python.parent
    runtime_root = target_dir.parent
    parent_dir = runtime_root.parent
    embedded_python_version = resolve_embedded_python_version(accelerator_mode)
    archive_path = Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap" / f"python-{embedded_python_version}-embed.zip"
    download_urls = choose_embedded_python_urls(embedded_python_version)

    logger("正在下载内置 Python 运行时（优先尝试国内镜像）")
    _, download_url = download_file(download_urls, archive_path, logger)
    logger(f"已下载内置 Python 压缩包：{archive_path}")

    parent_dir.mkdir(parents=True, exist_ok=True)
    backup_root = runtime_root.with_name(runtime_root.name + ".previous")
    if not runtime_root.exists() and backup_root.exists():
        logger("检测到上次中断留下的 runtime 备份，正在恢复。")
        shutil.move(str(backup_root), str(runtime_root))

    for stale_temp in parent_dir.glob(f"{runtime_root.name}.installing.*"):
        if stale_temp != runtime_root:
            shutil.rmtree(stale_temp, ignore_errors=True)

    temp_root = parent_dir / f"{runtime_root.name}.installing.{os.getpid()}"
    temp_target_dir = temp_root / target_dir.name
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(temp_target_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError(f"内置 Python 压缩包损坏：{archive_path}") from exc

    configure_embedded_python(temp_target_dir, embedded_python_version)
    temp_python = temp_target_dir / target_python.name
    if not temp_python.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError(f"内置 Python 创建失败，未找到解释器：{target_python}")

    if backup_root.exists():
        shutil.rmtree(backup_root, ignore_errors=True)
    if runtime_root.exists():
        shutil.move(str(runtime_root), str(backup_root))
    try:
        shutil.move(str(temp_root), str(runtime_root))
        if not target_python.exists():
            raise RuntimeError(f"内置 Python 创建失败，未找到解释器：{target_python}")
    except Exception:
        shutil.rmtree(runtime_root, ignore_errors=True)
        if backup_root.exists() and not runtime_root.exists():
            shutil.move(str(backup_root), str(runtime_root))
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root, ignore_errors=True)

    logger(f"内置 Python 已创建：{target_python}")
    return {
        "target_python": str(target_python),
        "runtime_root": str(runtime_root),
        "python_version": embedded_python_version,
        "download_url": download_url,
    }


def download_get_pip(target_dir: Path | None = None) -> Path:
    cache_dir = target_dir or (Path(tempfile.gettempdir()) / "yolo_local_desktop_bootstrap")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "get-pip.py"
    download_file(choose_get_pip_urls(), target_path)
    return target_path


def pip_is_usable() -> bool:
    if not has_pip():
        return False
    try:
        result = _run_capture([sys.executable, "-m", "pip", "--version"])
    except Exception:
        return False
    return result.returncode == 0


def site_packages_dir() -> Path:
    purelib = sysconfig.get_paths().get("purelib", "")
    if purelib:
        return Path(purelib)
    return Path(sys.prefix) / "Lib" / "site-packages"


def _cleanup_extracted_bootstrap_package(site_packages: Path, package: str) -> None:
    normalized = package.replace("-", "_").lower()
    candidates = [
        site_packages / normalized,
        site_packages / package.lower(),
    ]
    patterns = [
        f"{normalized}-*.dist-info",
        f"{package.lower()}-*.dist-info",
    ]
    if normalized == "setuptools":
        candidates.extend([site_packages / "_distutils_hack", site_packages / "distutils-precedence.pth"])
    for candidate in candidates:
        try:
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            elif candidate.exists():
                candidate.unlink(missing_ok=True)
        except OSError:
            pass
    for pattern in patterns:
        for candidate in site_packages.glob(pattern):
            try:
                if candidate.is_dir():
                    shutil.rmtree(candidate, ignore_errors=True)
                else:
                    candidate.unlink(missing_ok=True)
            except OSError:
                pass


def _extract_bootstrap_wheel(wheel_path: Path, site_packages: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(site_packages)


def bootstrap_pip_from_wheels(log: Callable[[str], None] | None = None) -> dict[str, str]:
    logger = log or (lambda _message: None)
    logger("正在尝试使用本地 wheel 引导 pip，避免 get-pip.py 在新版本 Python 中依赖 distutils 失败。")
    indexes = rank_candidate_urls(build_generic_pip_index_candidates(), logger)
    downloads = build_python_wheel_downloads(
        ("setuptools", "pip", "wheel"),
        indexes,
        cache_dirname=PYTHON_WHEEL_CACHE_DIRNAME,
        log=logger,
    )
    packages = {str(item.get("package") or "").lower(): item for item in downloads}
    required = ("setuptools", "pip", "wheel")
    missing = [package for package in required if package not in packages]
    if missing:
        raise RuntimeError("无法解析 pip 引导 wheel：" + ", ".join(missing))

    site_packages = site_packages_dir()
    site_packages.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    for package in required:
        item = packages[package]
        wheel_path, _chosen_url = download_file(
            [str(url) for url in item.get("urls") or []],
            Path(item["target_path"]),
            logger,
            prefer_segmented=True,
            rank_sources=False,
        )
        _cleanup_extracted_bootstrap_package(site_packages, package)
        _extract_bootstrap_wheel(wheel_path, site_packages)
        extracted.append(wheel_path.name)
        logger(f"已解压 pip 引导 wheel：{wheel_path.name}")

    importlib.invalidate_caches()
    if not pip_is_usable():
        raise RuntimeError("已解压 pip 引导 wheel，但当前解释器仍无法运行 pip。")
    logger("本地 wheel 引导 pip 成功。")
    return {
        "method": "wheel-bootstrap",
        "message": "已通过 pip/setuptools/wheel 本地 wheel 引导 pip。",
        "site_packages": str(site_packages),
        "wheels": ", ".join(extracted),
    }


def bootstrap_pip(log: Callable[[str], None] | None = None) -> dict[str, str]:
    logger = log or (lambda _message: None)

    if has_pip() and pip_is_usable():
        logger("已检测到 pip，跳过引导步骤。")
        return {"method": "existing", "message": "已检测到 pip，直接继续安装依赖。"}
    if has_pip():
        logger("检测到 pip 包但无法正常运行，将尝试修复 pip。")

    ensurepip_command = [sys.executable, "-m", "ensurepip", "--upgrade"]
    if has_ensurepip():
        logger("检测到 ensurepip，正在本地初始化 pip。")
        return_code = stream_command(ensurepip_command)
        if return_code == 0 and pip_is_usable():
            logger("ensurepip 初始化成功。")
            return {"method": "ensurepip", "message": "已通过 ensurepip 初始化 pip。"}
        logger("ensurepip 初始化失败，改为本地 wheel 引导 pip。")
    else:
        logger("当前 Python 不包含 ensurepip，改为本地 wheel 引导 pip。")

    try:
        return bootstrap_pip_from_wheels(logger)
    except Exception as wheel_exc:
        logger(f"本地 wheel 引导 pip 失败，改为在线下载 get-pip.py 安装 pip：{wheel_exc}")

    logger("正在下载 pip 引导脚本（优先尝试国内镜像）")
    script_path = download_get_pip()
    logger(f"已下载 pip 引导脚本：{script_path}")

    install_command = [sys.executable, str(script_path), "--disable-pip-version-check"]
    return_code = stream_command(install_command)
    if return_code != 0:
        raise RuntimeError(f"在线安装 pip 失败，退出码为 {return_code}。本地 wheel 引导也已失败，请查看上方日志。")
    if not pip_is_usable():
        raise RuntimeError("在线安装 pip 已执行完成，但当前解释器仍未检测到 pip。")

    logger("在线安装 pip 成功。")
    return {"method": "get-pip", "message": "已通过 get-pip.py 在线安装 pip。", "script": str(script_path)}


def build_install_plan(
    log: Callable[[str], None] | None = None,
    accelerator_mode: str = "auto",
) -> dict[str, object]:
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
    normalized_accelerator_mode = normalize_accelerator_mode(accelerator_mode)
    accelerator, index_url = choose_torch_index(
        str(gpu.get("cuda_version") or ""),
        str(gpu.get("gpu_architecture") or ""),
        normalized_accelerator_mode,
        str(gpu.get("gpu_name") or ""),
        str(gpu.get("compute_capability") or ""),
    )
    legacy_compatibility = legacy_cuda_compatibility(
        str(gpu.get("gpu_name") or ""),
        str(gpu.get("gpu_architecture") or ""),
        str(gpu.get("compute_capability") or ""),
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
            "--prefer-binary",
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

    logger("正在解析基础安装工具 wheel 下载地址。")
    tool_wheel_downloads = build_python_wheel_downloads(
        TOOL_WHEEL_PACKAGES,
        generic_indexes,
        cache_dirname=PYTHON_WHEEL_CACHE_DIRNAME,
        log=logger,
    )
    tool_fallback_steps = [build_pip_command(index_url_value, "pip", "setuptools", "wheel") for index_url_value in generic_indexes]
    if len(tool_wheel_downloads) >= len(TOOL_WHEEL_PACKAGES):
        tool_steps = [["__DOWNLOAD_TOOL_WHEELS__"]]
    else:
        tool_steps = tool_fallback_steps
    torch_packages = torch_packages_for_accelerator(accelerator)

    torch_steps: list[list[str]] = []
    torch_wheel_downloads: list[dict[str, object]] = []
    if accelerator == "cpu":
        torch_steps = [
            build_pip_command(index_url_value, *torch_packages, force_reinstall=force_torch_reinstall)
            for index_url_value in generic_indexes
        ]
    else:
        torch_wheel_downloads = build_torch_wheel_downloads(torch_indexes, torch_packages)
        if torch_wheel_downloads and accelerator == "cu118":
            runtime_wheel_downloads = build_python_wheel_downloads(
                TORCH_RUNTIME_WHEEL_PACKAGES,
                generic_indexes,
                cache_dirname=PYTHON_WHEEL_CACHE_DIRNAME,
                log=logger,
            )
            for item in runtime_wheel_downloads:
                item["category"] = "torch-runtime"
            torch_wheel_downloads.extend(runtime_wheel_downloads)
            torch_steps = [["__DOWNLOAD_TORCH_WHEELS__"]]
        else:
            for torch_index in torch_indexes:
                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--upgrade",
                    "--prefer-binary",
                    "--timeout",
                    "180",
                    "--retries",
                    "5",
                    "--index-url",
                    torch_index,
                ]
                if force_torch_reinstall:
                    command.append("--force-reinstall")
                command.extend(torch_packages)
                torch_steps.append(command)

    logger("正在解析常用 Python 依赖 wheel 下载地址。")
    app_wheel_downloads = build_app_wheel_downloads(generic_indexes, logger)
    app_fallback_steps = [build_pip_command(index_url_value, "ultralytics", "pillow", "pyyaml") for index_url_value in generic_indexes]
    if app_wheel_downloads:
        app_steps = [["__DOWNLOAD_APP_WHEELS__"]]
    else:
        app_steps = app_fallback_steps

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
        "accelerator_mode": normalized_accelerator_mode,
        "embedded_python_version": resolve_embedded_python_version(normalized_accelerator_mode),
        "legacy_cuda_compatibility": legacy_compatibility,
        "torch_index": " -> ".join(torch_indexes),
        "pip_indexes": " -> ".join(generic_indexes),
        "gpu": gpu,
        "installed_torch_accelerator": installed_torch_accelerator,
        "force_torch_reinstall": force_torch_reinstall,
        "torch_packages": " ".join(torch_packages),
        "tool_wheel_downloads": tool_wheel_downloads,
        "tool_fallback_steps": tool_fallback_steps,
        "torch_wheel_downloads": torch_wheel_downloads,
        "app_wheel_downloads": app_wheel_downloads,
        "app_fallback_steps": app_fallback_steps,
        "pip_dependency_index": generic_indexes[0] if generic_indexes else DEFAULT_PIP_INDEX_URL,
        "pip_dependency_indexes": generic_indexes,
        "command_extra_index_flags": [True, accelerator == "cpu", True],
        "pip_bootstrap": detect_pip_bootstrap_strategy(),
        "commands": commands,
        "notes": [
            "安装方案会根据 NVIDIA 环境自动判断。",
            "CUDA 版本到 PyTorch 下载源的映射策略：12.8+ -> cu128，12.6+ -> cu126，11.8+ -> cu118，否则回退 CPU。",
            "如果检测到 Maxwell / Pascal / Volta 这类老架构显卡，会回退 CPU 方案，避免安装当前 PyTorch 已不支持的 CUDA 轮子。",
            str(legacy_compatibility.get("message") or ""),
            "用户也可以手动选择老显卡 CUDA 兼容模式，程序会切换到更适合 cu118 的内置 Python 版本，并尝试 CUDA 11.8 轮子。",
            "如果检测到 Blackwell / Ada / Hopper / Ampere / Turing 等显卡，但 nvidia-smi 没有返回 CUDA 版本，会优先按 cu128 显卡方案处理。",
            "如果现有 Torch 类型和推荐方案不一致，会自动重装 Torch，避免 CPU 版残留导致显卡不可用。",
            "如果当前 Python 没有 pip，会自动尝试 ensurepip；若不可用，则优先用 pip/setuptools/wheel 的本地 wheel 引导，最后再回退 get-pip.py。",
            "内置 Python、pip 引导 wheel、get-pip 和 pip 镜像会先测速，再优先走响应更快的国内源；不可用时再自动回退。",
            "PyTorch CUDA 轮子会按测速结果优先尝试更快的镜像源，再回退官方源。",
            "显卡版 PyTorch 安装会使用隔离 pip 模式，避免普通 PyPI 源把 CUDA 方案替换成 CPU 版 Torch。",
            "Ultralytics / OpenCV / Matplotlib 等应用依赖会优先并行预下载 wheel，再从本地缓存安装，避免单个 pip 源拖慢整体流程。",
            "所有 pip 安装步骤都启用更长超时与重试，尽量减少弱网环境下的失败概率。",
        ],
    }


def stream_command(command: list[str], *, include_extra_index: bool = True) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path.cwd()),
        env=_process_env(include_extra_index=include_extra_index),
        creationflags=NO_WINDOW_FLAGS,
    )
    assert process.stdout is not None
    for line in process.stdout:
        try:
            print(line.rstrip(), flush=True)
        except (OSError, ValueError):
            # Windowed launcher processes may not have a usable stdout handle.
            # Do not let forwarding child logs crash environment setup.
            pass
    return process.wait()
