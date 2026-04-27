from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

REQUIRED_MODULES = ("ultralytics", "torch", "yaml")


def find_site_packages(search_paths: Iterable[str] | None = None) -> Path:
    for entry in search_paths or sys.path:
        normalized = str(entry).replace("\\", "/").lower()
        if not normalized.endswith("/site-packages"):
            continue
        candidate = Path(entry)
        if candidate.exists():
            return candidate
    raise RuntimeError(f"未在 sys.path 中找到 site-packages 路径：{list(search_paths or sys.path)!r}")


def inspect_required_modules(
    site_packages: Path,
    required_modules: Iterable[str] = REQUIRED_MODULES,
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    scandir: Callable[[os.PathLike[str] | str], object] = os.scandir,
) -> dict[str, dict[str, object]]:
    modules: dict[str, dict[str, object]] = {}
    for name in required_modules:
        module_root = site_packages / name
        if not module_root.exists():
            raise RuntimeError(f"缺少必需的包目录：{module_root}")
        if not module_root.is_dir():
            raise RuntimeError(f"必需的包路径不是目录：{module_root}")

        try:
            with scandir(module_root) as iterator:
                sample_entries = [entry.name for entry in islice(iterator, 5)]
        except OSError as exc:
            raise RuntimeError(f"无法枚举包目录 {module_root}：{exc}") from exc

        init_file = module_root / "__init__.py"
        if not init_file.exists():
            raise RuntimeError(f"缺少必需的包初始化文件：{init_file}")

        spec = find_spec(name)
        origin = getattr(spec, "origin", None) if spec is not None else None
        loader = getattr(spec, "loader", None) if spec is not None else None
        locations = list(getattr(spec, "submodule_search_locations", []) or []) if spec is not None else []
        if spec is None or origin is None or loader is None:
            raise RuntimeError(
                f"{name} 被解析为命名空间包或未知包："
                f"origin={origin!r}, loader={loader!r}, locations={locations!r}"
            )

        modules[name] = {
            "root": str(module_root),
            "origin": str(origin),
            "sample_entries": sample_entries,
        }
    return modules


def inspect_site_packages(
    search_paths: Iterable[str] | None = None,
    required_modules: Iterable[str] = REQUIRED_MODULES,
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    scandir: Callable[[os.PathLike[str] | str], object] = os.scandir,
) -> dict[str, object]:
    site_packages = find_site_packages(search_paths)
    return {
        "site_packages": str(site_packages),
        "modules": inspect_required_modules(site_packages, required_modules, find_spec=find_spec, scandir=scandir),
    }


def collect_runtime_versions() -> dict[str, object]:
    from ultralytics import YOLO
    import torch
    import yaml

    torch_cuda_version = str(getattr(torch.version, "cuda", None) or "")
    torch_cuda_available = False
    torch_device_count = 0
    torch_device_name = ""
    torch_device_capability = ""
    torch_cuda_warnings: list[str] = []
    torch_cuda_compatible = False
    runtime_backend = "cpu"
    runtime_backend_label = "当前运行环境使用 CPU"
    torch_build_label = "CPU 版 Torch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch_cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda_available = False

        if torch_cuda_available:
            try:
                torch_device_count = int(torch.cuda.device_count())
            except Exception:
                torch_device_count = 0
            if torch_device_count > 0:
                try:
                    torch_device_name = str(torch.cuda.get_device_name(0) or "")
                except Exception:
                    torch_device_name = ""
                try:
                    capability = torch.cuda.get_device_capability(0)
                    if capability and len(capability) >= 2:
                        torch_device_capability = f"{capability[0]}.{capability[1]}"
                except Exception:
                    torch_device_capability = ""

        for item in caught:
            message = str(item.message).strip()
            if message:
                torch_cuda_warnings.append(message)

    if torch_cuda_version:
        torch_build_label = f"NVIDIA 显卡版 Torch（CUDA {torch_cuda_version}）"

    incompatible_warning = any("not compatible with the current pytorch installation" in message.lower() for message in torch_cuda_warnings)
    torch_cuda_compatible = bool(torch_cuda_available and torch_device_count > 0 and not incompatible_warning)

    if torch_cuda_compatible:
        runtime_backend = "nvidia"
        runtime_backend_label = f"当前运行环境已启用 NVIDIA 显卡：{torch_device_name or '未知型号'}"
    elif torch_cuda_available and torch_device_count > 0 and incompatible_warning:
        runtime_backend = "nvidia-unsupported"
        capability_text = f"（算力 {torch_device_capability}）" if torch_device_capability else ""
        runtime_backend_label = (
            f"检测到 NVIDIA 显卡：{torch_device_name or '未知型号'}{capability_text}，"
            "但当前安装的 Torch 与这张显卡不兼容，暂时不能稳定使用显卡"
        )
    elif torch_cuda_version:
        runtime_backend = "cuda-build-no-device"
        runtime_backend_label = f"已安装显卡版 Torch（CUDA {torch_cuda_version}），但当前没有真正启用 NVIDIA 显卡"

    return {
        "yolo_class": YOLO.__name__,
        "torch_version": torch.__version__,
        "torch_build_label": torch_build_label,
        "torch_cuda_version": torch_cuda_version,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_compatible": torch_cuda_compatible,
        "torch_device_count": torch_device_count,
        "torch_device_name": torch_device_name,
        "torch_device_capability": torch_device_capability,
        "torch_cuda_warnings": torch_cuda_warnings,
        "runtime_backend": runtime_backend,
        "runtime_backend_label": runtime_backend_label,
        "yaml_module": getattr(yaml, "__file__", ""),
    }


def build_broken_runtime_report(error_message: str) -> dict[str, object]:
    return {
        "yolo_class": "",
        "torch_version": "",
        "torch_build_label": "当前 Torch / YOLO 依赖导入失败",
        "torch_cuda_version": "",
        "torch_cuda_available": False,
        "torch_cuda_compatible": False,
        "torch_device_count": 0,
        "torch_device_name": "",
        "torch_device_capability": "",
        "torch_cuda_warnings": [],
        "runtime_backend": "broken",
        "runtime_backend_label": "当前运行环境不完整，需要重新配置环境",
        "yaml_module": "",
        "preflight_error": error_message,
    }


def run_runtime_preflight() -> dict[str, object]:
    report: dict[str, object] = {}
    try:
        report.update(inspect_site_packages())
    except Exception as exc:
        report.update(
            {
                "site_packages": "",
                "modules": {},
                "site_packages_error": str(exc),
            }
        )

    try:
        report.update(collect_runtime_versions())
    except Exception as exc:
        report.update(build_broken_runtime_report(str(exc)))
    return report


def main() -> int:
    print(json.dumps(run_runtime_preflight(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
