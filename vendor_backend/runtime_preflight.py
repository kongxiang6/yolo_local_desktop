from __future__ import annotations

import importlib.util
import json
import os
import sys
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

    return {
        "yolo_class": YOLO.__name__,
        "torch_version": torch.__version__,
        "yaml_module": getattr(yaml, "__file__", ""),
    }


def run_runtime_preflight() -> dict[str, object]:
    report = inspect_site_packages()
    report.update(collect_runtime_versions())
    return report


def main() -> int:
    print(json.dumps(run_runtime_preflight(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
