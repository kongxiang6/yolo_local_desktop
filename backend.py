from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor_backend"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0


def _move_import_path_to_end(path: Path) -> None:
    """Keep bundled app modules importable without shadowing runtime stdlib extensions."""
    target = path.resolve(strict=False)
    kept_paths: list[str] = []
    for entry in sys.path:
        entry_path = Path(entry or os.getcwd()).resolve(strict=False)
        if entry_path == target:
            continue
        kept_paths.append(entry)
    kept_paths.append(str(target))
    sys.path[:] = kept_paths


_move_import_path_to_end(APP_DIR)
_move_import_path_to_end(VENDOR_DIR)

from annotation_support import AnnotationBox, ensure_class_names, list_annotation_images, save_class_names, save_yolo_boxes

import runtime_preflight
import runtime_installer

ALL_TASKS = ("detect", "segment", "classify", "pose", "obb")
TRACK_TASKS = ("detect", "segment", "pose", "obb")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_protocol_print(text: str) -> None:
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        # Windowed PyInstaller children can have an invalid stdout handle when
        # launched without an attached console. Logging must never crash the
        # backend command itself.
        pass


def emit(tag: str, payload: dict) -> None:
    safe_protocol_print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}")


def load_json(path: str) -> dict:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件必须是 JSON 对象。")
    return payload


def emit_status(message: str, **extra: object) -> None:
    payload = {"message": message}
    payload.update(extra)
    emit("STATUS", payload)


def _backend_process_env(*, include_extra_index: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_NO_INPUT"] = "1"
    if not include_extra_index:
        env.pop("PIP_EXTRA_INDEX_URL", None)
    return env


def run_runtime_preflight_subprocess(include_extra_index: bool = True) -> dict[str, object]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--backend-command", "runtime-preflight"]
    else:
        command = [sys.executable, "-u", str(VENDOR_DIR / "runtime_preflight.py")]
    process = subprocess.run(
        command,
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_backend_process_env(include_extra_index=include_extra_index),
        creationflags=NO_WINDOW_FLAGS,
        check=False,
    )
    output = (process.stdout or "").strip()
    if process.returncode != 0:
        detail = (process.stderr or output or f"退出码 {process.returncode}").strip()
        return runtime_preflight.build_broken_runtime_report(detail)
    if not output:
        return runtime_preflight.build_broken_runtime_report("运行环境自检没有返回结果。")
    try:
        payload = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return runtime_preflight.build_broken_runtime_report(f"运行环境自检结果无法解析：{exc}；输出：{output[-500:]}")
    if not isinstance(payload, dict):
        return runtime_preflight.build_broken_runtime_report("运行环境自检返回了非对象结果。")
    return payload


def run_runtime_preflight_command(_: argparse.Namespace) -> int:
    safe_protocol_print(json.dumps(runtime_preflight.run_runtime_preflight(), ensure_ascii=False))
    return 0


def validate_configured_runtime(plan: dict[str, object], report: dict[str, object]) -> None:
    expected_accelerator = str(plan.get("accelerator") or "cpu").strip().lower()
    runtime_backend = str(report.get("runtime_backend") or "cpu").strip().lower()
    if runtime_backend == "broken":
        detail = str(report.get("preflight_error") or report.get("runtime_backend_label") or "依赖导入失败").strip()
        raise RuntimeError(f"环境配置后自检失败：{detail}。请重新执行一键配置环境。")
    if expected_accelerator == "cpu" or runtime_backend == "nvidia":
        return

    expected_label = str(plan.get("accelerator_label") or expected_accelerator.upper()).strip()
    runtime_label = str(report.get("runtime_backend_label") or "当前运行环境使用 CPU").strip()
    torch_version = str(report.get("torch_version") or "未知").strip()
    torch_build = str(report.get("torch_build_label") or "").strip()
    torch_cuda_version = str(report.get("torch_cuda_version") or "").strip()
    torch_cuda_warnings = report.get("torch_cuda_warnings") or []
    warning_text = ""
    if isinstance(torch_cuda_warnings, list) and torch_cuda_warnings:
        warning_text = f"；CUDA 警告：{str(torch_cuda_warnings[0]).strip()}"
    cuda_text = f"，Torch CUDA: {torch_cuda_version}" if torch_cuda_version else ""
    raise RuntimeError(
        "你选择的是显卡 CUDA 方案，但安装完成后的自检没有启用 NVIDIA 显卡，"
        "程序已停止并且不会把这次环境配置当作成功。\n"
        f"计划方案：{expected_label}；实际结果：{runtime_label}；"
        f"Torch: {torch_version}（{torch_build or '未知构建'}）{cuda_text}{warning_text}。\n"
        "请重新点击“一键配置环境”后选择合适方案：1050 Ti / Pascal 可尝试“老显卡 CUDA 兼容模式”；"
        "如果仍失败，说明当前驱动、Python 或 PyTorch 轮子组合无法启用显卡，请改选“稳定 CPU 模式”。"
    )


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _public_install_plan(plan: dict[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"commands", "tool_fallback_steps", "app_fallback_steps"}
    }
    for key in ("tool_wheel_downloads", "torch_wheel_downloads", "app_wheel_downloads"):
        downloads = payload.get(key)
        if not isinstance(downloads, list):
            continue
        payload[f"{key}_count"] = len(downloads)
        payload[key] = [
            {
                item_key: item.get(item_key)
                for item_key in ("requirement", "package", "filename", "target_path")
                if isinstance(item, dict) and item.get(item_key)
            }
            for item in downloads
            if isinstance(item, dict)
        ]
    return payload


def summarize_download_sources(label: str, chosen_urls: list[str]) -> str:
    hosts: list[str] = []
    cache_count = 0
    for url in chosen_urls:
        if url == "cache":
            cache_count += 1
            continue
        host = urlparse(url).netloc or url
        if host and host not in hosts:
            hosts.append(host)
    parts = [f"{len(chosen_urls)} 个文件"]
    if cache_count:
        parts.append(f"{cache_count} 个复用缓存")
    if hosts:
        parts.append("来源：" + " / ".join(hosts[:4]) + (" ..." if len(hosts) > 4 else ""))
    return f"{label}下载完成：" + "，".join(parts)


def build_export_namespace(task: str, weights: str, config: dict) -> SimpleNamespace:
    defaults = {
        "task": task,
        "weights": weights,
        "format": "onnx",
        "imgsz": 640,
        "opset": 0,
        "workspace": 0.0,
        "batch": 1,
        "data": "",
        "fraction": 1.0,
        "device": "",
        "output_dir": "",
        "name": "",
        "dynamic": False,
        "simplify": False,
        "half": False,
        "int8": False,
        "nms": False,
        "optimize": False,
        "keras": False,
    }
    defaults.update(config)
    return SimpleNamespace(**defaults)


def run_check(_: argparse.Namespace) -> int:
    system_nvidia = runtime_installer.detect_nvidia_environment()
    recommended_accelerator, recommended_torch_index = runtime_installer.choose_torch_index(
        str(system_nvidia.get("cuda_version") or ""),
        str(system_nvidia.get("gpu_architecture") or ""),
        "auto",
        str(system_nvidia.get("gpu_name") or ""),
        str(system_nvidia.get("compute_capability") or ""),
    )
    report = run_runtime_preflight_subprocess()
    runtime_backend = str(report.get("runtime_backend") or "").strip().lower()
    active_accelerator = recommended_accelerator
    active_torch_index = recommended_torch_index
    if runtime_backend == "nvidia":
        torch_cuda_version = str(report.get("torch_cuda_version") or "").strip()
        active_accelerator = f"cu{torch_cuda_version.replace('.', '')}" if torch_cuda_version else "cuda"
        active_torch_index = ""

    report.update(runtime_installer.build_accelerator_summary(active_accelerator, system_nvidia))
    report["accelerator"] = active_accelerator
    report["torch_index"] = active_torch_index
    report["recommended_accelerator"] = recommended_accelerator
    report["recommended_torch_index"] = recommended_torch_index
    report["system_nvidia"] = system_nvidia
    report["legacy_cuda_compatibility"] = runtime_installer.legacy_cuda_compatibility(
        str(system_nvidia.get("gpu_name") or ""),
        str(system_nvidia.get("gpu_architecture") or ""),
        str(system_nvidia.get("compute_capability") or ""),
    )
    report["python"] = sys.executable
    emit("RESULT", {"kind": "check", **report})
    return 0


def run_configure_env(args: argparse.Namespace) -> int:
    emit_status("正在检测电脑配置和下载源")
    diagnostic_logger = lambda message: emit("APP_DIAGNOSTIC", {"level": "info", "message": message})
    runtime_installer.cleanup_download_cache(diagnostic_logger, remove_ready_wheels=False, max_age_hours=6.0)
    plan = runtime_installer.build_install_plan(
        log=diagnostic_logger,
        accelerator_mode=args.accelerator_mode,
    )
    accelerator_summary = runtime_installer.build_accelerator_summary(str(plan["accelerator"]), plan.get("gpu") or {})
    plan.update(accelerator_summary)
    emit_status(
        "已生成环境安装方案",
        python=plan["python"],
        accelerator=plan["accelerator"],
        accelerator_label=plan["accelerator_label"],
        torch_index=plan["torch_index"],
    )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Python: {plan['python']} ({plan['python_version']})"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Platform: {plan['platform']}"})
    gpu = plan.get("gpu") or {}
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"电脑硬件检测: {plan['hardware_label']}"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"计划安装方案: {plan['accelerator_label']}"})
    installed_torch_accelerator = str(plan.get("installed_torch_accelerator") or "").strip()
    if installed_torch_accelerator:
        emit("APP_DIAGNOSTIC", {"level": "info", "message": f"当前已安装 Torch 类型: {installed_torch_accelerator.upper()}"})
    if plan.get("force_torch_reinstall"):
        emit("APP_DIAGNOSTIC", {"level": "warning", "message": "当前 Torch 类型和推荐方案不一致，将自动重装 Torch。"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Torch 下载源: {plan['torch_index']}"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Pip 引导策略: {plan['pip_bootstrap']}"})
    for note in plan.get("notes", []):
        emit("APP_DIAGNOSTIC", {"level": "info", "message": str(note)})

    if args.dry_run:
        emit("RESULT", {"kind": "configure-env", **_public_install_plan(plan)})
        return 0

    emit_status("正在准备 pip 环境")
    bootstrap_info = runtime_installer.bootstrap_pip(
        diagnostic_logger
    )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Pip 准备完成: {bootstrap_info['message']}"})

    commands = list(plan.get("commands") or [])
    extra_index_flags = list(plan.get("command_extra_index_flags") or [])
    tool_wheel_downloads = list(plan.get("tool_wheel_downloads") or [])
    tool_fallback_steps = list(plan.get("tool_fallback_steps") or [])
    torch_wheel_downloads = list(plan.get("torch_wheel_downloads") or [])
    app_wheel_downloads = list(plan.get("app_wheel_downloads") or [])
    app_fallback_steps = list(plan.get("app_fallback_steps") or [])
    for index, command in enumerate(commands, start=1):
        if command == [["__DOWNLOAD_TOOL_WHEELS__"]]:
            emit_status(f"正在执行第 {index}/{len(commands)} 步", command="download tool wheels")
            try:
                wheel_paths, chosen_urls = runtime_installer.download_app_wheels(
                    tool_wheel_downloads,
                    diagnostic_logger,
                    label="基础安装工具",
                )
                install_command = runtime_installer.install_tool_wheels(wheel_paths)
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "info",
                        "message": summarize_download_sources("基础安装工具 wheel ", chosen_urls),
                    },
                )
                emit_status(
                    f"正在执行第 {index}/{len(commands)} 步（本地安装基础工具 wheel）",
                    command=subprocess.list2cmdline(install_command),
                )
                return_code = runtime_installer.stream_command(install_command, include_extra_index=False)
                if return_code == 0:
                    continue
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"基础工具本地 wheel 安装失败，退出码 {return_code}，自动回退到普通 pip 多源安装。",
                    },
                )
            except Exception as exc:
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"基础工具预下载失败，自动回退到普通 pip 多源安装：{exc}",
                    },
                )
            alternatives = tool_fallback_steps
            last_code = 0
            for attempt_index, attempt_command in enumerate(alternatives, start=1):
                emit_status(
                    f"正在执行第 {index}/{len(commands)} 步（回退尝试 {attempt_index}/{len(alternatives)}）",
                    command=subprocess.list2cmdline(attempt_command),
                )
                return_code = runtime_installer.stream_command(attempt_command, include_extra_index=True)
                if return_code == 0:
                    last_code = 0
                    break
                last_code = return_code
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"第 {index} 步基础工具回退源 {attempt_index} 失败，退出码 {return_code}，准备尝试下一个源。",
                    },
                )
            if last_code != 0:
                raise RuntimeError(f"环境配置失败，第 {index} 步退出码为 {last_code}。")
            continue
        if command == [["__DOWNLOAD_TORCH_WHEELS__"]]:
            emit_status(f"正在执行第 {index}/{len(commands)} 步", command="download torch wheels")
            wheel_paths, chosen_urls = runtime_installer.download_torch_wheels(torch_wheel_downloads, diagnostic_logger)
            install_command = runtime_installer.install_torch_wheels(
                wheel_paths,
                force_reinstall=bool(plan.get("force_torch_reinstall")),
                dependency_index=str(plan.get("pip_dependency_index") or ""),
                dependency_indexes=list(plan.get("pip_dependency_indexes") or []),
            )
            emit(
                "APP_DIAGNOSTIC",
                {
                    "level": "info",
                    "message": summarize_download_sources("Torch wheel ", chosen_urls),
                },
            )
            emit_status(
                f"正在执行第 {index}/{len(commands)} 步（本地安装 Torch wheel）",
                command=subprocess.list2cmdline(install_command),
            )
            return_code = runtime_installer.stream_command(install_command, include_extra_index=False)
            if return_code != 0:
                raise RuntimeError(f"环境配置失败，第 {index} 步退出码为 {return_code}。")
            continue
        if command == [["__DOWNLOAD_APP_WHEELS__"]]:
            emit_status(f"正在执行第 {index}/{len(commands)} 步", command="download app wheels")
            try:
                wheel_paths, chosen_urls = runtime_installer.download_app_wheels(app_wheel_downloads, diagnostic_logger)
                install_command = runtime_installer.install_app_wheels(
                    wheel_paths,
                    dependency_index=str(plan.get("pip_dependency_index") or ""),
                    dependency_indexes=list(plan.get("pip_dependency_indexes") or []),
                )
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "info",
                        "message": summarize_download_sources("应用依赖 wheel ", chosen_urls),
                    },
                )
                emit_status(
                    f"正在执行第 {index}/{len(commands)} 步（本地安装应用依赖 wheel）",
                    command=subprocess.list2cmdline(install_command),
                )
                return_code = runtime_installer.stream_command(install_command, include_extra_index=False)
                if return_code == 0:
                    continue
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"本地 wheel 安装失败，退出码 {return_code}，自动回退到普通 pip 多源安装。",
                    },
                )
            except Exception as exc:
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"应用依赖预下载失败，自动回退到普通 pip 多源安装：{exc}",
                    },
                )
            alternatives = app_fallback_steps
            include_extra_index = True
            last_code = 0
            for attempt_index, attempt_command in enumerate(alternatives, start=1):
                emit_status(
                    f"正在执行第 {index}/{len(commands)} 步（回退尝试 {attempt_index}/{len(alternatives)}）",
                    command=subprocess.list2cmdline(attempt_command),
                )
                return_code = runtime_installer.stream_command(attempt_command, include_extra_index=include_extra_index)
                if return_code == 0:
                    last_code = 0
                    break
                last_code = return_code
                emit(
                    "APP_DIAGNOSTIC",
                    {
                        "level": "warning",
                        "message": f"第 {index} 步回退源 {attempt_index} 失败，退出码 {return_code}，准备尝试下一个源。",
                    },
                )
            if last_code != 0:
                raise RuntimeError(f"环境配置失败，第 {index} 步退出码为 {last_code}。")
            continue
        alternatives = command if command and isinstance(command[0], list) else [command]
        include_extra_index = bool(extra_index_flags[index - 1]) if index <= len(extra_index_flags) else True
        last_code = 0
        for attempt_index, attempt_command in enumerate(alternatives, start=1):
            emit_status(
                f"正在执行第 {index}/{len(commands)} 步（尝试 {attempt_index}/{len(alternatives)}）",
                command=subprocess.list2cmdline(attempt_command),
            )
            return_code = runtime_installer.stream_command(attempt_command, include_extra_index=include_extra_index)
            if return_code == 0:
                last_code = 0
                break
            last_code = return_code
            emit(
                "APP_DIAGNOSTIC",
                {
                    "level": "warning",
                    "message": f"第 {index} 步第 {attempt_index} 个下载源失败，退出码 {return_code}，准备尝试下一个源。",
                },
            )
        if last_code != 0:
            raise RuntimeError(f"环境配置失败，第 {index} 步退出码为 {last_code}。")

    report = run_runtime_preflight_subprocess(include_extra_index=args.accelerator_mode != "legacy-cuda")
    report["system_nvidia"] = gpu
    report.update(accelerator_summary)
    runtime_backend = str(report.get("runtime_backend") or "cpu")
    if runtime_backend == "nvidia-unsupported":
        emit(
            "APP_DIAGNOSTIC",
            {
                "level": "warning",
                "message": "检测到了 NVIDIA 显卡，但当前安装的 Torch 与这张显卡不兼容；现在不会把它当成可正常使用的显卡环境。",
            },
        )
    elif gpu.get("available") and runtime_backend != "nvidia":
        emit(
            "APP_DIAGNOSTIC",
            {
                "level": "warning",
                "message": "电脑里检测到了 NVIDIA 显卡，但当前运行环境还没有真正启用显卡，本次运行会按 CPU 或降级模式处理。",
            },
        )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"当前运行结果: {report.get('runtime_backend_label')}"})
    validate_configured_runtime(plan, report)
    runtime_installer.cleanup_download_cache(diagnostic_logger, remove_ready_wheels=True)
    emit(
        "RESULT",
        {
            "kind": "configure-env",
            **_public_install_plan(plan),
            "pip_bootstrap_result": bootstrap_info,
            **report,
        },
    )
    return 0


def run_bootstrap_runtime(args: argparse.Namespace) -> int:
    target_python = Path(args.target_python).expanduser().resolve(strict=False)
    emit_status("正在准备内置 Python 运行时", target_python=str(target_python))
    report = runtime_installer.install_embedded_runtime(
        target_python,
        lambda message: emit("APP_DIAGNOSTIC", {"level": "info", "message": message}),
        getattr(args, "accelerator_mode", "auto"),
    )
    emit("RESULT", {"kind": "bootstrap-runtime", **report})
    return 0


def run_bootstrap_runtime_and_configure(args: argparse.Namespace) -> int:
    target_python = Path(args.target_python).expanduser().resolve(strict=False)
    emit_status("正在准备内置 Python 运行时", target_python=str(target_python))
    report = runtime_installer.install_embedded_runtime(
        target_python,
        lambda message: emit("APP_DIAGNOSTIC", {"level": "info", "message": message}),
        args.accelerator_mode,
    )
    emit("RESULT", {"kind": "bootstrap-runtime", **report})

    command = [
        str(target_python),
        "-u",
        str(Path(__file__).resolve()),
        "configure-env",
        "--accelerator-mode",
        args.accelerator_mode,
    ]
    emit_status("内置 Python 已就绪，开始配置运行环境", command=subprocess.list2cmdline(command))
    include_extra_index = args.accelerator_mode != "legacy-cuda"
    return_code = runtime_installer.stream_command(command, include_extra_index=include_extra_index)
    if return_code != 0:
        raise RuntimeError(f"内置 runtime 已创建，但环境配置失败，退出码为 {return_code}。")
    return 0


def run_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    config_payload = load_json(args.config_json)
    validated_config = yolo_runner.validate_train_config_payload(args.task, config_payload)
    train_kwargs = yolo_runner.clean_kwargs({"data": args.data, **validated_config})
    dataset_input = Path(args.data).expanduser().resolve()
    dataset_image_count, validation_image_count = yolo_runner.validate_training_dataset_input(dataset_input, args.task)

    emit_status("开始训练", task=args.task, model=args.model, data=args.data)
    emit_status("训练参数已加载", config=args.config_json)
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"数据集图片数：{dataset_image_count}，验证集图片数：{validation_image_count}"})

    weights_dir = yolo_runner.configure_weights_dir(args.weights_dir)
    if weights_dir is not None:
        emit_status("已配置权重缓存目录", weights_dir=str(weights_dir))
    yolo_runner.prepare_amp_check_weight(
        weights_dir,
        str(train_kwargs.get("device") or ""),
        lambda message, level="info": emit("APP_DIAGNOSTIC", {"level": level, "message": message}),
    )

    model = YOLO(args.model, task=args.task or None)
    result = model.train(**train_kwargs)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(
        getattr(result, "save_dir", "")
        or getattr(trainer, "save_dir", "")
        or Path.cwd()
    ).expanduser().resolve()
    summary = yolo_runner.summarize_training_results(save_dir)
    emit("RESULT", {"kind": "train", **summary})
    return 0


def run_val(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    if (args.task or "").strip().lower() not in ALL_TASKS:
        raise ValueError(f"不支持的验证任务：{args.task}")

    config_payload = load_json(args.config_json)
    validated_config = yolo_runner.clean_kwargs(config_payload)
    val_kwargs = yolo_runner.clean_kwargs({"data": args.data, **validated_config})
    dataset_input = Path(args.data).expanduser().resolve()
    dataset_image_count, validation_image_count = yolo_runner.validate_training_dataset_input(dataset_input, args.task)

    emit_status("开始验证", task=args.task, weights=args.weights, data=args.data)
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"数据集图片数：{dataset_image_count}，验证集图片数：{validation_image_count}"})
    model = YOLO(args.weights, task=args.task or None)
    result = model.val(**val_kwargs)
    trainer = getattr(model, "validator", None)
    save_dir = Path(
        getattr(result, "save_dir", "")
        or getattr(trainer, "save_dir", "")
        or Path.cwd()
    ).expanduser().resolve()
    summary = jsonable(getattr(result, "results_dict", {}) or getattr(result, "metrics", {}) or {})
    emit("RESULT", {"kind": "val", "save_dir": str(save_dir), "summary": summary})
    return 0


def run_predict(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    config_payload = load_json(args.config_json)
    validated_config = yolo_runner.clean_kwargs(config_payload)
    predict_kwargs = yolo_runner.clean_kwargs({"source": args.source, **validated_config})

    emit_status("开始预测", task=args.task, weights=args.weights, source=args.source)
    model = YOLO(args.weights, task=args.task or None)
    results = model.predict(**predict_kwargs)
    predictor = getattr(model, "predictor", None)
    save_dir = Path(getattr(predictor, "save_dir", "") or Path.cwd()).expanduser().resolve()
    count = len(results) if hasattr(results, "__len__") else None
    emit("RESULT", {"kind": "predict", "save_dir": str(save_dir), "count": count})
    return 0


def run_track(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    if (args.task or "").strip().lower() not in TRACK_TASKS:
        raise ValueError(f"不支持的跟踪任务：{args.task}")

    config_payload = load_json(args.config_json)
    validated_config = yolo_runner.clean_kwargs(config_payload)
    track_kwargs = yolo_runner.clean_kwargs({"source": args.source, **validated_config})

    emit_status("开始跟踪", task=args.task, weights=args.weights, source=args.source)
    model = YOLO(args.weights, task=args.task or None)
    results = model.track(**track_kwargs)
    predictor = getattr(model, "predictor", None)
    save_dir = Path(getattr(predictor, "save_dir", "") or Path.cwd()).expanduser().resolve()
    count = len(results) if hasattr(results, "__len__") else None
    emit("RESULT", {"kind": "track", "save_dir": str(save_dir), "count": count})
    return 0


def run_export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    config_payload = load_json(args.config_json)
    namespace = build_export_namespace(args.task, args.weights, config_payload)
    export_format = str(namespace.format).strip() or "onnx"
    supported_args = yolo_runner.contract_supported_export_arguments(export_format)

    emit_status("开始导出", task=args.task, weights=args.weights, format=export_format)

    warnings = yolo_runner.collect_unsupported_export_warnings(namespace, supported_args)
    export_kwargs = yolo_runner.build_export_kwargs(namespace, supported_args)
    model = YOLO(args.weights, task=args.task or None)
    exported_path = Path(model.export(**export_kwargs)).expanduser().resolve()
    final_output = yolo_runner.finalize_export_output(exported_path, namespace.output_dir)

    emit(
        "RESULT",
        {
            "kind": "export",
            "output": str(final_output),
            "format": export_format,
            "warnings": warnings,
        },
    )
    return 0


def run_auto_label_detect(args: argparse.Namespace) -> int:
    from PIL import Image
    from ultralytics import YOLO
    import yolo_runner

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"找不到图片目录：{image_dir}")

    image_paths = list_annotation_images(image_dir)
    if not image_paths:
        raise ValueError(f"{image_dir} 里没有可自动标注的图片。")

    target_image: Path | None = None
    if args.image:
        target_image = Path(args.image).expanduser().resolve()
        if target_image not in image_paths:
            raise ValueError(f"当前图片不在所选目录里：{target_image}")
        image_paths = [target_image]

    config_payload = load_json(args.config_json)
    project_class_names_raw = config_payload.pop("_project_class_names", [])
    if isinstance(project_class_names_raw, list):
        project_class_names = [str(item).strip() for item in project_class_names_raw if str(item).strip()]
    else:
        project_class_names = []
    predict_kwargs = yolo_runner.clean_kwargs(config_payload)
    predict_kwargs.update({"save": False, "verbose": False})

    emit_status("开始自动标注", image_dir=str(image_dir), image_count=len(image_paths), model=args.weights)
    model = YOLO(args.weights, task="detect")

    updated_files: list[str] = []
    total_box_count = 0
    max_detected_class_id = -1
    for index, image_path in enumerate(image_paths, start=1):
        emit_status(f"正在自动标注第 {index}/{len(image_paths)} 张图片", image=str(image_path))
        results = model.predict(source=str(image_path), **predict_kwargs)
        result = results[0] if results else None
        boxes: list[AnnotationBox] = []
        if result is not None and getattr(result, "boxes", None) is not None:
            for xyxy, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
                boxes.append(
                    AnnotationBox(
                        class_id=int(class_id),
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    )
                )

        image_width = 0
        image_height = 0
        orig_shape = getattr(result, "orig_shape", None) if result is not None else None
        if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
            image_height = int(orig_shape[0] or 0)
            image_width = int(orig_shape[1] or 0)
        if image_width <= 0 or image_height <= 0:
            with Image.open(image_path) as image:
                image_width = int(image.width)
                image_height = int(image.height)

        save_yolo_boxes(image_path.with_suffix(".txt"), boxes, image_width, image_height)
        updated_files.append(str(image_path))
        total_box_count += len(boxes)
        if boxes:
            max_detected_class_id = max(max_detected_class_id, max(box.class_id for box in boxes))

    model_names = getattr(model, "names", {}) or {}
    if isinstance(model_names, dict):
        ordered_names = [str(model_names[key]).strip() for key in sorted(model_names)]
    elif isinstance(model_names, list):
        ordered_names = [str(item).strip() for item in model_names]
    else:
        ordered_names = []
    detected_class_count = max_detected_class_id + 1 if max_detected_class_id >= 0 else 0
    target_class_count = max(len(project_class_names), len(ordered_names), detected_class_count, 1)
    generic_project_names = bool(project_class_names) and all(name == f"class{index}" for index, name in enumerate(project_class_names))

    if project_class_names and not generic_project_names:
        final_class_names = list(project_class_names)
        while len(final_class_names) < target_class_count:
            index = len(final_class_names)
            fallback_name = ordered_names[index] if index < len(ordered_names) and ordered_names[index] else f"class{index}"
            final_class_names.append(fallback_name)
    else:
        final_class_names = list(ordered_names or project_class_names or ["class0"])

    final_class_names = ensure_class_names(final_class_names, target_class_count - 1)
    save_class_names(image_dir, final_class_names)

    emit(
        "RESULT",
        {
            "kind": "auto-label-detect",
            "image_dir": str(image_dir),
            "target_image": "" if target_image is None else str(target_image),
            "image_count": len(updated_files),
            "box_count": total_box_count,
            "updated_files": updated_files,
            "model_class_names": final_class_names,
        },
    )
    return 0


def run_prepare_dataset(args: argparse.Namespace) -> int:
    from prepare_detection_dataset import run_from_args  # noqa: PLC0415

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()] if args.class_names else None
    prepare_args = SimpleNamespace(
        input=args.input,
        output=args.output,
        format=args.format,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_mode=args.copy_mode,
        class_names=class_names,
        class_names_file=args.class_names_file or None,
        force=args.force,
        strict=args.strict,
    )

    emit_status("开始整理数据集", input=args.input, output=args.output, format=args.format)
    previous_protocol = os.environ.get("YOLO_DESKTOP_PROTOCOL")
    os.environ["YOLO_DESKTOP_PROTOCOL"] = "1"
    try:
        return run_from_args(prepare_args)
    except Exception as exc:
        emit("ERROR", {"message": str(exc)})
        return 1
    finally:
        if previous_protocol is None:
            os.environ.pop("YOLO_DESKTOP_PROTOCOL", None)
        else:
            os.environ["YOLO_DESKTOP_PROTOCOL"] = previous_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 YOLO 工具后端")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="检查当前 Python 环境")
    subparsers.add_parser("runtime-preflight", help=argparse.SUPPRESS)
    configure = subparsers.add_parser("configure-env", help="自动安装推荐运行环境")
    configure.add_argument("--dry-run", action="store_true")
    configure.add_argument(
        "--accelerator-mode",
        default="auto",
        choices=("auto", "stable-cpu", "legacy-cuda"),
        help="环境安装策略：auto 自动推荐，stable-cpu 稳定 CPU，legacy-cuda 老显卡 CUDA 兼容模式",
    )
    bootstrap = subparsers.add_parser("bootstrap-runtime", help="在线创建内置 Python runtime")
    bootstrap.add_argument("--target-python", required=True)
    bootstrap.add_argument(
        "--accelerator-mode",
        default="auto",
        choices=("auto", "stable-cpu", "legacy-cuda"),
        help="创建内置 runtime 时使用的环境策略；legacy-cuda 会选择更适合 cu118 的内置 Python。",
    )
    bootstrap_and_configure = subparsers.add_parser(
        "bootstrap-runtime-and-configure",
        help="在线创建内置 Python runtime 并继续配置依赖环境",
    )
    bootstrap_and_configure.add_argument("--target-python", required=True)
    bootstrap_and_configure.add_argument(
        "--accelerator-mode",
        default="auto",
        choices=("auto", "stable-cpu", "legacy-cuda"),
    )

    train = subparsers.add_parser("train", help="训练模型")
    train.add_argument("--task", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--config-json", required=True)
    train.add_argument("--weights-dir", default="")

    val = subparsers.add_parser("val", help="验证模型")
    val.add_argument("--task", default="detect")
    val.add_argument("--weights", required=True)
    val.add_argument("--data", required=True)
    val.add_argument("--config-json", required=True)

    predict = subparsers.add_parser("predict", help="预测结果")
    predict.add_argument("--task", default="detect")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--config-json", required=True)

    track = subparsers.add_parser("track", help="目标跟踪")
    track.add_argument("--task", default="detect")
    track.add_argument("--weights", required=True)
    track.add_argument("--source", required=True)
    track.add_argument("--config-json", required=True)

    export = subparsers.add_parser("export", help="导出模型")
    export.add_argument("--task", default="")
    export.add_argument("--weights", required=True)
    export.add_argument("--config-json", required=True)
    auto_label = subparsers.add_parser("auto-label-detect", help="对检测图片目录执行自动标注")
    auto_label.add_argument("--weights", required=True)
    auto_label.add_argument("--image-dir", required=True)
    auto_label.add_argument("--config-json", required=True)
    auto_label.add_argument("--image", default="")

    prep = subparsers.add_parser("prepare-dataset", help="整理检测数据集")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--format", default="auto", choices=("auto", "yolo-flat", "labelme-json", "coco-json"))
    prep.add_argument("--val-ratio", type=float, default=0.2)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--copy-mode", default="copy", choices=("copy", "hardlink"))
    prep.add_argument("--class-names", default="")
    prep.add_argument("--class-names-file", default="")
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return run_check(args)
        if args.command == "runtime-preflight":
            return run_runtime_preflight_command(args)
        if args.command == "configure-env":
            return run_configure_env(args)
        if args.command == "bootstrap-runtime":
            return run_bootstrap_runtime(args)
        if args.command == "bootstrap-runtime-and-configure":
            return run_bootstrap_runtime_and_configure(args)
        if args.command == "train":
            return run_train(args)
        if args.command == "val":
            return run_val(args)
        if args.command == "predict":
            return run_predict(args)
        if args.command == "track":
            return run_track(args)
        if args.command == "export":
            return run_export(args)
        if args.command == "auto-label-detect":
            return run_auto_label_detect(args)
        if args.command == "prepare-dataset":
            return run_prepare_dataset(args)
        parser.error(f"未知命令: {args.command}")
        return 2
    except KeyboardInterrupt:
        emit("ERROR", {"message": "任务已取消。"})
        return 130
    except Exception as exc:
        emit("ERROR", {"message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
