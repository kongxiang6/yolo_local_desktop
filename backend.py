from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor_backend"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import runtime_preflight
import runtime_installer

ALL_TASKS = ("detect", "segment", "classify", "pose", "obb")
TRACK_TASKS = ("detect", "segment", "pose", "obb")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def emit(tag: str, payload: dict) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def load_json(path: str) -> dict:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件必须是 JSON 对象。")
    return payload


def emit_status(message: str, **extra: object) -> None:
    payload = {"message": message}
    payload.update(extra)
    emit("STATUS", payload)


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
    report = runtime_preflight.run_runtime_preflight()
    report["python"] = sys.executable
    emit("RESULT", {"kind": "check", **report})
    return 0


def run_configure_env(args: argparse.Namespace) -> int:
    plan = runtime_installer.build_install_plan()
    emit_status(
        "已生成环境安装方案",
        python=plan["python"],
        accelerator=plan["accelerator"],
        torch_index=plan["torch_index"],
    )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Python: {plan['python']} ({plan['python_version']})"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Platform: {plan['platform']}"})
    gpu = plan.get("gpu") or {}
    emit(
        "APP_DIAGNOSTIC",
        {
            "level": "info",
            "message": (
                f"NVIDIA: {'检测到' if gpu.get('available') else '未检测到'}"
                f"  GPU={gpu.get('gpu_name') or '无'}"
                f"  CUDA={gpu.get('cuda_version') or '无'}"
            ),
        },
    )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Torch 下载源: {plan['torch_index']}"})
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Pip 引导策略: {plan['pip_bootstrap']}"})
    for note in plan.get("notes", []):
        emit("APP_DIAGNOSTIC", {"level": "info", "message": str(note)})

    if args.dry_run:
        emit("RESULT", {"kind": "configure-env", **{k: v for k, v in plan.items() if k != "commands"}})
        return 0

    emit_status("正在准备 pip 环境")
    bootstrap_info = runtime_installer.bootstrap_pip(
        lambda message: emit("APP_DIAGNOSTIC", {"level": "info", "message": message})
    )
    emit("APP_DIAGNOSTIC", {"level": "info", "message": f"Pip 准备完成: {bootstrap_info['message']}"})

    commands = list(plan.get("commands") or [])
    for index, command in enumerate(commands, start=1):
        emit_status(f"正在执行第 {index}/{len(commands)} 步", command=subprocess.list2cmdline(command))
        return_code = runtime_installer.stream_command(command)
        if return_code != 0:
            raise RuntimeError(f"环境配置失败，第 {index} 步退出码为 {return_code}。")

    report = runtime_preflight.run_runtime_preflight()
    emit(
        "RESULT",
        {
            "kind": "configure-env",
            **{k: v for k, v in plan.items() if k != "commands"},
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
    )
    emit("RESULT", {"kind": "bootstrap-runtime", **report})
    return 0


def run_bootstrap_runtime_and_configure(args: argparse.Namespace) -> int:
    target_python = Path(args.target_python).expanduser().resolve(strict=False)
    emit_status("正在准备内置 Python 运行时", target_python=str(target_python))
    report = runtime_installer.install_embedded_runtime(
        target_python,
        lambda message: emit("APP_DIAGNOSTIC", {"level": "info", "message": message}),
    )
    emit("RESULT", {"kind": "bootstrap-runtime", **report})

    command = [str(target_python), "-u", str(Path(__file__).resolve()), "configure-env"]
    emit_status("内置 Python 已就绪，开始配置运行环境", command=subprocess.list2cmdline(command))
    return_code = runtime_installer.stream_command(command)
    if return_code != 0:
        raise RuntimeError(f"内置 runtime 已创建，但环境配置失败，退出码为 {return_code}。")
    return 0


def run_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import yolo_runner

    config_payload = load_json(args.config_json)
    validated_config = yolo_runner.validate_train_config_payload(args.task, config_payload)
    train_kwargs = yolo_runner.clean_kwargs({"data": args.data, **validated_config})

    emit_status("开始训练", task=args.task, model=args.model, data=args.data)
    emit_status("训练参数已加载", config=args.config_json)

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

    emit_status("开始验证", task=args.task, weights=args.weights, data=args.data)
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


def run_prepare_dataset(args: argparse.Namespace) -> int:
    prepare_script = VENDOR_DIR / "prepare_detection_dataset.py"
    command = [
        sys.executable,
        str(prepare_script),
        "--input",
        args.input,
        "--output",
        args.output,
        "--format",
        args.format,
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--copy-mode",
        args.copy_mode,
    ]
    if args.class_names:
        command.append("--class-names")
        command.extend([item.strip() for item in args.class_names.split(",") if item.strip()])
    if args.class_names_file:
        command.extend(["--class-names-file", args.class_names_file])
    if args.force:
        command.append("--force")
    if args.strict:
        command.append("--strict")

    emit_status("开始整理数据集", input=args.input, output=args.output, format=args.format)
    env = os.environ.copy()
    env["YOLO_DESKTOP_PROTOCOL"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=NO_WINDOW_FLAGS,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 YOLO 工具后端")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="检查当前 Python 环境")
    configure = subparsers.add_parser("configure-env", help="自动安装推荐运行环境")
    configure.add_argument("--dry-run", action="store_true")
    bootstrap = subparsers.add_parser("bootstrap-runtime", help="在线创建内置 Python runtime")
    bootstrap.add_argument("--target-python", required=True)
    bootstrap_and_configure = subparsers.add_parser(
        "bootstrap-runtime-and-configure",
        help="在线创建内置 Python runtime 并继续配置依赖环境",
    )
    bootstrap_and_configure.add_argument("--target-python", required=True)

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
