from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from export_capabilities import (
    export_format_choices,
    export_format_labels,
    supported_export_arguments as contract_supported_export_arguments,
)

AMP_CHECK_MODEL_NAME = "yolo26n.pt"
EXPORT_FORMAT_CHOICES = export_format_choices()
EXPORT_FORMAT_LABELS = export_format_labels()
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DETECTION_TASKS = {"detect", "segment", "pose", "obb"}
TRAIN_TASKS = {"detect", "segment", "classify", "pose", "obb"}
TRAIN_CAPABILITIES_PATH = Path(__file__).resolve().with_name("train_capabilities.json")


def clean_kwargs(values: dict) -> dict:
    return {key: value for key, value in values.items() if value not in (None, "")}


def desktop_protocol_enabled() -> bool:
    return str(os.environ.get("YOLO_DESKTOP_PROTOCOL") or "").strip() == "1"


def emit_json(tag: str, payload: dict) -> None:
    if desktop_protocol_enabled():
        print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def emit_app_diagnostic(message: str, level: str = "info") -> None:
    if desktop_protocol_enabled():
        emit_json("APP_DIAGNOSTIC", {"level": level, "message": message})

def _iter_dataset_entries(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _resolve_dataset_root(yaml_path: Path, payload: dict) -> Path:
    dataset_root = payload.get("path")
    if dataset_root:
        root_path = Path(str(dataset_root)).expanduser()
        if not root_path.is_absolute():
            root_path = (yaml_path.parent / root_path).resolve()
        return root_path
    return yaml_path.parent.resolve()


def _expand_dataset_entry(path: Path) -> list[Path]:
    if path.is_dir():
        return [
            candidate.resolve()
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        ]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path.resolve()]
    if path.is_file() and path.suffix.lower() == ".txt":
        resolved: list[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            nested = Path(entry).expanduser()
            if not nested.is_absolute():
                nested = (path.parent / nested).resolve()
            resolved.extend(_expand_dataset_entry(nested))
        return resolved
    return []


def _resolve_dataset_entries(entries: list[str], dataset_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for entry in entries:
        if any(char in entry for char in "*?["):
            pattern = entry if Path(entry).is_absolute() else str(dataset_root / entry)
            for match in glob.glob(pattern, recursive=True):
                resolved.extend(_expand_dataset_entry(Path(match).expanduser().resolve()))
            continue

        entry_path = Path(entry).expanduser()
        if not entry_path.is_absolute():
            entry_path = (dataset_root / entry_path).resolve()
        resolved.extend(_expand_dataset_entry(entry_path))
    return sorted({path.resolve() for path in resolved})


def _validate_classification_dataset(data_input: Path) -> tuple[int, int]:
    if not data_input.exists() or not data_input.is_dir():
        raise ValueError("分类任务只能使用 Ultralytics 官方分类数据集根目录。")

    train_dir = data_input / "train"
    if not train_dir.exists() or not train_dir.is_dir():
        raise ValueError("分类数据集根目录必须包含 train 文件夹。")

    image_count = sum(1 for candidate in data_input.rglob("*") if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES)
    return image_count, 0


def _validate_detection_dataset(data_input: Path, task: str) -> tuple[int, int]:
    if not data_input.exists() or not data_input.is_file() or data_input.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("检测、分割、姿态和 OBB 任务只能使用 Ultralytics 官方 dataset.yaml 文件。")

    payload = yaml.safe_load(data_input.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("dataset.yaml 内容无效，请检查文件格式。")

    train_entries = _iter_dataset_entries(payload.get("train"))
    if not train_entries:
        raise ValueError("dataset.yaml 缺少 train 配置。")

    val_entries = _iter_dataset_entries(payload.get("val"))
    if task in DETECTION_TASKS and not val_entries:
        raise ValueError("检测类数据集必须包含独立的 val 配置。")

    dataset_root = _resolve_dataset_root(data_input, payload)
    resolved_train = _resolve_dataset_entries(train_entries, dataset_root)
    resolved_val = _resolve_dataset_entries(val_entries, dataset_root)

    if not resolved_train:
        raise ValueError("dataset.yaml 中的 train 路径不存在或不包含图片。")
    if task in DETECTION_TASKS and not resolved_val:
        raise ValueError("dataset.yaml 中的 val 路径不存在或不包含图片。")
    if task in DETECTION_TASKS and resolved_train == resolved_val:
        raise ValueError("检测类数据集必须使用独立的 train 与 val 路径，不能让 val 指向 train。")

    resolved_test = _resolve_dataset_entries(_iter_dataset_entries(payload.get("test")), dataset_root)
    image_count = len({*resolved_train, *resolved_val, *resolved_test})
    return image_count, len(resolved_val)


def validate_training_dataset_input(data_input: Path, task: str) -> tuple[int, int]:
    normalized_task = (task or "").strip().lower()
    if normalized_task == "classify":
        return _validate_classification_dataset(data_input)
    if normalized_task in DETECTION_TASKS:
        return _validate_detection_dataset(data_input, normalized_task)
    raise ValueError(f"不支持的训练任务：{task}")


@lru_cache(maxsize=1)
def load_train_capabilities_contract() -> dict:
    if not TRAIN_CAPABILITIES_PATH.exists():
        raise FileNotFoundError(f"未找到训练参数契约文件：{TRAIN_CAPABILITIES_PATH}")
    payload = json.loads(TRAIN_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("训练参数契约文件格式无效。")
    return payload


@lru_cache(maxsize=1)
def load_train_capability_map() -> dict[str, dict]:
    contract = load_train_capabilities_contract()
    groups = contract.get("groups")
    if not isinstance(groups, list):
        raise ValueError("训练参数契约缺少 groups。")

    capabilities: dict[str, dict] = {}
    all_tasks = contract.get("tasks") or sorted(TRAIN_TASKS)
    for group in groups:
        if not isinstance(group, dict):
            continue
        for parameter in group.get("parameters", []):
            if not isinstance(parameter, dict):
                continue
            key = str(parameter.get("key") or "").strip()
            if not key:
                continue
            normalized = dict(parameter)
            normalized["tasks"] = list(parameter.get("tasks") or all_tasks)
            capabilities[key] = normalized
    if not capabilities:
        raise ValueError("训练参数契约没有可用的参数定义。")
    return capabilities


def _load_train_config_json(config_path: str) -> dict:
    resolved = Path(config_path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("训练配置 JSON 必须是对象。")
    return payload


def load_train_config_payload(args: argparse.Namespace) -> dict:
    return _load_train_config_json(args.train_config_json)


def _train_value_matches_type(value, type_name: str) -> bool:
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "list":
        return isinstance(value, list)
    return False


def _train_option_matches(value, option) -> bool:
    return value == option


def validate_train_config_payload(task: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("训练配置必须是对象。")

    normalized_task = (task or "").strip().lower()
    if normalized_task not in TRAIN_TASKS:
        raise ValueError(f"不支持的训练任务：{task}")

    capabilities = load_train_capability_map()
    validated: dict[str, object] = {}
    for key, value in payload.items():
        capability = capabilities.get(key)
        if capability is None:
            raise ValueError(f"训练配置包含未知参数：{key}")

        tasks = capability.get("tasks") or sorted(TRAIN_TASKS)
        if normalized_task not in tasks:
            raise ValueError(f"训练参数 {key} 不适用于任务 {normalized_task}")

        if value is None:
            if capability.get("optional"):
                validated[key] = None
                continue
            raise ValueError(f"训练参数 {key} 不能为空")

        if isinstance(value, str) and not value.strip() and capability.get("optional"):
            validated[key] = None
            continue

        allowed_types = capability.get("types") or []
        if not any(_train_value_matches_type(value, type_name) for type_name in allowed_types):
            raise ValueError(f"训练参数 {key} 类型无效，允许类型：{', '.join(allowed_types)}")

        options = capability.get("options") or []
        if options and not any(_train_option_matches(value, option) for option in options):
            raise ValueError(f"训练参数 {key} 的值不在允许范围内")

        validated[key] = value

    return validated


def build_train_kwargs(args: argparse.Namespace, config_payload: dict) -> dict:
    kwargs = {"data": args.data}
    for key, value in validate_train_config_payload(args.task, config_payload).items():
        kwargs[key] = value
    return clean_kwargs(kwargs)

def configure_weights_dir(weights_dir: str | None) -> Path | None:
    if not weights_dir:
        return None

    from ultralytics.utils import SETTINGS

    resolved = Path(weights_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    SETTINGS["weights_dir"] = str(resolved)
    return resolved


def should_prepare_amp_check_weight(device: str | None) -> bool:
    normalized = (device or "").strip().lower()
    if normalized in {"cpu", "mps", "-1"}:
        return False

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return True


def prepare_amp_check_weight(weights_dir: Path | None, device: str | None, logger) -> None:
    if weights_dir is None:
        return

    target = weights_dir / AMP_CHECK_MODEL_NAME
    if target.exists() or not should_prepare_amp_check_weight(device):
        return

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        logger(f"[信息] 正在准备 AMP 自检模型缓存：{target}")
        resolved = Path(attempt_download_asset(str(target))).expanduser().resolve()
        if resolved.exists():
            logger(f"[信息] AMP 自检模型已就绪：{resolved}")
    except ConnectionError:
        logger("[警告] 当前处于离线状态，无法准备 AMP 自检模型缓存；Ultralytics 可能会跳过 AMP 检查。")
    except Exception as exc:
        logger(f"[警告] 无法准备 AMP 自检模型缓存：{exc}")


def export_payload(status: str, export_format: str, simplified: bool, warnings: list[str], output: Path | None) -> dict:
    return {
        "status": status,
        "format": export_format,
        "format_label": EXPORT_FORMAT_LABELS.get(export_format, export_format),
        "simplified": simplified,
        "warnings": warnings,
        "output": "" if output is None else str(output),
    }


def finalize_export_output(output: Path, output_dir: str | None) -> Path:
    final_output = output.resolve()
    if not output_dir:
        return final_output

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    requested_output = target_dir / output.name
    if requested_output != final_output:
        if final_output.is_dir():
            if requested_output.exists():
                if requested_output.is_dir():
                    shutil.rmtree(requested_output)
                else:
                    requested_output.unlink()
            shutil.copytree(final_output, requested_output)
        else:
            shutil.copy2(final_output, requested_output)
        final_output = requested_output
    return final_output


def build_export_kwargs(args: argparse.Namespace, supported_args: set[str]) -> dict:
    return clean_kwargs(
        {
            "format": args.format,
            "imgsz": args.imgsz,
            "opset": (args.opset or None) if "opset" in supported_args else None,
            "workspace": (args.workspace or None) if "workspace" in supported_args else None,
            "batch": args.batch if "batch" in supported_args else None,
            "data": (args.data or None) if "data" in supported_args else None,
            "fraction": args.fraction if "fraction" in supported_args else None,
            "device": args.device or None,
            "dynamic": args.dynamic if "dynamic" in supported_args else None,
            "simplify": args.simplify if "simplify" in supported_args else None,
            "half": args.half if "half" in supported_args else None,
            "int8": args.int8 if "int8" in supported_args else None,
            "nms": args.nms if "nms" in supported_args else None,
            "optimize": args.optimize if "optimize" in supported_args else None,
            "keras": args.keras if "keras" in supported_args else None,
            "name": (args.name or None) if "name" in supported_args else None,
        }
    )


def collect_unsupported_export_warnings(args: argparse.Namespace, supported_args: set[str]) -> list[str]:
    label = EXPORT_FORMAT_LABELS.get(args.format, args.format)
    warnings: list[str] = []
    if args.workspace and "workspace" not in supported_args:
        warnings.append(f"{label} 不支持 workspace 参数，已忽略。")
    if args.batch != 1 and "batch" not in supported_args:
        warnings.append(f"{label} 不支持批次大小参数，已忽略。")
    if args.data and "data" not in supported_args:
        warnings.append(f"{label} 不支持 data 参数，已忽略。")
    if abs(args.fraction - 1.0) > 1e-9 and "fraction" not in supported_args:
        warnings.append(f"{label} 不支持采样比例参数，已忽略。")
    if args.name and "name" not in supported_args:
        warnings.append(f"{label} 不支持目标名称参数，已忽略。")
    if args.opset and "opset" not in supported_args:
        warnings.append(f"{label} 不支持 Opset 参数，已忽略。")
    if args.dynamic and "dynamic" not in supported_args:
        warnings.append(f"{label} 不支持动态输入，已忽略。")
    if args.half and "half" not in supported_args:
        warnings.append(f"{label} 不支持半精度导出，已忽略。")
    if args.simplify and "simplify" not in supported_args:
        warnings.append(f"{label} 不支持模型简化，已忽略。")
    if args.int8 and "int8" not in supported_args:
        warnings.append(f"{label} 不支持 INT8 量化，已忽略。")
    if args.nms and "nms" not in supported_args:
        warnings.append(f"{label} 不支持内置 NMS，已忽略。")
    if args.optimize and "optimize" not in supported_args:
        warnings.append(f"{label} 不支持图结构优化，已忽略。")
    if args.keras and "keras" not in supported_args:
        warnings.append(f"{label} 不支持 Keras 导出，已忽略。")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dear ImGui YOLO 桌面端后端执行器。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练 YOLO 模型。")
    train.add_argument("--task", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--train-config-json", required=True)
    train.add_argument("--weights-dir", default="")

    export = subparsers.add_parser("export", help="将 YOLO 权重导出为官方 Ultralytics 格式。")
    export.add_argument("--task", default="")
    export.add_argument("--format", choices=EXPORT_FORMAT_CHOICES, default="onnx")
    export.add_argument("--weights", required=True)
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--opset", type=int, default=0)
    export.add_argument("--workspace", type=float, default=0.0)
    export.add_argument("--batch", type=int, default=1)
    export.add_argument("--data", default="")
    export.add_argument("--fraction", type=float, default=1.0)
    export.add_argument("--device", default="")
    export.add_argument("--output-dir", default="")
    export.add_argument("--name", default="")
    export.add_argument("--dynamic", action="store_true")
    export.add_argument("--simplify", action="store_true")
    export.add_argument("--half", action="store_true")
    export.add_argument("--int8", action="store_true")
    export.add_argument("--nms", action="store_true")
    export.add_argument("--optimize", action="store_true")
    export.add_argument("--keras", action="store_true")
    return parser


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def summarize_training_results(save_dir: Path) -> dict:
    csv_path = save_dir / "results.csv"
    best_weights = save_dir / "weights" / "best.pt"
    summary = {
        "save_dir": str(save_dir),
        "results_csv": str(csv_path),
        "weights_best": str(best_weights),
        "weights_last": str(save_dir / "weights" / "last.pt"),
        "best_epoch": None,
    }

    if not csv_path.exists():
        return summary

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [{key.strip(): value for key, value in row.items()} for row in reader]

    if not rows:
        return summary

    keys = list(rows[-1].keys())
    epoch_key = "epoch" if "epoch" in keys else None
    primary_key = next((key for key in keys if "metrics/mAP50-95" in key), None)

    best_row = rows[-1]
    if primary_key:
        valid_rows = [row for row in rows if parse_float(row.get(primary_key)) is not None]
        if valid_rows:
            best_row = max(valid_rows, key=lambda row: parse_float(row.get(primary_key)) or float("-inf"))

    if epoch_key:
        epoch_value = best_row.get(epoch_key)
        summary["best_epoch"] = None if parse_float(epoch_value) is None else int(parse_float(epoch_value))

    return summary


def run_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    emit_app_diagnostic(f"Python 路径：{sys.executable}")
    emit_app_diagnostic(f"当前目录：{Path.cwd()}")
    emit_app_diagnostic(f"训练任务：{args.task}")
    emit_app_diagnostic(f"正在加载模型：{args.model}")

    dataset_input = Path(args.data).expanduser().resolve()
    dataset_image_count, validation_image_count = validate_training_dataset_input(dataset_input, args.task)
    if dataset_image_count < 2:
        raise ValueError("至少需要 2 张图片才能开始训练，请选择包含 2 张及以上图片的官方格式数据集。")
    emit_app_diagnostic(f"数据集图片数：{dataset_image_count}")
    if args.task in DETECTION_TASKS:
        emit_app_diagnostic(f"验证集图片数：{validation_image_count}")

    weights_dir = configure_weights_dir(args.weights_dir)
    if weights_dir is not None:
        emit_app_diagnostic(f"权重缓存目录：{weights_dir}")
    config_payload = load_train_config_payload(args)
    kwargs = build_train_kwargs(args, config_payload)
    emit_app_diagnostic(f"训练参数契约：{TRAIN_CAPABILITIES_PATH}")
    emit_app_diagnostic(f"训练配置文件：{Path(args.train_config_json).expanduser().resolve()}")
    prepare_amp_check_weight(weights_dir, str(kwargs.get("device") or ""), emit_app_diagnostic)

    model = YOLO(args.model, task=args.task or None)
    emit_app_diagnostic(f"训练参数：{kwargs}")

    result = model.train(**kwargs)
    save_dir = Path(getattr(result, "save_dir", Path.cwd()))
    summary = summarize_training_results(save_dir)
    emit_json("TRAIN_SUMMARY", summary)
    emit_app_diagnostic(f"训练已完成：{save_dir}")
    return 0


def run_export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    emit_app_diagnostic(f"Python 路径：{sys.executable}")
    emit_app_diagnostic(f"当前目录：{Path.cwd()}")
    emit_app_diagnostic(f"正在加载权重：{args.weights}")
    emit_app_diagnostic(f"导出格式：{args.format}")
    final_output: Path | None = None
    warnings: list[str] = []
    simplified = False

    try:
        model = YOLO(args.weights, task=args.task or None)
        supported_args = contract_supported_export_arguments(args.format)
        kwargs = build_export_kwargs(args, supported_args)
        warnings.extend(collect_unsupported_export_warnings(args, supported_args))
        emit_app_diagnostic(f"导出参数：{kwargs}")
        for warning in warnings:
            emit_app_diagnostic(warning, "warning")

        exported_path = Path(model.export(**kwargs)).expanduser().resolve()
        final_output = finalize_export_output(exported_path, args.output_dir)
        simplified = bool(kwargs.get("simplify"))

        status = "success_with_warning" if warnings else "success"
        emit_json("EXPORT_JSON", export_payload(status, args.format, simplified, warnings, final_output))
        emit_app_diagnostic(f"导出已完成：{final_output}", "warning" if warnings else "info")
        return 0
    except Exception as exc:
        warnings.append(str(exc))
        emit_json("EXPORT_JSON", export_payload("failed", args.format, simplified, warnings, final_output))
        emit_app_diagnostic(str(exc), "error")
        raise


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "train":
            return run_train(args)
        if args.command == "export":
            return run_export(args)
        parser.error(f"未知命令：{args.command}")
        return 2
    except KeyboardInterrupt:
        emit_app_diagnostic("用户中断了当前任务。", "warning")
        return 130
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
