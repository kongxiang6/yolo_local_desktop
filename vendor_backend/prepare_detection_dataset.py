from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TEXT_CLASS_NAME_FILES = ("classes.names", "classes.txt")
DATASET_PREP_TAG = "DATASET_PREP_JSON"


@dataclass
class PreparedSample:
    image_path: Path
    relative_image_path: Path
    label_lines: list[str]


@dataclass
class PreparationReport:
    dataset_format: str
    class_names: list[str]
    samples: list[PreparedSample]
    warnings: list[str]
    label_count: int
    used_class_ids: set[int]


def desktop_protocol_enabled() -> bool:
    return str(os.environ.get("YOLO_DESKTOP_PROTOCOL") or "").strip() == "1"


def emit_json(tag: str, payload: dict) -> None:
    if desktop_protocol_enabled():
        print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将原始检测数据整理为正式的 Ultralytics detect 数据集，"
            "生成 images/train、images/val、labels/train、labels/val 和 dataset.yaml。"
        )
    )
    parser.add_argument("--input", required=True, help="原始数据集目录。")
    parser.add_argument("--output", required=True, help="整理后的输出目录。")
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", "yolo-flat", "labelme-json", "coco-json"),
        help="原始数据格式。'auto' 会在平铺 YOLO 标注、LabelMe 矩形 JSON 和 COCO JSON 之间自动识别。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集划分比例。必须介于 0 和 1 之间，且输出始终保持 train/val 分离。",
    )
    parser.add_argument("--seed", type=int, default=42, help="train/val 划分使用的随机种子。")
    parser.add_argument(
        "--copy-mode",
        default="copy",
        choices=("copy", "hardlink"),
        help="整理后图片的放置方式。标签文件始终重新生成。",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="可选的显式类别顺序，例如：--class-names person car dog",
    )
    parser.add_argument(
        "--class-names-file",
        default=None,
        help="可选的类别顺序文件。支持纯文本、JSON，或带 names 字段的 YAML。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果输出目录下已存在 dataset.yaml、images/ 或 labels/，则覆盖它们。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="将未匹配文件和可恢复的标注问题提升为硬错误。",
    )
    return parser.parse_args()


def iter_flat_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def iter_recursive_images(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def iter_flat_json(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".json")


def iter_recursive_json(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*.json") if path.is_file())


def iter_flat_txt(root: Path) -> list[Path]:
    ignored = {name.lower() for name in TEXT_CLASS_NAME_FILES}
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and path.name.lower() not in ignored
    )


def read_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_class_names(values: Iterable[str]) -> list[str]:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        raise ValueError("解析得到的类别名称列表为空。")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"类别名称必须唯一，当前为 {normalized}。")
    return normalized


def load_class_names_from_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "names" in payload:
            payload = payload["names"]
        if isinstance(payload, dict):
            return normalize_class_names(payload[key] for key in sorted(payload, key=lambda item: int(item)))
        if isinstance(payload, list):
            return normalize_class_names(payload)
        raise ValueError(f"{path} 中的类别名称结构不受支持。")
    return normalize_class_names(read_text_lines(path))


def resolve_explicit_class_names(args: argparse.Namespace, input_root: Path) -> list[str] | None:
    if args.class_names:
        return normalize_class_names(args.class_names)
    if args.class_names_file:
        return load_class_names_from_file(Path(args.class_names_file).expanduser().resolve())
    for name in TEXT_CLASS_NAME_FILES:
        candidate = input_root / name
        if candidate.exists():
            return load_class_names_from_file(candidate)
    return None


def load_json_payload(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_coco_payload(payload: dict | None) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("images"), list)
        and isinstance(payload.get("annotations"), list)
        and isinstance(payload.get("categories"), list)
    )


def detect_dataset_format(root: Path, requested: str) -> str:
    if requested != "auto":
        return requested

    if iter_flat_txt(root):
        return "yolo-flat"

    coco_candidates = [path for path in iter_recursive_json(root) if is_coco_payload(load_json_payload(path))]
    if len(coco_candidates) == 1:
        return "coco-json"
    if len(coco_candidates) > 1:
        raise ValueError(
            "自动识别 COCO JSON 失败：检测到多个 COCO 风格 JSON 文件。"
            "请显式选择 --format coco-json，并确保所选目录下只保留一个 COCO 标注 JSON。"
        )

    if iter_flat_json(root):
        return "labelme-json"

    raise ValueError(
        "自动识别数据集格式失败：未找到受支持的原始标注。"
        "当前支持平铺 YOLO TXT、LabelMe 矩形框 JSON 和 COCO JSON。"
    )


def validate_yolo_label_lines(label_path: Path) -> tuple[list[str], set[int]]:
    lines: list[str] = []
    used_class_ids: set[int] = set()
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path} 第 {line_number} 行必须包含 5 列，实际为：{stripped!r}")

        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path} 第 {line_number} 行的类别 ID 不是整数：{parts[0]!r}") from exc
        if class_id < 0:
            raise ValueError(f"{label_path} 第 {line_number} 行的类别 ID 不能为负数：{class_id}")

        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path} 第 {line_number} 行的框坐标不是数字：{parts[1:]!r}") from exc

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                f"{label_path} 第 {line_number} 行必须使用 0..1 归一化坐标，实际为：{values!r}"
            )

        lines.append(f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
        used_class_ids.add(class_id)
    return lines, used_class_ids


def resolve_image_size(image_path: Path, width: int | None, height: int | None) -> tuple[int, int]:
    if width and height:
        return int(width), int(height)
    if Image is None:
        raise ValueError(
            f"{image_path} 缺少 imageWidth/imageHeight 元数据，且当前无法使用 Pillow 兜底读取。"
        )
    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def convert_rectangle_to_yolo(shape: dict, image_width: int, image_height: int, class_id: int, source: Path) -> str:
    points = shape.get("points") or []
    if len(points) < 2:
        raise ValueError(f"{source} 中存在少于 2 个点的矩形标注。")

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    left = clamp(min(xs), 0.0, float(image_width))
    right = clamp(max(xs), 0.0, float(image_width))
    top = clamp(min(ys), 0.0, float(image_height))
    bottom = clamp(max(ys), 0.0, float(image_height))

    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{source} 中存在裁剪到图像边界后退化的矩形标注。")

    x_center = (left + right) / 2.0 / float(image_width)
    y_center = (top + bottom) / 2.0 / float(image_height)
    box_width = width / float(image_width)
    box_height = height / float(image_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_coco_bbox_to_yolo(
    bbox: list[float], image_width: int, image_height: int, class_id: int, image_path: Path, annotation_id: object
) -> str:
    if len(bbox) != 4:
        raise ValueError(f"{image_path.name} 的标注 {annotation_id!r} 不包含 4 个数值的 bbox。")

    left = clamp(float(bbox[0]), 0.0, float(image_width))
    top = clamp(float(bbox[1]), 0.0, float(image_height))
    right = clamp(float(bbox[0]) + float(bbox[2]), 0.0, float(image_width))
    bottom = clamp(float(bbox[1]) + float(bbox[3]), 0.0, float(image_height))

    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{image_path.name} 的标注 {annotation_id!r} 在裁剪后为空。")

    x_center = (left + right) / 2.0 / float(image_width)
    y_center = (top + bottom) / 2.0 / float(image_height)
    box_width = width / float(image_width)
    box_height = height / float(image_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def resolve_relative_output_path(image_path: Path, input_root: Path) -> Path:
    try:
        return image_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return Path(image_path.name)


def resolve_labelme_image(
    json_path: Path, payload: dict, images_by_name: dict[str, Path], images_by_stem: dict[str, Path]
) -> Path | None:
    image_path = str(payload.get("imagePath") or "").strip()
    candidates: list[Path] = []
    if image_path:
        candidates.extend(
            [
                (json_path.parent / image_path).resolve(),
                Path(image_path).expanduser().resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    by_name = images_by_name.get(Path(image_path).name.lower())
    if by_name is not None:
        return by_name

    return images_by_stem.get(json_path.stem.lower())


def prepare_yolo_flat_dataset(input_root: Path, args: argparse.Namespace) -> PreparationReport:
    class_names = resolve_explicit_class_names(args, input_root)
    image_files = iter_flat_images(input_root)
    if not image_files:
        raise ValueError(f"{input_root} 下未找到图片文件。")

    warnings: list[str] = []
    samples: list[PreparedSample] = []
    used_class_ids: set[int] = set()
    label_count = 0
    paired_label_paths: set[Path] = set()

    for image_path in image_files:
        label_path = input_root / f"{image_path.stem}.txt"
        if label_path.exists():
            label_lines, label_ids = validate_yolo_label_lines(label_path)
            paired_label_paths.add(label_path.resolve())
            label_count += len(label_lines)
            used_class_ids.update(label_ids)
        else:
            label_lines = []
            warnings.append(f"图片 {image_path.name} 缺少标签，将按背景图处理。")
        samples.append(
            PreparedSample(
                image_path=image_path.resolve(),
                relative_image_path=Path(image_path.name),
                label_lines=label_lines,
            )
        )

    extra_labels = [path.name for path in iter_flat_txt(input_root) if path.resolve() not in paired_label_paths]
    if extra_labels:
        warnings.append(f"发现 {len(extra_labels)} 个 TXT 文件没有对应图片：{', '.join(extra_labels[:5])}")

    if class_names is None:
        if not used_class_ids:
            raise ValueError(
                "未发现任何带标签的类别，且未提供类别名称。"
                "请传入 --class-names 或 --class-names-file。"
            )
        class_names = [f"class{i}" for i in range(max(used_class_ids) + 1)]

    if used_class_ids and max(used_class_ids) >= len(class_names):
        raise ValueError(
            f"标签中的类别 ID {max(used_class_ids)} 超出了声明类别数 {len(class_names)}：{class_names!r}。"
        )

    unused_class_ids = sorted(set(range(len(class_names))) - used_class_ids)
    if unused_class_ids:
        unused_names = [class_names[index] for index in unused_class_ids]
        warnings.append(f"以下声明类别未在任何标签中使用：{unused_names}")

    return PreparationReport(
        dataset_format="yolo-flat",
        class_names=class_names,
        samples=samples,
        warnings=warnings,
        label_count=label_count,
        used_class_ids=used_class_ids,
    )


def prepare_labelme_dataset(input_root: Path, args: argparse.Namespace) -> PreparationReport:
    explicit_class_names = resolve_explicit_class_names(args, input_root)
    image_files = iter_flat_images(input_root)
    json_files = iter_flat_json(input_root)
    if not image_files:
        raise ValueError(f"{input_root} 下未找到图片文件。")
    if not json_files:
        raise ValueError(f"{input_root} 下未找到 JSON 标注文件。")

    images_by_name = {path.name.lower(): path.resolve() for path in image_files}
    images_by_stem = {path.stem.lower(): path.resolve() for path in image_files}

    warnings: list[str] = []
    encountered_class_names: list[str] = []
    encountered_set: set[str] = set()
    samples: list[PreparedSample] = []
    label_count = 0
    paired_images: set[Path] = set()
    pending_entries: list[tuple[Path, Path, list[tuple[str, dict]], int, int]] = []

    for json_path in json_files:
        payload = load_json_payload(json_path)
        if payload is None:
            message = f"跳过无法读取的 JSON 文件：{json_path.name}"
            if args.strict:
                raise ValueError(message)
            warnings.append(message)
            continue

        image_path = resolve_labelme_image(json_path, payload, images_by_name, images_by_stem)
        if image_path is None:
            warnings.append(f"跳过 {json_path.name}，因为无法解析出对应图片。")
            continue

        paired_images.add(image_path.resolve())
        image_width, image_height = resolve_image_size(
            image_path,
            payload.get("imageWidth"),
            payload.get("imageHeight"),
        )

        pending_shapes: list[tuple[str, dict]] = []
        for shape in payload.get("shapes", []):
            shape_type = str(shape.get("shape_type") or "rectangle").strip().lower()
            if shape_type != "rectangle":
                message = f"跳过 {json_path.name} 中的非矩形标注：{shape_type}"
                if args.strict:
                    raise ValueError(message)
                warnings.append(message)
                continue

            label = str(shape.get("label") or "").strip()
            if not label:
                message = f"跳过 {json_path.name} 中缺少标签的矩形标注。"
                if args.strict:
                    raise ValueError(message)
                warnings.append(message)
                continue

            if explicit_class_names is not None and label not in explicit_class_names:
                raise ValueError(
                    f"{json_path.name} 使用了标签 {label!r}，但声明的类别顺序是 {explicit_class_names!r}。"
                )

            if label not in encountered_set:
                encountered_set.add(label)
                encountered_class_names.append(label)
            pending_shapes.append((label, shape))

        pending_entries.append((image_path.resolve(), Path(image_path.name), pending_shapes, image_width, image_height))

    if explicit_class_names is not None:
        class_names = explicit_class_names
    else:
        class_names = normalize_class_names(encountered_class_names)

    class_index = {name: index for index, name in enumerate(class_names)}
    used_class_ids: set[int] = set()

    for image_path, relative_image_path, pending_shapes, image_width, image_height in pending_entries:
        label_lines: list[str] = []
        for label, shape in pending_shapes:
            if label not in class_index:
                raise ValueError(f"解析得到的类别映射中不存在标签 {label!r}。")
            line = convert_rectangle_to_yolo(shape, image_width, image_height, class_index[label], image_path)
            label_lines.append(line)
            label_count += 1
            used_class_ids.add(class_index[label])
        samples.append(PreparedSample(image_path=image_path, relative_image_path=relative_image_path, label_lines=label_lines))

    unmatched_images = [path.name for path in image_files if path.resolve() not in paired_images]
    if unmatched_images:
        warnings.append(f"发现 {len(unmatched_images)} 张图片没有对应的 JSON 标签：{', '.join(unmatched_images[:5])}")

    if not samples:
        raise ValueError("未整理出任何有效的图片/JSON 配对。")

    return PreparationReport(
        dataset_format="labelme-json",
        class_names=class_names,
        samples=samples,
        warnings=warnings,
        label_count=label_count,
        used_class_ids=used_class_ids,
    )


def resolve_unique_coco_annotation_file(input_root: Path) -> tuple[Path, dict]:
    matches: list[tuple[Path, dict]] = []
    for json_path in iter_recursive_json(input_root):
        payload = load_json_payload(json_path)
        if is_coco_payload(payload):
            matches.append((json_path, payload))

    if not matches:
        raise ValueError(f"{input_root} 下未找到 COCO JSON 文件。")
    if len(matches) > 1:
        options = ", ".join(path.name for path, _ in matches[:5])
        raise ValueError(
            f"{input_root} 下发现多个 COCO JSON 文件：{options}。"
            "该流程要求所选目录内只保留一个 COCO 标注 JSON。"
        )
    return matches[0]


def resolve_coco_image(
    file_name: str, input_root: Path, annotation_path: Path, images_by_relative: dict[str, Path], images_by_name: dict[str, list[Path]]
) -> Path | None:
    normalized = str(PurePosixPath(file_name.replace("\\", "/"))).strip()
    candidates = [
        input_root / normalized,
        annotation_path.parent / normalized,
        Path(normalized).expanduser(),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    lowered = normalized.lower()
    direct = images_by_relative.get(lowered)
    if direct is not None:
        return direct

    suffix_matches = [path for relative, path in images_by_relative.items() if relative.endswith("/" + lowered)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    basename = PurePosixPath(normalized).name.lower()
    by_name = images_by_name.get(basename, [])
    if len(by_name) == 1:
        return by_name[0]
    return None


def prepare_coco_dataset(input_root: Path, args: argparse.Namespace) -> PreparationReport:
    explicit_class_names = resolve_explicit_class_names(args, input_root)
    annotation_path, payload = resolve_unique_coco_annotation_file(input_root)
    image_files = iter_recursive_images(input_root)
    if not image_files:
        raise ValueError(f"{input_root} 下未找到图片文件。")

    images_by_relative: dict[str, Path] = {}
    images_by_name: dict[str, list[Path]] = {}
    for image_path in image_files:
        try:
            relative = image_path.relative_to(input_root).as_posix().lower()
            images_by_relative[relative] = image_path
        except ValueError:
            pass
        images_by_name.setdefault(image_path.name.lower(), []).append(image_path)

    warnings: list[str] = []
    categories = payload.get("categories") or []
    if not categories:
        raise ValueError(f"{annotation_path.name} 中不包含任何 COCO categories。")

    category_rows: list[tuple[int, str]] = []
    for category in categories:
        try:
            category_id = int(category["id"])
        except Exception as exc:
            raise ValueError(f"{annotation_path.name} 中存在缺少有效整数 id 的类别项。") from exc
        category_name = str(category.get("name") or "").strip()
        if not category_name:
            raise ValueError(f"{annotation_path.name} 中的类别 id {category_id} 缺少名称。")
        category_rows.append((category_id, category_name))

    category_rows.sort(key=lambda item: item[0])
    default_class_names = normalize_class_names(name for _, name in category_rows)
    category_id_to_name = {category_id: name for category_id, name in category_rows}

    class_names = explicit_class_names if explicit_class_names is not None else default_class_names
    class_index = {name: index for index, name in enumerate(class_names)}

    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in payload.get("annotations") or []:
        try:
            image_id = int(annotation["image_id"])
        except Exception:
            warnings.append(f"跳过缺少有效图片 id（image_id）的 COCO 标注：{annotation!r}")
            continue
        annotations_by_image_id.setdefault(image_id, []).append(annotation)

    samples: list[PreparedSample] = []
    used_class_ids: set[int] = set()
    label_count = 0
    paired_images: set[Path] = set()

    for image_record in payload.get("images") or []:
        try:
            image_id = int(image_record["id"])
        except Exception:
            warnings.append(f"跳过缺少有效 id 的 COCO 图片记录：{image_record!r}")
            continue

        file_name = str(image_record.get("file_name") or "").strip()
        if not file_name:
            warnings.append(f"跳过 COCO 图片 id {image_id}，因为文件名（file_name）为空。")
            continue

        image_path = resolve_coco_image(file_name, input_root, annotation_path, images_by_relative, images_by_name)
        if image_path is None:
            warnings.append(f"跳过 COCO 图片 {file_name}，因为无法解析出对应图片文件。")
            continue

        paired_images.add(image_path.resolve())
        image_width, image_height = resolve_image_size(
            image_path,
            image_record.get("width"),
            image_record.get("height"),
        )

        label_lines: list[str] = []
        for annotation in annotations_by_image_id.get(image_id, []):
            annotation_id = annotation.get("id", "<unknown>")
            try:
                category_id = int(annotation["category_id"])
            except Exception:
                warnings.append(f"跳过 COCO 标注 {annotation_id!r}，因为类别 id（category_id）缺失或无效。")
                continue

            category_name = category_id_to_name.get(category_id)
            if not category_name:
                warnings.append(
                    f"跳过 {file_name} 的 COCO 标注 {annotation_id!r}，因为类别 {category_id} 未定义。"
                )
                continue
            if category_name not in class_index:
                raise ValueError(
                    f"标注中出现了 COCO 类别 {category_name!r}，但声明的类别顺序是 {class_names!r}。"
                )

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list):
                warnings.append(f"跳过 {file_name} 的 COCO 标注 {annotation_id!r}，因为边界框（bbox）缺失。")
                continue

            try:
                line = convert_coco_bbox_to_yolo(
                    bbox, image_width, image_height, class_index[category_name], image_path, annotation_id
                )
            except ValueError as exc:
                warnings.append(f"跳过 {file_name} 的 COCO 标注 {annotation_id!r}：{exc}")
                continue

            label_lines.append(line)
            label_count += 1
            used_class_ids.add(class_index[category_name])

        samples.append(
            PreparedSample(
                image_path=image_path,
                relative_image_path=resolve_relative_output_path(image_path, input_root),
                label_lines=label_lines,
            )
        )

    unmatched_images = [path.name for path in image_files if path.resolve() not in paired_images]
    if unmatched_images:
        warnings.append(
            f"在所选目录中发现 {len(unmatched_images)} 张图片未被 COCO images 引用："
            + ", ".join(unmatched_images[:5])
        )

    if not samples:
        raise ValueError(f"未从 {annotation_path.name} 整理出任何有效的 COCO 图片条目。")

    return PreparationReport(
        dataset_format="coco-json",
        class_names=class_names,
        samples=samples,
        warnings=warnings,
        label_count=label_count,
        used_class_ids=used_class_ids,
    )


def compute_split_counts(total: int, val_ratio: float) -> tuple[int, int]:
    if total < 2:
        raise ValueError("detect 数据集至少需要 2 个样本，才能保持 train 和 val 分离。")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio 必须介于 0 和 1 之间，当前为 {val_ratio}。")

    val_count = int(round(total * val_ratio))
    if val_count <= 0:
        val_count = 1
    if val_count >= total:
        val_count = total - 1
    return total - val_count, val_count


def ensure_output_root(output_root: Path, force: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    targets = [
        output_root / "dataset.yaml",
        output_root / "images",
        output_root / "labels",
    ]
    existing = [target for target in targets if target.exists()]
    if existing and not force:
        raise ValueError(
            f"输出目录 {output_root} 中已存在整理后的数据文件。"
            "如需覆盖，请传入 --force。"
        )

    if force:
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()


def place_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as exc:
            raise RuntimeError(
                f"无法将 {src} 硬链接到 {dst}。如果路径位于不同磁盘分区，请改用 --copy-mode copy。"
            ) from exc
        return
    shutil.copy2(src, dst)


def validate_unique_output_names(samples: list[PreparedSample]) -> None:
    normalized_paths: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        normalized = sample.relative_image_path.as_posix().lower()
        if normalized in normalized_paths:
            duplicates.add(sample.relative_image_path.as_posix())
        normalized_paths.add(normalized)
    if duplicates:
        raise ValueError(
            "整理后的数据集中存在重复输出图片路径冲突：" + ", ".join(sorted(duplicates))
        )


def write_split(samples: list[PreparedSample], split_name: str, output_root: Path, copy_mode: str) -> None:
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        dst_image = image_dir / sample.relative_image_path
        dst_label = label_dir / sample.relative_image_path.with_suffix(".txt")
        place_image(sample.image_path, dst_image, copy_mode)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(sample.label_lines)
        if text:
            text += "\n"
        dst_label.write_text(text, encoding="utf-8")


def write_dataset_yaml(output_root: Path, class_names: list[str]) -> Path:
    resolved_output_root = output_root.resolve()
    payload = {
        "path": resolved_output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    dataset_yaml = resolved_output_root / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def validate_generated_dataset(dataset_yaml: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1] / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    import yolo_runner  # noqa: PLC0415

    yolo_runner.validate_training_dataset_input(dataset_yaml, "detect")


def emit_report(report: PreparationReport, train_count: int, val_count: int, dataset_yaml: Path, output_root: Path) -> None:
    print(f"整理后的数据格式：{report.dataset_format}")
    print(f"输出目录：{output_root}")
    print(f"生成的 dataset.yaml：{dataset_yaml}")
    print(f"类别顺序：{report.class_names}")
    print(f"样本数量：总计={len(report.samples)}，train={train_count}，val={val_count}")
    print(f"标注框数量：{report.label_count}")
    if report.used_class_ids:
        print(f"使用到的类别 ID：{sorted(report.used_class_ids)}")
    else:
        print("使用到的类别 ID：[]")

    for warning in report.warnings:
        print(f"警告：{warning}", file=sys.stderr)

    emit_json(
        DATASET_PREP_TAG,
        {
            "dataset_yaml": str(dataset_yaml),
            "output_dir": str(output_root),
            "format": report.dataset_format,
            "train_count": train_count,
            "val_count": val_count,
            "warnings": report.warnings,
            "class_names": report.class_names,
        },
    )


def run() -> int:
    args = parse_args()
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise ValueError(f"输入目录不存在：{input_root}")

    dataset_format = detect_dataset_format(input_root, args.format)
    if dataset_format == "yolo-flat":
        report = prepare_yolo_flat_dataset(input_root, args)
    elif dataset_format == "labelme-json":
        report = prepare_labelme_dataset(input_root, args)
    elif dataset_format == "coco-json":
        report = prepare_coco_dataset(input_root, args)
    else:
        raise ValueError(f"不支持的数据集格式：{dataset_format}")

    if args.strict and report.warnings:
        raise ValueError("严格模式失败：处理过程中产生了警告。\n" + "\n".join(report.warnings))

    validate_unique_output_names(report.samples)
    train_count, val_count = compute_split_counts(len(report.samples), args.val_ratio)

    ensure_output_root(output_root, args.force)
    shuffled = sorted(report.samples, key=lambda sample: sample.relative_image_path.as_posix().lower())
    random.Random(args.seed).shuffle(shuffled)
    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count:]

    write_split(train_samples, "train", output_root, args.copy_mode)
    write_split(val_samples, "val", output_root, args.copy_mode)
    dataset_yaml = write_dataset_yaml(output_root, report.class_names)
    validate_generated_dataset(dataset_yaml)
    emit_report(report, len(train_samples), len(val_samples), dataset_yaml, output_root)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
