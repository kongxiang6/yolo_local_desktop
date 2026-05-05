from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAME_FILES = ("classes.txt", "classes.names")


@dataclass
class AnnotationBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self, image_width: int, image_height: int) -> tuple[int, float, float, float, float]:
        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        top = min(self.y1, self.y2)
        bottom = max(self.y1, self.y2)
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        x_center = (left + right) / 2.0 / float(image_width)
        y_center = (top + bottom) / 2.0 / float(image_height)
        return (
            int(self.class_id),
            x_center,
            y_center,
            width / float(image_width),
            height / float(image_height),
        )


@dataclass
class AnnotationPolygon:
    class_id: int
    points: list[tuple[float, float]]

    def normalized_points(self, image_width: int, image_height: int) -> list[float]:
        normalized: list[float] = []
        for x, y in self.points:
            x_value = min(max(float(x) / float(image_width), 0.0), 1.0)
            y_value = min(max(float(y) / float(image_height), 0.0), 1.0)
            normalized.extend([x_value, y_value])
        return normalized


def list_annotation_images(folder: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_project_annotation_images(folder: Path) -> list[Path]:
    image_paths = list_annotation_images(folder)
    if image_paths:
        return image_paths
    images_dir = folder / "images"
    if images_dir.exists() and images_dir.is_dir():
        return list_annotation_images(images_dir)
    return []


def label_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def classes_path_for_folder(folder: Path) -> Path:
    return folder / CLASS_NAME_FILES[0]


def parse_class_names_text(text: str) -> list[str]:
    raw = [line.strip() for line in text.replace(",", "\n").splitlines()]
    names = [item for item in raw if item]
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def load_class_names(folder: Path) -> list[str]:
    for file_name in CLASS_NAME_FILES:
        candidate = folder / file_name
        if not candidate.exists():
            continue
        try:
            return parse_class_names_text(candidate.read_text(encoding="utf-8"))
        except OSError:
            continue
    return []


def save_class_names(folder: Path, class_names: list[str]) -> Path:
    target = classes_path_for_folder(folder)
    target.write_text("\n".join(parse_class_names_text("\n".join(class_names))), encoding="utf-8")
    return target


def ensure_class_names(class_names: list[str], max_class_id: int | None = None) -> list[str]:
    normalized = list(parse_class_names_text("\n".join(class_names)))
    if not normalized:
        normalized = ["class0"]
    if max_class_id is not None:
        while len(normalized) <= max_class_id:
            normalized.append(f"class{len(normalized)}")
    return normalized


def load_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[AnnotationBox]:
    if not label_path.exists():
        return []

    boxes: list[AnnotationBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        box_width = width * image_width
        box_height = height * image_height
        center_x = x_center * image_width
        center_y = y_center * image_height
        left = center_x - box_width / 2.0
        top = center_y - box_height / 2.0
        right = center_x + box_width / 2.0
        bottom = center_y + box_height / 2.0
        boxes.append(AnnotationBox(class_id=class_id, x1=left, y1=top, x2=right, y2=bottom))
    return boxes


def save_yolo_boxes(label_path: Path, boxes: list[AnnotationBox], image_width: int, image_height: int) -> None:
    lines: list[str] = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box.normalized(image_width, image_height)
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)
        if width <= 0.0 or height <= 0.0:
            continue
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def load_yolo_polygons(label_path: Path, image_width: int, image_height: int) -> list[AnnotationPolygon]:
    if not label_path.exists():
        return []

    polygons: list[AnnotationPolygon] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        try:
            class_id = int(parts[0])
            raw_values = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        points: list[tuple[float, float]] = []
        for index in range(0, len(raw_values), 2):
            x_value = min(max(raw_values[index], 0.0), 1.0) * image_width
            y_value = min(max(raw_values[index + 1], 0.0), 1.0) * image_height
            points.append((x_value, y_value))
        if len(points) >= 3:
            polygons.append(AnnotationPolygon(class_id=class_id, points=points))
    return polygons


def save_yolo_polygons(
    label_path: Path,
    polygons: list[AnnotationPolygon],
    image_width: int,
    image_height: int,
) -> None:
    lines: list[str] = []
    for polygon in polygons:
        if len(polygon.points) < 3:
            continue
        values = polygon.normalized_points(image_width, image_height)
        if len(values) < 6:
            continue
        lines.append(f"{int(polygon.class_id)} {' '.join(f'{value:.6f}' for value in values)}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def infer_max_class_id_from_label_lines(lines: list[str]) -> int:
    max_class_id = -1
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id > max_class_id:
            max_class_id = class_id
    return max_class_id


def infer_max_class_id_from_label_file(label_path: Path) -> int:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return -1
    return infer_max_class_id_from_label_lines(lines)


def load_session_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_session_store(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project_session(path: Path, folder: Path) -> dict[str, Any]:
    store = load_session_store(path)
    payload = store.get(str(folder.resolve()))
    return payload if isinstance(payload, dict) else {}


def save_project_session(
    path: Path,
    folder: Path,
    *,
    current_image: str,
    classes: list[str],
) -> None:
    store = load_session_store(path)
    store[str(folder.resolve())] = {
        "current_image": current_image,
        "classes": list(classes),
    }
    save_session_store(path, store)
