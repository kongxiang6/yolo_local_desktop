from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from annotation_support import CLASS_NAME_FILES, IMAGE_SUFFIXES, load_class_names, save_class_names


MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript", ".bin"}


@dataclass
class AiCapability:
    capability_id: str
    label: str
    status: str
    summary: str
    detail: str


@dataclass
class ModelRecord:
    path: Path
    source_label: str
    task_hint: str
    size_bytes: int

    @property
    def display_name(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024) if self.size_bytes else 0.0
        return f"{self.path.name}  ·  {self.task_hint}  ·  {size_mb:.1f} MB"


@dataclass
class TileReport:
    output_dir: Path
    tile_count: int
    source_count: int
    tile_size: int
    overlap: int


AI_CAPABILITIES: list[AiCapability] = [
    AiCapability(
        capability_id="detect_auto",
        label="检测自动标注",
        status="ready",
        summary="直接调用现有检测模型，对整批图片先打一轮框。",
        detail="适合先把目标框快速铺出来，再回到检测工作区做人工修订。",
    ),
    AiCapability(
        capability_id="large_image_tiling",
        label="大图切片准备",
        status="ready",
        summary="把超大图切成小块，方便做小目标标注和后续训练。",
        detail="适合航拍、大图巡检、小目标密集场景。当前先做稳定切片流程，后续再扩切片自动标注。",
    ),
    AiCapability(
        capability_id="prompt_detect",
        label="文本提示检测",
        status="planned",
        summary="按文字描述找目标并回写标注。",
        detail="为后续接 GroundingDINO / Florence2 之类的文本提示模型预留接口。",
    ),
    AiCapability(
        capability_id="prompt_segment",
        label="文本提示分割",
        status="planned",
        summary="按文字描述直接生成掩码。",
        detail="为后续接文本提示分割工作流预留接口，准备放到第三期继续扩。",
    ),
    AiCapability(
        capability_id="ocr_prep",
        label="OCR 数据准备",
        status="planned",
        summary="面向文字检测/识别的数据处理入口。",
        detail="后续准备接文字区域标注、文本识别样本整理、票据/文档场景预处理。",
    ),
]


OFFICIAL_MODEL_SUGGESTIONS = {
    "detect": [
        ("yolo11n.pt", "轻量检测，适合先快速自动标注"),
        ("yolo11s.pt", "比 n 更稳一些，速度和效果平衡"),
        ("yolo11m.pt", "精度更高，适合显卡较强的机器"),
    ],
    "segment": [
        ("yolo11n-seg.pt", "轻量分割，适合先试流程"),
        ("yolo11s-seg.pt", "分割效果和速度更均衡"),
        ("yolo11m-seg.pt", "显卡够用时更适合正式批量处理"),
    ],
    "pose": [
        ("yolo11n-pose.pt", "轻量姿态估计"),
        ("yolo11s-pose.pt", "更均衡的姿态估计"),
    ],
    "obb": [
        ("yolo11n-obb.pt", "轻量旋转框检测"),
        ("yolo11s-obb.pt", "更均衡的旋转框检测"),
    ],
}


def infer_task_from_model_name(name: str) -> str:
    lowered = name.lower()
    if "-seg" in lowered or "segment" in lowered:
        return "segment"
    if "-pose" in lowered or "pose" in lowered:
        return "pose"
    if "-obb" in lowered or "obb" in lowered:
        return "obb"
    if "-cls" in lowered or "classify" in lowered or "classifier" in lowered:
        return "classify"
    if "ocr" in lowered or "text" in lowered:
        return "ocr"
    return "detect"


def scan_model_files(search_roots: list[Path], *, limit: int = 200) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    seen: set[Path] = set()
    for root in search_roots:
        resolved_root = root.expanduser().resolve(strict=False)
        if not resolved_root.exists():
            continue
        for path in resolved_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            resolved_path = path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            try:
                stat = resolved_path.stat()
            except OSError:
                continue
            records.append(
                ModelRecord(
                    path=resolved_path,
                    source_label=resolved_root.name or str(resolved_root),
                    task_hint=infer_task_from_model_name(resolved_path.name),
                    size_bytes=int(stat.st_size),
                )
            )
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    records.sort(key=lambda item: (item.task_hint, item.path.name.lower()))
    return records


def slice_large_images(
    *,
    source_dir: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    progress: Callable[[str], None] | None = None,
) -> TileReport:
    if tile_size <= 32:
        raise ValueError("切片尺寸太小，建议至少大于 32。")
    if overlap < 0:
        raise ValueError("重叠尺寸不能小于 0。")
    if overlap >= tile_size:
        raise ValueError("重叠尺寸必须小于切片尺寸。")

    images = sorted(
        path.resolve()
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("源目录里没有找到图片。")

    output_dir.mkdir(parents=True, exist_ok=True)
    step = tile_size - overlap
    tile_count = 0
    tile_root = output_dir / "images"
    tile_root.mkdir(parents=True, exist_ok=True)

    for image_index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            if progress is not None:
                progress(f"正在切片 {image_index}/{len(images)}：{image_path.name}")

            x_positions = _build_positions(width, tile_size, step)
            y_positions = _build_positions(height, tile_size, step)

            for y_value in y_positions:
                for x_value in x_positions:
                    tile = rgb_image.crop((x_value, y_value, x_value + tile_size, y_value + tile_size))
                    tile_name = f"{image_path.stem}_x{x_value:05d}_y{y_value:05d}.jpg"
                    tile.save(tile_root / tile_name, quality=95)
                    tile_count += 1

    copied_names = load_class_names(source_dir)
    if copied_names:
        save_class_names(output_dir, copied_names)

    meta_path = output_dir / "tile_manifest.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "tile_size": tile_size,
                "overlap": overlap,
                "source_count": len(images),
                "tile_count": tile_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return TileReport(
        output_dir=output_dir,
        tile_count=tile_count,
        source_count=len(images),
        tile_size=tile_size,
        overlap=overlap,
    )


def _build_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), step))
    last_start = max(0, length - tile_size)
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def default_model_scan_roots(project_root: Path) -> list[Path]:
    candidates: list[Path] = [project_root]
    home = Path.home()
    desktop = home / "Desktop"
    downloads = home / "Downloads"
    if desktop.exists():
        candidates.append(desktop)
    if downloads.exists():
        candidates.append(downloads)
    return candidates


def load_model_hub_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_model_hub_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_model_roots(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        resolved = str(Path(text).expanduser().resolve(strict=False))
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result

