import sys
import tempfile
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_platform_support import infer_task_from_model_name, scan_model_files, slice_large_images


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "yolo11n.pt").write_bytes(b"detect-model")
    (model_root / "yolo11s-seg.pt").write_bytes(b"segment-model")

    records = scan_model_files([model_root], limit=20)
    assert len(records) == 2
    assert any(item.task_hint == "detect" for item in records)
    assert any(item.task_hint == "segment" for item in records)
    assert infer_task_from_model_name("abc-pose.pt") == "pose"
    assert infer_task_from_model_name("demo-obb.engine") == "obb"

    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "classes.txt").write_text("cat\ndog", encoding="utf-8")
    Image.new("RGB", (1800, 1200), color=(255, 255, 255)).save(image_root / "big.jpg")

    output_root = tmp_path / "tiles"
    report = slice_large_images(
        source_dir=image_root,
        output_dir=output_root,
        tile_size=800,
        overlap=100,
    )
    assert report.tile_count > 1
    assert (output_root / "images").exists()
    assert (output_root / "classes.txt").exists()
    assert (output_root / "tile_manifest.json").exists()

print("PHASE3_AI_TOOLS_SELFTEST_OK")
