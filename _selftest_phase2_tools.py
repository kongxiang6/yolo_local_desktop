import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from multitask_dataset_panel import prepare_classification_dataset, prepare_yolo_task_dataset
from video_annotation_panel import extract_video_frames, inspect_video


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    detect_src = tmp_path / "segment_src"
    detect_src.mkdir()
    (detect_src / "classes.txt").write_text("cat\ndog", encoding="utf-8")
    Image.new("RGB", (100, 80), color=(255, 255, 255)).save(detect_src / "a.jpg")
    (detect_src / "a.txt").write_text("1 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8", encoding="utf-8")
    Image.new("RGB", (100, 80), color=(240, 240, 240)).save(detect_src / "b.jpg")
    (detect_src / "b.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.7 0.2 0.7", encoding="utf-8")

    segment_out = tmp_path / "segment_out"
    report = prepare_yolo_task_dataset(
        task_id="segment",
        source_dir=detect_src,
        output_dir=segment_out,
        val_ratio=0.5,
        seed=42,
        class_names=[],
        copy_mode="copy",
        strict=True,
    )
    assert report.sample_count == 2
    assert report.train_count == 1
    assert report.val_count == 1
    dataset_yaml = yaml.safe_load((segment_out / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["names"] == ["cat", "dog"]
    assert (segment_out / "labels" / "train").exists()
    assert (segment_out / "labels" / "val").exists()

    mismatch_src = tmp_path / "mismatch_src"
    mismatch_src.mkdir()
    Image.new("RGB", (80, 60), color=(220, 220, 220)).save(mismatch_src / "only.jpg")
    (mismatch_src / "only.txt").write_text("2 0.5 0.5 0.4 0.4", encoding="utf-8")
    mismatch_out = tmp_path / "mismatch_out"
    mismatch_report = prepare_yolo_task_dataset(
        task_id="detect",
        source_dir=mismatch_src,
        output_dir=mismatch_out,
        val_ratio=0.0,
        seed=42,
        class_names=["cat"],
        copy_mode="copy",
        strict=True,
    )
    mismatch_yaml = yaml.safe_load((mismatch_out / "dataset.yaml").read_text(encoding="utf-8"))
    assert mismatch_report.class_names == ["cat", "class1", "class2"]
    assert mismatch_yaml["names"] == ["cat", "class1", "class2"]
    assert mismatch_yaml["nc"] == 3

    classify_src = tmp_path / "classify_src"
    (classify_src / "apple").mkdir(parents=True)
    (classify_src / "banana").mkdir(parents=True)
    Image.new("RGB", (40, 40), color=(255, 0, 0)).save(classify_src / "apple" / "1.jpg")
    Image.new("RGB", (40, 40), color=(255, 10, 10)).save(classify_src / "apple" / "2.jpg")
    Image.new("RGB", (40, 40), color=(255, 255, 0)).save(classify_src / "banana" / "1.jpg")
    Image.new("RGB", (40, 40), color=(250, 250, 0)).save(classify_src / "banana" / "2.jpg")
    classify_out = tmp_path / "classify_out"
    classify_report = prepare_classification_dataset(
        source_dir=classify_src,
        output_dir=classify_out,
        val_ratio=0.5,
        seed=42,
        copy_mode="copy",
    )
    assert classify_report.sample_count == 4
    assert (classify_out / "train" / "apple").exists()
    assert (classify_out / "val" / "banana").exists()

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    if cv2 is not None:
        import numpy as np

        video_path = tmp_path / "sample.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        assert writer.isOpened(), "video writer not available"
        for index in range(30):
            frame = (index * 5) % 255
            image = np.full((48, 64, 3), frame, dtype=np.uint8)
            writer.write(image)
        writer.release()

        info = inspect_video(video_path)
        assert info.frame_count >= 30
        extract_report = extract_video_frames(video_path=video_path, output_dir=tmp_path / "frames", frame_step=5)
        assert extract_report.extracted_count >= 6

print("PHASE2_TOOLS_SELFTEST_OK")
