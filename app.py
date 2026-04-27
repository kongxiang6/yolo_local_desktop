from __future__ import annotations

import importlib.util
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import urllib.request
import zipfile
from datetime import datetime
from urllib.parse import quote, unquote
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any


APP_NAME = "YoloTool"
WINDOW_BG = "#eef4ff"
PANEL_BG = "#f7fbff"
CARD_BG = "#ffffff"
CARD_SOFT = "#fbfdff"
PRIMARY = "#5b8cff"
PRIMARY_DARK = "#4f7de6"
PRIMARY_SOFT = "#edf3ff"
BORDER = "#dce7fb"
TEXT = "#243348"
TEXT_MUTED = "#6c7c96"
SUCCESS = "#31c48d"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NO_WINDOW_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0
LOG_POLL_ACTIVE_MS = 40
LOG_QUEUE_ITEMS_PER_TICK = 240
LOG_VISIBLE_LINE_LIMIT = 2000

PARAMETER_HINTS: dict[str, str] = {
    "epochs": "训练多少轮。数值越大，训练越久；新手一般先从 50 或 100 开始。",
    "time": "限制最长训练时间，单位是小时。填了它以后，会优先按时间停止，而不是按轮次停止。",
    "batch": "每次同时喂给显卡/CPU 多少张图。太大容易爆显存，太小训练会更慢。",
    "imgsz": "输入图片尺寸。常用 640；想更清楚可以调大，但会更吃显存和时间。",
    "cache": "是否提前把图片缓存起来。内存够就选 ram，硬盘快可选 disk，不确定就关掉。",
    "device": "运行设备。`0` 表示第一张显卡，`0,1` 表示多卡，`cpu` 表示只用 CPU。",
    "workers": "数据读取线程数。电脑卡顿或硬盘慢时，可以适当调小。",
    "project": "结果保存到哪个总文件夹。留空就按软件默认目录保存。",
    "name": "这次任务的子文件夹名字，方便区分不同实验。",
    "exist_ok": "如果结果目录已经存在，是否允许继续写进去。新手一般保持关闭更安全。",
    "pretrained": "是否使用预训练权重。一般建议开着，训练更快、更容易收敛。",
    "verbose": "是否输出更详细的日志。看不懂可以不管，默认开着即可。",
    "seed": "随机种子。想复现同一套结果时，尽量保持不变。",
    "deterministic": "尽量让结果更稳定、更可复现，但训练速度可能稍慢。",
    "resume": "是否从上一次中断的位置继续训练。只有断点续训时才建议打开。",
    "single_cls": "把所有类别都当成一个类别来训练。只有单类别场景才建议打开。",
    "rect": "按原图长宽比做更省显存的矩形训练/验证。图片比例差异很大时更有用。",
    "cos_lr": "使用余弦学习率衰减。一般用于想让训练后期更平滑时。",
    "close_mosaic": "训练最后多少轮关闭 Mosaic 增强，常用于后期稳定收敛。",
    "amp": "自动混合精度。大多数显卡建议开启，通常更快也更省显存。",
    "fraction": "只使用一部分数据。比如填 0.5 表示只用 50% 数据做测试。",
    "profile": "额外统计模型速度信息。一般排查性能时再开。",
    "save_period": "每隔多少轮额外保存一次 checkpoint。填 -1 表示只保留默认的 best / last。",
    "val": "训练过程中是否顺手跑验证。新手一般建议开着，这样更容易看出模型有没有变好。",
    "freeze": "冻结前几层参数。适合小数据集微调，完全新手建议先留空。",
    "patience": "早停耐心值。模型长期没有提升时，会提前结束训练，避免浪费时间。",
    "save": "是否保存可视化结果。新手一般建议打开。",
    "save_txt": "是否额外保存成 txt 标注结果，便于后续处理。",
    "save_conf": "保存 txt 时，是否把置信度一起写进去。",
    "save_crop": "把识别到的目标单独裁剪保存出来。",
    "show": "实时弹窗显示结果。远程桌面或长时间跑任务时通常建议关闭。",
    "conf": "置信度阈值。越高越严格，误检会少一些，但也可能漏检。",
    "iou": "去重阈值。主要影响重叠框如何合并，一般保持默认就行。",
    "agnostic_nms": "不同类别之间也互相去重。只有特殊场景才需要开。",
    "classes": "只保留指定类别。比如填 `0,1` 表示只看第 0 和第 1 类。",
    "max_det": "每张图最多保留多少个结果。目标很多时可以调大。",
    "save_json": "把验证结果额外导出成 COCO JSON。通常是做评测、对接别的工具时才需要。",
    "vid_stride": "视频隔多少帧处理一次。数值越大越省性能，但可能漏掉瞬间目标。",
    "tracker": "跟踪算法选择。新手不确定时保持默认即可。",
    "persist": "连续视频中尽量沿用上一帧的跟踪结果，适合正常视频流。",
    "stream_buffer": "流媒体时先多缓存一些帧，减少丢帧，但延迟会变大。",
    "line_width": "框线粗细。留空会自动适配图片大小。",
    "augment": "预测时做增强测试，结果可能更稳，但速度会慢。",
    "retina_masks": "分割任务输出更精细的掩码，画面更好看，但更占资源。",
    "half": "半精度运行。显卡支持时通常更快；不确定就保持默认。",
    "dnn": "仅特定 ONNX 场景才需要。一般不要改。",
    "plots": "是否生成曲线图、混淆矩阵等图表，方便看结果。",
    "split": "验证时使用哪一部分数据，一般选 val。",
    "optimizer": "优化器选择。不懂就保持 auto，让程序自动挑更合适的方案。",
    "format": "导出格式。按你的部署环境选，不确定时一般选 ONNX。",
    "opset": "ONNX 导出版本号。除非部署端有明确要求，否则别改。",
    "workspace": "TensorRT 可用工作空间大小。显存紧张时不要乱调大。",
    "int8": "是否做 INT8 量化。会更快更小，但通常需要校准数据。",
    "nms": "是否把 NMS 一起打包进导出模型。部署端没有后处理时才建议开。",
    "optimize": "做额外导出优化。一般保持默认。",
    "keras": "导出 TensorFlow/Keras 相关格式时才可能用到。",
    "data": "数据集配置文件路径，通常是 `dataset.yaml`。",
    "dynamic": "导出后是否支持动态输入尺寸。部署更灵活，但兼容性视平台而定。",
    "simplify": "尝试简化导出的模型结构，通常建议开着。",
    "source": "输入来源，可以是图片、视频、文件夹、摄像头编号或流地址。",
    "visualize": "可视化中间特征图。一般调试模型时才打开。",
    "embed": "导出中间特征向量。普通训练/预测用户通常用不到。",
    "multi_scale": "训练时自动在多个输入尺寸间切换。能增强泛化，但训练速度通常会变慢。",
    "compile": "是否启用 PyTorch 编译加速。兼容性没问题时可能更快，出错就关掉。",
    "lr0": "初始学习率。不会调参的新手建议保持默认。",
    "lrf": "最终学习率比例。和学习率衰减有关，一般别乱改。",
    "momentum": "动量参数，影响训练更新平滑程度，通常保持默认。",
    "weight_decay": "权重衰减，主要用来抑制过拟合，默认值通常够用。",
    "warmup_epochs": "训练前几轮慢慢升学习率，帮助训练更稳。",
    "warmup_momentum": "预热阶段的动量参数，新手一般不用改。",
    "warmup_bias_lr": "预热阶段偏置项学习率，新手一般不用改。",
    "box": "检测框损失权重，只在你明确知道调参目的时再动。",
    "cls": "分类损失权重。类别很多、分类困难时才会考虑调整。",
    "dfl": "边框分布损失权重。一般保持默认。",
    "pose": "姿态任务关键点损失权重，仅姿态任务相关。",
    "kobj": "关键点目标损失权重，仅姿态任务相关。",
    "label_smoothing": "标签平滑，用于缓解过拟合；新手一般保持默认。",
    "nbs": "标准批大小，用于内部缩放，通常不需要改。",
    "hsv_h": "色相增强强度，控制颜色变化幅度。",
    "hsv_s": "饱和度增强强度，控制颜色浓淡变化。",
    "hsv_v": "亮度增强强度，控制明暗变化。",
    "degrees": "随机旋转角度范围，图片方向变化大时才适合调大。",
    "translate": "随机平移幅度，适合目标位置变化大的数据。",
    "scale": "随机缩放幅度，适合目标大小变化大的数据。",
    "shear": "随机错切变换强度，普通场景一般用默认即可。",
    "perspective": "透视变换强度，过大容易让图片失真。",
    "flipud": "上下翻转概率，只有上下颠倒也合理的数据才建议开。",
    "fliplr": "左右翻转概率，大多数场景都可以适当使用。",
    "bgr": "RGB/BGR 通道扰动概率，通常保持默认。",
    "mosaic": "四图拼接增强。对小目标和少量数据常有帮助。",
    "mixup": "两张图混合增强。数据复杂时可提高泛化，但不一定总有效。",
    "cutmix": "把两张图局部拼接到一起做增强。能提升泛化，但不一定适合所有数据集。",
    "copy_paste": "复制粘贴增强，多用于分割任务。",
    "copy_paste_mode": "分割任务里复制粘贴增强的方式。新手一般保持默认的 flip 即可。",
    "auto_augment": "分类任务的自动增强策略。不懂就保持默认，让程序按常见方案处理。",
    "erasing": "随机擦除一块区域，提升遮挡鲁棒性。",
    "overlap_mask": "分割任务里是否允许多个掩码按顺序重叠显示。一般保持默认即可。",
    "mask_ratio": "分割掩码缩放比例。值越小越省显存，但掩码细节会少一些。",
    "dropout": "分类任务的随机失活比例，用来降低过拟合风险。不会调参就先保持默认。",
    "crop_fraction": "裁剪保留比例，分类任务更常见。",
    "rle": "是否把分割结果编码成 RLE 格式。通常只在特定评测或导出场景才用到。",
    "angle": "旋转框角度损失权重，仅 OBB 任务相关。没有明确需要时建议保持默认。",
}

PARAMETER_EXAMPLES: dict[str, str] = {
    "epochs": "例如：`100`",
    "time": "例如：`2.5` 表示最多训练 2.5 小时",
    "cache": "例如：`False`、`ram`、`disk`",
    "device": "例如：`0`、`0,1`、`cpu`",
    "classes": "例如：`0,1` 或 `[0, 1]`",
    "project": "例如：`D:/yolo_runs`",
    "name": "例如：`exp_cat_dog_v1`",
    "imgsz": "例如：`640`",
    "batch": "例如：`8`、`16`",
    "workers": "例如：`4`、`8`",
    "seed": "例如：`0`",
    "resume": "例如：勾选=继续上次训练，不勾选=重新开始",
    "save_period": "例如：`10` 表示每 10 轮额外保存一次",
    "val": "例如：勾选=训练时同步跑验证",
    "split": "例如：`val`、`test`、`train`",
    "save_json": "例如：勾选后额外生成 COCO JSON 文件",
    "conf": "例如：`0.25`",
    "iou": "例如：`0.7`",
    "max_det": "例如：`300`",
    "half": "例如：勾选=尝试半精度运行",
    "plots": "例如：勾选后会生成曲线图和混淆矩阵",
    "patience": "例如：`20`、`50`",
    "optimizer": "例如：`auto`、`SGD`、`AdamW`",
    "cos_lr": "例如：勾选=使用余弦学习率",
    "close_mosaic": "例如：`10` 表示最后 10 轮关闭 Mosaic",
    "amp": "例如：勾选=开启混合精度",
    "fraction": "例如：`0.5` 表示只用一半数据",
    "freeze": "例如：`10` 表示冻结前 10 层",
    "multi_scale": "例如：勾选=训练时自动切换输入尺寸",
    "compile": "例如：`False`、`default`",
    "source": "例如：图片路径、视频路径、`0`、RTSP 地址",
    "tracker": "例如：`botsort.yaml`、`bytetrack.yaml`",
    "dynamic": "例如：勾选后支持动态输入尺寸",
    "opset": "例如：`13`",
    "simplify": "例如：勾选=自动简化导出结构",
    "nms": "例如：勾选=把 NMS 一起导出",
    "int8": "例如：勾选=尝试 INT8 量化",
    "workspace": "例如：`4.0`",
    "optimize": "例如：勾选=启用额外优化",
    "data": "例如：`D:/dataset/dataset.yaml`",
}

BUILTIN_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "train": {
        "推荐-轻量快速训练": {
            "task": "detect",
            "train_mode_label": "官方预训练",
            "family": "YOLO11",
            "size": "n",
            "config": {"epochs": 50, "batch": 8, "imgsz": 640, "workers": 4, "cache": False, "amp": True, "patience": 20},
        },
        "推荐-均衡通用训练": {
            "task": "detect",
            "train_mode_label": "官方预训练",
            "family": "YOLO11",
            "size": "s",
            "config": {"epochs": 100, "batch": 16, "imgsz": 640, "workers": 8, "cache": "ram", "amp": True, "patience": 35},
        },
        "推荐-高精度训练": {
            "task": "detect",
            "train_mode_label": "官方预训练",
            "family": "YOLO11",
            "size": "m",
            "config": {"epochs": 180, "batch": 8, "imgsz": 960, "workers": 8, "cache": "disk", "amp": True, "cos_lr": True, "patience": 50},
        },
    },
    "val": {
        "推荐-快速验收": {"task": "detect", "config": {"imgsz": 640, "batch": 8, "plots": True, "verbose": True}},
        "推荐-完整评估": {"task": "detect", "config": {"imgsz": 640, "batch": 16, "save_json": True, "plots": True, "verbose": True}},
    },
    "predict": {
        "推荐-图片通用预测": {"task": "detect", "config": {"imgsz": 640, "conf": 0.25, "save": True, "show": False, "verbose": True}},
        "推荐-视频实时预测": {"task": "detect", "config": {"imgsz": 640, "conf": 0.35, "vid_stride": 1, "save": True, "show": False, "verbose": False}},
    },
    "track": {
        "推荐-实时跟踪": {"task": "detect", "config": {"imgsz": 640, "conf": 0.25, "tracker": "bytetrack.yaml", "persist": True, "save": True}},
        "推荐-稳健跟踪": {"task": "detect", "config": {"imgsz": 640, "conf": 0.2, "tracker": "botsort.yaml", "persist": True, "save": True}},
    },
    "export": {
        "推荐-ONNX通用部署": {
            "task_label": "自动识别",
            "format_label": "通用交换格式（ONNX）",
            "imgsz": "640",
            "device": "0",
            "config": {"batch": 1, "dynamic": False, "half": False, "simplify": True, "nms": False},
        },
        "推荐-TensorRT高速部署": {
            "task_label": "自动识别",
            "format_label": "TensorRT 引擎（TensorRT Engine）",
            "imgsz": "640",
            "device": "0",
            "config": {"batch": 1, "dynamic": False, "half": True, "simplify": True, "int8": False},
        },
    },
}

TASK_LABEL_TO_ID = {
    "目标检测": "detect",
    "实例分割": "segment",
    "图像分类": "classify",
    "姿态估计": "pose",
    "旋转框检测": "obb",
}
TASK_ID_TO_LABEL = {value: key for key, value in TASK_LABEL_TO_ID.items()}
TRACK_TASK_IDS = ("detect", "segment", "pose", "obb")
TRACK_TASK_LABELS = [TASK_ID_TO_LABEL[item] for item in TRACK_TASK_IDS]
EXPORT_TASK_LABEL_TO_ID = {
    "自动识别": "",
    "目标检测": "detect",
    "实例分割": "segment",
    "图像分类": "classify",
    "姿态估计": "pose",
    "旋转框检测": "obb",
}
PREP_COPY_MODE_LABEL_TO_ID = {
    "复制文件": "copy",
    "硬链接": "hardlink",
}

ACTION_ID_TO_LABEL = {
    "train": "训练",
    "val": "验证",
    "predict": "预测",
    "track": "跟踪",
}

TASK_SUFFIX = {
    "detect": "",
    "segment": "-seg",
    "classify": "-cls",
    "pose": "-pose",
    "obb": "-obb",
}

MODEL_FAMILY_TO_TEMPLATE = {
    "YOLO11": "yolo11{size}",
    "YOLOv10": "yolov10{size}",
    "YOLOv9": "yolov9{size}",
    "YOLOv8": "yolov8{size}",
    "YOLOv5u": "yolov5{size}u",
    "YOLO26": "yolo26{size}",
}
MODEL_SIZES = ["n", "s", "m", "l", "x"]

TRAIN_PRESET_SCOPES = ("train", "val", "predict", "track")

VAL_SPECS = [
    {"key": "imgsz", "types": ["int"], "default": 640, "description": "验证图像尺寸，通常与训练尺寸保持一致。"},
    {"key": "batch", "types": ["int"], "default": 16, "description": "验证批次大小，显存不够时调小。"},
    {"key": "device", "types": ["string"], "default": "", "optional": True, "description": "运行设备，例如 cpu、0、0,1。"},
    {"key": "workers", "types": ["int"], "default": 8, "description": "数据加载线程数，机械硬盘或卡顿时可适当减小。"},
    {"key": "split", "types": ["string"], "default": "val", "options": ["val", "test", "train"], "description": "选择要评估的数据集划分。"},
    {"key": "save_json", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "把验证结果另存为 COCO JSON。"},
    {"key": "conf", "types": ["float"], "default": None, "optional": True, "tasks": ["detect", "segment", "pose", "obb"], "description": "验证时使用的置信度阈值，留空则用官方默认值。"},
    {"key": "iou", "types": ["float"], "default": 0.7, "tasks": ["detect", "segment", "pose", "obb"], "description": "NMS 使用的 IoU 阈值。"},
    {"key": "max_det", "types": ["int"], "default": 300, "tasks": ["detect", "segment", "pose", "obb"], "description": "每张图最多保留多少个检测结果。"},
    {"key": "half", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "验证时尝试使用半精度，显卡支持时更快。"},
    {"key": "dnn", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "仅在 ONNX 场景下使用 OpenCV DNN。"},
    {"key": "plots", "types": ["bool"], "default": True, "description": "保存 PR 曲线、混淆矩阵等图表。"},
    {"key": "rect", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "使用矩形批次，适合长宽比差异较大的图片。"},
    {"key": "project", "types": ["string"], "default": "", "optional": True, "description": "结果根目录，留空时默认写入 runs。"},
    {"key": "name", "types": ["string"], "default": "", "optional": True, "description": "当前验证任务的结果文件夹名称。"},
    {"key": "exist_ok", "types": ["bool"], "default": False, "description": "允许结果写入已存在的目录。"},
    {"key": "verbose", "types": ["bool"], "default": True, "description": "输出更详细的验证日志。"},
]

PREDICT_SPECS = [
    {"key": "imgsz", "types": ["int"], "default": 640, "description": "预测输入尺寸，越大越清晰但越吃显存。"},
    {"key": "conf", "types": ["float"], "default": 0.25, "description": "只保留高于该阈值的结果。"},
    {"key": "iou", "types": ["float"], "default": 0.7, "tasks": ["detect", "segment", "pose", "obb"], "description": "NMS 的 IoU 阈值。"},
    {"key": "max_det", "types": ["int"], "default": 300, "tasks": ["detect", "segment", "pose", "obb"], "description": "每张图最多输出多少个目标。"},
    {"key": "device", "types": ["string"], "default": "", "optional": True, "description": "运行设备，例如 cpu、0、0,1。"},
    {"key": "half", "types": ["bool"], "default": False, "description": "尝试用 FP16 推理。"},
    {"key": "augment", "types": ["bool"], "default": False, "description": "测试时增强，精度略高但速度更慢。"},
    {"key": "agnostic_nms", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "不同类别之间也执行 NMS。"},
    {"key": "retina_masks", "types": ["bool"], "default": False, "tasks": ["segment"], "description": "导出更细的分割掩码。"},
    {"key": "save", "types": ["bool"], "default": True, "description": "保存可视化结果到输出目录。"},
    {"key": "show", "types": ["bool"], "default": False, "description": "实时弹窗显示预测结果。"},
    {"key": "save_txt", "types": ["bool"], "default": False, "description": "额外保存 TXT 结果。"},
    {"key": "save_conf", "types": ["bool"], "default": False, "description": "保存 TXT 时附带置信度。"},
    {"key": "save_crop", "types": ["bool"], "default": False, "tasks": ["detect", "segment", "pose", "obb"], "description": "把检测到的目标单独裁剪保存。"},
    {"key": "classes", "types": ["int", "list"], "default": None, "optional": True, "description": "只保留指定类别，可填 0,1 或 [0,1]。"},
    {"key": "vid_stride", "types": ["int"], "default": 1, "description": "视频抽帧步长，2 表示隔帧处理。"},
    {"key": "stream_buffer", "types": ["bool"], "default": False, "description": "流媒体场景下缓存更多帧，避免丢帧。"},
    {"key": "line_width", "types": ["int"], "default": None, "optional": True, "description": "绘制框线宽，留空自动适配。"},
    {"key": "project", "types": ["string"], "default": "", "optional": True, "description": "结果根目录，留空时默认写入 runs。"},
    {"key": "name", "types": ["string"], "default": "", "optional": True, "description": "当前预测任务的结果文件夹名称。"},
    {"key": "exist_ok", "types": ["bool"], "default": False, "description": "允许结果写入已存在目录。"},
    {"key": "verbose", "types": ["bool"], "default": True, "description": "输出更详细的预测日志。"},
]

TRACK_SPECS = [
    {"key": "imgsz", "types": ["int"], "default": 640, "description": "跟踪输入尺寸，越大越清晰但越吃显存。"},
    {"key": "conf", "types": ["float"], "default": 0.1, "description": "只保留高于该阈值的结果。"},
    {"key": "iou", "types": ["float"], "default": 0.7, "description": "NMS 的 IoU 阈值。"},
    {"key": "device", "types": ["string"], "default": "", "optional": True, "description": "运行设备，例如 cpu、0、0,1。"},
    {"key": "half", "types": ["bool"], "default": False, "description": "尝试用 FP16 推理。"},
    {"key": "augment", "types": ["bool"], "default": False, "description": "测试时增强，精度略高但速度更慢。"},
    {"key": "agnostic_nms", "types": ["bool"], "default": False, "description": "不同类别之间也执行 NMS。"},
    {"key": "save", "types": ["bool"], "default": True, "description": "保存带跟踪 ID 的可视化结果。"},
    {"key": "show", "types": ["bool"], "default": False, "description": "实时弹窗显示跟踪结果。"},
    {"key": "save_txt", "types": ["bool"], "default": False, "description": "额外保存 TXT 结果。"},
    {"key": "save_conf", "types": ["bool"], "default": False, "description": "保存 TXT 时附带置信度。"},
    {"key": "save_crop", "types": ["bool"], "default": False, "description": "把跟踪到的目标单独裁剪保存。"},
    {"key": "classes", "types": ["int", "list"], "default": None, "optional": True, "description": "只保留指定类别，可填 0,1 或 [0,1]。"},
    {"key": "vid_stride", "types": ["int"], "default": 1, "description": "视频抽帧步长，2 表示隔帧处理。"},
    {"key": "tracker", "types": ["string"], "default": "botsort.yaml", "options": ["botsort.yaml", "bytetrack.yaml"], "description": "跟踪器选择：BoT-SORT 或 ByteTrack。"},
    {"key": "persist", "types": ["bool"], "default": True, "description": "持续沿用上一帧轨迹信息。"},
    {"key": "project", "types": ["string"], "default": "", "optional": True, "description": "结果根目录，留空时默认写入 runs。"},
    {"key": "name", "types": ["string"], "default": "", "optional": True, "description": "当前跟踪任务的结果文件夹名称。"},
    {"key": "exist_ok", "types": ["bool"], "default": False, "description": "允许结果写入已存在目录。"},
    {"key": "verbose", "types": ["bool"], "default": True, "description": "输出更详细的跟踪日志。"},
]

def resolve_resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


def resolve_work_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


RESOURCE_DIR = resolve_resource_dir()
WORK_DIR = resolve_work_dir()
BACKEND = RESOURCE_DIR / "backend.py"
CONTRACT_DIR = RESOURCE_DIR / "contracts"
PRESET_ROOT = WORK_DIR / "presets"
ICON_PNG = RESOURCE_DIR / "assets" / "yolotool_icon.png"
ICON_ICO = RESOURCE_DIR / "assets" / "yolotool_icon.ico"
BUNDLED_RUNTIME_DIR = WORK_DIR / "runtime" / "python"
BUNDLED_RUNTIME_PYTHON = BUNDLED_RUNTIME_DIR / "python.exe"
BACKEND_LAUNCH_FLAG = "--backend-command"
SUPPORTED_WINDOWS_PYTHON = ("3.13", "3.12", "3.11", "3.10", "3.9")


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundled_runtime_exists() -> bool:
    return BUNDLED_RUNTIME_PYTHON.exists()


def is_python_interpreter_path(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name in {"python.exe", "pythonw.exe", "py.exe", "py"}


def is_app_launcher_path(path: str | Path) -> bool:
    try:
        return Path(path).resolve() == Path(sys.executable).resolve() and is_frozen_app()
    except OSError:
        return False


def probe_python_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command + ["-c", "import json,sys; print(json.dumps({'exe': sys.executable, 'ver': list(sys.version_info[:2])}))"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads((result.stdout or "").strip().splitlines()[-1])
    except Exception:
        return None

    version = tuple(payload.get("ver") or [])
    executable = str(payload.get("exe") or "").strip()
    if version < (3, 9) or version > (3, 13):
        return None
    if not executable or not Path(executable).exists():
        return None
    if is_app_launcher_path(executable):
        return None
    if not is_python_interpreter_path(executable):
        return None
    return executable


def find_supported_system_python() -> str:
    candidates: list[list[str]] = []
    if shutil.which("py"):
        for version in SUPPORTED_WINDOWS_PYTHON:
            candidates.append(["py", f"-{version}"])
    if shutil.which("python"):
        candidates.append(["python"])
    if shutil.which("python3"):
        candidates.append(["python3"])

    seen: set[str] = set()
    for candidate in candidates:
        key = " ".join(candidate)
        if key in seen:
            continue
        seen.add(key)
        executable = probe_python_command(candidate)
        if executable:
            return executable
    return ""


def recommended_python_path() -> str:
    if bundled_runtime_exists():
        return str(BUNDLED_RUNTIME_PYTHON)
    system_python = find_supported_system_python()
    if system_python:
        return system_python
    if is_frozen_app():
        return str(BUNDLED_RUNTIME_PYTHON)
    return sys.executable


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON 对象。")
    return payload


def stringify_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def describe_types(allowed_types: list[str]) -> str:
    mapping = {
        "int": "整数",
        "float": "小数",
        "bool": "开关",
        "string": "文本",
        "list": "列表",
    }
    labels = [mapping[item] for item in allowed_types if item in mapping]
    return " / ".join(labels) if labels else "文本"


def describe_default_value(default: object, optional: bool) -> str:
    if default is None:
        return "可留空" if optional else "无默认值"
    rendered = stringify_value(default)
    return rendered if rendered else ("可留空" if optional else "空")


def build_spec_description(spec: dict[str, Any]) -> str:
    key = str(spec.get("key") or "").strip()
    optional = bool(spec.get("optional"))
    allowed_types = list(spec.get("types") or [])
    hint = PARAMETER_HINTS.get(key) or str(spec.get("description") or "").strip() or "按实际需要填写，拿不准时先保持默认。"

    parts = [
        hint,
        f"类型：{describe_types(allowed_types)}。",
        f"默认：{describe_default_value(spec.get('default'), optional)}。",
    ]

    options = spec.get("options") or []
    if options:
        rendered_options = "、".join(stringify_value(item) for item in options[:6])
        if len(options) > 6:
            rendered_options += "……"
        parts.append(f"可选值：{rendered_options}。")

    if key in PARAMETER_EXAMPLES:
        parts.append(f"{PARAMETER_EXAMPLES[key]}。")
    elif "bool" in allowed_types and allowed_types == ["bool"]:
        parts.append("填写方式：勾选=开，不勾选=关。")
    elif "list" in allowed_types:
        parts.append("列表可写成 `0,1` 或 `[0, 1]`。")

    return " ".join(part for part in parts if part)


def parse_scalar(text: str, allowed_types: list[str]) -> object:
    raw = text.strip()
    lowered = raw.lower()
    if "bool" in allowed_types and lowered in {"true", "false"}:
        return lowered == "true"
    if "list" in allowed_types and (raw.startswith("[") or "," in raw):
        if raw.startswith("["):
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError("列表参数必须是 JSON 数组。")
            return payload
        parsed: list[object] = []
        for item in [part.strip() for part in raw.split(",") if part.strip()]:
            if item.lower() in {"true", "false"}:
                parsed.append(item.lower() == "true")
                continue
            try:
                parsed.append(int(item))
                continue
            except ValueError:
                pass
            try:
                parsed.append(float(item))
                continue
            except ValueError:
                pass
            parsed.append(item)
        return parsed
    if "int" in allowed_types:
        try:
            if "." not in raw:
                return int(raw)
        except ValueError:
            pass
    if "float" in allowed_types:
        try:
            return float(raw)
        except ValueError:
            pass
    if "string" in allowed_types:
        return raw
    if "bool" in allowed_types:
        raise ValueError("布尔值请填写 true 或 false。")
    raise ValueError(f"无法识别参数值：{raw}")


class ToolTip:
    _open_instances: set["ToolTip"] = set()

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text.strip()
        self.window: tk.Toplevel | None = None
        if not self.text:
            return
        self.widget.bind("<Enter>", self._show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<Destroy>", self._hide, add="+")

    def _show(self, _: tk.Event | None = None) -> None:
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.window,
            text=self.text,
            bg=CARD_BG,
            fg=TEXT,
            relief="solid",
            bd=1,
            padx=8,
            pady=6,
            justify="left",
            wraplength=340,
            font=("Microsoft YaHei UI", 9),
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        label.pack()
        ToolTip._open_instances.add(self)

    def _hide(self, _: tk.Event | None = None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
        ToolTip._open_instances.discard(self)

    @classmethod
    def hide_all(cls) -> None:
        for instance in list(cls._open_instances):
            instance._hide()


class SmartComboBox(tk.Frame):
    _open_instances: set["SmartComboBox"] = set()
    _bound_root: tk.Misc | None = None

    def __init__(self, parent: tk.Widget, variable: tk.StringVar, values: list[str], *, readonly: bool = True) -> None:
        super().__init__(parent, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER, highlightcolor=PRIMARY)
        self.variable = variable
        self.values = list(values)
        self.readonly = readonly
        self.popup: tk.Toplevel | None = None
        self.listbox: tk.Listbox | None = None
        self._trace_id = self.variable.trace_add("write", lambda *_: self._sync_display())

        self.grid_columnconfigure(0, weight=1)
        self.display_var = tk.StringVar(value=self.variable.get())
        self.entry = tk.Entry(
            self,
            textvariable=self.display_var,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            readonlybackground=CARD_BG,
            font=("Microsoft YaHei UI", 10),
            insertbackground=TEXT,
            state="readonly" if readonly else "normal",
        )
        self.entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(8, 0))
        self.button = tk.Button(
            self,
            text="▾",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=4,
            font=("Microsoft YaHei UI", 10),
        )
        self.button.grid(row=0, column=1, sticky="ns")

        for widget in (self, self.entry, self.button):
            widget.bind("<Button-1>", self._open_from_click, add="+")
        self.bind("<Destroy>", lambda _event: self.close_popup(), add="+")

        self._ensure_root_binding()

    def _ensure_root_binding(self) -> None:
        root = self.winfo_toplevel()
        bound_root = SmartComboBox._bound_root
        if bound_root is not None:
            try:
                if bound_root.winfo_exists() and bound_root is root:
                    return
            except tk.TclError:
                pass
            SmartComboBox._bound_root = None
        if SmartComboBox._bound_root is root:
            return
        root.bind("<Configure>", lambda _event: SmartComboBox.close_all(), add="+")
        root.bind_all("<ButtonRelease-1>", SmartComboBox._close_on_global_click, add="+")
        SmartComboBox._bound_root = root

    @classmethod
    def _close_on_global_click(cls, event: tk.Event) -> None:
        target = event.widget
        for instance in list(cls._open_instances):
            if instance.popup is None:
                continue
            if str(target).startswith(str(instance)) or str(target).startswith(str(instance.popup)):
                continue
            instance.close_popup()

    def _sync_display(self) -> None:
        self.display_var.set(self.variable.get())
        if self.popup is not None:
            self._sync_selection()

    def _sync_selection(self) -> None:
        if self.listbox is None:
            return
        current = self.variable.get()
        self.listbox.selection_clear(0, "end")
        if current in self.values:
            index = self.values.index(current)
            self.listbox.selection_set(index)
            self.listbox.see(index)

    def _open_from_click(self, _event: tk.Event | None = None) -> str:
        self.toggle_popup()
        return "break"

    def toggle_popup(self) -> None:
        if self.popup is not None:
            self.close_popup()
        else:
            self.open_popup()

    def open_popup(self) -> None:
        if not self.values or not self.winfo_ismapped():
            return
        SmartComboBox.close_all(except_instance=self)
        popup = tk.Toplevel(self.winfo_toplevel())
        popup.wm_overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.attributes("-topmost", True)
        frame = tk.Frame(popup, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(fill="both", expand=True)
        height = min(max(len(self.values), 1), 8)
        listbox = tk.Listbox(
            frame,
            height=height,
            relief="flat",
            bd=0,
            activestyle="none",
            exportselection=False,
            bg=CARD_BG,
            fg=TEXT,
            selectbackground=PRIMARY,
            selectforeground="white",
            font=("Microsoft YaHei UI", 10),
        )
        listbox.pack(fill="both", expand=True, padx=1, pady=1)
        for value in self.values:
            listbox.insert("end", value)
        self.popup = popup
        self.listbox = listbox
        self._sync_selection()
        listbox.bind("<ButtonRelease-1>", self._confirm_selection, add="+")
        listbox.bind("<Double-Button-1>", self._confirm_selection, add="+")
        listbox.bind("<Return>", self._confirm_selection, add="+")
        listbox.bind("<Escape>", lambda _e: self.close_popup(), add="+")
        for widget in (popup,):
            widget.bind("<MouseWheel>", self._forward_mousewheel, add="+")
            widget.bind("<Button-4>", self._forward_mousewheel, add="+")
            widget.bind("<Button-5>", self._forward_mousewheel, add="+")
        for widget in (listbox,):
            widget.bind("<MouseWheel>", self._scroll_popup_listbox, add="+")
            widget.bind("<Button-4>", self._scroll_popup_listbox, add="+")
            widget.bind("<Button-5>", self._scroll_popup_listbox, add="+")
        self.reposition_popup()
        listbox.focus_set()
        SmartComboBox._open_instances.add(self)

    def _confirm_selection(self, _event: tk.Event | None = None) -> str:
        if self.listbox is None:
            return "break"
        selection = self.listbox.curselection()
        if not selection:
            return "break"
        self.variable.set(str(self.listbox.get(selection[0])))
        self.event_generate("<<ComboboxSelected>>")
        self.close_popup()
        return "break"

    def _find_scrollable_parent(self) -> "ScrollableFrame | None":
        parent = self.master
        while parent is not None:
            if isinstance(parent, ScrollableFrame):
                return parent
            parent = parent.master
        return None

    def _forward_mousewheel(self, event: tk.Event) -> str | None:
        scrollable = self._find_scrollable_parent()
        if scrollable is None:
            return None
        return scrollable._on_mousewheel(event)

    def _scroll_popup_listbox(self, event: tk.Event) -> str | None:
        if self.listbox is None:
            return "break"
        visible_rows = int(self.listbox.cget("height") or 0)
        if self.listbox.size() <= visible_rows:
            return self._forward_mousewheel(event)

        delta = 0
        if getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if not delta:
            return "break"
        self.listbox.yview_scroll(delta, "units")
        return "break"

    def reposition_popup(self) -> None:
        if self.popup is None:
            return
        if not self.winfo_ismapped():
            self.close_popup()
            return
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        width = max(self.winfo_width(), 120)
        height = min(max(len(self.values), 1), 8) * 28 + 4
        self.popup.geometry(f"{width}x{height}+{x}+{y}")

    def close_popup(self) -> None:
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
        self.popup = None
        self.listbox = None
        SmartComboBox._open_instances.discard(self)

    @classmethod
    def reposition_all(cls) -> None:
        for instance in list(cls._open_instances):
            instance.reposition_popup()

    @classmethod
    def close_all(cls, except_instance: "SmartComboBox | None" = None) -> None:
        for instance in list(cls._open_instances):
            if instance is except_instance:
                continue
            instance.close_popup()

    def configure(self, cnf: dict | None = None, **kwargs) -> None:
        options = {}
        if cnf:
            options.update(cnf)
        options.update(kwargs)
        if "values" in options:
            self.values = list(options.pop("values"))
            self._sync_selection()
        if "state" in options:
            self.readonly = options.pop("state") == "readonly"
            self.entry.configure(state="readonly" if self.readonly else "normal")
        super().configure(**options)

    config = configure


class ScrollableFrame(tk.Frame):
    _active_instance: "ScrollableFrame | None" = None
    _bound_root: tk.Misc | None = None

    def __init__(self, parent: tk.Widget, *, background: str) -> None:
        super().__init__(parent, bg=background)
        self.canvas = tk.Canvas(self, bg=background, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._scroll_canvas, style="App.Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=background)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_width)
        for widget in (self, self.canvas, self.inner):
            widget.bind("<Enter>", self._bind_mousewheel, add="+")
            widget.bind("<Leave>", self._unbind_mousewheel, add="+")
        self._ensure_global_mousewheel_binding()

    @classmethod
    def _resolve_from_widget(cls, widget: tk.Widget | None) -> "ScrollableFrame | None":
        current = widget
        while current is not None:
            if isinstance(current, cls):
                return current
            current = getattr(current, "master", None)
        return None

    def _ensure_global_mousewheel_binding(self) -> None:
        root = self.winfo_toplevel()
        bound_root = ScrollableFrame._bound_root
        if bound_root is not None:
            try:
                if bound_root.winfo_exists() and bound_root is root:
                    return
            except tk.TclError:
                pass
            ScrollableFrame._bound_root = None
        if ScrollableFrame._bound_root is root:
            return
        root.bind_all("<MouseWheel>", ScrollableFrame._dispatch_mousewheel, add="+")
        root.bind_all("<Button-4>", ScrollableFrame._dispatch_mousewheel, add="+")
        root.bind_all("<Button-5>", ScrollableFrame._dispatch_mousewheel, add="+")
        ScrollableFrame._bound_root = root

    @classmethod
    def _dispatch_mousewheel(cls, event: tk.Event) -> str | None:
        instance = cls._resolve_from_widget(getattr(event, "widget", None))
        if instance is None:
            instance = cls._active_instance
        if instance is None:
            return None
        return instance._on_mousewheel(event)

    def _on_canvas_scroll(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        ToolTip.hide_all()
        SmartComboBox.reposition_all()

    def _scroll_canvas(self, *args: object) -> None:
        ToolTip.hide_all()
        self.canvas.yview(*args)

    def _bind_mousewheel(self, _: tk.Event | None = None) -> None:
        ScrollableFrame._active_instance = self

    def _unbind_mousewheel(self, _: tk.Event | None = None) -> None:
        pointer_x = self.winfo_pointerx()
        pointer_y = self.winfo_pointery()
        widget_under_pointer = self.winfo_containing(pointer_x, pointer_y)
        if widget_under_pointer and str(widget_under_pointer).startswith(str(self)):
            return
        if ScrollableFrame._active_instance is self:
            ScrollableFrame._active_instance = None

    def _sync_scrollregion(self, _: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        ToolTip.hide_all()
        SmartComboBox.reposition_all()

    def _sync_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)
        ToolTip.hide_all()
        SmartComboBox.reposition_all()

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        if ScrollableFrame._active_instance not in {None, self}:
            return None
        if not self.winfo_ismapped():
            return None
        delta = 0
        if getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if delta:
            self.canvas.yview_scroll(delta, "units")
            ToolTip.hide_all()
            return "break"
        return None


class AccordionSection(tk.Frame):
    def __init__(self, parent: tk.Widget, title: str, *, expanded: bool = False) -> None:
        super().__init__(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        self.expanded = expanded

        self.header = tk.Frame(self, bg=CARD_BG, height=42)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        self.header.grid_columnconfigure(1, weight=1)

        self.accent = tk.Frame(self.header, width=4, bg=PRIMARY if expanded else CARD_BG)
        self.accent.grid(row=0, column=0, sticky="ns")

        self.title_label = tk.Label(
            self.header,
            text=title,
            bg=PRIMARY_SOFT if expanded else CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
            padx=12,
        )
        self.title_label.grid(row=0, column=1, sticky="nsew")

        header_bg = PRIMARY_SOFT if expanded else CARD_BG
        self.arrow_box = tk.Frame(self.header, bg=header_bg, width=46)
        self.arrow_box.grid(row=0, column=2, sticky="nsew")
        self.arrow_box.grid_propagate(False)

        self.arrow_label = tk.Label(
            self.arrow_box,
            text="▾" if expanded else "▸",
            bg=header_bg,
            fg=PRIMARY_DARK if expanded else TEXT_MUTED,
            font=("Microsoft YaHei UI", 12, "bold"),
            bd=0,
            relief="flat",
            anchor="center",
        )
        self.arrow_label.pack(fill="both", expand=True)

        self.body = tk.Frame(self, bg=CARD_BG)
        if expanded:
            self.body.pack(fill="both", expand=True)

        for widget in (self.header, self.title_label, self.arrow_box, self.arrow_label, self.accent):
            widget.bind("<Button-1>", self._toggle, add="+")

    def _toggle(self, _: tk.Event | None = None) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, expanded: bool) -> None:
        if self.expanded == expanded:
            return
        SmartComboBox.close_all()
        self.expanded = expanded
        if expanded:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()
        header_bg = PRIMARY_SOFT if self.expanded else CARD_BG
        self.accent.configure(bg=PRIMARY if self.expanded else CARD_BG)
        self.title_label.configure(bg=header_bg)
        self.arrow_box.configure(bg=header_bg)
        self.arrow_label.configure(
            text="▾" if self.expanded else "▸",
            bg=header_bg,
            fg=PRIMARY_DARK if self.expanded else TEXT_MUTED,
        )


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1600x980")
        self.root.minsize(1320, 860)
        self.root.configure(bg=WINDOW_BG)

        self.train_contract = load_json(CONTRACT_DIR / "train_capabilities.json")
        self.export_contract = load_json(CONTRACT_DIR / "export_capabilities.json")

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._log_after_id: str | None = None
        self._visible_log_lines = 0
        self._log_sequence = 0
        self.temp_files: list[Path] = []
        self.current_log_path: Path | None = None
        self.current_log_handle: object | None = None
        self.last_result_path: Path | None = None

        self.active_tab = tk.StringVar(value="train")
        self.train_action_var = tk.StringVar(value="train")

        self.python_var = tk.StringVar(value=str(BUNDLED_RUNTIME_PYTHON) if is_frozen_app() else sys.executable)

        self.process_status_var = tk.StringVar(value="等待开始")
        self.left_log_state_var = tk.StringVar(value="暂无")
        self.left_result_var = tk.StringVar(value="暂无结果")
        self.result_location_var = tk.StringVar(value="")

        self.summary_label1 = tk.StringVar(value="数据集")
        self.summary_label2 = tk.StringVar(value="模型")
        self.summary_label3 = tk.StringVar(value="预计输出目录")
        self.summary_value1 = tk.StringVar(value="未选择")
        self.summary_value2 = tk.StringVar(value="yolov5nu.pt")
        self.summary_value3 = tk.StringVar(value=str(WORK_DIR / "runs" / "detect" / "train"))

        self.selected_train_action_label_var = tk.StringVar(value=ACTION_ID_TO_LABEL["train"])
        self.export_mode_label_var = tk.StringVar(value="导出")

        self.train_task_label_var = tk.StringVar(value="目标检测")
        self.train_mode_label_var = tk.StringVar(value="官方预训练")
        self.train_family_var = tk.StringVar(value="YOLO11")
        self.train_size_var = tk.StringVar(value="n")
        self.train_use_local_weights_var = tk.BooleanVar(value=False)
        self.train_model_var = tk.StringVar(value="yolov5nu.pt")
        self.train_data_var = tk.StringVar(value="暂无")

        self.val_task_label_var = tk.StringVar(value="目标检测")
        self.val_weights_var = tk.StringVar(value="未选择")
        self.val_data_var = tk.StringVar(value="暂无")

        self.predict_task_label_var = tk.StringVar(value="目标检测")
        self.predict_weights_var = tk.StringVar(value="未选择")
        self.predict_source_var = tk.StringVar(value="")

        self.track_task_label_var = tk.StringVar(value="目标检测")
        self.track_weights_var = tk.StringVar(value="未选择")
        self.track_source_var = tk.StringVar(value="")

        self.prep_input_var = tk.StringVar(value="暂无")
        self.prep_format_var = tk.StringVar(value="自动判断")
        self.prep_output_preview_var = tk.StringVar(value="暂无")
        self.prep_val_ratio_var = tk.StringVar(value="0.2")
        self.prep_seed_var = tk.StringVar(value="42")
        self.prep_copy_mode_var = tk.StringVar(value="复制文件")
        self.prep_class_names_file_var = tk.StringVar(value="")
        self.prep_strict_var = tk.BooleanVar(value=False)
        self.prep_overwrite_var = tk.BooleanVar(value=False)

        self.export_task_label_var = tk.StringVar(value="自动识别")
        self.export_weights_var = tk.StringVar(value="未选择")
        self.export_output_dir_var = tk.StringVar(value="")
        self.export_format_label_var = tk.StringVar()
        self.export_imgsz_var = tk.StringVar(value="640")
        self.export_device_var = tk.StringVar(value="0")

        self.train_preset_var = tk.StringVar(value="")
        self.export_preset_var = tk.StringVar(value="")

        self.train_fields: dict[str, dict[str, Any]] = {}
        self.val_fields: dict[str, dict[str, Any]] = {}
        self.predict_fields: dict[str, dict[str, Any]] = {}
        self.track_fields: dict[str, dict[str, Any]] = {}
        self.export_fields: dict[str, dict[str, Any]] = {}

        self.train_group_meta: dict[str, dict[str, Any]] = {}
        self.train_section_modes: dict[str, set[str]] = {}

        self.export_format_map = {item["label"]: item["id"] for item in self.export_contract.get("formats", [])}
        self.export_label_map = {item["id"]: item["label"] for item in self.export_contract.get("formats", [])}
        first_export_label = self.export_contract["formats"][1]["label"]
        self.export_format_label_var.set(first_export_label)

        self.train_action_buttons: dict[str, tk.Button] = {}
        self.train_preset_combo: SmartComboBox | None = None
        self.export_preset_combo: SmartComboBox | None = None
        self.train_start_button: tk.Button | None = None
        self.export_start_button: tk.Button | None = None
        self.train_stop_button: tk.Button | None = None
        self.export_stop_button: tk.Button | None = None
        self.train_open_result_button: tk.Button | None = None
        self.export_open_result_button: tk.Button | None = None
        self.train_open_log_button: tk.Button | None = None
        self.export_open_log_button: tk.Button | None = None
        self.train_preset_buttons: dict[str, tk.Button] = {}
        self.export_preset_buttons: dict[str, tk.Button] = {}
        self.process_status_labels: list[tk.Label] = []
        self.left_result_entry: tk.Entry | None = None
        self.left_log_state_entry: tk.Entry | None = None
        self.train_recommended_preset_var = tk.StringVar(value="")
        self.export_recommended_preset_var = tk.StringVar(value="")

        self._configure_styles()
        self._build_ui()
        self._bind_traces()

        self._refresh_model_name()
        self._on_local_weights_toggle()
        self._refresh_prep_output_preview()
        self._refresh_train_task_visibility()
        self._refresh_action_task_visibility("val")
        self._refresh_action_task_visibility("predict")
        self._refresh_action_task_visibility("track")
        self._refresh_export_visibility()
        self._refresh_train_preset_choices()
        self._refresh_export_preset_choices()
        self._show_train_action("train")
        self._show_tab("train")
        self._set_log_placeholder("train")
        self._refresh_summary()
        self._refresh_run_action_buttons()
        self._refresh_status_visuals()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "App.TCombobox",
            padding=7,
            foreground=TEXT,
            fieldbackground=CARD_BG,
            background=CARD_BG,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            arrowcolor=TEXT,
            relief="flat",
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[("readonly", CARD_BG)],
            background=[("readonly", CARD_BG), ("active", CARD_BG)],
            foreground=[("readonly", TEXT)],
            bordercolor=[("readonly", BORDER), ("focus", PRIMARY)],
            lightcolor=[("focus", PRIMARY), ("readonly", BORDER)],
            darkcolor=[("focus", PRIMARY), ("readonly", BORDER)],
            arrowcolor=[("readonly", TEXT), ("active", TEXT)],
        )
        style.configure(
            "App.Vertical.TScrollbar",
            troughcolor=PANEL_BG,
            background=PRIMARY_SOFT,
            bordercolor=PANEL_BG,
            arrowcolor=TEXT_MUTED,
            lightcolor=PRIMARY_SOFT,
            darkcolor=PRIMARY_SOFT,
            relief="flat",
        )
        style.map("App.Vertical.TScrollbar", background=[("active", PRIMARY_SOFT)])
        style.configure("TCheckbutton", background=CARD_BG, foreground=TEXT)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._build_shell_header()

        self.content = tk.Frame(self.root, bg=WINDOW_BG, padx=16, pady=0)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)

        footer = tk.Label(
            self.root,
            text="设计关键词：简约、留白、低干扰、参数解释清晰、适合新手操作",
            bg=WINDOW_BG,
            fg=TEXT_MUTED,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
            padx=18,
            pady=10,
        )
        footer.grid(row=2, column=0, sticky="ew")

        self.left_panel = tk.Frame(self.content, bg=CARD_BG, padx=14, pady=14, highlightbackground=BORDER, highlightthickness=1)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_rowconfigure(2, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.right_panel = tk.Frame(self.content, bg=CARD_BG, padx=14, pady=14, highlightbackground=BORDER, highlightthickness=1)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_shell_header(self) -> None:
        header = tk.Frame(self.root, bg=WINDOW_BG, height=52)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)
        header.pack_propagate(False)

        tk.Label(
            header,
            text=APP_NAME,
            bg=WINDOW_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        dot_bar = tk.Frame(header, bg=WINDOW_BG)
        dot_bar.grid(row=0, column=1, sticky="e")
        for _ in range(3):
            tk.Canvas(dot_bar, width=12, height=12, bg=WINDOW_BG, highlightthickness=0, bd=0).pack(side="left", padx=4)
            dot_canvas = dot_bar.winfo_children()[-1]
            dot_canvas.create_oval(1, 1, 11, 11, fill=BORDER, outline=BORDER)

    def _build_left_panel(self) -> None:
        current_box = self._create_box(self.left_panel, "当前入口")
        current_box["frame"].grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._create_summary_row(current_box["body"], self.summary_label1, self.summary_value1, 0)
        self._create_summary_row(current_box["body"], self.summary_label2, self.summary_value2, 1)
        self._create_summary_row(current_box["body"], self.summary_label3, self.summary_value3, 2)

        result_box = self._create_box(self.left_panel, "最近结果")
        result_box["frame"].grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.left_result_entry = self._create_summary_row(result_box["body"], tk.StringVar(value="结果"), self.left_result_var, 0)
        ToolTip(self.left_result_entry, "这里显示最近一次成功任务的输出位置或结果摘要。")

        log_box = self._create_box(self.left_panel, "实时日志")
        log_box["frame"].grid(row=2, column=0, sticky="nsew")
        log_box["body"].grid_rowconfigure(1, weight=1)
        log_box["body"].grid_columnconfigure(1, weight=1)
        self.left_log_state_entry = self._create_summary_row(log_box["body"], tk.StringVar(value="当前日志"), self.left_log_state_var, 0)
        ToolTip(self.left_log_state_entry, "这里显示当前任务的日志状态，例如等待开始、运行中、已完成或任务失败。")

        self.log_text = scrolledtext.ScrolledText(
            log_box["body"],
            wrap="word",
            bg=CARD_SOFT,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(6, 12))
        self.log_text.configure(state="disabled")

    def _build_right_panel(self) -> None:
        tab_bar = tk.Frame(self.right_panel, bg=CARD_BG)
        tab_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab_bar.grid_columnconfigure(0, weight=1)
        tab_bar.grid_columnconfigure(1, weight=1)

        self.train_tab_button = tk.Button(
            tab_bar,
            text="训练工作台",
            command=lambda: self._show_tab("train"),
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
            pady=12,
            cursor="hand2",
        )
        self.train_tab_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.export_tab_button = tk.Button(
            tab_bar,
            text="导出工作台",
            command=lambda: self._show_tab("export"),
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
            pady=12,
            cursor="hand2",
        )
        self.export_tab_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.train_scroll = ScrollableFrame(self.right_panel, background=CARD_BG)
        self.train_scroll.grid(row=1, column=0, sticky="nsew")

        self.export_scroll = ScrollableFrame(self.right_panel, background=CARD_BG)
        self.export_scroll.grid(row=1, column=0, sticky="nsew")

        self._build_train_view(self.train_scroll.inner)
        self._build_export_view(self.export_scroll.inner)

    def _build_train_view(self, parent: tk.Widget) -> None:
        parent.grid_columnconfigure(0, weight=1)
        row = 0

        self._create_workspace_banner(parent, row, "玻璃轻拟态 · 新手友好布局")
        row += 1

        action_bar = tk.Frame(parent, bg=CARD_BG)
        action_bar.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        for index, action_id in enumerate(("train", "val", "predict", "track")):
            action_bar.grid_columnconfigure(index, weight=1)
            button = tk.Button(
                action_bar,
                text=ACTION_ID_TO_LABEL[action_id],
                command=lambda item=action_id: self._show_train_action(item),
                bg=CARD_BG,
                fg=TEXT,
                activebackground=PRIMARY_SOFT,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=PRIMARY,
                font=("Microsoft YaHei UI", 10, "bold"),
                pady=8,
                cursor="hand2",
            )
            button.grid(row=0, column=index, sticky="ew", padx=(0, 6) if index < 3 else 0)
            self.train_action_buttons[action_id] = button
        row += 1

        self.train_sections: dict[str, AccordionSection] = {}

        self.train_sections["env"] = self._create_section(parent, row, "运行环境", True)
        self.train_section_modes["env"] = {"train", "val", "predict", "track"}
        self._build_env_section(self.train_sections["env"].body, scope="train")
        row += 1

        self.train_sections["train_entry"] = self._create_section(parent, row, "训练入口", True)
        self.train_section_modes["train_entry"] = {"train"}
        self._build_train_entry_section(self.train_sections["train_entry"].body)
        row += 1

        self.train_sections["val_entry"] = self._create_section(parent, row, "验证入口", True)
        self.train_section_modes["val_entry"] = {"val"}
        self._build_val_entry_section(self.train_sections["val_entry"].body)
        row += 1

        self.train_sections["predict_entry"] = self._create_section(parent, row, "预测入口", True)
        self.train_section_modes["predict_entry"] = {"predict"}
        self._build_predict_entry_section(self.train_sections["predict_entry"].body)
        row += 1

        self.train_sections["track_entry"] = self._create_section(parent, row, "跟踪入口", True)
        self.train_section_modes["track_entry"] = {"track"}
        self._build_track_entry_section(self.train_sections["track_entry"].body)
        row += 1

        self.train_sections["prep"] = self._create_section(parent, row, "整理原始检测数据")
        self.train_section_modes["prep"] = {"train"}
        self._build_prep_section(self.train_sections["prep"].body)
        row += 1

        for group in self.train_contract.get("groups", []):
            section_id = str(group.get("id") or f"group_{row}")
            section = self._create_section(parent, row, str(group.get("label") or section_id))
            self.train_sections[section_id] = section
            self.train_section_modes[section_id] = {"train"}
            self._build_train_group_section(section.body, group)
            row += 1

        self.train_sections["val_params"] = self._create_section(parent, row, "验证参数")
        self.train_section_modes["val_params"] = {"val"}
        self._build_param_group_section(self.train_sections["val_params"].body, VAL_SPECS, self.val_fields, "val")
        row += 1

        self.train_sections["predict_params"] = self._create_section(parent, row, "预测参数")
        self.train_section_modes["predict_params"] = {"predict"}
        self._build_param_group_section(self.train_sections["predict_params"].body, PREDICT_SPECS, self.predict_fields, "predict")
        row += 1

        self.train_sections["track_params"] = self._create_section(parent, row, "跟踪参数")
        self.train_section_modes["track_params"] = {"track"}
        self._build_param_group_section(self.train_sections["track_params"].body, TRACK_SPECS, self.track_fields, "track")
        row += 1

        self.train_sections["run"] = self._create_section(parent, row, "运行操作")
        self.train_section_modes["run"] = {"train", "val", "predict", "track"}
        self._build_run_section(self.train_sections["run"].body, scope="train")

    def _build_export_view(self, parent: tk.Widget) -> None:
        parent.grid_columnconfigure(0, weight=1)
        row = 0

        self._create_workspace_banner(parent, row, "玻璃轻拟态 · 导出工作台")
        row += 1

        self.export_sections: dict[str, AccordionSection] = {}

        self.export_sections["env"] = self._create_section(parent, row, "运行环境", True)
        self._build_env_section(self.export_sections["env"].body, scope="export")
        row += 1

        self.export_sections["entry"] = self._create_section(parent, row, "导出入口", True)
        self._build_export_entry_section(self.export_sections["entry"].body)
        row += 1

        self.export_sections["params"] = self._create_section(parent, row, "官方导出参数")
        self._build_export_params_section(self.export_sections["params"].body)
        row += 1

        self.export_sections["run"] = self._create_section(parent, row, "运行操作")
        self._build_run_section(self.export_sections["run"].body, scope="export")

    def _create_section(self, parent: tk.Widget, row: int, title: str, expanded: bool = False) -> AccordionSection:
        section = AccordionSection(parent, title, expanded=expanded)
        section.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        return section

    def _create_workspace_banner(self, parent: tk.Widget, row: int, title: str) -> None:
        banner = tk.Frame(parent, bg=PRIMARY_SOFT, highlightbackground=BORDER, highlightthickness=1)
        banner.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        banner.grid_columnconfigure(0, weight=1)
        tk.Label(
            banner,
            text=title,
            bg=PRIMARY_SOFT,
            fg=TEXT,
            font=("Microsoft YaHei UI", 15, "bold"),
            anchor="w",
            padx=18,
            pady=12,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            banner,
            text="v2",
            bg=PRIMARY_SOFT,
            fg=PRIMARY,
            font=("Microsoft YaHei UI", 9),
            padx=12,
        ).grid(row=0, column=1, sticky="e")

    def _create_box(self, parent: tk.Widget, title: str) -> dict[str, tk.Widget]:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        frame.grid_columnconfigure(0, weight=1)

        header = tk.Frame(frame, bg=PRIMARY_SOFT, height=38)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        accent = tk.Frame(header, bg=PRIMARY, width=4)
        accent.pack(side="left", fill="y")

        title_label = tk.Label(
            header,
            text=title,
            bg=PRIMARY_SOFT,
            fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
            padx=10,
        )
        title_label.pack(side="left", fill="both", expand=True)

        body = tk.Frame(frame, bg=CARD_BG)
        body.grid(row=1, column=0, sticky="nsew")
        return {"frame": frame, "body": body}

    def _create_summary_row(self, parent: tk.Widget, label_var: tk.StringVar, value_var: tk.StringVar, row: int) -> tk.Entry:
        parent.grid_columnconfigure(1, weight=1)
        label = tk.Label(
            parent,
            textvariable=label_var,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10),
            anchor="w",
        )
        label.grid(row=row, column=0, sticky="w", padx=(14, 8), pady=6)

        entry = tk.Entry(
            parent,
            textvariable=value_var,
            relief="flat",
            bd=0,
            bg=CARD_SOFT,
            readonlybackground=CARD_SOFT,
            fg=TEXT,
            font=("Microsoft YaHei UI", 10),
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            insertbackground=TEXT,
        )
        entry.configure(state="readonly")
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 14), pady=6, ipady=6)
        return entry

    def _build_env_section(self, parent: tk.Widget, *, scope: str) -> None:
        check_text = "检查当前环境"
        install_button = tk.Button(
            parent,
            text="在线一键配置环境（自动识别 CPU / NVIDIA）",
            command=self.start_configure_environment,
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=PRIMARY,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
            pady=9,
            cursor="hand2",
        )
        install_button.pack(fill="x", padx=14, pady=(0, 8))
        ToolTip(
            install_button,
            "只会检查并配置软件内置 runtime。缺少内置环境时会自动在线创建，不会往别的路径里安装环境。",
        )

        check_button = tk.Button(
            parent,
            text=check_text,
            command=self.start_check_environment,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11),
            pady=8,
            cursor="hand2",
        )
        check_button.pack(fill="x", padx=14, pady=(0, 8))
        ToolTip(check_button, "检查的是软件当前内置 runtime 是否完整可用，并把版本信息输出到左侧日志。")

    def _build_train_entry_section(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10, pady=10)
        for column in range(2):
            grid.grid_columnconfigure(column * 2 + 1, weight=1)

        self._create_form_row(
            grid,
            row=0,
            label="任务类型",
            widget=self._create_combo(grid, self.train_task_label_var, list(TASK_LABEL_TO_ID.keys())),
            description="选择要训练的任务类型，会自动过滤不适用的参数。",
        )
        self._create_form_row(
            grid,
            row=1,
            label="训练模式",
            widget=self._create_combo(grid, self.train_mode_label_var, ["官方预训练", "本地权重继续训练"]),
            description="官方预训练适合从头开新项目；本地权重继续训练适合断点续训或迁移训练。",
        )
        self._create_form_row(
            grid,
            row=2,
            label="模型系列",
            widget=self._create_combo(grid, self.train_family_var, list(MODEL_FAMILY_TO_TEMPLATE.keys())),
            description="这里决定基础模型族，下拉顺序已按更常用、更容易上手的系列重新排好；拿不准时优先选 YOLO11。",
        )
        self._create_form_row(
            grid,
            row=3,
            label="模型尺寸",
            widget=self._create_combo(grid, self.train_size_var, MODEL_SIZES),
            description="n 最轻，x 最大。显存不够优先选 n / s。",
        )

        check_row = tk.Frame(grid, bg=PANEL_BG)
        self.train_use_local_weights_check = ttk.Checkbutton(check_row, variable=self.train_use_local_weights_var)
        self.train_use_local_weights_check.pack(side="left")
        self._create_form_row(
            grid,
            row=4,
            label="使用本地权重",
            widget=check_row,
            description="勾选后可手动指定已有 .pt 权重文件。",
        )

        model_row = tk.Frame(grid, bg=PANEL_BG)
        model_row.grid_columnconfigure(0, weight=1)
        self.train_model_entry = self._create_entry(model_row, self.train_model_var, readonly=True)
        self.train_model_entry.grid(row=0, column=0, sticky="ew")
        self.train_model_browse_button = self._create_small_button(model_row, "浏览", self._pick_train_model)
        self.train_model_browse_button.grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(
            grid,
            row=5,
            label="当前模型",
            widget=model_row,
            description="不勾选本地权重时，会自动按系列 + 尺寸 + 任务拼出官方模型名。",
        )

        data_row = tk.Frame(grid, bg=PANEL_BG)
        data_row.grid_columnconfigure(0, weight=1)
        self.train_data_entry = self._create_entry(data_row, self.train_data_var, readonly=True)
        self.train_data_entry.grid(row=0, column=0, sticky="ew")
        self._create_small_button(data_row, "选择", self._pick_train_dataset).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(
            grid,
            row=6,
            label="训练入口",
            widget=data_row,
            description="detect / segment / pose / obb 请选择 dataset.yaml；classify 请选择分类数据集根目录。",
        )

        hint = tk.Label(
            parent,
            text="detect 可以直接选择正式 dataset.yaml，也可以先在“整理原始检测数据”区把 YOLO TXT / LabelMe 标注 JSON / COCO JSON 转成正式训练结构。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            wraplength=430,
            justify="left",
            font=("Microsoft YaHei UI", 10),
        )
        hint.pack(fill="x", padx=14, pady=(0, 12))

    def _build_val_entry_section(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10, pady=10)
        grid.grid_columnconfigure(1, weight=1)

        self._create_form_row(
            grid,
            row=0,
            label="任务类型",
            widget=self._create_combo(grid, self.val_task_label_var, list(TASK_LABEL_TO_ID.keys())),
            description="验证任务必须与权重实际任务对应，否则指标会不正确。",
        )

        weight_row = tk.Frame(grid, bg=PANEL_BG)
        weight_row.grid_columnconfigure(0, weight=1)
        self._create_entry(weight_row, self.val_weights_var, readonly=True).grid(row=0, column=0, sticky="ew")
        self._create_small_button(weight_row, "浏览", self._pick_val_weights).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(grid, row=1, label="验证权重", widget=weight_row, description="选择已经训练好的 .pt 权重。")

        data_row = tk.Frame(grid, bg=PANEL_BG)
        data_row.grid_columnconfigure(0, weight=1)
        self._create_entry(data_row, self.val_data_var, readonly=True).grid(row=0, column=0, sticky="ew")
        self._create_small_button(data_row, "选择", self._pick_val_dataset).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(
            grid,
            row=2,
            label="验证数据",
            widget=data_row,
            description="detect / segment / pose / obb 请选择 dataset.yaml；classify 请选择分类数据集根目录。",
        )

    def _build_predict_entry_section(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10, pady=10)
        grid.grid_columnconfigure(1, weight=1)

        self._create_form_row(
            grid,
            row=0,
            label="任务类型",
            widget=self._create_combo(grid, self.predict_task_label_var, list(TASK_LABEL_TO_ID.keys())),
            description="预测任务必须与权重实际任务对应。",
        )

        weight_row = tk.Frame(grid, bg=PANEL_BG)
        weight_row.grid_columnconfigure(0, weight=1)
        self._create_entry(weight_row, self.predict_weights_var, readonly=True).grid(row=0, column=0, sticky="ew")
        self._create_small_button(weight_row, "浏览", self._pick_predict_weights).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(grid, row=1, label="预测权重", widget=weight_row, description="选择已经训练好的 .pt 权重。")

        source_row = tk.Frame(grid, bg=PANEL_BG)
        source_row.grid_columnconfigure(0, weight=1)
        self._create_entry(source_row, self.predict_source_var, readonly=False).grid(row=0, column=0, sticky="ew")
        self._create_small_button(source_row, "文件", lambda: self._pick_source_file(self.predict_source_var)).grid(row=0, column=1, padx=(8, 0))
        self._create_small_button(source_row, "目录", lambda: self._pick_source_dir(self.predict_source_var)).grid(row=0, column=2, padx=(8, 0))
        self._create_small_button(source_row, "清空", lambda: self.predict_source_var.set("")).grid(row=0, column=3, padx=(8, 0))
        self._create_form_row(
            grid,
            row=2,
            label="预测源",
            widget=source_row,
            description="可填图片、视频、文件夹、摄像头编号（如 0）、RTSP/HTTP 流地址。",
        )

    def _build_track_entry_section(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10, pady=10)
        grid.grid_columnconfigure(1, weight=1)

        self._create_form_row(
            grid,
            row=0,
            label="任务类型",
            widget=self._create_combo(grid, self.track_task_label_var, TRACK_TASK_LABELS),
            description="跟踪只支持 detect / segment / pose / obb。",
        )

        weight_row = tk.Frame(grid, bg=PANEL_BG)
        weight_row.grid_columnconfigure(0, weight=1)
        self._create_entry(weight_row, self.track_weights_var, readonly=True).grid(row=0, column=0, sticky="ew")
        self._create_small_button(weight_row, "浏览", self._pick_track_weights).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(grid, row=1, label="跟踪权重", widget=weight_row, description="选择已经训练好的 .pt 权重。")

        source_row = tk.Frame(grid, bg=PANEL_BG)
        source_row.grid_columnconfigure(0, weight=1)
        self._create_entry(source_row, self.track_source_var, readonly=False).grid(row=0, column=0, sticky="ew")
        self._create_small_button(source_row, "文件", lambda: self._pick_source_file(self.track_source_var)).grid(row=0, column=1, padx=(8, 0))
        self._create_small_button(source_row, "目录", lambda: self._pick_source_dir(self.track_source_var)).grid(row=0, column=2, padx=(8, 0))
        self._create_small_button(source_row, "清空", lambda: self.track_source_var.set("")).grid(row=0, column=3, padx=(8, 0))
        self._create_form_row(
            grid,
            row=2,
            label="跟踪源",
            widget=source_row,
            description="可填视频、图片目录、摄像头编号（如 0）、RTSP/HTTP 流地址。",
        )

    def _build_prep_section(self, parent: tk.Widget) -> None:
        note = tk.Label(
            parent,
            text="仅用于 detect 任务。整理完成后会自动回填正式 dataset.yaml，方便继续训练。",
            bg=PANEL_BG,
            fg=TEXT,
            wraplength=430,
            justify="left",
            font=("Microsoft YaHei UI", 10),
        )
        note.pack(fill="x", padx=14, pady=(10, 8))

        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10)
        grid.grid_columnconfigure(1, weight=1)

        raw_row = tk.Frame(grid, bg=PANEL_BG)
        raw_row.grid_columnconfigure(0, weight=1)
        self.prep_input_entry = self._create_entry(raw_row, self.prep_input_var, readonly=True)
        self.prep_input_entry.grid(row=0, column=0, sticky="ew")
        self._create_small_button(raw_row, "选择原始数据目录", self._pick_prep_input).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(grid, row=0, label="原始目录", widget=raw_row, description="这里放原始图片和标注。")

        self._create_form_row(
            grid,
            row=1,
            label="原始格式",
            widget=self._create_combo(grid, self.prep_format_var, ["自动判断", "YOLO TXT", "LabelMe 矩形框 JSON", "COCO JSON"]),
            description="不确定时选自动判断即可。",
        )

        ratio_row = self._create_step_entry_row(grid, self.prep_val_ratio_var, {"types": ["float"]})
        self._create_form_row(
            grid,
            row=2,
            label="验证集比例",
            widget=ratio_row,
            description="例如 0.2 表示自动切出 20% 图片作为 val。",
        )

        seed_row = self._create_step_entry_row(grid, self.prep_seed_var, {"types": ["int"]})
        self._create_form_row(
            grid,
            row=3,
            label="随机种子",
            widget=seed_row,
            description="保持同样的划分结果时不要改它。",
        )

        self._create_form_row(
            grid,
            row=4,
            label="整理方式",
            widget=self._create_combo(grid, self.prep_copy_mode_var, list(PREP_COPY_MODE_LABEL_TO_ID.keys())),
            description="复制文件更稳妥；硬链接更省空间但对原目录依赖更强。",
        )

        names_frame = tk.Frame(grid, bg=PANEL_BG)
        self.prep_class_names_text = tk.Text(
            names_frame,
            height=4,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 10),
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
        )
        self.prep_class_names_text.pack(fill="x")
        self._create_form_row(
            grid,
            row=5,
            label="类别顺序（可选）",
            widget=names_frame,
            description="留空时自动推断；可一行一个，也可用逗号分隔。",
        )

        class_file_row = tk.Frame(grid, bg=PANEL_BG)
        class_file_row.grid_columnconfigure(0, weight=1)
        self._create_entry(class_file_row, self.prep_class_names_file_var, readonly=False).grid(row=0, column=0, sticky="ew")
        self._create_small_button(class_file_row, "浏览", lambda: self._pick_file(self.prep_class_names_file_var)).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(
            grid,
            row=6,
            label="类别文件（可选）",
            widget=class_file_row,
            description="可额外指定类别名文件，适合类别很多时统一维护。",
        )

        output_entry = self._create_entry(grid, self.prep_output_preview_var, readonly=True)
        self._create_form_row(grid, row=7, label="输出预览", widget=output_entry, description="将自动生成标准 YOLO 训练目录。")

        strict_frame = tk.Frame(grid, bg=PANEL_BG)
        ttk.Checkbutton(strict_frame, variable=self.prep_strict_var).pack(side="left")
        self._create_form_row(grid, row=8, label="严格模式", widget=strict_frame, description="标注异常时直接报错，方便排查脏数据。")

        overwrite_frame = tk.Frame(grid, bg=PANEL_BG)
        ttk.Checkbutton(overwrite_frame, variable=self.prep_overwrite_var).pack(side="left")
        self._create_form_row(grid, row=9, label="覆盖已有输出", widget=overwrite_frame, description="勾选后会覆盖同名输出目录。")

        info = tk.Label(
            parent,
            text="支持的原始标注：YOLO TXT / LabelMe 矩形框 JSON / COCO JSON；支持图片格式：bmp / jpeg / jpg / png / tif / tiff / webp。",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            wraplength=430,
            justify="left",
            font=("Microsoft YaHei UI", 10),
        )
        info.pack(fill="x", padx=14, pady=(8, 12))

        start_button = tk.Button(
            parent,
            text="开始整理",
            command=self.start_prepare_dataset,
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=PRIMARY,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
            pady=9,
            cursor="hand2",
        )
        start_button.pack(fill="x", padx=14, pady=(0, 14))

    def _build_train_group_section(self, parent: tk.Widget, group: dict) -> None:
        group_id = str(group.get("id") or "")
        reset_button = self._create_ghost_button(parent, "恢复官方默认值", lambda gid=group_id: self._reset_train_group(gid))
        reset_button.pack(fill="x", padx=10, pady=(10, 10))

        fields_box = tk.Frame(parent, bg=PANEL_BG)
        fields_box.pack(fill="x", padx=10, pady=(0, 10))
        fields_box.grid_columnconfigure(0, weight=1)

        keys: list[str] = []
        for row_index, spec in enumerate(group.get("parameters", [])):
            self._create_spec_row(fields_box, row_index, spec, self.train_fields, group_id=group_id)
            keys.append(str(spec["key"]))
        self.train_group_meta[group_id] = {"keys": keys}

    def _build_param_group_section(self, parent: tk.Widget, specs: list[dict[str, Any]], store: dict[str, dict[str, Any]], group_id: str) -> None:
        reset_button = self._create_ghost_button(parent, "恢复官方默认值", lambda: self._reset_spec_store(store))
        reset_button.pack(fill="x", padx=10, pady=(10, 10))

        fields_box = tk.Frame(parent, bg=PANEL_BG)
        fields_box.pack(fill="x", padx=10, pady=(0, 10))
        fields_box.grid_columnconfigure(0, weight=1)

        for row_index, spec in enumerate(specs):
            self._create_spec_row(fields_box, row_index, spec, store, group_id=group_id)

    def _build_export_entry_section(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL_BG)
        grid.pack(fill="x", padx=10, pady=10)
        grid.grid_columnconfigure(1, weight=1)

        weight_row = tk.Frame(grid, bg=PANEL_BG)
        weight_row.grid_columnconfigure(0, weight=1)
        self.export_weights_entry = self._create_entry(weight_row, self.export_weights_var, readonly=True)
        self.export_weights_entry.grid(row=0, column=0, sticky="ew")
        self._create_small_button(weight_row, "浏览", self._pick_export_weights).grid(row=0, column=1, padx=(8, 0))
        self._create_form_row(grid, row=0, label="权重文件", widget=weight_row, description="选择要导出的 .pt 权重。")

        out_row = tk.Frame(grid, bg=PANEL_BG)
        out_row.grid_columnconfigure(0, weight=1)
        self.export_output_entry = self._create_entry(out_row, self.export_output_dir_var, readonly=False)
        self.export_output_entry.grid(row=0, column=0, sticky="ew")
        self._create_small_button(out_row, "选择", self._pick_export_output_dir).grid(row=0, column=1, padx=(8, 0))
        self._create_small_button(out_row, "清空", self._clear_export_output_dir).grid(row=0, column=2, padx=(8, 0))
        self._create_form_row(
            grid,
            row=1,
            label="导出目录",
            widget=out_row,
            description="不填写时，导出产物会按官方默认行为写到权重文件所在目录。",
        )

        self._create_form_row(
            grid,
            row=2,
            label="导出格式",
            widget=self._create_combo(grid, self.export_format_label_var, list(self.export_format_map.keys())),
            description="这里只显示 Ultralytics 当前支持的导出格式。",
        )

        self._create_form_row(
            grid,
            row=3,
            label="任务类型",
            widget=self._create_combo(grid, self.export_task_label_var, list(EXPORT_TASK_LABEL_TO_ID.keys())),
            description="默认自动识别；如果你的权重文件名不规范，可以手动指定任务类型。",
        )

        imgsz_row = self._create_step_entry_row(grid, self.export_imgsz_var, {"types": ["int"]})
        self._create_form_row(grid, row=4, label="输入尺寸", widget=imgsz_row, description="部署端常用的输入尺寸。")

        device_entry = self._create_entry(grid, self.export_device_var, readonly=False)
        self._create_form_row(grid, row=5, label="设备", widget=device_entry, description="例如 cpu、0、0,1。")

    def _build_export_params_section(self, parent: tk.Widget) -> None:
        reset_button = self._create_ghost_button(parent, "恢复官方默认值", self._reset_export_params)
        reset_button.pack(fill="x", padx=10, pady=(10, 10))

        fields_box = tk.Frame(parent, bg=PANEL_BG)
        fields_box.pack(fill="x", padx=10, pady=(0, 10))
        fields_box.grid_columnconfigure(0, weight=1)

        export_specs = [
            {"key": "batch", "types": ["int"], "default": 1, "description": "导出时假定的 batch 大小。"},
            {"key": "dynamic", "types": ["bool"], "default": False, "description": "允许模型接收可变尺寸输入。"},
            {"key": "half", "types": ["bool"], "default": False, "description": "使用 FP16 导出，硬件支持时更省空间。"},
            {"key": "opset", "types": ["int"], "default": 13, "description": "ONNX 导出使用的 opset 版本。"},
            {"key": "simplify", "types": ["bool"], "default": True, "description": "自动简化计算图，通常建议开启。"},
            {"key": "nms", "types": ["bool"], "default": False, "description": "把 NMS 一并写入导出模型。"},
            {"key": "int8", "types": ["bool"], "default": False, "description": "尝试使用 INT8 量化。"},
            {"key": "fraction", "types": ["float"], "default": 1.0, "description": "量化或校准时使用多少比例的数据。"},
            {"key": "workspace", "types": ["float"], "default": 4.0, "description": "TensorRT 使用的 workspace 大小（GB）。"},
            {"key": "optimize", "types": ["bool"], "default": False, "description": "启用额外图优化。"},
            {"key": "keras", "types": ["bool"], "default": False, "description": "TensorFlow 家族导出时走 Keras 兼容路径。"},
            {"key": "data", "types": ["string"], "default": "", "optional": True, "browse": "file", "description": "INT8 校准等场景需要的数据集 YAML。"},
            {"key": "name", "types": ["string"], "default": "", "optional": True, "description": "导出结果名称或后缀。"},
        ]

        for row_index, spec in enumerate(export_specs):
            self._create_spec_row(fields_box, row_index, spec, self.export_fields, group_id="export")

    def _build_run_section(self, parent: tk.Widget, *, scope: str) -> None:
        wrapper = tk.Frame(parent, bg=PANEL_BG)
        wrapper.pack(fill="x", padx=10, pady=10)
        wrapper.grid_columnconfigure(1, weight=1)

        mode_var = self.selected_train_action_label_var if scope == "train" else self.export_mode_label_var
        preset_var = self.train_preset_var if scope == "train" else self.export_preset_var
        recommended_var = self.train_recommended_preset_var if scope == "train" else self.export_recommended_preset_var

        tk.Label(wrapper, text="功能", bg=PANEL_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Label(wrapper, textvariable=mode_var, bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=1, sticky="w")

        tk.Label(wrapper, text="状态", bg=PANEL_BG, fg=TEXT_MUTED, font=("Microsoft YaHei UI", 10)).grid(row=1, column=0, sticky="w", pady=(2, 0))
        status_label = tk.Label(wrapper, textvariable=self.process_status_var, bg=PANEL_BG, fg=TEXT, font=("Microsoft YaHei UI", 10, "bold"))
        status_label.grid(row=1, column=1, sticky="w", pady=(2, 0))
        self.process_status_labels.append(status_label)

        preset_name_row = tk.Frame(wrapper, bg=PANEL_BG)
        preset_name_row.grid_columnconfigure(0, weight=1)
        self._create_entry(preset_name_row, preset_var, readonly=False).grid(row=0, column=0, sticky="ew")
        load_button = self._create_small_button(preset_name_row, "加载", lambda item=scope: self._load_preset(item))
        load_button.grid(row=0, column=1, padx=(8, 0))
        save_button = self._create_small_button(preset_name_row, "保存", lambda item=scope: self._save_preset(item))
        save_button.grid(row=0, column=2, padx=(8, 0))
        delete_button = self._create_small_button(preset_name_row, "删除", lambda item=scope: self._delete_preset(item))
        delete_button.grid(row=0, column=3, padx=(8, 0))
        ToolTip(load_button, "按左侧输入的预设名加载自定义预设；如果名字与推荐预设同名，也会直接套用推荐方案。")
        ToolTip(save_button, "把当前参数保存成自定义预设。需要先在输入框里填一个自己的预设名称。")
        ToolTip(delete_button, "删除当前输入名称对应的自定义预设。内置推荐预设不会被删除。")
        self._create_form_row(
            wrapper,
            row=2,
            label="参数预设",
            widget=preset_name_row,
            description="这里默认留空，可直接输入自己的预设名称后保存；加载和删除也是按这里输入的名称执行。",
        )

        recommended_row = tk.Frame(wrapper, bg=PANEL_BG)
        recommended_row.grid_columnconfigure(0, weight=1)
        preset_combo = self._create_combo(recommended_row, recommended_var, [])
        preset_combo.grid(row=0, column=0, sticky="ew")
        preset_combo.bind("<<ComboboxSelected>>", lambda _event, item=scope: self._load_recommended_preset(item), add="+")
        self._create_form_row(
            wrapper,
            row=3,
            label="推荐预设",
            widget=recommended_row,
            description="这里是软件内置推荐方案，下拉选择后会立即应用，不会覆盖你自己保存的预设名称。",
        )

        if scope == "train":
            self.train_preset_combo = preset_combo
            self.train_start_button = tk.Button(
                wrapper,
                text="开始训练",
                command=self._run_current_train_action,
                bg=PRIMARY,
                fg="white",
                activebackground=PRIMARY_DARK,
                activeforeground="white",
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=PRIMARY,
                highlightcolor=PRIMARY,
                font=("Microsoft YaHei UI", 11, "bold"),
                pady=8,
                cursor="hand2",
            )
            start_button = self.train_start_button
        else:
            self.export_preset_combo = preset_combo
            self.export_start_button = tk.Button(
                wrapper,
                text="开始导出",
                command=self.start_export,
                bg=PRIMARY,
                fg="white",
                activebackground=PRIMARY_DARK,
                activeforeground="white",
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=PRIMARY,
                highlightcolor=PRIMARY,
                font=("Microsoft YaHei UI", 11, "bold"),
                pady=8,
                cursor="hand2",
            )
            start_button = self.export_start_button

        start_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ToolTip(start_button, "确认当前入口、模型、数据和参数后，从这里启动任务。")

        stop_button = tk.Button(
            wrapper,
            text="停止任务",
            command=self.stop_process,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11),
            pady=8,
            cursor="hand2",
        )
        stop_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ToolTip(stop_button, "任务运行中可点这里发送停止信号；空闲时按钮会自动置灰。")

        open_result_button = tk.Button(
            wrapper,
            text="打开结果",
            command=self.open_result_location,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11),
            pady=8,
            cursor="hand2",
        )
        open_result_button.grid(row=10, column=0, sticky="ew", pady=(0, 4), padx=(0, 5))
        ToolTip(open_result_button, "打开最近一次任务的结果目录；没有结果时按钮会自动置灰。")

        open_log_button = tk.Button(
            wrapper,
            text="打开日志",
            command=self.open_log_file,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11),
            pady=8,
            cursor="hand2",
        )
        open_log_button.grid(row=10, column=1, sticky="ew", pady=(0, 4), padx=(5, 0))
        ToolTip(open_log_button, "打开当前任务对应的日志文件；尚未生成日志时按钮会自动置灰。")

        result_entry = self._create_entry(wrapper, self.result_location_var, readonly=True)
        self._create_form_row(wrapper, row=11, label="结果位置", widget=result_entry)

        if scope == "train":
            self.train_preset_buttons = {"load": load_button, "save": save_button, "delete": delete_button}
            self.train_stop_button = stop_button
            self.train_open_result_button = open_result_button
            self.train_open_log_button = open_log_button
        else:
            self.export_preset_buttons = {"load": load_button, "save": save_button, "delete": delete_button}
            self.export_stop_button = stop_button
            self.export_open_result_button = open_result_button
            self.export_open_log_button = open_log_button

    def _create_form_row(
        self,
        parent: tk.Widget,
        *,
        row: int,
        label: str,
        widget: tk.Widget,
        description: str = "",
    ) -> None:
        label_widget = tk.Label(
            parent,
            text=label,
            bg=parent.cget("bg"),
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10),
            anchor="w",
        )
        label_widget.grid(row=row * 2, column=0, sticky="w", padx=(0, 10), pady=(0, 4))
        widget.grid(row=row * 2, column=1, sticky="ew", pady=(0, 4))
        if description:
            ToolTip(label_widget, description)
            ToolTip(widget, description)
            tk.Label(
                parent,
                text=description,
                bg=parent.cget("bg"),
                fg=TEXT_MUTED,
                wraplength=360,
                justify="left",
                font=("Microsoft YaHei UI", 10),
            ).grid(row=row * 2 + 1, column=1, sticky="w", pady=(0, 8))

    def _create_spec_row(self, parent: tk.Widget, row: int, spec: dict[str, Any], store: dict[str, dict[str, Any]], *, group_id: str) -> None:
        container = tk.Frame(parent, bg=parent.cget("bg"))
        container.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        container.grid_columnconfigure(1, weight=1)

        label_widget = tk.Label(
            container,
            text=str(spec.get("key", "")),
            bg=container.cget("bg"),
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 10),
            anchor="w",
            width=12,
        )
        label_widget.grid(row=0, column=0, sticky="nw", padx=(0, 10))

        types = list(spec.get("types") or [])
        default = spec.get("default")
        options = spec.get("options") or []

        if types == ["bool"]:
            var: tk.Variable = tk.BooleanVar(value=bool(default))
            widget = ttk.Checkbutton(container, variable=var)
            widget.grid(row=0, column=1, sticky="w")
        elif options:
            var = tk.StringVar(value=stringify_value(default))
            widget = self._create_combo(container, var, [stringify_value(option) for option in options])
            widget.grid(row=0, column=1, sticky="ew")
        elif spec.get("browse") in {"file", "dir"}:
            var = tk.StringVar(value=stringify_value(default))
            row_widget = tk.Frame(container, bg=PANEL_BG)
            row_widget.grid_columnconfigure(0, weight=1)
            entry = self._create_entry(row_widget, var, readonly=False)
            entry.grid(row=0, column=0, sticky="ew")
            if spec["browse"] == "file":
                self._create_small_button(row_widget, "浏览", lambda variable=var: self._pick_file(variable)).grid(row=0, column=1, padx=(8, 0))
            else:
                self._create_small_button(row_widget, "选择", lambda variable=var: self._pick_dir(variable)).grid(row=0, column=1, padx=(8, 0))
            widget = row_widget
            widget.grid(row=0, column=1, sticky="ew")
        else:
            var = tk.StringVar(value=stringify_value(default))
            if any(kind in types for kind in ("int", "float")) and "list" not in types and "string" not in types:
                widget = self._create_step_entry_row(container, var, spec)
                widget.grid(row=0, column=1, sticky="ew")
            else:
                widget = self._create_entry(container, var, readonly=False)
                widget.grid(row=0, column=1, sticky="ew")

        description = build_spec_description(spec)
        if description:
            ToolTip(label_widget, description)
            ToolTip(widget, description)
        tk.Label(
            container,
            text=description,
            bg=container.cget("bg"),
            fg=TEXT_MUTED,
            wraplength=360,
            justify="left",
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        store[str(spec["key"])] = {
            "spec": spec,
            "group_id": group_id,
            "var": var,
            "container": container,
            "active": True,
        }

    def _create_combo(self, parent: tk.Widget, variable: tk.StringVar, values: list[str]) -> SmartComboBox:
        return SmartComboBox(parent, variable, values, readonly=True)

    def _create_entry(self, parent: tk.Widget, variable: tk.StringVar, *, readonly: bool) -> tk.Entry:
        entry = tk.Entry(
            parent,
            textvariable=variable,
            relief="flat",
            bd=0,
            bg=CARD_BG,
            fg=TEXT,
            readonlybackground=CARD_BG,
            font=("Microsoft YaHei UI", 10),
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
        )
        entry.configure(state="readonly" if readonly else "normal")
        return entry

    def _create_small_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 10),
            cursor="hand2",
            padx=14,
            pady=4,
        )

    def _create_ghost_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=CARD_BG,
            fg=TEXT,
            activebackground=PRIMARY_SOFT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PRIMARY,
            font=("Microsoft YaHei UI", 11),
            cursor="hand2",
            pady=7,
        )

    def _create_step_entry_row(self, parent: tk.Widget, variable: tk.StringVar, spec: dict[str, Any]) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.grid_columnconfigure(0, weight=1)

        entry = self._create_entry(frame, variable, readonly=False)
        entry.grid(row=0, column=0, sticky="ew")

        minus_button = self._create_small_button(frame, "-", lambda: self._step_numeric_value(variable, spec, -1))
        minus_button.grid(row=0, column=1, padx=(8, 0))

        plus_button = self._create_small_button(frame, "+", lambda: self._step_numeric_value(variable, spec, 1))
        plus_button.grid(row=0, column=2, padx=(8, 0))
        return frame

    def _step_numeric_value(self, variable: tk.StringVar, spec: dict[str, Any], direction: int) -> None:
        types = list(spec.get("types") or [])
        current = str(variable.get()).strip()
        if "int" in types and "float" not in types:
            value = int(current or 0) + direction
            variable.set(str(value))
            return
        base = float(current or 0.0) + (0.1 * direction)
        variable.set(f"{base:.6f}")

    def _bind_traces(self) -> None:
        self.train_task_label_var.trace_add("write", lambda *_: self._on_train_task_changed())
        self.train_mode_label_var.trace_add("write", lambda *_: self._on_train_mode_changed())
        self.train_family_var.trace_add("write", lambda *_: self._refresh_model_name())
        self.train_size_var.trace_add("write", lambda *_: self._refresh_model_name())
        self.train_use_local_weights_var.trace_add("write", lambda *_: self._on_local_weights_toggle())
        self.train_model_var.trace_add("write", lambda *_: self._refresh_summary())
        self.train_data_var.trace_add("write", lambda *_: self._refresh_summary())

        self.val_task_label_var.trace_add("write", lambda *_: self._on_action_task_changed("val"))
        self.val_weights_var.trace_add("write", lambda *_: self._refresh_summary())
        self.val_data_var.trace_add("write", lambda *_: self._refresh_summary())

        self.predict_task_label_var.trace_add("write", lambda *_: self._on_action_task_changed("predict"))
        self.predict_weights_var.trace_add("write", lambda *_: self._refresh_summary())
        self.predict_source_var.trace_add("write", lambda *_: self._refresh_summary())

        self.track_task_label_var.trace_add("write", lambda *_: self._on_action_task_changed("track"))
        self.track_weights_var.trace_add("write", lambda *_: self._refresh_summary())
        self.track_source_var.trace_add("write", lambda *_: self._refresh_summary())

        self.prep_input_var.trace_add("write", lambda *_: self._refresh_prep_output_preview())
        self.prep_output_preview_var.trace_add("write", lambda *_: self._refresh_summary())

        self.export_task_label_var.trace_add("write", lambda *_: self._refresh_summary())
        self.export_weights_var.trace_add("write", lambda *_: self._refresh_summary())
        self.export_output_dir_var.trace_add("write", lambda *_: self._refresh_summary())
        self.export_format_label_var.trace_add("write", lambda *_: self._refresh_export_visibility())
        self.export_imgsz_var.trace_add("write", lambda *_: self._refresh_summary())
        self.export_device_var.trace_add("write", lambda *_: self._refresh_summary())
        self.process_status_var.trace_add("write", lambda *_: self._refresh_status_visuals())
        self.left_log_state_var.trace_add("write", lambda *_: self._refresh_status_visuals())
        self.result_location_var.trace_add("write", lambda *_: self._refresh_run_action_buttons())
        self.train_preset_var.trace_add("write", lambda *_: self._refresh_run_action_buttons())
        self.export_preset_var.trace_add("write", lambda *_: self._refresh_run_action_buttons())

    def _show_tab(self, tab: str) -> None:
        ToolTip.hide_all()
        SmartComboBox.close_all()
        self.active_tab.set(tab)
        if tab == "train":
            self.train_tab_button.configure(bg=PRIMARY, fg="white")
            self.export_tab_button.configure(bg=CARD_BG, fg=TEXT)
            self.train_scroll.grid()
            self.export_scroll.grid_remove()
            self._reset_scroll_to_top(self.train_scroll)
        else:
            self.train_tab_button.configure(bg=CARD_BG, fg=TEXT)
            self.export_tab_button.configure(bg=PRIMARY, fg="white")
            self.export_scroll.grid()
            self.train_scroll.grid_remove()
            self._reset_scroll_to_top(self.export_scroll)
        if not self.process and self.left_log_state_var.get() in {"暂无", "已完成", "已结束", "已取消"}:
            self._set_log_placeholder(tab if tab == "export" else self.train_action_var.get())
        self._refresh_summary()

    def _show_train_action(self, action: str) -> None:
        ToolTip.hide_all()
        SmartComboBox.close_all()
        self.train_action_var.set(action)
        self.selected_train_action_label_var.set(ACTION_ID_TO_LABEL[action])
        for action_id, button in self.train_action_buttons.items():
            is_active = action_id == action
            button.configure(
                bg=PRIMARY if is_active else CARD_BG,
                fg="white" if is_active else TEXT,
                activebackground=PRIMARY_DARK if is_active else PRIMARY_SOFT,
                activeforeground="white" if is_active else TEXT,
            )
        for section_id, section in self.train_sections.items():
            visible_actions = self.train_section_modes.get(section_id, {action})
            if action in visible_actions:
                section.grid()
            else:
                section.grid_remove()
        self._refresh_train_preset_choices()
        self._refresh_train_task_visibility()
        self._refresh_action_task_visibility("val")
        self._refresh_action_task_visibility("predict")
        self._refresh_action_task_visibility("track")
        self._refresh_train_run_button()
        self._reset_scroll_to_top(self.train_scroll)
        if self.active_tab.get() == "train" and not self.process and self.left_log_state_var.get() in {"暂无", "已完成", "已结束", "已取消"}:
            self._set_log_placeholder(action)
        self._refresh_summary()

    def _on_train_task_changed(self) -> None:
        self._refresh_model_name()
        self._refresh_train_task_visibility()
        self._refresh_summary()

    def _on_action_task_changed(self, action: str) -> None:
        self._refresh_action_task_visibility(action)
        self._refresh_summary()

    def _on_train_mode_changed(self) -> None:
        if self.train_mode_label_var.get() == "本地权重继续训练" and not self.train_use_local_weights_var.get():
            self.train_use_local_weights_var.set(True)
        elif self.train_mode_label_var.get() == "官方预训练" and self.train_use_local_weights_var.get():
            self.train_use_local_weights_var.set(False)

    def _on_local_weights_toggle(self) -> None:
        is_local = self.train_use_local_weights_var.get()
        self.train_model_entry.configure(state="normal" if is_local else "readonly")
        self.train_model_browse_button.configure(state="normal" if is_local else "disabled")
        if is_local:
            self.train_mode_label_var.set("本地权重继续训练")
        else:
            self.train_mode_label_var.set("官方预训练")
            self._refresh_model_name()
        self._refresh_summary()

    def _current_train_task(self) -> str:
        return TASK_LABEL_TO_ID.get(self.train_task_label_var.get(), "detect")

    def _current_action_task(self, action: str) -> str:
        if action == "train":
            return self._current_train_task()
        if action == "val":
            return TASK_LABEL_TO_ID.get(self.val_task_label_var.get(), "detect")
        if action == "predict":
            return TASK_LABEL_TO_ID.get(self.predict_task_label_var.get(), "detect")
        if action == "track":
            return TASK_LABEL_TO_ID.get(self.track_task_label_var.get(), "detect")
        return "detect"

    def _build_official_model_name(self) -> str:
        template = MODEL_FAMILY_TO_TEMPLATE.get(self.train_family_var.get(), "yolo11{size}")
        size = self.train_size_var.get().strip() or "n"
        suffix = TASK_SUFFIX.get(self._current_train_task(), "")
        return f"{template.format(size=size)}{suffix}.pt"

    def _refresh_model_name(self) -> None:
        if not self.train_use_local_weights_var.get():
            self.train_model_var.set(self._build_official_model_name())

    def _refresh_train_task_visibility(self) -> None:
        task = self._current_train_task()
        self._refresh_store_visibility(self.train_fields, task, list(TASK_LABEL_TO_ID.values()))

    def _refresh_action_task_visibility(self, action: str) -> None:
        if action == "val":
            self._refresh_store_visibility(self.val_fields, self._current_action_task("val"), list(TASK_LABEL_TO_ID.values()))
        elif action == "predict":
            self._refresh_store_visibility(self.predict_fields, self._current_action_task("predict"), list(TASK_LABEL_TO_ID.values()))
        elif action == "track":
            self._refresh_store_visibility(self.track_fields, self._current_action_task("track"), list(TRACK_TASK_IDS))

    def _refresh_store_visibility(self, store: dict[str, dict[str, Any]], task: str, fallback_tasks: list[str]) -> None:
        for entry in store.values():
            tasks = entry["spec"].get("tasks") or fallback_tasks
            is_active = task in tasks
            entry["active"] = is_active
            if is_active:
                entry["container"].grid()
            else:
                entry["container"].grid_remove()

    def _current_export_format_id(self) -> str:
        return self.export_format_map.get(self.export_format_label_var.get(), "onnx")

    def _refresh_export_visibility(self) -> None:
        supported = set()
        for item in self.export_contract.get("formats", []):
            if item["id"] == self._current_export_format_id():
                supported = set(item.get("arguments") or [])
                break
        for key, entry in self.export_fields.items():
            is_active = key in supported
            entry["active"] = is_active
            if is_active:
                entry["container"].grid()
            else:
                entry["container"].grid_remove()
        self._reset_scroll_to_top(self.export_scroll)

    def _reset_scroll_to_top(self, scrollable: ScrollableFrame) -> None:
        def apply() -> None:
            try:
                if not scrollable.winfo_exists():
                    return
                scrollable.update_idletasks()
                bbox = scrollable.canvas.bbox("all")
                if bbox:
                    scrollable.canvas.configure(scrollregion=bbox)
                scrollable.canvas.yview_moveto(0)
            except tk.TclError:
                return

        self.root.after_idle(apply)
        self._refresh_summary()

    def _refresh_prep_output_preview(self) -> None:
        raw_input = self.prep_input_var.get().strip()
        if not raw_input or raw_input == "暂无":
            self.prep_output_preview_var.set("暂无")
            return
        input_path = Path(raw_input).expanduser()
        output_path = input_path.parent / f"{input_path.name}_yolo_dataset"
        self.prep_output_preview_var.set(str(output_path))

    def _expected_run_dir(self, task: str, fields: dict[str, dict[str, Any]], default_name: str) -> str:
        project = ""
        name = default_name
        if "project" in fields:
            project = str(fields["project"]["var"].get()).strip()
        if "name" in fields:
            name_value = str(fields["name"]["var"].get()).strip()
            if name_value:
                name = name_value
        base = Path(project).expanduser() if project else (WORK_DIR / "runs" / task)
        return str(base / name)

    def _expected_train_output_dir(self) -> str:
        return self._expected_run_dir(self._current_train_task(), self.train_fields, "train")

    def _expected_val_output_dir(self) -> str:
        return self._expected_run_dir(self._current_action_task("val"), self.val_fields, "val")

    def _expected_predict_output_dir(self) -> str:
        return self._expected_run_dir(self._current_action_task("predict"), self.predict_fields, "predict")

    def _expected_track_output_dir(self) -> str:
        return self._expected_run_dir(self._current_action_task("track"), self.track_fields, "track")

    def _expected_export_target(self) -> str:
        output_dir = self.export_output_dir_var.get().strip()
        if output_dir:
            return output_dir
        weight_path = self.export_weights_var.get().strip()
        if weight_path and weight_path != "未选择":
            return str(Path(weight_path).expanduser().resolve().parent)
        return "跟随权重目录"

    def _refresh_summary(self) -> None:
        if self.active_tab.get() == "export":
            self.summary_label1.set("权重")
            self.summary_label2.set("任务类型")
            self.summary_label3.set("输出目录")
            self.summary_value1.set(self.export_weights_var.get() or "未选择")
            self.summary_value2.set(self.export_task_label_var.get() or "自动识别")
            preview = str(self.last_result_path) if self.last_result_path and self.process_status_var.get() == "导出完成" else ""
            self.summary_value3.set(preview or self._expected_export_target())
            self.result_location_var.set(preview or self._expected_export_target())
            self._refresh_run_action_buttons()
            return

        action = self.train_action_var.get()
        if action == "train":
            self.summary_label1.set("数据集")
            self.summary_label2.set("模型")
            self.summary_label3.set("预计输出目录")
            self.summary_value1.set(self.train_data_var.get() or "暂无")
            self.summary_value2.set(self.train_model_var.get() or "暂无")
            preview = self._expected_train_output_dir()
        elif action == "val":
            self.summary_label1.set("验证数据")
            self.summary_label2.set("权重")
            self.summary_label3.set("预计输出目录")
            self.summary_value1.set(self.val_data_var.get() or "暂无")
            self.summary_value2.set(self.val_weights_var.get() or "未选择")
            preview = self._expected_val_output_dir()
        elif action == "predict":
            self.summary_label1.set("预测源")
            self.summary_label2.set("权重")
            self.summary_label3.set("预计输出目录")
            self.summary_value1.set(self.predict_source_var.get() or "未填写")
            self.summary_value2.set(self.predict_weights_var.get() or "未选择")
            preview = self._expected_predict_output_dir()
        else:
            self.summary_label1.set("跟踪源")
            self.summary_label2.set("权重")
            self.summary_label3.set("预计输出目录")
            self.summary_value1.set(self.track_source_var.get() or "未填写")
            self.summary_value2.set(self.track_weights_var.get() or "未选择")
            preview = self._expected_track_output_dir()

        self.summary_value3.set(str(self.last_result_path) if self.last_result_path and self.process_status_var.get().endswith("完成") else preview)
        self.result_location_var.set(str(self.last_result_path) if self.last_result_path and self.process_status_var.get().endswith("完成") else preview)
        self._refresh_run_action_buttons()

    def _set_log_placeholder(self, mode: str) -> None:
        text_map = {
            "train": "暂无运行日志。\n\n开始训练后，这里显示 Ultralytics 实时输出。",
            "val": "暂无运行日志。\n\n开始验证后，这里显示验证过程与指标摘要。",
            "predict": "暂无运行日志。\n\n开始预测后，这里显示预测过程和保存位置。",
            "track": "暂无运行日志。\n\n开始跟踪后，这里显示跟踪过程和保存位置。",
            "export": "暂无运行日志。\n\n开始导出后，这里显示官方导出输出。",
        }
        self._clear_log_text()
        self.log_text.configure(state="normal")
        placeholder = text_map.get(mode, text_map["train"])
        self.log_text.insert("1.0", placeholder)
        self.log_text.configure(state="disabled")
        self._visible_log_lines = len(placeholder.splitlines())

    def _collect_field_payload(self, fields: dict[str, dict[str, Any]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, entry in fields.items():
            if not entry.get("active", True):
                continue
            spec = entry["spec"]
            var = entry["var"]
            if isinstance(var, tk.BooleanVar):
                payload[key] = bool(var.get())
                continue
            text = str(var.get()).strip()
            if not text:
                if spec.get("optional"):
                    continue
                continue
            options = spec.get("options") or []
            if options:
                matched = next((option for option in options if stringify_value(option) == text), None)
                payload[key] = matched if matched is not None else parse_scalar(text, list(spec.get("types") or []))
            else:
                payload[key] = parse_scalar(text, list(spec.get("types") or []))
        return payload

    def _collect_train_config(self) -> dict[str, object]:
        return self._collect_field_payload(self.train_fields)

    def _collect_val_config(self) -> dict[str, object]:
        return self._collect_field_payload(self.val_fields)

    def _collect_predict_config(self) -> dict[str, object]:
        return self._collect_field_payload(self.predict_fields)

    def _collect_track_config(self) -> dict[str, object]:
        return self._collect_field_payload(self.track_fields)

    def _collect_export_config(self) -> dict[str, object]:
        payload = self._collect_field_payload(self.export_fields)
        payload["format"] = self._current_export_format_id()
        payload["imgsz"] = parse_scalar(self.export_imgsz_var.get().strip() or "640", ["int"])
        device = self.export_device_var.get().strip()
        if device:
            payload["device"] = device
        output_dir = self.export_output_dir_var.get().strip()
        if output_dir:
            payload["output_dir"] = output_dir
        return payload

    def _reset_train_group(self, group_id: str) -> None:
        for entry in self.train_fields.values():
            if entry["group_id"] != group_id:
                continue
            self._set_var_to_default(entry["var"], entry["spec"].get("default"))
        self._refresh_summary()

    def _reset_spec_store(self, store: dict[str, dict[str, Any]]) -> None:
        for entry in store.values():
            self._set_var_to_default(entry["var"], entry["spec"].get("default"))
        self._refresh_summary()

    def _set_var_to_default(self, var: tk.Variable, default: Any) -> None:
        if isinstance(var, tk.BooleanVar):
            var.set(bool(default))
        else:
            var.set(stringify_value(default))

    def _reset_export_params(self) -> None:
        self._reset_spec_store(self.export_fields)
        self._refresh_export_visibility()

    def _pick_python(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 Python 解释器",
            filetypes=[("Python", "python.exe;pythonw.exe"), ("所有文件", "*.*")],
        )
        if path:
            self.python_var.set(path)

    def _use_default_runtime(self) -> None:
        self.python_var.set(str(BUNDLED_RUNTIME_PYTHON))
        if not bundled_runtime_exists():
            self.left_result_var.set("未检测到内置环境，正在准备自动配置")
            self.process_status_var.set("准备自动配置内置环境")
            self.start_configure_environment(auto_triggered=True)
            return

        probe = self._probe_environment_state(str(BUNDLED_RUNTIME_PYTHON))
        if str(probe.get("state") or "") == "ready":
            self.left_result_var.set(f"已切换到内置环境：{BUNDLED_RUNTIME_PYTHON}")
            self.process_status_var.set("内置环境可用")
            return

        self.left_result_var.set("内置环境不完整，正在自动配置")
        self.process_status_var.set("准备自动修复内置环境")
        self.start_configure_environment(auto_triggered=True)

    def _use_current_python(self) -> None:
        if not is_frozen_app():
            self.python_var.set(sys.executable)
            return

        system_python = find_supported_system_python()
        if not system_python:
            messagebox.showwarning(
                "未找到系统 Python",
                "当前电脑没有检测到可直接使用的 Python 3.9 ~ 3.13。\n"
                "如果你是打包版用户，建议直接点击“在线一键配置环境”，程序会自动创建内置 runtime。",
            )
            return
        self.python_var.set(system_python)

    def _is_builtin_runtime_target(self, path: str) -> bool:
        try:
            return Path(path).expanduser().resolve(strict=False) == BUNDLED_RUNTIME_PYTHON.resolve(strict=False)
        except OSError:
            return False

    def _resolve_python_path(self, *, allow_missing_builtin: bool = False) -> str | None:
        if is_frozen_app():
            python_path = str(BUNDLED_RUNTIME_PYTHON)
            self.python_var.set(python_path)
            if BUNDLED_RUNTIME_PYTHON.exists():
                return python_path
            if allow_missing_builtin:
                return python_path
            messagebox.showwarning(
                "内置环境不存在",
                "当前还没有可用的内置 runtime。\n\n请先点击“在线一键配置环境”自动创建并配置软件内置环境。",
            )
            return None

        python_path = self.python_var.get().strip() or sys.executable
        self.python_var.set(python_path)
        if is_app_launcher_path(python_path):
            messagebox.showwarning("Python 选择无效", "当前选择的是软件主程序本身，不是 Python 解释器。")
            return None
        if not is_python_interpreter_path(python_path):
            messagebox.showwarning("Python 选择无效", f"所选文件不是 python.exe：\n{python_path}")
            return None
        path_obj = Path(python_path).expanduser()
        if path_obj.exists():
            return str(path_obj)
        messagebox.showwarning("Python 不存在", f"找不到解释器：\n{python_path}")
        return None

    def _system_environment_label(self, payload: dict[str, Any]) -> str:
        gpu = payload.get("system_nvidia") or payload.get("gpu") or {}
        if not isinstance(gpu, dict):
            gpu = {}
        if gpu.get("available"):
            gpu_name = str(gpu.get("gpu_name") or "NVIDIA 显卡").strip()
            gpu_architecture = str(gpu.get("gpu_architecture") or "").strip()
            cuda_version = str(gpu.get("cuda_version") or "").strip()
            driver_version = str(gpu.get("driver_version") or "").strip()
            details: list[str] = [gpu_name]
            if gpu_architecture:
                details.append(gpu_architecture)
            if cuda_version:
                details.append(f"CUDA {cuda_version}")
            if driver_version:
                details.append(f"驱动 {driver_version}")
            return "NVIDIA 显卡（" + "，".join(details) + "）"
        return "CPU（未检测到可用 NVIDIA 显卡）"

    def _runtime_environment_label(self, payload: dict[str, Any]) -> str:
        explicit = str(payload.get("runtime_backend_label") or "").strip()
        if explicit:
            return explicit

        runtime_backend = str(payload.get("runtime_backend") or "").strip().lower()
        torch_device_name = str(payload.get("torch_device_name") or "").strip()
        torch_cuda_version = str(payload.get("torch_cuda_version") or "").strip()
        torch_device_capability = str(payload.get("torch_device_capability") or "").strip()
        if runtime_backend == "nvidia":
            return f"当前运行环境已启用 NVIDIA 显卡：{torch_device_name or '未知型号'}"
        if runtime_backend == "nvidia-unsupported":
            capability_text = f"（算力 {torch_device_capability}）" if torch_device_capability else ""
            return f"检测到 NVIDIA 显卡：{torch_device_name or '未知型号'}{capability_text}，但当前 Torch 与这张显卡不兼容"
        if runtime_backend == "cuda-build-no-device":
            return f"已安装显卡版 Torch（CUDA {torch_cuda_version or '未知'}），但当前没有真正启用 NVIDIA 显卡"
        if runtime_backend == "broken":
            return "当前运行环境不完整，需要重新配置环境"
        return "当前运行环境使用 CPU"

    def _runtime_environment_brief(self, payload: dict[str, Any]) -> str:
        runtime_backend = str(payload.get("runtime_backend") or "").strip().lower()
        torch_device_name = str(payload.get("torch_device_name") or "").strip()
        torch_cuda_version = str(payload.get("torch_cuda_version") or "").strip()
        torch_device_capability = str(payload.get("torch_device_capability") or "").strip()
        if runtime_backend == "nvidia":
            return f"NVIDIA 显卡可用（{torch_device_name or '已启用'}）"
        if runtime_backend == "nvidia-unsupported":
            capability_text = f"，算力 {torch_device_capability}" if torch_device_capability else ""
            return f"检测到 NVIDIA，但当前 Torch 不兼容（{torch_device_name or '未知型号'}{capability_text}）"
        if runtime_backend == "cuda-build-no-device":
            return f"显卡版 Torch 已装，但当前未启用显卡（CUDA {torch_cuda_version or '未知'}）"
        if runtime_backend == "broken":
            return "环境损坏，需要重配"
        return "CPU 模式"

    def _install_plan_label(self, payload: dict[str, Any]) -> str:
        explicit = str(payload.get("accelerator_label") or "").strip()
        if explicit:
            return explicit
        accelerator = str(payload.get("accelerator") or "cpu").strip().lower()
        return f"NVIDIA 显卡版（{accelerator.upper()}）" if accelerator and accelerator != "cpu" else "CPU 版"

    def _recommended_accelerator(self, payload: dict[str, Any]) -> str:
        accelerator = str(payload.get("accelerator") or "").strip().lower()
        if accelerator:
            return accelerator
        accelerator_label = str(payload.get("accelerator_label") or "").strip().lower()
        if "cpu" in accelerator_label:
            return "cpu"
        for token in accelerator_label.replace("（", " ").replace("）", " ").split():
            if token.startswith("cu") and token[2:].isdigit():
                return token
        return "cpu"

    def _runtime_needs_configuration(self, payload: dict[str, Any]) -> bool:
        if str(payload.get("preflight_error") or "").strip():
            return True
        if str(payload.get("site_packages_error") or "").strip():
            return True
        runtime_backend = str(payload.get("runtime_backend") or "").strip().lower()
        if runtime_backend == "nvidia-unsupported":
            return True
        if runtime_backend == "broken":
            return True
        recommended_accelerator = self._recommended_accelerator(payload)
        if recommended_accelerator == "cpu":
            return False
        return runtime_backend != "nvidia"

    def _environment_dialog_message(self, payload: dict[str, Any], *, include_plan: bool = False) -> str:
        python_path = str(payload.get("python") or self.python_var.get()).strip()
        parts = [f"Python：{python_path}", f"电脑硬件：{self._system_environment_label(payload)}"]
        if include_plan:
            parts.append(f"安装计划：{self._install_plan_label(payload)}")
        parts.append(f"当前环境：{self._runtime_environment_label(payload)}")

        torch_version = str(payload.get("torch_version") or "").strip()
        if torch_version:
            parts.append(f"Torch 版本：{torch_version}")
        torch_build_label = str(payload.get("torch_build_label") or "").strip()
        if torch_build_label:
            parts.append(f"Torch 类型：{torch_build_label}")
        torch_device_capability = str(payload.get("torch_device_capability") or "").strip()
        if torch_device_capability:
            parts.append(f"显卡算力：{torch_device_capability}")
        torch_device_count = payload.get("torch_device_count")
        if torch_device_count not in (None, ""):
            parts.append(f"可用显卡数量：{torch_device_count}")
        torch_cuda_warnings = payload.get("torch_cuda_warnings") or []
        if isinstance(torch_cuda_warnings, list) and torch_cuda_warnings:
            first_warning = str(torch_cuda_warnings[0]).strip()
            if first_warning:
                parts.append(f"兼容性提醒：{first_warning.splitlines()[0]}")
        preflight_error = str(payload.get("preflight_error") or payload.get("site_packages_error") or "").strip()
        if preflight_error:
            parts.append(f"环境异常：{preflight_error}")
        return "\n".join(parts)

    def _environment_dialog_needs_warning(self, payload: dict[str, Any]) -> bool:
        return self._runtime_needs_configuration(payload)

    def _run_quiet_backend_command(self, command: list[str]) -> tuple[int, list[str]]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            command,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=NO_WINDOW_FLAGS,
            check=False,
        )
        lines = [line.rstrip() for line in ((process.stdout or "") + "\n" + (process.stderr or "")).splitlines() if line.strip()]
        return process.returncode, lines

    def _probe_environment_state(self, python_path: str) -> dict[str, Any]:
        if self._is_builtin_runtime_target(python_path) and not Path(python_path).exists():
            return {
                "state": "missing-runtime",
                "message": "内置 runtime 不存在，需要先在线创建并配置。",
                "python": python_path,
            }

        command = [python_path, "-u", str(BACKEND), "check"]
        try:
            return_code, lines = self._run_quiet_backend_command(command)
        except Exception as exc:
            return {
                "state": "launch-error",
                "message": str(exc),
                "python": python_path,
            }

        result_payload: dict[str, Any] | None = None
        error_message = ""
        for line in lines:
            if not line.startswith("[") or "] " not in line:
                continue
            tag, payload_text = line.split("] ", 1)
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if tag == "[RESULT" and payload.get("kind") == "check":
                result_payload = payload
            elif tag == "[ERROR":
                error_message = str(payload.get("message") or "")

        if return_code == 0 and result_payload:
            if self._runtime_needs_configuration(result_payload):
                return {
                    "state": "needs-configure",
                    "message": self._runtime_environment_label(result_payload),
                    "python": str(result_payload.get("python") or python_path),
                    "torch_version": str(result_payload.get("torch_version") or ""),
                    "payload": result_payload,
                }
            return {
                "state": "ready",
                "message": "内置环境已可直接使用。",
                "python": str(result_payload.get("python") or python_path),
                "torch_version": str(result_payload.get("torch_version") or ""),
                "payload": result_payload,
            }

        return {
            "state": "needs-configure",
            "message": error_message or "内置环境不完整，需要安装或更新依赖。",
            "python": python_path,
            "returncode": return_code,
            "lines": lines[-12:],
        }

    def _should_continue_after_probe(self, probe: dict[str, Any]) -> bool:
        state = str(probe.get("state") or "")
        if state == "ready":
            python_path = str(probe.get("python") or self.python_var.get())
            torch_version = str(probe.get("torch_version") or "")
            payload = probe.get("payload") if isinstance(probe.get("payload"), dict) else {}
            merged_payload = {"python": python_path, "torch_version": torch_version, **payload}
            message = "已检测到内置环境可直接使用：\n" + self._environment_dialog_message(merged_payload)
            message += "\n\n如果继续，程序会联网检查并更新内置 runtime 的依赖。\n要继续更新吗？"
            return messagebox.askyesno("内置环境已可用", message)

        if state == "needs-configure":
            detail = str(probe.get("message") or "内置环境不完整，需要安装或更新依赖。")
            return messagebox.askyesno(
                "需要配置内置环境",
                f"{detail}\n\n程序将自动联网安装或更新软件内置 runtime。\n确定继续吗？",
            )

        if state == "missing-runtime":
            return messagebox.askyesno(
                "需要创建内置环境",
                "当前没有检测到内置 runtime。\n程序将先在线创建内置 Python，再继续自动配置依赖环境。\n确定继续吗？",
            )

        detail = str(probe.get("message") or "环境检查失败。")
        return messagebox.askyesno(
            "环境检查失败",
            f"{detail}\n\n仍要尝试自动配置环境吗？",
        )

    def _start_configure_process(self, python_path: str) -> None:
        if not Path(python_path).exists() and self._is_builtin_runtime_target(python_path) and is_frozen_app():
            command = [
                str(Path(sys.executable).resolve()),
                BACKEND_LAUNCH_FLAG,
                "bootstrap-runtime-and-configure",
                "--target-python",
                str(BUNDLED_RUNTIME_PYTHON),
            ]
            self.python_var.set(str(BUNDLED_RUNTIME_PYTHON))
        else:
            command = [python_path, "-u", str(BACKEND), "configure-env"]
        self._start_process(command, title="环境配置", mode_label="环境配置", preview_path=None)

    def _pick_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            variable.set(path)

    def _pick_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _pick_source_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="选择图片或视频")
        if path:
            variable.set(path)

    def _pick_source_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="选择图片目录")
        if path:
            variable.set(path)

    def _pick_train_model(self) -> None:
        if not self.train_use_local_weights_var.get():
            return
        path = filedialog.askopenfilename(
            title="选择本地模型权重",
            filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            self.train_model_var.set(path)

    def _pick_train_dataset(self) -> None:
        if self._current_train_task() == "classify":
            path = filedialog.askdirectory(title="选择分类数据集根目录")
        else:
            path = filedialog.askopenfilename(
                title="选择 dataset.yaml",
                filetypes=[("YAML 文件", "*.yaml;*.yml"), ("所有文件", "*.*")],
            )
        if path:
            self.train_data_var.set(path)

    def _pick_val_dataset(self) -> None:
        if self._current_action_task("val") == "classify":
            path = filedialog.askdirectory(title="选择分类数据集根目录")
        else:
            path = filedialog.askopenfilename(
                title="选择 dataset.yaml",
                filetypes=[("YAML 文件", "*.yaml;*.yml"), ("所有文件", "*.*")],
            )
        if path:
            self.val_data_var.set(path)

    def _pick_prep_input(self) -> None:
        path = filedialog.askdirectory(title="选择原始数据目录")
        if path:
            self.prep_input_var.set(path)

    def _pick_val_weights(self) -> None:
        path = self._pick_weight_file("选择验证权重")
        if path:
            self.val_weights_var.set(path)
            guessed = self._guess_task_from_weight(Path(path))
            self.val_task_label_var.set(TASK_ID_TO_LABEL.get(guessed, "目标检测"))

    def _pick_predict_weights(self) -> None:
        path = self._pick_weight_file("选择预测权重")
        if path:
            self.predict_weights_var.set(path)
            guessed = self._guess_task_from_weight(Path(path))
            self.predict_task_label_var.set(TASK_ID_TO_LABEL.get(guessed, "目标检测"))

    def _pick_track_weights(self) -> None:
        path = self._pick_weight_file("选择跟踪权重")
        if path:
            self.track_weights_var.set(path)
            guessed = self._guess_task_from_weight(Path(path))
            if guessed not in TRACK_TASK_IDS:
                guessed = "detect"
            self.track_task_label_var.set(TASK_ID_TO_LABEL.get(guessed, "目标检测"))

    def _pick_export_weights(self) -> None:
        path = self._pick_weight_file("选择导出权重")
        if path:
            self.export_weights_var.set(path)

    def _pick_weight_file(self, title: str) -> str:
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")],
        )

    def _pick_export_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择导出目录")
        if path:
            self.export_output_dir_var.set(path)

    def _clear_export_output_dir(self) -> None:
        self.export_output_dir_var.set("")

    def _guess_task_from_weight(self, path: Path) -> str:
        lower_name = path.name.lower()
        if "-seg" in lower_name:
            return "segment"
        if "-cls" in lower_name:
            return "classify"
        if "-pose" in lower_name:
            return "pose"
        if "-obb" in lower_name:
            return "obb"
        return "detect"

    def _refresh_train_run_button(self) -> None:
        if not self.train_start_button:
            return
        action = self.train_action_var.get()
        self.train_start_button.configure(text=f"开始{ACTION_ID_TO_LABEL[action]}")

    def _set_button_enabled(self, button: tk.Button | None, enabled: bool, *, primary: bool) -> None:
        if button is None:
            return
        if primary:
            button.configure(
                state="normal" if enabled else "disabled",
                bg=PRIMARY if enabled else "#dbe6ff",
                fg="white" if enabled else "#9caecc",
                activebackground=PRIMARY_DARK if enabled else "#dbe6ff",
                activeforeground="white" if enabled else "#9caecc",
                disabledforeground="#9caecc",
                highlightbackground=PRIMARY if enabled else BORDER,
                cursor="hand2" if enabled else "arrow",
            )
            return
        button.configure(
            state="normal" if enabled else "disabled",
            bg=CARD_BG if enabled else PANEL_BG,
            fg=TEXT if enabled else "#9caecc",
            activebackground=PRIMARY_SOFT if enabled else PANEL_BG,
            activeforeground=TEXT if enabled else "#9caecc",
            disabledforeground="#9caecc",
            highlightbackground=BORDER,
            cursor="hand2" if enabled else "arrow",
        )

    def _refresh_run_action_buttons(self) -> None:
        is_running = self.process is not None and self.process.poll() is None
        has_result = False
        result_location = self.result_location_var.get().strip()
        if result_location:
            try:
                has_result = Path(result_location).expanduser().exists()
            except OSError:
                has_result = False
        has_log = bool(self.current_log_path and self.current_log_path.exists())

        for button in (self.train_start_button, self.export_start_button):
            self._set_button_enabled(button, not is_running, primary=True)
        for button in (self.train_stop_button, self.export_stop_button):
            self._set_button_enabled(button, is_running, primary=False)
        for button in (self.train_open_result_button, self.export_open_result_button):
            self._set_button_enabled(button, has_result, primary=False)
        for button in (self.train_open_log_button, self.export_open_log_button):
            self._set_button_enabled(button, has_log, primary=False)

        for context, buttons in (("train", self.train_preset_buttons), ("export", self.export_preset_buttons)):
            preset_name = (self.train_preset_var if context == "train" else self.export_preset_var).get().strip()
            scope = self._preset_scope_from_context(context)
            has_name = bool(preset_name)
            is_builtin = has_name and self._is_builtin_preset(scope, preset_name)
            has_preset = False
            if has_name:
                has_preset = is_builtin or self._find_preset_path(scope, preset_name).exists()
            self._set_button_enabled(buttons.get("save"), has_name and not is_builtin, primary=False)
            self._set_button_enabled(buttons.get("load"), has_preset, primary=False)
            self._set_button_enabled(buttons.get("delete"), has_preset and not is_builtin, primary=False)

    def _status_color(self, status: str) -> str:
        normalized = status.strip()
        if not normalized:
            return TEXT_MUTED
        if any(keyword in normalized for keyword in ("\u5931\u8d25", "\u9519\u8bef")):
            return "#d14343"
        if any(keyword in normalized for keyword in ("\u8fd0\u884c", "\u542f\u52a8", "\u5904\u7406\u4e2d", "\u914d\u7f6e\u4e2d", "\u68c0\u67e5\u4e2d")):
            return PRIMARY
        if any(keyword in normalized for keyword in ("\u5b8c\u6210", "\u6b63\u5e38")):
            return SUCCESS
        if any(keyword in normalized for keyword in ("\u53d6\u6d88", "\u7ed3\u675f", "\u505c\u6b62")):
            return "#f59e0b"
        return TEXT

    def _refresh_status_visuals(self) -> None:
        process_color = self._status_color(self.process_status_var.get())
        log_color = self._status_color(self.left_log_state_var.get())
        for label in self.process_status_labels:
            try:
                label.configure(fg=process_color)
            except tk.TclError:
                pass
        for entry, color in ((self.left_log_state_entry, log_color), (self.left_result_entry, TEXT)):
            if entry is None:
                continue
            try:
                entry.configure(fg=color)
            except tk.TclError:
                pass

    def _preset_scope_from_context(self, context: str) -> str:
        return self.train_action_var.get() if context == "train" else "export"

    def _builtin_preset_map(self, scope: str) -> dict[str, dict[str, Any]]:
        return BUILTIN_PRESETS.get(scope, {})

    def _is_builtin_preset(self, scope: str, name: str) -> bool:
        return name in self._builtin_preset_map(scope)

    def _recommended_preset_var(self, context: str) -> tk.StringVar:
        return self.train_recommended_preset_var if context == "train" else self.export_recommended_preset_var

    def _preset_dir(self, scope: str) -> Path:
        directory = PRESET_ROOT / scope
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _preset_path(self, scope: str, preset_name: str) -> Path:
        safe_name = quote(preset_name, safe="")
        return self._preset_dir(scope) / f"{safe_name}.json"

    def _find_preset_path(self, scope: str, preset_name: str) -> Path:
        encoded_path = self._preset_path(scope, preset_name)
        if encoded_path.exists():
            return encoded_path
        legacy_path = self._preset_dir(scope) / f"{preset_name}.json"
        return legacy_path

    def _list_presets(self, scope: str) -> list[str]:
        directory = self._preset_dir(scope)
        custom_names = sorted(
            unquote(item.stem)
            for item in directory.glob("*.json")
        )
        return custom_names

    def _refresh_train_preset_choices(self) -> None:
        if not self.train_preset_combo:
            return
        presets = list(self._builtin_preset_map(self.train_action_var.get()).keys())
        self.train_preset_combo.configure(values=presets)
        if self.train_recommended_preset_var.get() not in presets:
            self.train_recommended_preset_var.set("")

    def _refresh_export_preset_choices(self) -> None:
        if not self.export_preset_combo:
            return
        presets = list(self._builtin_preset_map("export").keys())
        self.export_preset_combo.configure(values=presets)
        if self.export_recommended_preset_var.get() not in presets:
            self.export_recommended_preset_var.set("")

    def _load_recommended_preset(self, context: str) -> None:
        scope = self._preset_scope_from_context(context)
        preset_name = self._recommended_preset_var(context).get().strip()
        if not preset_name:
            return
        payload = self._builtin_preset_map(scope).get(preset_name)
        if not payload:
            return
        self._apply_preset_payload(scope, payload)

    def _save_preset(self, context: str) -> None:
        scope = self._preset_scope_from_context(context)
        name_var = self.train_preset_var if context == "train" else self.export_preset_var
        preset_name = name_var.get().strip()
        if not preset_name:
            messagebox.showinfo("请输入名称", "请先在“参数预设”里输入自定义名称，再点击保存。")
            return
        if self._is_builtin_preset(scope, preset_name):
            messagebox.showwarning("名称占用", "这个名称已被内置推荐预设使用，请换一个名称。")
            return
        payload = self._collect_preset_payload(scope)
        preset_path = self._preset_path(scope, preset_name)
        preset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        name_var.set(preset_name)
        if context == "train":
            self._refresh_train_preset_choices()
        else:
            self._refresh_export_preset_choices()
        messagebox.showinfo("保存成功", f"已保存预设：{preset_name}")

    def _load_preset(self, context: str, notify: bool = True) -> None:
        scope = self._preset_scope_from_context(context)
        name_var = self.train_preset_var if context == "train" else self.export_preset_var
        preset_name = name_var.get().strip()
        if not preset_name:
            messagebox.showinfo("请输入名称", "请先输入要加载的自定义预设名称。")
            return
        if self._is_builtin_preset(scope, preset_name):
            self._apply_preset_payload(scope, self._builtin_preset_map(scope)[preset_name])
            if notify:
                messagebox.showinfo("加载成功", f"已加载同名推荐预设：{preset_name}")
            return
        preset_path = self._find_preset_path(scope, preset_name)
        if not preset_path.exists():
            messagebox.showwarning("预设不存在", f"找不到预设：{preset_path}")
            return
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        self._apply_preset_payload(scope, payload)
        self._recommended_preset_var(context).set("")
        if notify:
            messagebox.showinfo("加载成功", f"已加载预设：{preset_name}")

    def _delete_preset(self, context: str) -> None:
        scope = self._preset_scope_from_context(context)
        name_var = self.train_preset_var if context == "train" else self.export_preset_var
        preset_name = name_var.get().strip()
        if not preset_name:
            messagebox.showinfo("请输入名称", "请先输入要删除的自定义预设名称。")
            return
        if self._is_builtin_preset(scope, preset_name):
            messagebox.showinfo("内置预设", "推荐预设是软件内置的，不能删除。你可以直接加载，或另存为自己的版本。")
            return
        preset_path = self._find_preset_path(scope, preset_name)
        if not preset_path.exists():
            messagebox.showwarning("预设不存在", f"找不到预设：{preset_path}")
            return
        if not messagebox.askyesno("确认删除", f"确定删除预设“{preset_name}”吗？"):
            return
        preset_path.unlink(missing_ok=True)
        name_var.set("")
        if context == "train":
            self._refresh_train_preset_choices()
        else:
            self._refresh_export_preset_choices()

    def _collect_preset_payload(self, scope: str) -> dict[str, Any]:
        if scope == "train":
            return {
                "task": self._current_train_task(),
                "train_mode_label": self.train_mode_label_var.get(),
                "family": self.train_family_var.get(),
                "size": self.train_size_var.get(),
                "use_local_weights": bool(self.train_use_local_weights_var.get()),
                "model": self.train_model_var.get(),
                "data": self.train_data_var.get(),
                "config": self._collect_train_config(),
            }
        if scope == "val":
            return {
                "task": self._current_action_task("val"),
                "weights": self.val_weights_var.get(),
                "data": self.val_data_var.get(),
                "config": self._collect_val_config(),
            }
        if scope == "predict":
            return {
                "task": self._current_action_task("predict"),
                "weights": self.predict_weights_var.get(),
                "source": self.predict_source_var.get(),
                "config": self._collect_predict_config(),
            }
        if scope == "track":
            return {
                "task": self._current_action_task("track"),
                "weights": self.track_weights_var.get(),
                "source": self.track_source_var.get(),
                "config": self._collect_track_config(),
            }
        return {
            "task_label": self.export_task_label_var.get(),
            "weights": self.export_weights_var.get(),
            "output_dir": self.export_output_dir_var.get(),
            "format_label": self.export_format_label_var.get(),
            "imgsz": self.export_imgsz_var.get(),
            "device": self.export_device_var.get(),
            "config": self._collect_field_payload(self.export_fields),
        }

    def _apply_preset_payload(self, scope: str, payload: dict[str, Any]) -> None:
        if scope == "train":
            self.train_task_label_var.set(TASK_ID_TO_LABEL.get(str(payload.get("task") or "detect"), "目标检测"))
            self.train_mode_label_var.set(str(payload.get("train_mode_label") or "官方预训练"))
            self.train_family_var.set(str(payload.get("family") or "YOLO11"))
            self.train_size_var.set(str(payload.get("size") or "n"))
            self.train_use_local_weights_var.set(bool(payload.get("use_local_weights")))
            self.train_model_var.set(str(payload.get("model") or self.train_model_var.get()))
            self.train_data_var.set(str(payload.get("data") or self.train_data_var.get()))
            self._apply_store_payload(self.train_fields, payload.get("config") or {})
            return
        if scope == "val":
            self.val_task_label_var.set(TASK_ID_TO_LABEL.get(str(payload.get("task") or "detect"), "目标检测"))
            self.val_weights_var.set(str(payload.get("weights") or "未选择"))
            self.val_data_var.set(str(payload.get("data") or "暂无"))
            self._apply_store_payload(self.val_fields, payload.get("config") or {})
            return
        if scope == "predict":
            self.predict_task_label_var.set(TASK_ID_TO_LABEL.get(str(payload.get("task") or "detect"), "目标检测"))
            self.predict_weights_var.set(str(payload.get("weights") or "未选择"))
            self.predict_source_var.set(str(payload.get("source") or ""))
            self._apply_store_payload(self.predict_fields, payload.get("config") or {})
            return
        if scope == "track":
            task = str(payload.get("task") or "detect")
            if task not in TRACK_TASK_IDS:
                task = "detect"
            self.track_task_label_var.set(TASK_ID_TO_LABEL.get(task, "目标检测"))
            self.track_weights_var.set(str(payload.get("weights") or "未选择"))
            self.track_source_var.set(str(payload.get("source") or ""))
            self._apply_store_payload(self.track_fields, payload.get("config") or {})
            return

        if str(payload.get("task_label") or "") in EXPORT_TASK_LABEL_TO_ID:
            self.export_task_label_var.set(str(payload.get("task_label")))
        self.export_weights_var.set(str(payload.get("weights") or "未选择"))
        self.export_output_dir_var.set(str(payload.get("output_dir") or ""))
        if str(payload.get("format_label") or "") in self.export_format_map:
            self.export_format_label_var.set(str(payload.get("format_label")))
        self.export_imgsz_var.set(str(payload.get("imgsz") or "640"))
        self.export_device_var.set(str(payload.get("device") or "0"))
        self._apply_store_payload(self.export_fields, payload.get("config") or {})

    def _apply_store_payload(self, store: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
        for key, entry in store.items():
            if key not in payload:
                continue
            value = payload[key]
            var = entry["var"]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(stringify_value(value))

    def _write_temp_json(self, payload: dict[str, Any], prefix: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", prefix=prefix, delete=False)
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        path = Path(handle.name)
        self.temp_files.append(path)
        return path

    def _ensure_python(self) -> str | None:
        return self._resolve_python_path()

    def start_check_environment(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        self.left_result_var.set(f"正在检查内置环境：{python_path}")
        self.process_status_var.set("正在检查内置环境")
        command = [python_path, "-u", str(BACKEND), "check"]
        self._start_process(command, title="环境检查", mode_label="环境检查", preview_path=None)

    def start_configure_environment(self, *, auto_triggered: bool = False) -> None:
        python_path = str(BUNDLED_RUNTIME_PYTHON)
        self.python_var.set(python_path)

        probe = self._probe_environment_state(python_path)
        if auto_triggered:
            if str(probe.get("state") or "") == "ready":
                self.left_result_var.set(f"内置环境已可用：{python_path}")
                self.process_status_var.set("内置环境已可用")
                return
        elif not self._should_continue_after_probe(probe):
            return

        self.left_result_var.set(f"正在配置内置环境：{python_path}")
        self.process_status_var.set("正在配置内置环境")
        self._start_configure_process(python_path)

    def start_prepare_dataset(self) -> None:
        if self._current_train_task() != "detect":
            messagebox.showwarning("当前任务不支持", "整理原始检测数据仅用于 detect 任务。")
            return
        python_path = self._ensure_python()
        if not python_path:
            return
        raw_input = self.prep_input_var.get().strip()
        output_preview = self.prep_output_preview_var.get().strip()
        if not raw_input or raw_input == "暂无" or not output_preview or output_preview == "暂无":
            messagebox.showwarning("缺少参数", "请先选择原始数据目录。")
            return

        format_map = {
            "自动判断": "auto",
            "YOLO TXT": "yolo-flat",
            "LabelMe 矩形框 JSON": "labelme-json",
            "COCO JSON": "coco-json",
        }
        class_names = self._get_prep_class_names()
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "prepare-dataset",
            "--input",
            raw_input,
            "--output",
            output_preview,
            "--format",
            format_map.get(self.prep_format_var.get(), "auto"),
            "--val-ratio",
            self.prep_val_ratio_var.get().strip() or "0.2",
            "--seed",
            self.prep_seed_var.get().strip() or "42",
            "--copy-mode",
            PREP_COPY_MODE_LABEL_TO_ID.get(self.prep_copy_mode_var.get(), "copy"),
        ]
        if class_names:
            command.extend(["--class-names", class_names])
        class_names_file = self.prep_class_names_file_var.get().strip()
        if class_names_file:
            command.extend(["--class-names-file", class_names_file])
        if self.prep_strict_var.get():
            command.append("--strict")
        if self.prep_overwrite_var.get():
            command.append("--force")
        self._start_process(command, title="整理数据", mode_label="整理数据", preview_path=Path(output_preview))

    def start_train(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        model_path = self.train_model_var.get().strip()
        dataset_path = self.train_data_var.get().strip()
        if not model_path or model_path == "暂无" or not dataset_path or dataset_path == "暂无":
            messagebox.showwarning("缺少参数", "请先选择模型和训练入口。")
            return
        try:
            config_path = self._write_temp_json(self._collect_train_config(), "yolo_train_")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "train",
            "--task",
            self._current_train_task(),
            "--model",
            model_path,
            "--data",
            dataset_path,
            "--config-json",
            str(config_path),
        ]
        self._start_process(command, title="开始训练", mode_label="训练", preview_path=Path(self._expected_train_output_dir()))

    def start_val(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        weight_path = self.val_weights_var.get().strip()
        data_path = self.val_data_var.get().strip()
        if not weight_path or weight_path == "未选择" or not data_path or data_path == "暂无":
            messagebox.showwarning("缺少参数", "请先选择验证权重和验证数据。")
            return
        try:
            config_path = self._write_temp_json(self._collect_val_config(), "yolo_val_")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "val",
            "--task",
            self._current_action_task("val"),
            "--weights",
            weight_path,
            "--data",
            data_path,
            "--config-json",
            str(config_path),
        ]
        self._start_process(command, title="开始验证", mode_label="验证", preview_path=Path(self._expected_val_output_dir()))

    def start_predict(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        weight_path = self.predict_weights_var.get().strip()
        source = self.predict_source_var.get().strip()
        if not weight_path or weight_path == "未选择" or not source:
            messagebox.showwarning("缺少参数", "请先选择预测权重和预测源。")
            return
        try:
            config_path = self._write_temp_json(self._collect_predict_config(), "yolo_predict_")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "predict",
            "--task",
            self._current_action_task("predict"),
            "--weights",
            weight_path,
            "--source",
            source,
            "--config-json",
            str(config_path),
        ]
        self._start_process(command, title="开始预测", mode_label="预测", preview_path=Path(self._expected_predict_output_dir()))

    def start_track(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        weight_path = self.track_weights_var.get().strip()
        source = self.track_source_var.get().strip()
        if not weight_path or weight_path == "未选择" or not source:
            messagebox.showwarning("缺少参数", "请先选择跟踪权重和跟踪源。")
            return
        try:
            config_path = self._write_temp_json(self._collect_track_config(), "yolo_track_")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "track",
            "--task",
            self._current_action_task("track"),
            "--weights",
            weight_path,
            "--source",
            source,
            "--config-json",
            str(config_path),
        ]
        self._start_process(command, title="开始跟踪", mode_label="跟踪", preview_path=Path(self._expected_track_output_dir()))

    def start_export(self) -> None:
        python_path = self._ensure_python()
        if not python_path:
            return
        weight_path = self.export_weights_var.get().strip()
        if not weight_path or weight_path == "未选择":
            messagebox.showwarning("缺少参数", "请先选择导出权重。")
            return
        try:
            config_path = self._write_temp_json(self._collect_export_config(), "yolo_export_")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        command = [
            python_path,
            "-u",
            str(BACKEND),
            "export",
            "--task",
            EXPORT_TASK_LABEL_TO_ID.get(self.export_task_label_var.get(), "") or self._guess_task_from_weight(Path(weight_path)),
            "--weights",
            weight_path,
            "--config-json",
            str(config_path),
        ]
        self._start_process(command, title="开始导出", mode_label="导出", preview_path=Path(self._expected_export_target()))

    def _run_current_train_action(self) -> None:
        action = self.train_action_var.get()
        if action == "train":
            self.start_train()
        elif action == "val":
            self.start_val()
        elif action == "predict":
            self.start_predict()
        else:
            self.start_track()

    def _get_prep_class_names(self) -> str:
        raw_text = self.prep_class_names_text.get("1.0", "end").strip()
        if not raw_text:
            return ""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) == 1:
            return lines[0]
        return ",".join(lines)

    def _start_process(self, command: list[str], *, title: str, mode_label: str, preview_path: Path | None) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("任务进行中", "当前已有任务在运行，请先停止或等待完成。")
            return

        self._close_log_handle()
        self.current_log_path = self._create_log_path(title)
        self.current_log_handle = self.current_log_path.open("w", encoding="utf-8", buffering=1)
        self.last_result_path = preview_path if preview_path and preview_path.exists() else None
        self.process_status_var.set("任务已启动")
        self.left_log_state_var.set("运行中")
        self.left_result_var.set("任务已启动")
        self.result_location_var.set(str(preview_path) if preview_path else "")
        self._clear_log_text()
        self._append_log(f"=== {title} ===")
        self._append_log("命令: " + subprocess.list2cmdline(command))
        self._refresh_run_action_buttons()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(WORK_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=NO_WINDOW_FLAGS,
            )
        except Exception as exc:
            self.process_status_var.set("启动失败")
            self.left_log_state_var.set("启动失败")
            self._append_log(str(exc))
            self._close_log_handle()
            self._refresh_run_action_buttons()
            messagebox.showerror("启动失败", str(exc))
            return

        threading.Thread(target=self._pump_stdout, daemon=True).start()
        self._schedule_log_drain(LOG_POLL_ACTIVE_MS, reset=True)

    def _create_log_path(self, title: str) -> Path:
        logs_dir = WORK_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.replace(" ", "_")
        self._log_sequence += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{safe_title}_{timestamp}_{self._log_sequence:03d}.log"
        return logs_dir / file_name

    def _pump_stdout(self) -> None:
        assert self.process is not None
        process = self.process
        assert process.stdout is not None
        for line in process.stdout:
            self.log_queue.put(("line", line.rstrip()))
        self.log_queue.put(("exit", process.wait()))

    def _schedule_log_drain(self, delay_ms: int, *, reset: bool = False) -> None:
        if self._log_after_id and not reset:
            return
        if self._log_after_id:
            try:
                self.root.after_cancel(self._log_after_id)
            except tk.TclError:
                pass
            self._log_after_id = None
        try:
            if self.root.winfo_exists():
                self._log_after_id = self.root.after(delay_ms, self._drain_logs)
        except tk.TclError:
            self._log_after_id = None

    def _drain_logs(self) -> None:
        self._log_after_id = None
        pending_lines: list[str] = []
        processed = 0

        def flush_lines() -> None:
            if not pending_lines:
                return
            batch = pending_lines[:]
            pending_lines.clear()
            self._append_log_batch(batch)
            for line in batch:
                self._handle_backend_line(line)

        while processed < LOG_QUEUE_ITEMS_PER_TICK:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "line":
                pending_lines.append(str(payload))
            elif kind == "exit":
                flush_lines()
                self._append_log(f"退出码: {payload}")
                self.process = None
                if str(payload) == "0":
                    self.left_log_state_var.set("已完成")
                    if self.process_status_var.get() == "任务已启动":
                        self.process_status_var.set("已完成")
                else:
                    if self.process_status_var.get() not in {"任务失败", "已取消"}:
                        self.process_status_var.set("已结束")
                self._close_log_handle()
        flush_lines()

        has_backlog = not self.log_queue.empty()
        keep_draining = (self.process and self.process.poll() is None) or has_backlog
        try:
            if keep_draining and self.root.winfo_exists():
                self._log_after_id = self.root.after(LOG_POLL_ACTIVE_MS, self._drain_logs)
        except tk.TclError:
            self._log_after_id = None

    def _handle_backend_line(self, line: str) -> None:
        if not line.startswith("[") or "] " not in line:
            return
        tag, payload_text = line.split("] ", 1)
        tag = tag[1:]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return

        if tag == "STATUS":
            message = str(payload.get("message") or "运行中")
            self.process_status_var.set(message)
            self.left_log_state_var.set(message)
            return

        if tag == "RESULT" and payload.get("kind") == "train":
            output_dir = Path(str(payload.get("save_dir"))).expanduser()
            self.last_result_path = output_dir
            self.left_result_var.set(f"训练完成：{output_dir}")
            self.result_location_var.set(str(output_dir))
            self.process_status_var.set("训练完成")
            self._refresh_summary()
            messagebox.showinfo("训练完成", f"输出目录：{output_dir}")
            return

        if tag == "RESULT" and payload.get("kind") == "val":
            output_dir = Path(str(payload.get("save_dir"))).expanduser()
            summary = payload.get("summary") or {}
            self.last_result_path = output_dir
            self.left_result_var.set(f"验证完成：{output_dir}")
            self.result_location_var.set(str(output_dir))
            self.process_status_var.set("验证完成")
            self._refresh_summary()
            metric_text = self._format_metric_summary(summary)
            messagebox.showinfo("验证完成", f"输出目录：{output_dir}\n{metric_text}".strip())
            return

        if tag == "RESULT" and payload.get("kind") == "predict":
            output_dir = Path(str(payload.get("save_dir"))).expanduser()
            count = payload.get("count")
            self.last_result_path = output_dir
            self.left_result_var.set(f"预测完成：{output_dir}")
            self.result_location_var.set(str(output_dir))
            self.process_status_var.set("预测完成")
            self._refresh_summary()
            extra = f"\n处理结果数：{count}" if count is not None else ""
            messagebox.showinfo("预测完成", f"输出目录：{output_dir}{extra}")
            return

        if tag == "RESULT" and payload.get("kind") == "track":
            output_dir = Path(str(payload.get("save_dir"))).expanduser()
            count = payload.get("count")
            self.last_result_path = output_dir
            self.left_result_var.set(f"跟踪完成：{output_dir}")
            self.result_location_var.set(str(output_dir))
            self.process_status_var.set("跟踪完成")
            self._refresh_summary()
            extra = f"\n处理结果数：{count}" if count is not None else ""
            messagebox.showinfo("跟踪完成", f"输出目录：{output_dir}{extra}")
            return

        if tag == "RESULT" and payload.get("kind") == "export":
            output_path = Path(str(payload.get("output"))).expanduser()
            self.last_result_path = output_path
            self.left_result_var.set(f"导出完成：{output_path}")
            self.result_location_var.set(str(output_path.parent if output_path.is_file() else output_path))
            self.process_status_var.set("导出完成")
            self._refresh_summary()
            messagebox.showinfo("导出完成", f"输出结果：{output_path}")
            return

        if tag == "RESULT" and payload.get("kind") == "check":
            python_path = str(payload.get("python") or self.python_var.get())
            self.python_var.set(python_path)
            runtime_brief = self._runtime_environment_brief(payload)
            self.left_result_var.set(f"内置环境正常：{runtime_brief}")
            self.process_status_var.set("内置环境检查完成")
            self.result_location_var.set("")
            message = self._environment_dialog_message(payload)
            dialog = messagebox.showwarning if self._environment_dialog_needs_warning(payload) else messagebox.showinfo
            dialog("环境检查完成", message)
            return

        if tag == "RESULT" and payload.get("kind") == "configure-env":
            python_path = str(payload.get("python") or self.python_var.get())
            self.python_var.set(python_path)
            runtime_brief = self._runtime_environment_brief(payload)
            self.left_result_var.set(f"内置环境配置完成：{runtime_brief}")
            self.process_status_var.set("内置环境配置完成")
            self.result_location_var.set("")
            message = self._environment_dialog_message(payload, include_plan=True)
            torch_index = str(payload.get("torch_index") or "默认").strip()
            if torch_index:
                message += f"\nTorch 源：{torch_index}"
            dialog = messagebox.showwarning if self._environment_dialog_needs_warning(payload) else messagebox.showinfo
            dialog("环境配置完成", message)
            return

        if tag == "DATASET_PREP_JSON":
            dataset_yaml = str(payload.get("dataset_yaml") or "")
            output_dir = str(payload.get("output_dir") or "")
            if dataset_yaml:
                self.train_data_var.set(dataset_yaml)
            if output_dir:
                self.last_result_path = Path(output_dir)
                self.result_location_var.set(output_dir)
            self.left_result_var.set(f"整理完成：{dataset_yaml or output_dir}")
            self.process_status_var.set("数据整理完成")
            self._refresh_summary()
            messagebox.showinfo("整理完成", f"已生成：{dataset_yaml or output_dir}")
            return

        if tag == "ERROR":
            message = str(payload.get("message") or "未知错误")
            self.process_status_var.set("任务失败")
            self.left_log_state_var.set("任务失败")
            self.left_result_var.set(message)
            messagebox.showerror("任务失败", message)

    def _format_metric_summary(self, summary: Any) -> str:
        if not isinstance(summary, dict) or not summary:
            return ""
        parts: list[str] = []
        for key, value in list(summary.items())[:6]:
            try:
                if isinstance(value, (int, float)):
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value}")
            except Exception:
                parts.append(f"{key}={value}")
        return "指标摘要：" + "，".join(parts) if parts else ""

    def _append_log(self, text: str) -> None:
        self._append_log_batch([text])

    def _append_log_batch(self, lines: list[str]) -> None:
        if not lines:
            return
        text = "".join(f"{line}\n" for line in lines)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self._visible_log_lines += len(lines)
        overflow = self._visible_log_lines - LOG_VISIBLE_LINE_LIMIT
        if overflow > 0:
            self.log_text.delete("1.0", f"{overflow + 1}.0")
            self._visible_log_lines -= overflow
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        if self.current_log_handle is not None:
            self.current_log_handle.write(text)
            self.current_log_handle.flush()

    def _clear_log_text(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._visible_log_lines = 0

    def _close_log_handle(self) -> None:
        if self.current_log_handle is not None:
            try:
                self.current_log_handle.close()
            except OSError:
                pass
            self.current_log_handle = None
        self._refresh_run_action_buttons()

    def stop_process(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process_status_var.set("已取消")
            self.left_log_state_var.set("已取消")
            self._append_log("已发送停止信号。")
            self._refresh_run_action_buttons()

    def open_result_location(self) -> None:
        location = self.result_location_var.get().strip()
        if not location:
            messagebox.showinfo("暂无结果", "当前还没有可打开的结果位置。")
            return
        self._open_path(Path(location))

    def open_log_file(self) -> None:
        if not self.current_log_path or not self.current_log_path.exists():
            messagebox.showinfo("暂无日志", "当前还没有生成日志文件。")
            return
        self._open_path(self.current_log_path)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("路径不存在", f"找不到：{path}")
            return
        try:
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _on_close(self) -> None:
        SmartComboBox.close_all()
        if self._log_after_id:
            try:
                self.root.after_cancel(self._log_after_id)
            except tk.TclError:
                pass
            self._log_after_id = None
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("确认退出", "当前任务仍在运行，确定退出吗？"):
                return
            self.process.terminate()
        self._close_log_handle()
        for path in self.temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.root.destroy()


def apply_window_icon(root: tk.Tk) -> None:
    try:
        if ICON_ICO.exists():
            root.iconbitmap(default=str(ICON_ICO))
    except Exception:
        pass
    try:
        if ICON_PNG.exists():
            icon_image = tk.PhotoImage(file=str(ICON_PNG))
            root.iconphoto(True, icon_image)
            root._app_icon_image = icon_image  # type: ignore[attr-defined]
            root.after(50, lambda: root.iconphoto(True, icon_image))
    except Exception:
        pass


def run_backend_command_from_launcher(argv: list[str]) -> int:
    if not BACKEND.exists():
        print(f"[ERROR] 后端脚本不存在：{BACKEND}", flush=True)
        return 1

    spec = importlib.util.spec_from_file_location("yolo_local_desktop_backend", BACKEND)
    if spec is None or spec.loader is None:
        print(f"[ERROR] 无法加载后端脚本：{BACKEND}", flush=True)
        return 1

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend_main = getattr(module, "main", None)
    if not callable(backend_main):
        print(f"[ERROR] 后端入口无效：{BACKEND}", flush=True)
        return 1
    return int(backend_main(argv))


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == BACKEND_LAUNCH_FLAG:
        raise SystemExit(run_backend_command_from_launcher(sys.argv[2:]))

    root = tk.Tk()
    apply_window_icon(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
