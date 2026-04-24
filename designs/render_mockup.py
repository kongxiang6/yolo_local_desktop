from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent
W, H = 1600, 980

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
if not BOLD_FONT_PATH.exists():
    BOLD_FONT_PATH = FONT_PATH

F_H1 = ImageFont.truetype(str(BOLD_FONT_PATH), 28)
F_H2 = ImageFont.truetype(str(BOLD_FONT_PATH), 20)
F_H3 = ImageFont.truetype(str(BOLD_FONT_PATH), 16)
F_BODY = ImageFont.truetype(str(FONT_PATH), 14)
F_SMALL = ImageFont.truetype(str(FONT_PATH), 12)


STYLES = {
    "minimal_blue": {
        "name": "简约浅蓝",
        "bg": "#F6F8FC",
        "panel": "#FFFFFF",
        "panel_soft": "#F9FBFF",
        "line": "#E6ECF5",
        "text": "#1F2A37",
        "muted": "#6B7A90",
        "primary": "#3B82F6",
        "primary_soft": "#EAF2FF",
        "primary_text": "#FFFFFF",
        "success": "#22C55E",
        "shadow": (20, 38, 63, 18),
        "ghost": "#FFFFFF",
        "ghost_text": "#1F2A37",
    },
    "dark_tech": {
        "name": "深色科技",
        "bg": "#0B1220",
        "panel": "#121B2B",
        "panel_soft": "#0F1726",
        "line": "#23324A",
        "text": "#E5EEF9",
        "muted": "#8EA0BC",
        "primary": "#4F8CFF",
        "primary_soft": "#15233D",
        "primary_text": "#FFFFFF",
        "success": "#2ED47A",
        "shadow": (0, 0, 0, 60),
        "ghost": "#182235",
        "ghost_text": "#E5EEF9",
    },
    "business_green": {
        "name": "商务绿色",
        "bg": "#F5F8F3",
        "panel": "#FFFFFF",
        "panel_soft": "#FAFDF8",
        "line": "#DFE9E0",
        "text": "#203126",
        "muted": "#6E8573",
        "primary": "#2F9E63",
        "primary_soft": "#E8F7EF",
        "primary_text": "#FFFFFF",
        "success": "#1E8A55",
        "shadow": (20, 38, 63, 16),
        "ghost": "#FFFFFF",
        "ghost_text": "#203126",
    },
    "glass_light": {
        "name": "玻璃轻拟态",
        "bg": "#EEF4FF",
        "panel": "#FFFFFFCC",
        "panel_soft": "#FFFFFFD8",
        "line": "#DCE7FB",
        "text": "#243348",
        "muted": "#6C7C96",
        "primary": "#5B8CFF",
        "primary_soft": "#EDF3FF",
        "primary_text": "#FFFFFF",
        "success": "#31C48D",
        "shadow": (91, 140, 255, 26),
        "ghost": "#FFFFFFC8",
        "ghost_text": "#243348",
    },
    "premium_beige": {
        "name": "暖米高级",
        "bg": "#F6F1EA",
        "panel": "#FFFDF9",
        "panel_soft": "#FBF6EF",
        "line": "#E8DDD0",
        "text": "#3A2F28",
        "muted": "#8A786B",
        "primary": "#C78C52",
        "primary_soft": "#F6E9DA",
        "primary_text": "#FFFFFF",
        "success": "#5DAA6F",
        "shadow": (62, 45, 33, 18),
        "ghost": "#FFFDF9",
        "ghost_text": "#3A2F28",
    },
    "mono_pro": {
        "name": "黑白极简",
        "bg": "#F3F4F6",
        "panel": "#FFFFFF",
        "panel_soft": "#FAFAFA",
        "line": "#D8D8D8",
        "text": "#111111",
        "muted": "#666666",
        "primary": "#111111",
        "primary_soft": "#ECECEC",
        "primary_text": "#FFFFFF",
        "success": "#3B3B3B",
        "shadow": (0, 0, 0, 14),
        "ghost": "#FFFFFF",
        "ghost_text": "#111111",
    },
    "soft_purple": {
        "name": "柔和紫灰",
        "bg": "#F5F3FB",
        "panel": "#FFFFFF",
        "panel_soft": "#FBFAFE",
        "line": "#E5DFF4",
        "text": "#2F2A3D",
        "muted": "#7B7490",
        "primary": "#8B78E6",
        "primary_soft": "#EEEAFE",
        "primary_text": "#FFFFFF",
        "success": "#53B58B",
        "shadow": (84, 71, 140, 18),
        "ghost": "#FFFFFF",
        "ghost_text": "#2F2A3D",
    },
    "focus_orange": {
        "name": "活力橙彩",
        "bg": "#FFF7F1",
        "panel": "#FFFFFF",
        "panel_soft": "#FFF9F3",
        "line": "#F2DFC9",
        "text": "#31251C",
        "muted": "#8C6F57",
        "primary": "#F28C28",
        "primary_soft": "#FFF0DF",
        "primary_text": "#FFFFFF",
        "success": "#2BAA7A",
        "shadow": (97, 58, 17, 16),
        "ghost": "#FFFFFF",
        "ghost_text": "#31251C",
    },
}


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline, radius=18, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def shadow(draw: ImageDraw.ImageDraw, xy, fill, radius=24, dx=8, dy=10):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1 + dx, y1 + dy, x2 + dx, y2 + dy), radius=radius, fill=fill)


def draw_value(draw: ImageDraw.ImageDraw, theme: dict, x: int, y: int, label: str, value: str, width: int = 420):
    draw.text((x, y), label, font=F_SMALL, fill=theme["muted"])
    rounded(draw, (x + 76, y - 6, x + width, y + 28), theme["ghost"], theme["line"], 12)
    draw.text((x + 92, y + 2), value, font=F_BODY, fill=theme["text"])


def draw_badge(draw: ImageDraw.ImageDraw, theme: dict, xy, text: str):
    rounded(draw, xy, theme["primary_soft"], None, 12)
    draw.text((xy[0] + 10, xy[1] + 6), text, font=F_SMALL, fill=theme["primary"])


def build_mockup(theme: dict):
    image = Image.new("RGBA", (W, H), theme["bg"])
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    shadow(draw, (16, 16, W - 16, H - 16), theme["shadow"], 28)
    rounded(draw, (16, 16, W - 16, H - 16), theme["panel_soft"], theme["line"], 28)

    rounded(draw, (16, 16, W - 16, 68), theme["panel"], theme["line"], 28)
    draw.text((36, 28), "YoloTool", font=F_H2, fill=theme["text"])
    for index in range(3):
        left = W - 110 + index * 22
        draw.ellipse((left, 30, left + 14, 44), fill=theme["line"])

    left_panel = (28, 84, 1006, H - 28)
    right_panel = (1022, 84, W - 28, H - 28)
    shadow(draw, left_panel, theme["shadow"])
    shadow(draw, right_panel, theme["shadow"])
    rounded(draw, left_panel, theme["panel"], theme["line"], 24)
    rounded(draw, right_panel, theme["panel"], theme["line"], 24)

    cards = [
        (42, 100, 992, 196, "当前入口"),
        (42, 208, 992, 304, "最近结果"),
        (42, 316, 992, H - 44, "实时日志"),
    ]
    for x1, y1, x2, y2, title in cards:
        rounded(draw, (x1, y1, x2, y2), theme["panel_soft"], theme["line"], 18)
        draw.rounded_rectangle((x1, y1, x2, y1 + 40), radius=18, fill=theme["primary_soft"], outline=theme["line"])
        draw.rounded_rectangle((x1, y1 + 22, x2, y1 + 40), radius=0, fill=theme["primary_soft"])
        draw.rounded_rectangle((x1 + 12, y1 + 10, x1 + 16, y1 + 30), radius=2, fill=theme["primary"])
        draw.text((x1 + 26, y1 + 9), title, font=F_H3, fill=theme["text"])

    draw_value(draw, theme, 58, 112, "数据集", "coco128.yaml")
    draw_value(draw, theme, 58, 146, "模型", "yolo11n.pt")
    draw_value(draw, theme, 58, 180, "输出目录", r"runs\detect\train-2026")
    draw_value(draw, theme, 58, 254, "结果", "训练完成，mAP50 = 0.842，权重已保存")
    draw_value(draw, theme, 58, 362, "当前日志", "等待开始新的操作...")

    log_x1, log_y1, log_x2, log_y2 = 58, 402, 974, H - 66
    rounded(draw, (log_x1, log_y1, log_x2, log_y2), theme["panel"], theme["line"], 16)
    logs = [
        "[STATUS] 环境检查通过，已识别 CUDA:0",
        "[TRAIN] epochs=100  batch=16  imgsz=640",
        "[VAL] precision=0.81  recall=0.77  mAP50=0.84",
        "[EXPORT] onnx 已生成，输出目录已同步",
        "[TIP] 鼠标悬停参数项可查看说明",
    ]
    for index, text in enumerate(logs):
        yy = log_y1 + 18 + index * 32
        if index == 0:
            draw.rounded_rectangle((log_x1 + 12, yy - 6, log_x2 - 12, yy + 22), radius=10, fill=theme["primary_soft"])
        color = theme["primary"] if index == 4 else theme["text"]
        draw.text((log_x1 + 20, yy), text, font=F_BODY, fill=color)

    rounded(draw, (1038, 100, 1572, 150), theme["panel_soft"], theme["line"], 18)
    rounded(draw, (1046, 108, 1298, 142), theme["primary"], None, 14)
    rounded(draw, (1310, 108, 1564, 142), theme["ghost"], theme["line"], 14)
    draw.text((1142, 116), "训练工作台", font=F_H3, fill=theme["primary_text"])
    draw.text((1408, 116), "导出工作台", font=F_H3, fill=theme["ghost_text"])

    rounded(draw, (1038, 164, 1572, 218), theme["primary_soft"], theme["line"], 16)
    draw.text((1060, 178), f"{theme['name']} · 新手友好布局", font=F_H1, fill=theme["text"])
    draw_badge(draw, theme, (1472, 174, 1522, 202), "v2")

    sections = [
        ("训练入口", True, 132),
        ("训练参数", True, 310),
        ("验证 / 预测 / 跟踪", False, 56),
        ("导出与预设", False, 56),
        ("运行操作", True, 150),
    ]
    section_y = 236
    for index, (title, is_open, height) in enumerate(sections):
        x1, x2 = 1038, 1572
        y1, y2 = section_y, section_y + height
        rounded(draw, (x1, y1, x2, y2), theme["panel"], theme["line"], 18)
        draw.rounded_rectangle((x1, y1, x2, y1 + 42), radius=18, fill=theme["panel"], outline=theme["line"])
        draw.rounded_rectangle((x1, y1 + 22, x2, y1 + 42), radius=0, fill=theme["panel"])
        draw.rounded_rectangle((x1 + 10, y1 + 10, x1 + 14, y1 + 30), radius=2, fill=theme["primary"] if is_open else theme["line"])
        draw.text((x1 + 24, y1 + 10), title, font=F_H3, fill=theme["text"])
        draw.text((x2 - 26, y1 + 10), "∨" if is_open else ">", font=F_H3, fill=theme["muted"])

        if index == 0:
            items = [
                ("任务类型", "目标检测"),
                ("训练模式", "官方预训练"),
                ("模型系列", "YOLO11"),
                ("模型尺寸", "n"),
            ]
            for sub_index, (label, value) in enumerate(items):
                xx = x1 + 18 + (sub_index % 2) * 258
                yy = y1 + 58 + (sub_index // 2) * 34
                draw.text((xx, yy), label, font=F_SMALL, fill=theme["muted"])
                rounded(draw, (xx + 78, yy - 6, xx + 230, yy + 24), theme["ghost"], theme["line"], 12)
                draw.text((xx + 92, yy), value, font=F_BODY, fill=theme["text"])

        if index == 1:
            rows = [
                ("epochs", "100", "训练轮次"),
                ("batch", "16", "批次大小"),
                ("imgsz", "640", "输入尺寸"),
                ("device", "0", "显卡编号"),
                ("optimizer", "auto", "优化器"),
                ("amp", "true", "混合精度"),
                ("project", "runs/detect", "保存目录"),
                ("name", "train_minimal", "实验名称"),
            ]
            for row_index, (key, value, desc) in enumerate(rows):
                yy = y1 + 56 + row_index * 30
                draw.text((x1 + 18, yy), key, font=F_SMALL, fill=theme["muted"])
                rounded(draw, (x1 + 98, yy - 6, x1 + 240, yy + 22), theme["ghost"], theme["line"], 10)
                draw.text((x1 + 112, yy), value, font=F_BODY, fill=theme["text"])
                draw.text((x1 + 254, yy), desc, font=F_SMALL, fill=theme["muted"])

        if index == 4:
            rounded(draw, (x1 + 20, y1 + 56, x1 + 514, y1 + 98), theme["primary"], None, 14)
            draw.text((x1 + 228, y1 + 66), "开始训练", font=F_H3, fill=theme["primary_text"])
            rounded(draw, (x1 + 20, y1 + 106, x1 + 260, y1 + 142), theme["ghost"], theme["line"], 14)
            rounded(draw, (x1 + 274, y1 + 106, x1 + 514, y1 + 142), theme["ghost"], theme["line"], 14)
            draw.text((x1 + 98, y1 + 114), "打开结果", font=F_BODY, fill=theme["ghost_text"])
            draw.text((x1 + 352, y1 + 114), "保存预设", font=F_BODY, fill=theme["ghost_text"])
            draw.ellipse((x1 + 452, y1 + 64, x1 + 468, y1 + 80), fill=theme["success"])
            draw.text((x1 + 474, y1 + 62), "状态：准备就绪", font=F_SMALL, fill=theme["muted"])

        section_y = y2 + 12

    draw.text((42, H - 34), "设计关键词：简约、留白、低干扰、参数解释清晰、适合新手操作", font=F_SMALL, fill=theme["muted"])
    return Image.alpha_composite(image, canvas).convert("RGB")


def build_board(outputs: list[tuple[str, Path]]):
    board_w, board_h = 2200, 2500
    board = Image.new("RGB", (board_w, board_h), "#EEF2F7")
    draw = ImageDraw.Draw(board)

    draw.text((70, 48), "YoloTool UI 风格对比稿", font=F_H1, fill="#17212F")
    draw.text((70, 92), "统一布局，仅替换视觉风格，方便直接比较气质与可读性。", font=F_BODY, fill="#5E6B7A")

    cols = 2
    gap_x, gap_y = 50, 60
    margin_x, margin_y = 70, 150
    card_w, card_h = 1005, 540
    thumb_w, thumb_h = 965, 400

    for index, (style_name, path) in enumerate(outputs):
        col = index % cols
        row = index // cols
        x1 = margin_x + col * (card_w + gap_x)
        y1 = margin_y + row * (card_h + gap_y)
        x2 = x1 + card_w
        y2 = y1 + card_h

        draw.rounded_rectangle((x1 + 10, y1 + 14, x2 + 10, y2 + 14), radius=28, fill=(24, 39, 75, 18))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=28, fill="#FFFFFF", outline="#DCE3EE", width=2)
        draw.text((x1 + 26, y1 + 22), style_name, font=F_H2, fill="#17212F")
        draw.text((x1 + 26, y1 + 58), "适合同一套功能界面的视觉方向选择", font=F_SMALL, fill="#6A7686")

        thumb = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        board.paste(thumb, (x1 + 20, y1 + 104))
        draw.rounded_rectangle((x1 + 20, y1 + 104, x1 + 20 + thumb_w, y1 + 104 + thumb_h), radius=18, outline="#DCE3EE", width=2)

    return board


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for style_id, theme in STYLES.items():
        image = build_mockup(theme)
        output = OUTPUT_DIR / f"ui_{style_id}.png"
        image.save(output, quality=95)
        outputs.append((theme["name"], output))
        print(output)
    board = build_board(outputs)
    board_output = OUTPUT_DIR / "ui_style_board.png"
    board.save(board_output, quality=95)
    print(board_output)


if __name__ == "__main__":
    main()
