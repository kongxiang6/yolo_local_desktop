from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
OPTION_DIR = ASSET_DIR / "icon_options"
PNG_PATH = ASSET_DIR / "yolotool_icon.png"
ICO_PATH = ASSET_DIR / "yolotool_icon.ico"
BOARD_PATH = ROOT / "designs" / "icon_style_board.png"
SIZE = 1024

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(path: Path, size: int):
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return None


def rounded_mask(size: int, radius: int, inset: int = 0) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((inset, inset, size - inset, size - inset), radius=radius, fill=255)
    return mask


def gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        ratio = y / max(size - 1, 1)
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line((0, y, size, y), fill=(*color, 255))
    return image


def make_base(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    inset = 92
    radius = 230
    base = gradient(SIZE, top, bottom)
    base.putalpha(rounded_mask(SIZE, radius, inset))

    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle((inset, inset, SIZE - inset, SIZE - inset), radius=radius, fill=(94, 205, 255, 36))
    glow = glow.filter(ImageFilter.GaussianBlur(36))
    base.alpha_composite(glow)

    top_glass = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    tg = ImageDraw.Draw(top_glass)
    tg.rounded_rectangle((inset + 34, inset + 24, SIZE - inset - 34, SIZE // 2 - 18), radius=170, fill=(255, 255, 255, 68))
    top_glass = top_glass.filter(ImageFilter.GaussianBlur(8))
    base.alpha_composite(top_glass)

    edge = ImageDraw.Draw(base)
    edge.rounded_rectangle((inset, inset, SIZE - inset, SIZE - inset), radius=radius, outline=(232, 246, 255, 235), width=8)
    return base


def add_soft_shadow(image: Image.Image, points: list[tuple[float, float]], *, blur: int = 18, offset: tuple[int, int] = (18, 22)) -> None:
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shifted = [(x + offset[0], y + offset[1]) for x, y in points]
    shadow_draw.polygon(shifted, fill=(20, 114, 226, 78))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    image.alpha_composite(shadow)


def draw_variant_soft_y() -> Image.Image:
    image = make_base((138, 206, 255), (24, 144, 241))
    draw = ImageDraw.Draw(image)
    left = [(332, 290), (470, 290), (582, 496), (510, 560), (378, 364)]
    right = [(690, 290), (806, 290), (648, 744), (512, 830), (566, 670)]
    stem = [(474, 510), (560, 510), (686, 740), (600, 790)]
    add_soft_shadow(image, left)
    add_soft_shadow(image, right)
    add_soft_shadow(image, stem, blur=16, offset=(14, 18))
    draw.polygon(left, fill=(255, 255, 255, 245))
    draw.polygon(right, fill=(236, 247, 255, 248))
    draw.polygon(stem, fill=(251, 253, 255, 250))
    return image


def draw_variant_folded_y() -> Image.Image:
    image = make_base((127, 196, 255), (15, 134, 238))
    draw = ImageDraw.Draw(image)
    left = [(300, 288), (430, 288), (560, 490), (470, 548), (338, 360)]
    right = [(710, 288), (830, 288), (642, 762), (520, 838), (614, 620)]
    center = [(468, 510), (560, 510), (642, 650), (556, 700)]
    add_soft_shadow(image, left, blur=14, offset=(16, 20))
    add_soft_shadow(image, right, blur=16, offset=(20, 24))
    draw.polygon(left, fill=(255, 255, 255, 245))
    draw.polygon(right, fill=(244, 251, 255, 245))
    draw.polygon(center, fill=(227, 241, 255, 235))
    return image


def draw_variant_ribbon_y() -> Image.Image:
    image = make_base((146, 214, 255), (19, 145, 243))
    draw = ImageDraw.Draw(image)
    left = [(320, 272), (468, 272), (590, 500), (522, 552), (394, 372)]
    right = [(694, 272), (816, 272), (648, 776), (534, 850), (594, 644)]
    notch = [(504, 528), (556, 528), (672, 732), (620, 764)]
    add_soft_shadow(image, left, blur=18, offset=(16, 20))
    add_soft_shadow(image, right, blur=20, offset=(20, 24))
    draw.polygon(left, fill=(255, 255, 255, 248))
    draw.polygon(right, fill=(239, 248, 255, 248))
    draw.polygon(notch, fill=(250, 252, 255, 245))
    return image


def draw_variant_glossy_y() -> Image.Image:
    image = make_base((140, 208, 255), (22, 148, 245))
    draw = ImageDraw.Draw(image)
    left = [(322, 282), (458, 282), (598, 506), (514, 568), (376, 372)]
    right = [(700, 282), (828, 282), (648, 786), (520, 860), (604, 630)]
    add_soft_shadow(image, left, blur=20, offset=(22, 26))
    add_soft_shadow(image, right, blur=22, offset=(24, 30))
    draw.polygon(left, fill=(255, 255, 255, 250))
    draw.polygon(right, fill=(233, 245, 255, 248))
    draw.rounded_rectangle((428, 512, 592, 612), radius=42, fill=(245, 251, 255, 210))
    return image


VARIANTS = {
    "soft_y": ("柔光Y", draw_variant_soft_y),
    "folded_y": ("折角Y", draw_variant_folded_y),
    "ribbon_y": ("丝带Y", draw_variant_ribbon_y),
    "glossy_y": ("高光Y", draw_variant_glossy_y),
}


def save_icon_files(image: Image.Image, png_path: Path, ico_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    image.save(ico_path, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])


def build_board(outputs: list[tuple[str, Path]]) -> Image.Image:
    board = Image.new("RGB", (1800, 1100), "#eef4ff")
    draw = ImageDraw.Draw(board)
    title_font = font(FONT_BOLD, 56)
    subtitle_font = font(FONT_REGULAR, 28)
    label_font = font(FONT_BOLD, 32)

    draw.text((60, 40), "YoloTool 图标方案对比", fill="#243348", font=title_font)
    draw.text((60, 112), "按你给的参考图，统一改成蓝底 + 白色立体 Y 的 glossy 风格。", fill="#6c7c96", font=subtitle_font)

    for index, (label, png_path) in enumerate(outputs):
        row = index // 2
        col = index % 2
        x = 80 + col * 850
        y = 190 + row * 420
        draw.rounded_rectangle((x, y, x + 760, y + 340), radius=34, fill="#ffffff", outline="#dce7fb", width=3)
        draw.text((x + 34, y + 28), label, fill="#243348", font=label_font)
        icon = Image.open(png_path).convert("RGBA").resize((220, 220))
        board.paste(icon, (x + 270, y + 78), icon)
    return board


def main() -> None:
    OPTION_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, Path]] = []
    selected_key = "soft_y"

    for key, (label, builder) in VARIANTS.items():
        image = builder()
        png_path = OPTION_DIR / f"{key}.png"
        ico_path = OPTION_DIR / f"{key}.ico"
        save_icon_files(image, png_path, ico_path)
        outputs.append((label, png_path))
        if key == selected_key:
            save_icon_files(image, PNG_PATH, ICO_PATH)

    board = build_board(outputs)
    board.save(BOARD_PATH)

    print(PNG_PATH)
    print(ICO_PATH)
    print(BOARD_PATH)


if __name__ == "__main__":
    main()
