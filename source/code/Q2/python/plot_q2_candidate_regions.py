"""Render the Q2 candidate-region figure from saved selector outputs."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "Q2" / "generic_selector"
FIGURES = ROOT / "paper" / "figures"

PRIMARY = "#1A6FC4"
PRIMARY_LIGHT = "#5B9BD5"
PRIMARY_PALE = "#D9EAF8"
ENHANCED = "#138A72"
ENHANCED_LIGHT = "#49A993"
ENHANCED_PALE = "#CFEDE6"
ACCENT = "#E28E2C"
ACCENT_PALE = "#F5D9A0"
BASELINE = "#767676"
GRID = "#D8D8D8"
TEXT = "#333333"
WHITE = "#FFFFFF"

WIDTH, HEIGHT = 3264, 1472  # 10.2 x 4.6 inches at 320 dpi
MARGIN_X, TOP, BOTTOM, GAP = 160, 110, 310, 170
PANEL_W = (WIDTH - 2 * MARGIN_X - GAP) // 2
PANEL_H = HEIGHT - TOP - BOTTOM
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def transform(point, box, limits) -> tuple[int, int]:
    left, top, right, bottom = box
    xmin, xmax, ymin, ymax = limits
    scale = min((right - left) / (xmax - xmin), (bottom - top) / (ymax - ymin))
    x0 = (left + right - scale * (xmax - xmin)) / 2.0
    y0 = (top + bottom + scale * (ymax - ymin)) / 2.0
    return (
        round(x0 + (point[0] - xmin) * scale),
        round(y0 - (point[1] - ymin) * scale),
    )


def draw_dashed_ellipse(draw, bbox, color: str, width: int = 3) -> None:
    start = 0
    while start < 360:
        draw.arc(bbox, start=start, end=min(360, start + 8), fill=color, width=width)
        start += 13


def polygon_pixels(points, box, limits) -> list[tuple[int, int]]:
    return [transform(point, box, limits) for point in points]


def draw_case(
    image,
    draw,
    panel_left: int,
    data: dict,
    title: str,
    limits,
    conservative: dict | None = None,
) -> None:
    plot_box = (panel_left, TOP, panel_left + PANEL_W, TOP + PANEL_H)
    first = data["input"]["first_point_m"]
    budget = data["input"]["movement_budget_m"]
    source = data["first_source_outer_polygon_world_m"]
    safe = data["guaranteed_reception_inner_polygon_world_m"]
    best = data["recommended_point"]["world_m"]
    near = [row["world_m"] for row in data["near_optimal_candidate_points"]]

    draw.rectangle(plot_box, fill="#FAFAFA", outline=BASELINE, width=2)
    xmin, xmax, ymin, ymax = limits
    tick_font = font(23)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        xv = xmin + fraction * (xmax - xmin)
        yv = ymin + fraction * (ymax - ymin)
        xpixel, _ = transform((xv, ymin), plot_box, limits)
        _, ypixel = transform((xmin, yv), plot_box, limits)
        draw.line((xpixel, plot_box[1], xpixel, plot_box[3]), fill=GRID, width=2)
        draw.line((plot_box[0], ypixel, plot_box[2], ypixel), fill=GRID, width=2)
        draw.text((xpixel, plot_box[3] + 12), f"{xv:.0f}", fill=TEXT, font=tick_font, anchor="ma")
        draw.text((plot_box[0] - 14, ypixel), f"{yv:.0f}", fill=TEXT, font=tick_font, anchor="rm")

    def circle_bbox(center, radius):
        p1 = transform((center[0] - radius, center[1] - radius), plot_box, limits)
        p2 = transform((center[0] + radius, center[1] + radius), plot_box, limits)
        return (p1[0], p2[1], p2[0], p1[1])

    layer = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))
    geometry = ImageDraw.Draw(layer)
    geometry.ellipse(circle_bbox((0.0, 0.0), 1800.0), outline=BASELINE, width=4)
    geometry.polygon(polygon_pixels(safe, plot_box, limits), fill=ENHANCED_PALE, outline=ENHANCED)
    geometry.line(polygon_pixels(safe + [safe[0]], plot_box, limits), fill=ENHANCED, width=4, joint="curve")
    if conservative is not None:
        conservative_safe = conservative["guaranteed_reception_inner_polygon_world_m"]
        geometry.polygon(
            polygon_pixels(conservative_safe, plot_box, limits),
            fill=PRIMARY_PALE,
            outline=PRIMARY,
        )
        geometry.line(
            polygon_pixels(conservative_safe + [conservative_safe[0]], plot_box, limits),
            fill=PRIMARY,
            width=4,
            joint="curve",
        )
    geometry.polygon(polygon_pixels(source, plot_box, limits), fill=ACCENT_PALE, outline=ACCENT)
    geometry.line(polygon_pixels(source + [source[0]], plot_box, limits), fill=ACCENT, width=5, joint="curve")
    draw_dashed_ellipse(geometry, circle_bbox(first, budget), BASELINE, width=4)

    for point in near:
        x, y = transform(point, plot_box, limits)
        geometry.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ENHANCED_LIGHT)

    fx, fy = transform(first, plot_box, limits)
    bx, by = transform(best, plot_box, limits)
    if conservative is not None:
        fallback_best = conservative["recommended_point"]["world_m"]
        cx, cy = transform(fallback_best, plot_box, limits)
        geometry.line((fx, fy, cx, cy), fill=PRIMARY, width=4)
        geometry.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=WHITE, outline=PRIMARY, width=4)
        geometry.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=PRIMARY)
    geometry.line((fx, fy, bx, by), fill=ENHANCED, width=5)
    geometry.rectangle((fx - 8, fy - 8, fx + 8, fy + 8), fill=TEXT)
    geometry.ellipse((bx - 16, by - 16, bx + 16, by + 16), fill=WHITE, outline=ENHANCED, width=5)
    geometry.ellipse((bx - 8, by - 8, bx + 8, by + 8), fill=ENHANCED)
    crop = layer.crop(plot_box)
    image.paste(crop, (plot_box[0], plot_box[1]), crop)
    draw.rectangle(plot_box, fill=None, outline=BASELINE, width=2)

    draw.text((panel_left + PANEL_W // 2, 42), title, fill=TEXT, font=font(31, bold=True), anchor="ma")
    draw.text((panel_left + PANEL_W // 2, plot_box[3] + 62), "x（m）", fill=TEXT, font=font(26), anchor="ma")
    draw.text((panel_left - 92, TOP + PANEL_H // 2), "y（m）", fill=TEXT, font=font(26), anchor="mm")


def draw_legend(draw) -> None:
    items = [
        ("line", BASELINE, "目标圆域边界"),
        ("fill", ACCENT_PALE, "首次观测可行集外包"),
        ("fill", PRIMARY_PALE, "固定1000米保守域"),
        ("fill", ENHANCED_PALE, "相关接收增强域"),
        ("dash", BASELINE, "移动预算圆"),
        ("best", PRIMARY, "保守域推荐点"),
        ("best", ENHANCED, "增强域推荐点"),
    ]
    positions = [
        (130, HEIGHT - 125),
        (850, HEIGHT - 125),
        (1660, HEIGHT - 125),
        (2440, HEIGHT - 125),
        (500, HEIGHT - 58),
        (1320, HEIGHT - 58),
        (2140, HEIGHT - 58),
    ]
    for (kind, color, label), (x, y) in zip(items, positions):
        if kind == "fill":
            draw.rectangle((x, y - 12, x + 42, y + 12), fill=color, outline=TEXT, width=1)
        elif kind == "best":
            draw.ellipse((x + 7, y - 13, x + 33, y + 13), fill=WHITE, outline=color, width=4)
            draw.ellipse((x + 15, y - 5, x + 25, y + 5), fill=color)
        elif kind == "dash":
            draw.line((x, y, x + 18, y), fill=color, width=4)
            draw.line((x + 29, y, x + 47, y), fill=color, width=4)
        else:
            draw.line((x, y, x + 47, y), fill=color, width=4)
        draw.text((x + 58, y), label, fill=TEXT, font=font(25), anchor="lm")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    origin = load("origin_east_budget600.json")
    conservative = load("origin_east_budget600_conservative.json")
    boundary = load("boundary_outward.json")
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw_case(
        image,
        draw,
        MARGIN_X,
        origin,
        "(a) 600米预算：相关接收约束扩大安全域",
        (-150, 1150, -900, 900),
        conservative=conservative,
    )
    draw_case(
        image,
        draw,
        MARGIN_X + PANEL_W + GAP,
        boundary,
        "(b) 圆域截断：增强策略自动缩短移动",
        (650, 2650, -1050, 1050),
    )
    draw_legend(draw)
    output = FIGURES / "q2_candidate_regions_v3.png"
    image.save(output, dpi=(320, 320), optimize=True)
    print(output)


if __name__ == "__main__":
    main()
