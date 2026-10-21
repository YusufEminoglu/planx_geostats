# -*- coding: utf-8 -*-
"""Draw the 64x64 algorithm icons added from v3.11.0 on.

Reproduces the house style of the existing icons - a rounded tile with a
group-coloured border and a light tint of the same colour, a tab carrying the
group number, a pictogram and a short label in a white pill - so a new tool
does not stand out in the toolbox.  Drawn at 4x and reduced with a Lanczos
filter.  Development-only: needs Pillow, is excluded from the release zip, and
only writes the icons listed in ICONS, never the hand-made older ones.

    py -3 tools/make_algorithm_icons.py            # write missing icons
    py -3 tools/make_algorithm_icons.py --force    # rewrite the listed icons
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "icons" / "algorithms"
S = 4                     # supersampling factor
SIZE = 64 * S

GROUPS = {
    "01": ("#0891B2", "#CFFAFE"),
    "02": ("#2563EB", "#DBEAFE"),
    "03": ("#DC2626", "#FEE2E2"),
    "05": ("#059669", "#D1FAE5"),
    "07": ("#0D9488", "#CCFBF1"),
    "08": ("#B45309", "#FEF3C7"),
}


def _font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _p(x, y):
    """Icon coordinates (0-64) to canvas pixels."""
    return (x * S, y * S)


def _box(x0, y0, x1, y1):
    return [x0 * S, y0 * S, x1 * S, y1 * S]


# --------------------------------------------------------------------------
# pictograms - each draws into the 12..46 vertical band
# --------------------------------------------------------------------------

def g_rate_smoothing(d, c, t):
    heights = [22, 9, 26, 14, 19]
    for i, h in enumerate(heights):
        x = 14 + i * 7.5
        d.rectangle(_box(x, 44 - h, x + 5, 44), fill=t, outline=c, width=S)
    pts = [_p(14 + i * 1.5, 44 - (17 + 2.5 * math.sin(i / 6.0))) for i in range(25)]
    d.line(pts, fill=c, width=3 * S)


def g_eb_rate_lisa(d, c, t):
    for r in range(3):
        for q in range(3):
            x, y = 16 + q * 11, 15 + r * 10
            fill = c if (r, q) in ((0, 0), (0, 1), (1, 0)) else "white"
            d.rectangle(_box(x, y, x + 9, y + 8), fill=fill, outline=c, width=S)
    d.line([_p(44, 15), _p(49, 15)], fill=c, width=2 * S)


def g_mv_local_geary(d, c, t):
    for k, dy in enumerate((0, 8, 16)):
        y = 14 + dy
        poly = [_p(16, y + 6), _p(32, y), _p(48, y + 6), _p(32, y + 12)]
        d.polygon(poly, fill="white" if k < 2 else t, outline=c)
        d.line(poly + [poly[0]], fill=c, width=S * 2)
    d.ellipse(_box(29, 29, 35, 35), fill=c)


def g_local_join_counts(d, c, t):
    nodes = [(18, 18), (32, 15), (46, 20), (22, 34), (37, 31), (47, 40)]
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (3, 4), (4, 5), (2, 5)]
    for a, b in edges:
        d.line([_p(*nodes[a]), _p(*nodes[b])], fill=c, width=2 * S)
    for i, (x, y) in enumerate(nodes):
        filled = i in (1, 3, 4)
        d.ellipse(_box(x - 4, y - 4, x + 4, y + 4), fill=c if filled else "white", outline=c, width=S * 2)


def g_losh(d, c, t):
    d.polygon([_p(14, 30), _p(30, 22), _p(50, 14), _p(50, 44), _p(30, 36), _p(14, 32)], fill=t)
    pts = [_p(14 + i * 1.5, 30 + 6 * math.sin(i / 2.2) * (i / 24.0)) for i in range(25)]
    d.line(pts, fill=c, width=3 * S)
    d.line([_p(14, 30), _p(50, 14)], fill=c, width=S)
    d.line([_p(14, 32), _p(50, 44)], fill=c, width=S)


def g_partial_moran(d, c, t):
    for r in range(3):
        for q in range(3):
            x, y = 19 + q * 10, 16 + r * 9
            filled = (r, q) == (1, 1)
            d.ellipse(_box(x - 3.5, y - 3.5, x + 3.5, y + 3.5), fill=c if filled else "white", outline=c, width=S * 2)
    d.line([_p(43, 26), _p(51, 26)], fill=c, width=3 * S)


def g_gamma(d, c, t):
    d.line([_p(22, 15), _p(40, 15)], fill=c, width=4 * S)
    d.line([_p(24, 15), _p(24, 44)], fill=c, width=4 * S)
    d.ellipse(_box(38, 32, 46, 40), fill=c)
    d.ellipse(_box(46, 22, 52, 28), fill="white", outline=c, width=2 * S)
    d.line([_p(42, 36), _p(49, 25)], fill=c, width=S * 2)


def g_smaup(d, c, t):
    for r in range(4):
        for q in range(4):
            x, y = 16 + q * 8, 13 + r * 8
            d.rectangle(_box(x, y, x + 7, y + 7), fill=t, outline=c, width=S)
    d.rectangle(_box(16, 13, 31, 28), outline=c, width=3 * S)
    d.rectangle(_box(32, 29, 47, 44), outline=c, width=3 * S)


def g_max_p(d, c, t):
    regions = [((15, 14, 31, 29), t), ((32, 14, 49, 25), "white"), ((15, 30, 27, 45), "white"), ((28, 26, 49, 45), t)]
    for box, fill in regions:
        d.rectangle(_box(*box), fill=fill, outline=c, width=2 * S)
    d.text(_p(23, 21.5), "≥", fill=c, font=_font(9 * S), anchor="mm")


def g_azp(d, c, t):
    d.polygon([_p(15, 15), _p(33, 13), _p(30, 30), _p(14, 28)], fill=t, outline=c)
    d.polygon([_p(33, 13), _p(49, 17), _p(47, 32), _p(30, 30)], fill="white", outline=c)
    d.polygon([_p(14, 28), _p(30, 30), _p(29, 44), _p(15, 43)], fill="white", outline=c)
    d.polygon([_p(30, 30), _p(47, 32), _p(48, 44), _p(29, 44)], fill=t, outline=c)
    for poly in ([(15, 15), (33, 13), (30, 30), (14, 28)], [(33, 13), (49, 17), (47, 32), (30, 30)],
                 [(14, 28), (30, 30), (29, 44), (15, 43)], [(30, 30), (47, 32), (48, 44), (29, 44)]):
        d.line([_p(*q) for q in poly + [poly[0]]], fill=c, width=2 * S)
    d.line([_p(36, 23), _p(24, 37)], fill=c, width=2 * S)
    d.polygon([_p(24, 37), _p(24, 32), _p(28, 35)], fill=c)


def g_redcap(d, c, t):
    pts = {"r": (32, 14), "a": (21, 25), "b": (43, 25), "c": (15, 38), "d": (27, 38), "e": (39, 38), "f": (49, 38)}
    for u, v in (("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("b", "f")):
        d.line([_p(*pts[u]), _p(*pts[v])], fill=c, width=2 * S)
    for k, (x, y) in pts.items():
        d.ellipse(_box(x - 3.5, y - 3.5, x + 3.5, y + 3.5), fill=c if k in "acd" else "white", outline=c, width=2 * S)
    d.line([_p(30, 19), _p(36, 15)], fill="white", width=3 * S)


def g_region_quality(d, c, t):
    d.rectangle(_box(14, 14, 50, 44), fill="white", outline=c, width=2 * S)
    for k, h in enumerate((24, 18, 27, 12)):
        x = 18 + k * 8
        d.rectangle(_box(x, 44 - h, x + 5, 44), fill=c if k % 2 == 0 else t, outline=c, width=S)
    d.line([_p(14, 30), _p(50, 30)], fill=c, width=S)


ICONS = {
    "rate_smoothing": ("01", "EB", g_rate_smoothing),
    "eb_rate_lisa": ("03", "EBL", g_eb_rate_lisa),
    "multivariate_local_geary": ("03", "MLG", g_mv_local_geary),
    "local_join_counts": ("03", "LJC", g_local_join_counts),
    "losh_heteroskedasticity": ("03", "LOSH", g_losh),
    "local_partial_moran": ("03", "PLM", g_partial_moran),
    "gamma_index": ("02", "GAM", g_gamma),
    "smaup_test": ("02", "MAUP", g_smaup),
    "max_p_regions": ("03", "MAXP", g_max_p),
    "azp_regionalization": ("03", "AZP", g_azp),
    "redcap_regionalization": ("03", "RDCP", g_redcap),
    "regionalization_quality": ("03", "RQA", g_region_quality),
}


def draw(group: str, label: str, glyph) -> Image.Image:
    color, tint = GROUPS[group]
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(_box(2, 2, 62, 62), radius=9 * S, fill=tint, outline=color, width=2 * S)
    d.rounded_rectangle(_box(4, 4, 25, 15), radius=3 * S, fill=color)
    f_tab = _font(8 * S)
    d.text(_p(14.5, 9.8), group, fill="white", font=f_tab, anchor="mm")
    glyph(d, color, tint)
    f_label = _font((9 if len(label) <= 3 else 8) * S)
    width = d.textlength(label, font=f_label) / S + 10
    x0 = 32 - width / 2
    d.rounded_rectangle(_box(x0, 47, x0 + width, 59), radius=6 * S, fill=(255, 255, 255, 230))
    d.text(_p(32, 53.4), label, fill=color, font=f_label, anchor="mm")
    return img.resize((64, 64), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (group, label, glyph) in ICONS.items():
        path = OUT / f"{name}.png"
        if path.exists() and not args.force:
            continue
        draw(group, label, glyph).save(path, optimize=True)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
