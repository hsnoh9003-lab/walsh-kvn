"""Normalize native SVG figure labels for Nature's final artwork typography.

Outlined historical geometry is preserved. Native standard labels use Arial
or its metric-compatible Liberation Sans fallback; mathematical glyphs keep
their math font, with isolated Greek glyphs using Standard Symbols PS.
"""

from pathlib import Path
from lxml import etree as E
import re, subprocess, fitz

S = "http://www.w3.org/2000/svg"
NS = {"s": S}
WIDTH = {
    "F1": 180,
    "F2": 180,
    "F3": 88,
    "S3": 180,
    "S4": 180,
    "S7": 180,
    "S8": 180,
    "S9": 180,
    "S11": 180,
    "S10": 180,
}


def pts(value):
    v = float(re.match(r"[-+.0-9eE]+", value).group())
    return v * (
        72 / 25.4
        if value.endswith("mm")
        else 72 if value.endswith("in") else 1 if value.endswith("pt") else 0.75
    )


def size(style):
    m = re.search(r"font-size\s*:\s*([.0-9]+)(?:px|pt)?", style)
    if m:
        return float(m.group(1))
    m = re.search(r"font\s*:[^;]*?([.0-9]+)(?:px|pt)", style)
    return float(m.group(1)) if m else None


def resized(style, old, new):
    if old is None:
        return style
    return (
        re.sub(r"(font-size\s*:\s*)[.0-9]+", lambda m: m.group(1) + f"{new:g}", style)
        if "font-size" in style
        else re.sub(
            r"(font\s*:[^;]*?)[.0-9]+((?:px|pt))",
            lambda m: m.group(1) + f"{new:g}" + m.group(2),
            style,
            count=1,
        )
    )


def normalize(svg, key):
    p = Path(svg)
    tree = E.parse(str(p))
    r = tree.getroot()
    v = list(map(float, r.get("viewBox").split()))
    oldw = pts(r.get("width"))
    target = WIDTH.get(key, oldw * 25.4 / 72)
    pp = target * 72 / 25.4 / v[2]
    r.set("width", f"{target:.8g}mm")
    r.set("height", f"{target*v[3]/v[2]:.8g}mm")
    for text in r.findall(".//s:text", NS):
        content = "".join(text.itertext()).strip()
        parent_style = text.get("style", "")
        panel = (
            len(content) == 1
            and content in "abcdefgh"
            and ("bold" in parent_style or "700" in parent_style)
        )
        math = text.find("s:tspan", NS) is not None
        items = [text] + list(text.findall(".//s:tspan", NS))
        for e in items:
            style = e.get("style", "")
            old = size(style)
            is_serif = (
                "DejaVu Serif" in style
                or "font-family:serif" in style
                or "font-family: serif" in style
            )
            if is_serif:
                style = style.replace("DejaVu Serif", "Liberation Sans")
                style = re.sub(
                    r"font-family\s*:\s*serif",
                    "font-family:'Liberation Sans',sans-serif",
                    style,
                )
            if old is not None:
                if e is text and not math:
                    desired = 8 if panel else min(7, max(5, old * pp))
                    # Main schematics are displayed at about 4/5 of the 180-mm master width.
                    if key in ["F1", "F2"] and not panel:
                        desired = min(7, max(5 / 0.79, desired))
                    style = resized(style, old, desired / pp)
                elif math:
                    # Preserve subordinate mathematical script proportions.
                    all_sizes = [size(x.get("style", "")) for x in items]
                    base = max(x for x in all_sizes if x is not None)
                    newbase = min(7, max(5, base * pp)) / pp
                    style = resized(style, old, old * newbase / base)
            visible = "".join(e.itertext()).strip()
            if (
                e.tag.endswith("tspan")
                and visible
                and all(("\u0370" <= x <= "\u03ff") or x.isspace() for x in visible)
            ):
                style = re.sub(
                    r"font-family\s*:[^;]+", "font-family:'Standard Symbols PS'", style
                )
                style = re.sub(
                    r"'STIXGeneral'|STIXGeneral", "'Standard Symbols PS'", style
                )
            if style:
                e.set("style", style)
    r.set("data-ncs-font-style", "sans-5-7-panel8")
    tree.write(str(p), encoding="UTF-8", xml_declaration=True)
    return target


def export(svg, key, pdf=None, png=None):
    svg = Path(svg)
    normalize(svg, key)
    pdf = Path(pdf) if pdf else svg.with_suffix(".pdf")
    png = Path(png) if png else svg.with_suffix(".png")
    subprocess.run(
        ["inkscape", str(svg), "--export-area-page", "--export-filename=" + str(pdf)],
        check=True,
        capture_output=True,
    )
    with fitz.open(pdf) as d:
        d[0].get_pixmap(matrix=fitz.Matrix(450 / 72, 450 / 72), alpha=False).save(png)
    return str(pdf)


def normalize_folder(output, key):
    out = Path(output)
    stem = ("Figure_" + key[1:]) if key.startswith("F") else "Figure_" + key
    for p in out.glob("*.svg"):
        if p.stem in [stem, "Figure" + key[1:], "Figure_" + key]:
            export(p, key)
