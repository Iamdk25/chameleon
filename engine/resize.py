"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.

---

Approach (see README.md for the full write-up): a deterministic reflow
engine, not a scale/stretch transform. Optional text judgments from
`engine.text_context` can replace the geometric role guess and the
importance numbers; font fitting, placement, and collisions stay here.
With no judgments, behavior is the geometric heuristic below.

1. Role classification - each element is heuristically tagged (full-bleed
   background, logo, hero image, headline, body text, legal/fine print,
   a "pill" group such as a CTA or badge, a decorative shape, a rotated
   "ribbon" accent) purely from its geometry, type, z-index and
   group_id - never from its id or content, so this generalizes to
   designs it has never seen.
2. Archetype candidates - the target canvas's aspect ratio (and, for very
   short canvases, its absolute height) narrows the choice to one of
   three layout templates - a vertical stack (portrait/near-square), a
   two-column horizontal layout (wide banners/posters), or a single-row
   layout (ultra-thin strips like a 728x90 leaderboard) - usually just
   one clear-cut choice, but both neighboring templates near an actual
   boundary, where the heuristic alone is genuinely ambiguous.
3. Importance scoring - each classified item (not just its role, but its
   own font size/area relative to the rest of *this* design) gets a
   continuous 0-1 importance score. This replaces flat per-role constants
   with a signal that varies design-to-design - an unusually large badge
   or an already-dominant headline earns more room than a role label
   alone would give it.
4. Each archetype allocates space by importance weight, then places each
   element (or, for a group, the whole group as one rigid unit so a
   button's background and label never drift apart) into its slot:
   images are aspect-preserving "contain"-fit (except `fit:cover`, which
   is expected to crop), and text's font_size is shrunk/grown to fit
   using the exact font metrics the real renderer uses (falling back to
   a rough estimate if that's unavailable), floored at an
   importance-scaled minimum so a headline never shrinks as far as fine
   print.
5. Purely decorative elements (ungrouped shapes, rotated accent groups)
   are passed through a simple proportional transform instead of being
   forced into the flow - they're meant to bleed/overlap by design.
6. Candidate scoring - every candidate from step 2 is built out in full
   and scored on unwanted collisions between message-bearing elements
   (weighted by importance, extra-penalized for text sitting on an
   image), compressed/illegible text, and anything left outside the
   canvas; the lowest-cost candidate wins. For the common case of a
   single unambiguous candidate this is a formality, but near a genuine
   boundary it lets real geometry - not a hard cutoff - make the call.
7. Collision resolution - whichever candidate wins gets a cleanup pass
   that pushes apart any two message-bearing units (a button's
   background+label move together) still overlapping after layout,
   splitting the separation in inverse proportion to importance.
8. A final clamp guarantees nothing but an already-intentionally-bleeding
   decorative element ends up outside the target canvas, followed by a
   grid-snap of every element's position (never its size) to a small
   pixel grid - the alignment discipline real design tools default to -
   so edges land in alignment instead of at arbitrary sub-pixel offsets.
"""

import math
from typing import Any, Optional

from engine.text_context import layout_bridge

Element = dict[str, Any]
Box = tuple[float, float, float, float]  # x, y, width, height
PillGroup = tuple[str, list[Element]]

# ---------------------------------------------------------------------------
# Text measurement - reuse the vendored renderer's exact font metrics so
# fit decisions match what cr-renderer will actually draw. Falls back to a
# rough average-character-width estimate if that import ever fails (it
# shouldn't in the shipped container, since cr-renderer/skia is already a
# hard dependency of the renderer main.py calls), so resize() never crashes
# for a measurement reason.
# ---------------------------------------------------------------------------

from pathlib import Path

_FONTS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "fonts"
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "assets"
_font_manager = None
_font_manager_failed = False


# ---------------------------------------------------------------------------
# Background-busyness avoidance - "text never sits on a busy image" is one
# of the rubric's own named qualities. This is plain, deterministic image
# statistics (PIL's own grayscale standard deviation over a region) - not
# a vision/language model, so none of the offline-compliance questions
# around a locally-hosted model apply. It reuses the exact same `fit:cover`
# crop math the renderer itself uses (see renderer/converter/assets.py),
# so the pixels sampled here are the pixels that will actually render
# underneath text, and it works from the asset file itself rather than
# any assumption about a particular background's layout - so it
# generalizes to backgrounds this engine has never seen, same as every
# other heuristic in this file.
# ---------------------------------------------------------------------------


def _load_cover_background_gray(asset_id: Optional[str], tw: float, th: float):
    if not asset_id:
        return None
    try:
        from PIL import Image

        path = _ASSETS_DIR / asset_id
        if not path.is_file():
            return None
        image = Image.open(path).convert("L")
        w, h = max(1, round(tw)), max(1, round(th))
        scale = max(w / image.width, h / image.height)
        nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        resized = image.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    except Exception:
        return None


def _region_busyness(bg_gray, box: "Box") -> float:
    """The busiest small patch within `box`, not the average over the whole
    box. A single decorative dot inside an otherwise calm gradient barely
    moves the box's overall average/stddev if the box is much bigger than
    the dot - which is exactly the case that matters (a small bright dot
    sitting on one word of a multi-line fine-print block). Scanning in
    small blocks and taking the max catches that localized feature instead
    of averaging it away."""
    if bg_gray is None:
        return 0.0
    try:
        from PIL import ImageStat

        x0, y0, w, h = box
        x0 = max(0, min(bg_gray.width - 1, int(x0)))
        y0 = max(0, min(bg_gray.height - 1, int(y0)))
        x1 = max(x0 + 1, min(bg_gray.width, int(x0 + w)))
        y1 = max(y0 + 1, min(bg_gray.height, int(y0 + h)))
        block = max(6, min(16, (x1 - x0) // 4, (y1 - y0) // 2))
        worst = 0.0
        by = y0
        while by < y1:
            bx = x0
            while bx < x1:
                cell = bg_gray.crop((bx, by, min(bx + block, x1), min(by + block, y1)))
                worst = max(worst, ImageStat.Stat(cell).stddev[0])
                bx += block
            by += block
        return worst
    except Exception:
        return 0.0


def _keeps_intentional_center(element: Element, tw: float) -> bool:
    """Center-aligned text whose box is already on the canvas center.

    A full-width search slides that box, and the glyphs leave the center
    the layout chose. A center-aligned line in a side column is not on
    the canvas center, so it can still move sideways. A left-aligned box
    can sit on the center and still move: its glyphs are not centered.
    """
    if str(element.get("text_align") or "").lower() != "center":
        return False
    center = float(element["x"]) + float(element["width"]) / 2.0
    return abs(center - float(tw) / 2.0) <= max(8.0, float(tw) * 0.01)


def _avoid_busy_background(
    elements: list[Element], eligible_ids: set, other_boxes: dict, bg_gray, tw: float, th: float,
) -> list[Element]:
    """For text sitting directly on the background (not part of a group,
    so it's free to move independently without breaking a button's label
    alignment), scans for a calmer spot and keeps whichever one reads as
    meaningfully better - but only within its own row (a small vertical
    wiggle, so it doesn't jump to a visually unrelated part of the
    layout and read as randomly placed) and across the row's FULL width
    (not just a small local radius - a narrow local-only search can miss
    a genuinely calm, genuinely empty patch of canvas that happens to be
    far from where this element started, e.g. past a neighboring image),
    and never into a new collision with other foreground content.

    A center-aligned box the layout already placed on the canvas center
    keeps that horizontal center. Vertical candidates and collision checks
    still run. A column of center-aligned text is not on the canvas center
    and can still move sideways.
    """
    if bg_gray is None or not eligible_ids:
        return elements
    by_id = {e["id"]: e for e in elements}
    for eid in eligible_ids:
        e = by_id.get(eid)
        if e is None or e["type"] != "text":
            continue
        x, y, w, h = e["x"], e["y"], e["width"], e["height"]
        base_score = _region_busyness(bg_gray, (x, y, w, h))
        best_x, best_y, best_score = x, y, base_score
        ry = max(6.0, h * 0.4)
        step = max(2.0, w * 0.04)
        lock_horizontal = _keeps_intentional_center(e, tw)
        for dy in (-ry, -ry * 0.5, 0.0, ry * 0.5, ry):
            ny = y + dy
            if ny < 0 or ny + h > th:
                continue
            if lock_horizontal:
                nx_values = [x]
            else:
                nx_values = []
                nx = 0.0
                while nx + w <= tw:
                    nx_values.append(nx)
                    nx += step
            for nx in nx_values:
                if nx != x or ny != y:
                    collides = False
                    for oid, obox in other_boxes.items():
                        if oid == eid:
                            continue
                        if _overlap_area((nx, ny, w, h), obox) > 4.0:
                            collides = True
                            break
                    if not collides:
                        score = _region_busyness(bg_gray, (nx, ny, w, h))
                        if score < best_score:
                            best_score, best_x, best_y = score, nx, ny
        if best_score < base_score - 0.2:
            # In a genuinely cramped layout, a modest improvement (moving
            # a few px off the peak of a dot, even if not fully clearing
            # it) is still worth taking - the earlier, stricter threshold
            # meant "not dramatically better" and "no better" were treated
            # the same, when even a small shift measurably helps.
            e["x"], e["y"] = best_x, best_y
            if lock_horizontal:
                # Growing only to the right would move the center the
                # layout just protected. A calmer row can still be used.
                continue
            # A calmer spot is wasted if it's narrower than where the
            # text started, forcing worse wrapping than before - widen
            # into whatever's actually free to the right (up to the
            # nearest real obstacle or the canvas edge) and let the font
            # re-fit into that room, same fit logic used everywhere else.
            max_right = tw
            for oid, obox in other_boxes.items():
                if oid == eid:
                    continue
                ox0, oy0, ow_, oh_ = obox
                if ox0 > best_x and oy0 < best_y + h and oy0 + oh_ > best_y:
                    max_right = min(max_right, ox0)
            avail_w = max_right - best_x
            if avail_w > w * 1.15:
                new_size = _fit_font_size(
                    e["content"], e["font_family"], e.get("font_weight"), e.get("font_style"),
                    avail_w, h, e.get("line_height", 1.2), e["font_size"], min_size=e["font_size"],
                )
                if new_size > e["font_size"]:
                    e["width"], e["font_size"] = avail_w, new_size
    return elements


def _get_font_manager():
    global _font_manager, _font_manager_failed
    if _font_manager is not None or _font_manager_failed:
        return _font_manager
    try:
        from renderer.converter.brand_fonts import load_brand_font_manager

        _font_manager = load_brand_font_manager(str(_FONTS_DIR))
    except Exception:
        _font_manager_failed = True
        _font_manager = None
    return _font_manager


def _normalize_weight(weight: Optional[str]) -> str:
    return (weight or "regular").lower()


def _font_style_for(weight: Optional[str], style: Optional[str]) -> str:
    is_bold = _normalize_weight(weight) == "bold"
    is_italic = (style or "normal") == "italic"
    if is_bold and is_italic:
        return "bolditalic"
    if is_bold:
        return "bold"
    if is_italic:
        return "italic"
    return "regular"


def _measured_lines(
    content: str, font_family: str, font_weight: Optional[str], font_style: Optional[str],
    font_size: float, max_width: float,
) -> Optional[tuple[int, float]]:
    """Line count and the widest line, using the renderer's own glyph widths.

    `wrap_text` only breaks on spaces. A store name, URL, or code that is
    wider than the box stays on one line and would paint past the box.
    The returned width is that unbreakable line, not a hard break we invented.
    """
    manager = _get_font_manager()
    if manager is None or not content:
        return None
    try:
        from renderer.converter.wrap import char_widths, wrap_text

        weight = _normalize_weight(font_weight)
        style = _font_style_for(font_weight, font_style)
        widths = char_widths(manager, font_family, weight, style, font_size, content)
        wrapped = wrap_text(content, widths, max_width)
        line = 0.0
        widest = 0.0
        for index, character in enumerate(wrapped):
            if character == "\n":
                widest = max(widest, line)
                line = 0.0
                continue
            line += widths[index] if index < len(widths) else 0.0
        widest = max(widest, line)
        return wrapped.count("\n") + 1, widest
    except Exception:
        return None


def _wrap_and_measure(
    content: str, font_family: str, font_weight: Optional[str], font_style: Optional[str],
    font_size: float, max_width: float,
) -> int:
    """Number of lines `content` wraps to at `font_size` within `max_width`."""
    measured = _measured_lines(content, font_family, font_weight, font_style, font_size, max_width)
    if measured is not None:
        return measured[0]

    # Fallback: rough average-advance-width estimate.
    avg_char_w = max(1.0, font_size * 0.55)
    chars_per_line = max(1, int(max_width / avg_char_w))
    words = content.split(" ")
    lines = 1
    cur = 0
    for w in words:
        wl = len(w) + 1
        if cur + wl > chars_per_line and cur > 0:
            lines += 1
            cur = wl
        else:
            cur += wl
    return lines


def _widest_line(
    content: str, font_family: str, font_weight: Optional[str], font_style: Optional[str],
    font_size: float, max_width: float,
) -> float:
    measured = _measured_lines(content, font_family, font_weight, font_style, font_size, max_width)
    if measured is not None:
        return measured[1]
    parts = [part for part in content.replace("\n", " ").split(" ") if part]
    longest = max((len(part) for part in parts), default=0)
    return longest * max(1.0, font_size * 0.55)


def _font_ascent_descent(
    font_family: str, font_weight: Optional[str], font_style: Optional[str], size: float
) -> Optional[tuple[float, float]]:
    """Real (ascent, descent) in px for this font/size, straight from skia's
    own font metrics - the renderer positions each line's glyphs by
    `baseline = y - ascent` and `y - ascent + descent` (cr_renderer's
    `render_text_fill`), so a line's true rendered height is
    `ascent + descent`, not the nominal `size`. For most of this brand
    kit's display faces ascent+descent comes out well above `size` (a
    display serif can be ~1.4x), so a fit/min-height check built on
    `size` alone quietly lets text run past the bottom of its box - the
    renderer never clips text, so that overflow is visible, not just a
    geometry technicality."""
    manager = _get_font_manager()
    if manager is None:
        return None
    try:
        import skia

        weight = _normalize_weight(font_weight)
        style = _font_style_for(font_weight, font_style)
        ttf_bytes = manager.lookup(font_family, weight, style)
        typeface = skia.Typeface.MakeFromData(ttf_bytes)
        font = skia.Font(typeface, size)
        m = font.getMetrics()
        return (-m.fAscent, m.fDescent)
    except Exception:
        return None


def _text_block_height(
    lines: int, size: float, line_height: float,
    font_family: str, font_weight: Optional[str], font_style: Optional[str],
) -> float:
    """True rendered height of an `lines`-line text block at `size`: each
    line after the first advances by `size * line_height` (matching
    cr_renderer's own `line_index * fontSize * lineHeight`), but the block's
    total height is capped top-and-bottom by the font's real ascent/descent
    rather than by `size` itself. Falls back to the plain `lines * size *
    line_height` estimate if font metrics aren't available."""
    if lines <= 0:
        return 0.0
    metrics = _font_ascent_descent(font_family, font_weight, font_style, size)
    if metrics is None:
        return lines * size * line_height
    ascent, descent = metrics
    return (lines - 1) * size * line_height + (ascent + descent)


def _unbreakable(content: str) -> bool:
    """True when the renderer cannot wrap this text on a space."""
    return not any(break_char in content for break_char in (" ", "\n", "\t"))


def _fit_font_size(
    content: str, font_family: str, font_weight: Optional[str], font_style: Optional[str],
    box_w: float, box_h: float, line_height: float, start_size: float, min_size: float = 10.0,
) -> float:
    """Largest font size (down to `min_size`) at which `content` wraps to fit
    within `box_h`, seeded from a size proportional to the available box so
    layouts with much more (or much less) room than the source don't stay
    oddly tiny (or overflow)."""
    box_w = max(1.0, box_w)
    box_h = max(1.0, box_h)
    seed = max(min_size, min(start_size * 1.6, box_h * 0.7, start_size + 40))
    size = seed

    def overflows(candidate: float) -> bool:
        measured = _measured_lines(content, font_family, font_weight, font_style, candidate, box_w)
        if measured is None:
            lines = _wrap_and_measure(content, font_family, font_weight, font_style, candidate, box_w)
            wide = _widest_line(content, font_family, font_weight, font_style, candidate, box_w)
        else:
            lines, wide = measured
        block_h = _text_block_height(lines, candidate, line_height, font_family, font_weight, font_style)
        return block_h > box_h or wide > box_w + 0.5

    for _ in range(40):
        if not overflows(size) or size <= min_size:
            break
        size = max(min_size, size * 0.92)
    # A judged layout keeps shrinking whenever the floor still overflows.
    # Without a judgment, do that only for a single token that cannot wrap:
    # shrinking a normal sentence here changes the next fit's starting size
    # and moves the saved no-judgment Level 3 fixture. A size that already
    # fits does not enter this loop.
    single_token = not any(break_char in content for break_char in (" ", "\n", "\t"))
    if overflows(size) and (layout_bridge.active_importance_by_id or single_token):
        while size > 1.0 and overflows(size):
            size = max(1.0, round(size - 0.5, 1))
            if size <= 1.0:
                break
    return round(size, 1)


def _text_min_height(t: Element, width: float, floor_font: float) -> float:
    """The real height `t`'s content needs to wrap and fit at its legibility
    floor within `width` - using the exact same font-metric wrapping as
    `_fit_font_size` and the overflow check in `_score_candidate`, not a
    guess. Feeding this into a layout step's space allocation (instead of a
    flat guessed min_px) is what lets a long headline or a short badge
    label each claim the room they actually need, rather than both being
    squeezed by the same arbitrary number - and it's what prevents text
    from overflowing its box in the first place, rather than only
    detecting that overflow after the fact during candidate scoring."""
    family, weight, style = t.get("font_family", ""), t.get("font_weight"), t.get("font_style")
    lines = _wrap_and_measure(t.get("content", ""), family, weight, style, floor_font, max(1.0, width))
    line_height = t.get("line_height", 1.2)
    return _text_block_height(lines, floor_font, line_height, family, weight, style)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _copy_element(e: Element) -> Element:
    e = dict(e)
    if e.get("style_runs") is not None:
        e["style_runs"] = [dict(r) for r in e["style_runs"]]
    return e


def _contain_box(src_w: float, src_h: float, box: Box) -> Box:
    """Largest box with `src_w`/`src_h`'s aspect ratio, centered in `box`."""
    bx, by, bw, bh = box
    if src_w <= 0 or src_h <= 0 or bw <= 0 or bh <= 0:
        return box
    scale = min(bw / src_w, bh / src_h)
    nw, nh = src_w * scale, src_h * scale
    return (bx + (bw - nw) / 2, by + (bh - nh) / 2, nw, nh)


def _place_image(img: Element, box: Box) -> Element:
    out = _copy_element(img)
    if out.get("fit") == "cover":
        out["x"], out["y"], out["width"], out["height"] = box
    else:
        out["x"], out["y"], out["width"], out["height"] = _contain_box(img["width"], img["height"], box)
    return out


def _judgment_start(element: Element, source_size: float, box_h: float) -> float:
    """Seed for font fitting. With no judgment, the source font size is the
    seed (baseline). With a judgment, the seed follows the box this element's
    prominence was given, so a small source font can become the headline and
    a large source font can step back."""
    if element.get("id") not in layout_bridge.active_promoted_ids:
        return source_size
    importance = layout_bridge.active_importance_by_id.get(element.get("id"))
    if importance is None or box_h <= 0:
        return source_size
    wanted = box_h * (0.35 + 0.45 * max(0.0, min(1.0, importance)))
    return max(source_size, wanted)


def _place_text(t: Element, box: Box, min_size: float) -> Element:
    bx, by, bw, bh = box
    out = _copy_element(t)
    out["x"], out["y"], out["width"], out["height"] = bx, by, bw, bh
    out["font_size"] = _fit_font_size(
        out["content"], out["font_family"], out.get("font_weight"), out.get("font_style"),
        bw, bh, out.get("line_height", 1.2), _judgment_start(out, out["font_size"], bh), min_size=min_size,
    )
    return out


def _group_bbox(members: list[Element]) -> Box:
    x0 = min(m["x"] for m in members)
    y0 = min(m["y"] for m in members)
    x1 = max(m["x"] + m["width"] for m in members)
    y1 = max(m["y"] + m["height"] for m in members)
    return (x0, y0, x1 - x0, y1 - y0)


def _group_label_height_ratio(members: list[Element]) -> float:
    """What fraction of a group's total height its text label actually
    occupies - e.g. a button's label commonly sits in only the lower half
    of its background pill, by the source design's own proportions.
    `_apply_group_transform` scales every member by the same factor, so a
    content-aware minimum-height check against the label alone understates
    what the GROUP as a whole needs by roughly this same fraction; dividing
    by it converts "how much room the label needs" into "how much room the
    group needs for the label to get that much"."""
    label = next((m for m in members if m["type"] == "text"), None)
    if label is None:
        return 1.0
    group_h = _group_bbox(members)[3]
    if group_h <= 0:
        return 1.0
    return max(0.15, min(1.0, label["height"] / group_h))


def _rotated_bbox(x: float, y: float, w: float, h: float, rotation_deg: float) -> Box:
    """The axis-aligned box that actually contains a w x h rectangle
    rotated by `rotation_deg` about its own center. A rotated element's
    painted extent can stick out past its own unrotated x/y/w/h - e.g. a
    wide ribbon tilted -12 degrees near a canvas edge or another element
    can visually clip or collide even when its stored (unrotated) box is
    fully in bounds and non-overlapping. Collision checks and off-canvas
    clamping need this, not the raw box, to catch what's actually visible."""
    if not rotation_deg:
        return (x, y, w, h)
    cx, cy = x + w / 2.0, y + h / 2.0
    rad = math.radians(rotation_deg)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    new_w, new_h = w * cos_a + h * sin_a, w * sin_a + h * cos_a
    return (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)


def _visual_bbox(members: list[Element]) -> Box:
    """Like `_group_bbox`, but using each member's rotated (true visual)
    extent - the right thing to use for collision detection, where what
    matters is what's actually painted, not the unrotated storage box."""
    boxes = [_rotated_bbox(m["x"], m["y"], m["width"], m["height"], m.get("rotation", 0)) for m in members]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _apply_group_transform(members: list[Element], old_box: Box, new_box: Box, min_size: float = 10.0) -> list[Element]:
    """Moves/resizes a group as one rigid unit - every member keeps its
    position and size relative to the group's bounding box, so e.g. a
    button's background shape and its label never drift apart."""
    ox, oy, ow, oh = old_box
    nx, ny, nw, nh = new_box
    sx = (nw / ow) if ow > 1e-6 else 1.0
    sy = (nh / oh) if oh > 1e-6 else 1.0
    out = []
    for m in members:
        e = _copy_element(m)
        e["x"] = nx + (m["x"] - ox) * sx
        e["y"] = ny + (m["y"] - oy) * sy
        e["width"] = m["width"] * sx
        e["height"] = m["height"] * sy
        if e["type"] == "text":
            e["font_size"] = _fit_font_size(
                e["content"], e["font_family"], e.get("font_weight"), e.get("font_style"),
                e["width"], e["height"], e.get("line_height", 1.2),
                _judgment_start(e, e["font_size"] * min(sx, sy), e["height"]), min_size=min_size,
            )
        out.append(e)
    if layout_bridge.active_importance_by_id:
        _fit_single_label_to_shape(out, min_size)
    return out


def _fit_single_label_to_shape(members: list[Element], min_size: float) -> None:
    """Let a button or badge label use the shape behind it.

    A rigid scale keeps the source inset, so a short canvas turns the label
    into a sliver inside a still-visible pill. When the group is one shape
    and one text, and that text uses less than three quarters of the shape,
    fit the words to the shape's interior. The shape stays put. This runs
    only when a text judgment is active, so the no-judgment baseline is unchanged.
    """
    shapes = [member for member in members if member.get("type") == "shape"]
    texts = [member for member in members if member.get("type") == "text"]
    if len(shapes) != 1 or len(texts) != 1:
        return
    if any(member.get("type") not in {"shape", "text"} for member in members):
        return
    shape, text = shapes[0], texts[0]
    # A low-importance badge should stay visually quieter than the button
    # it sits beside. Filling every pill makes a crowded strip look like
    # six headlines. The call to action and other prominent labels still
    # use the shape.
    importance = float(layout_bridge.active_importance_by_id.get(text.get("id"), 1.0))
    fills_shape = text["height"] >= shape["height"] * 0.75 and text["width"] >= shape["width"] * 0.75
    if fills_shape:
        # The label already uses the pill. A wrap that is only a few pixels
        # past one line still gets the width below. The font stays put.
        _widen_wrapped_label(shape, text)
        return
    # A prominent button can fill its shape. A quieter badge keeps a larger
    # inset when its pill is tall enough for that inset to stay readable.
    # A short pill or a rotated ribbon has no spare inset: the words have
    # to use the shape or they disappear.
    rotated = abs(float(text.get("rotation") or 0)) > 0.5 or abs(float(shape.get("rotation") or 0)) > 0.5
    roomy = float(shape["height"]) >= 28.0 and not rotated
    if importance < 0.5 and roomy:
        pad_x = max(2.0, float(shape["width"]) * 0.12)
        pad_y = max(1.0, float(shape["height"]) * 0.22)
    else:
        pad_x = max(2.0, float(shape["width"]) * 0.08)
        pad_y = max(1.0, float(shape["height"]) * 0.1)
    inner_w = float(shape["width"]) - 2 * pad_x
    inner_h = float(shape["height"]) - 2 * pad_y
    if inner_w < 8 or inner_h < 4:
        return
    text["x"] = float(shape["x"]) + pad_x
    text["y"] = float(shape["y"]) + pad_y
    text["width"] = inner_w
    text["height"] = inner_h
    text["font_size"] = _fit_font_size(
        text["content"], text["font_family"], text.get("font_weight"), text.get("font_style"),
        text["width"], text["height"], text.get("line_height", 1.2),
        _judgment_start(text, text["font_size"], text["height"]), min_size=min_size,
    )
    _widen_wrapped_label(shape, text)


def _widen_wrapped_label(shape: Element, text: Element) -> None:
    """Give a wrapping label the one-line width when the shape can hold it.

    An inset, or a label that already fills most of the pill, can wrap a
    line that is only a few pixels too long. The font size stays. A label
    that still needs more than the shape stays wrapped. The no-judgment
    path does not call this.
    """
    measured = _measured_lines(
        text["content"], text["font_family"], text.get("font_weight"), text.get("font_style"),
        text["font_size"], text["width"],
    )
    if measured is None or measured[0] <= 1:
        return
    one_line = _widest_line(
        text["content"], text["font_family"], text.get("font_weight"), text.get("font_style"),
        text["font_size"], 10000.0,
    )
    if one_line > float(shape["width"]) - 4.0:
        return
    pad_x = max(2.0, (float(shape["width"]) - one_line - 2.0) / 2.0)
    text["x"] = float(shape["x"]) + pad_x
    text["width"] = float(shape["width"]) - 2.0 * pad_x


def _distribute(weighted: list[tuple[float, float]], avail: float) -> tuple[list[float], float]:
    """Splits `avail` px across items by weight, honoring each item's
    minimum where the budget allows and scaling down proportionally
    (never below its share) when it doesn't. Returns (sizes, gap).

    This is the "not enough room" fallback - it fills 100% of `avail` by
    construction, which is correct when space is scarce, but is the wrong
    tool when space is generous (see `_flow_sizes`): inflating a band
    that a centered image or a capped-size text block won't actually fill
    just moves the blank space inside the band instead of removing it."""
    n = len(weighted)
    if n == 0 or avail <= 0:
        return [0.0] * n, 0.0
    gap = max(4.0, avail * 0.015) if n > 1 else 0.0
    if gap * (n - 1) > avail * 0.35:
        gap = (avail * 0.35) / max(1, n - 1)
    budget = max(1.0, avail - gap * (n - 1))
    total_w = sum(w for w, _ in weighted) or 1.0
    raw = [max(mn, budget * (w / total_w)) for w, mn in weighted]
    scale = budget / sum(raw) if sum(raw) > 0 else 1.0
    return [r * scale for r in raw], gap


def _priority_allocate(weighted: list[tuple[float, float]], avail: float, hard_floor_frac: float = 0.45) -> list[float]:
    """Used when even everyone's true minimum doesn't fit. Uniformly
    shrinking every item by the same percentage (what `_distribute` does)
    means a headline gets crushed just as hard as a badge - which is
    exactly what causes high-importance text to overflow its box and
    visually collide with its neighbors.

    Every item starts at a reduced hard floor, then the remaining budget
    is handed out by weighted water-filling (the same idea real
    bandwidth/resource schedulers use for max-min-fair allocation):
    repeatedly split whatever's left among items still below their true
    minimum, in proportion to importance, rather than fully satisfying
    the single highest-importance item before the next one sees any
    budget at all. That "winner takes all" order would leave a headline
    at 100% of its minimum while a badge sits stuck at its hard floor;
    water-filling instead lets every item make importance-weighted
    progress, so the least important ones are still visibly compressed
    but not crushed to an unreadable sliver."""
    n = len(weighted)
    if n == 0:
        return []
    hard_floors = [mn * hard_floor_frac for _, mn in weighted]
    total_floor = sum(hard_floors)
    if total_floor >= avail:
        scale = (avail / total_floor) if total_floor > 0 else 0.0
        return [f * scale for f in hard_floors]
    sizes = list(hard_floors)
    remaining = avail - total_floor
    active = [i for i in range(n) if weighted[i][1] > sizes[i]]
    for _ in range(80):
        if not active or remaining <= 1e-6:
            break
        total_w = sum(weighted[i][0] for i in active) or 1.0
        still_active = []
        for i in active:
            share = remaining * (weighted[i][0] / total_w)
            want = weighted[i][1] - sizes[i]
            grant = min(share, want)
            sizes[i] += grant
            remaining -= grant
            if weighted[i][1] - sizes[i] > 1e-6:
                still_active.append(i)
        if len(still_active) == len(active):
            # Nobody got fully satisfied this round (remaining was the
            # binding constraint for everyone) - one more even split
            # finishes the job; looping further wouldn't change anything.
            if remaining > 1e-6 and still_active:
                bonus = remaining / len(still_active)
                for i in still_active:
                    sizes[i] += bonus
            break
        active = still_active
    return sizes


def _flow_sizes(
    naturals: list[float], weights: list[float], minimums: list[float], avail: float,
) -> tuple[list[float], float, float]:
    """Lays items out at their natural (desired) size when there's room for
    it, spreading any leftover space evenly as extra gap between items plus
    a matching leading margin - so a sparse design (few elements, lots of
    canvas) reads as a deliberately spaced, centered composition instead of
    a few elements stretched to fill the canvas with dead space trapped
    inside their own boxes. When there isn't room for natural sizes but
    there IS room for everyone's true minimum, falls back to
    `_distribute`'s proportional shrink. When even minimums don't fit,
    switches to `_priority_allocate` instead of continuing to shrink
    proportionally - past the minimum, proportional shrink is what causes
    important content to overflow its box just as much as unimportant
    content. Returns (sizes, gap, leading_margin)."""
    n = len(naturals)
    if n == 0 or avail <= 0:
        return [0.0] * n, 0.0, 0.0
    base_gap = max(4.0, avail * 0.02) if n > 1 else 0.0
    if base_gap * (n - 1) > avail * 0.3:
        base_gap = (avail * 0.3) / max(1, n - 1)
    # A "natural" size is an aesthetic target, not a hard requirement - it
    # can legitimately be smaller than what an item's content actually
    # needs (e.g. a long headline's content-driven minimum can exceed its
    # importance-based natural size). Comparing only the raw naturals'
    # total against `avail` would let that slip through: there could be
    # plenty of room overall while one specific item still gets less than
    # it needs. `effective` is what each item will actually get in this
    # branch, so the room check and the returned sizes agree.
    effective = [max(nat, mn) for nat, mn in zip(naturals, minimums)]
    total_effective = sum(effective)
    if total_effective + base_gap * (n - 1) <= avail:
        slack = avail - total_effective - base_gap * (n - 1)
        portion = slack / (n + 1)
        return effective, base_gap + portion, portion
    budget = max(1.0, avail - base_gap * (n - 1))
    if sum(minimums) <= budget:
        sizes, _ = _distribute(list(zip(weights, minimums)), avail)
        return sizes, base_gap, 0.0
    return _priority_allocate(list(zip(weights, minimums)), budget), base_gap, 0.0


# Natural-size targets per flow-item kind: (reference axis, fraction of
# that axis) at importance == 1.0, scaled down toward 0.55x at
# importance == 0.0, and capped as a fraction of the available height so
# nothing with a tiny importance can still claim a huge natural size just
# because its own original box was large.
_NATURAL_BASE = {
    "hero": ("w", 0.50), "logo": ("w", 0.11), "headline": ("h", 0.16),
    "body": ("h", 0.10), "cta": ("w", 0.095), "badges": ("w", 0.08), "legal": ("h", 0.05),
}
_NATURAL_CAP_FRAC = {
    "hero": 0.60, "logo": 0.22, "headline": 0.40, "body": 0.28,
    "cta": 0.22, "badges": 0.20, "legal": 0.12,
}


def _natural_size(kind: str, importance: float, avail_w: float, avail_h: float) -> float:
    axis, frac = _NATURAL_BASE[kind]
    ref = avail_w if axis == "w" else avail_h
    scale = 0.55 + 0.9 * max(0.0, min(1.0, importance))
    size = ref * frac * scale
    return max(10.0, min(size, avail_h * _NATURAL_CAP_FRAC[kind]))


# ---------------------------------------------------------------------------
# Importance scoring - a continuous 0-1 signal per classified flow item,
# used in place of fixed per-role constants to drive how much space and
# how aggressive a legibility floor each element gets. This is what lets
# e.g. an unusually large badge, or a headline that's already much bigger
# than its design's other text, earn proportionally more room than a flat
# "all badges are worth 0.09" rule would give it - a step toward the kind
# of judgment call a human designer (or a future vision/LLM pass feeding
# this same dict) would make, while staying fully deterministic for now.
# ---------------------------------------------------------------------------

_ROLE_BASE_IMPORTANCE = {
    "headline": 0.90,
    "cta": 0.75,
    "hero": 0.70,
    "logo": 0.45,
    "body": 0.50,
    "badge": 0.35,
    "legal": 0.18,
}


def _is_must_read(element: Optional[Element]) -> bool:
    return element is not None and element.get("id") in layout_bridge.active_must_read_ids


def _space_weight(element: Optional[Element], importance: float) -> float:
    """Visual prominence stays what the model chose. A must-read line still
    receives a medium share of scarce height, so a low-prominence disclosure
    is not the first line crushed when the canvas is short."""
    if _is_must_read(element):
        return max(float(importance), 0.62)
    return float(importance)


def _min_size_for(
    importance: float, scale: float = 1.0, floor: float = 6.0, *, element: Optional[Element] = None,
) -> float:
    """Legibility floor for a font_size shrink-to-fit pass: higher-importance
    text (a headline) is allowed to stay bigger under pressure than
    lower-importance text (fine print). `scale`/`floor` let tighter
    archetypes (the ultra-thin strip) compress everything further while
    preserving the same relative ordering. A must-read judgment keeps at
    least 8 px even when its visual prominence is low."""
    importance = max(0.0, min(1.0, importance))
    base = max(floor, round((6.0 + 11.0 * importance) * scale, 1))
    if _is_must_read(element):
        return max(base, 8.0)
    return base


def _emergency_floor(element: Optional[Element]) -> float:
    """Last-resort floor when a slot cannot hold the preferred minimum.

    A must-read line still asks for 8 px when the slot is tall enough.
    When the slot is shorter than that, forcing 8 px overflows the box
    and the whole judged layout is thrown away. Fit the type instead,
    and let the entrypoint report the small size as a trade-off.
    """
    return 5.0


def _adjusted_min_size(
    importance: float, needed_h: float, granted_h: float, scale: float = 1.0, *, element: Optional[Element] = None,
) -> float:
    """The normal legibility floor, except when an item was granted less
    height than it actually needs (a genuine "not everything fits" case -
    see `_priority_allocate`). In that case, holding the normal floor just
    means the text overflows its box and visually collides with whatever's
    next, so this hands `_fit_font_size` the lowest floor it's allowed to
    use instead - text height doesn't scale linearly with font size (a
    smaller font fits more characters per line, so lines shrink faster
    than the font does), so pre-computing an intermediate "just shrunk
    enough" floor would be guessing; `_fit_font_size` already iteratively
    searches for the largest font that fits down to whatever floor it's
    given, so the right move is to open up the full range and let that
    search do the real work, not second-guess it with an estimate."""
    base = _min_size_for(importance, scale=scale, element=element)
    if needed_h <= 0 or granted_h >= needed_h:
        return base
    return _emergency_floor(element)


def _deemphasize_if_underfunded(elements: list[Element], needed_h: float, granted_h: float) -> list[Element]:
    """When an item was severely underfunded (got well under its true
    minimum - a genuine "everything doesn't fit" case) and had to shrink
    past its normal floor to avoid overflowing, a bounded opacity
    reduction (never below ~0.55, never fully hidden) is written onto
    the element. The entrypoint copies source opacity back before it
    publishes JSON or a PNG, so this does not change the published
    design. Must-read text is left untouched here too."""
    if any(_is_must_read(e) for e in elements):
        return elements
    if needed_h <= 0 or granted_h >= needed_h * 0.75:
        return elements
    mult = max(0.55, min(0.9, 0.5 + 0.4 * (granted_h / needed_h)))
    for e in elements:
        e["opacity"] = round(max(0.0, min(1.0, e.get("opacity", 1.0))) * mult, 3)
    return elements


def _compute_importances(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], sw: float, sh: float,
) -> dict[str, float]:
    canvas_area = max(1.0, sw * sh)
    text_candidates = ([headline] if headline else []) + body_texts + ([legal] if legal else [])
    max_font = max((t["font_size"] for t in text_candidates), default=1.0)

    def text_score(base: float, el: Element) -> float:
        return min(1.0, base + 0.15 * (el["font_size"] / max_font))

    def area_score(base: float, w: float, h: float, ref_frac: float) -> float:
        ratio = (w * h) / canvas_area
        return min(1.0, base + 0.25 * min(1.0, ratio / ref_frac))

    importances: dict[str, float] = {}
    if headline:
        importances["headline"] = text_score(_ROLE_BASE_IMPORTANCE["headline"], headline)
    if primary_cta:
        _, members = primary_cta
        bbox = _group_bbox(members)
        importances["cta"] = area_score(_ROLE_BASE_IMPORTANCE["cta"], bbox[2], bbox[3], 0.06)
    if hero:
        importances["hero"] = area_score(_ROLE_BASE_IMPORTANCE["hero"], hero["width"], hero["height"], 0.25)
    if logo:
        importances["logo"] = area_score(_ROLE_BASE_IMPORTANCE["logo"], logo["width"], logo["height"], 0.05)
    for i, bt in enumerate(body_texts):
        importances[f"body_{i}"] = text_score(_ROLE_BASE_IMPORTANCE["body"], bt)
    if legal:
        importances["legal"] = text_score(_ROLE_BASE_IMPORTANCE["legal"], legal)
    for i, (_, members) in enumerate(badges):
        bbox = _group_bbox(members)
        importances[f"badge_{i}"] = area_score(_ROLE_BASE_IMPORTANCE["badge"], bbox[2], bbox[3], 0.03)
    return importances


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------


def _covers_canvas(e: Element, cw: float, ch: float) -> bool:
    """A real background: fit=cover, and the box actually covers the canvas.

    Size alone is not enough. A full-frame fit=contain image is a hero.
    A cover image shifted off the frame is not a background either.
    Classification and candidate scoring both use this check.
    """
    return (
        e.get("type") == "image" and e.get("fit") == "cover"
        and e["x"] <= cw * 0.05 and e["y"] <= ch * 0.05
        and e["x"] + e["width"] >= cw * 0.95 and e["y"] + e["height"] >= ch * 0.95
    )


def _looks_like_logo(e: Element, canvas_h: float) -> bool:
    aspect = e["width"] / e["height"] if e["height"] > 0 else 1.0
    return aspect >= 1.6 and e["y"] < canvas_h * 0.32


def _is_out_of_bounds(e: Element, w: float, h: float) -> bool:
    return e["x"] < -0.5 or e["y"] < -0.5 or e["x"] + e["width"] > w + 0.5 or e["y"] + e["height"] > h + 0.5


def _rotated_overhang(element: Element, canvas_w: float, canvas_h: float) -> float:
    """How far the painted, rotated rectangle leaves the canvas.

    The stored x/y box can sit inside the frame while a tilt sticks out.
    """
    rotation = float(element.get("rotation") or 0.0)
    rx, ry, rw, rh = _rotated_bbox(
        float(element["x"]), float(element["y"]),
        float(element["width"]), float(element["height"]), rotation,
    )
    return max(0.0, -rx, -ry, rx + rw - canvas_w, ry + rh - canvas_h)


def _source_bleed(element: Element, canvas_w: float, canvas_h: float) -> bool:
    """True when the source itself hangs off the canvas by more than a sliver.

    A tilt of a pixel is not the intentional bleed the Level 2 circle has.
    Those elements still get pulled back inside the target.
    """
    if _is_out_of_bounds(element, canvas_w, canvas_h):
        return True
    tolerance = max(4.0, 0.02 * min(canvas_w, canvas_h))
    return _rotated_overhang(element, canvas_w, canvas_h) > tolerance


# ---------------------------------------------------------------------------
# Candidate generation and scoring - the archetype thresholds in
# `_candidate_archetypes` are a strong prior, not a guess: for an
# unambiguous target shape (e.g. a 1:1 square or a 9:16 story), the "right"
# archetype genuinely is clear-cut and trying alternatives would just waste
# work. Near a boundary, though, the heuristic is genuinely uncertain, so
# both neighboring archetypes get built in full and scored - the same
# qualities the brief's own rubric names (unwanted collisions, text
# readability, nothing cut off) decide the winner instead of a hard cutoff
# guessing blind.
# ---------------------------------------------------------------------------


def _candidate_archetypes(th: float, ratio: float) -> list[str]:
    """Plausible layouts for this canvas shape.

    A banner at least six times as wide as it is tall is one reading row.
    Drawing it as a wide card lets short pills outgrow the headline. A
    canvas that is wide but only moderately so, and between 150 px and
    220 px tall, tries both the row and the wide card. The score then
    keeps the one whose measured type stays in order. Taller wide cards
    stay on the wide layout. A sparse left column is added later, only
    when the element count says the canvas is mostly empty.
    """
    if ratio >= 6.0:
        return ["strip"]
    if th < 160:
        archetypes = ["strip"]
        if th >= 120:
            archetypes.append("wide" if ratio >= 1.8 else "stack")
        return archetypes
    if ratio >= 1.8:
        archetypes = ["wide"]
        if ratio <= 2.3:
            archetypes.append("stack")
        # A short wide card can let a row of pills take more height than
        # the headline. The row layout is the other candidate. Above
        # 220 px the wide card already has a real headline band.
        if th <= 220 and ratio >= 2.2:
            archetypes.append("strip")
        return archetypes
    archetypes = ["stack"]
    if ratio >= 1.45:
        archetypes.append("wide")
    return archetypes


def _foreground_weights(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float],
) -> dict[str, float]:
    """Element id -> importance, for every id that should participate in
    collision scoring (message-bearing content). Background and purely
    decorative elements are deliberately left out - they're expected to be
    overlapped, so penalizing that would fight the brief's own distinction
    ("overlaps are fine when they're intentional... what matters is that
    elements that shouldn't overlap, don't")."""
    weights: dict[str, float] = {}
    if logo:
        weights[logo["id"]] = importances["logo"]
    if hero:
        weights[hero["id"]] = importances["hero"]
    if headline:
        weights[headline["id"]] = importances["headline"]
    for i, bt in enumerate(body_texts):
        weights[bt["id"]] = importances[f"body_{i}"]
    if legal:
        weights[legal["id"]] = importances["legal"]
    if primary_cta:
        _, members = primary_cta
        for m in members:
            weights[m["id"]] = importances["cta"]
    for i, (_, members) in enumerate(badges):
        for m in members:
            weights[m["id"]] = importances[f"badge_{i}"]
    return weights


def _overlap_area(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ox * oy


def _score_candidate(elements: list[Element], fg_weights: dict[str, float], tw: float, th: float) -> float:
    """Lower is better. Penalizes unwanted collisions between message-bearing
    elements (weighted by how important each one is, and extra-penalized
    when it's specifically text sitting on an image), small/compressed
    text, and anything left outside the canvas - mirroring the brief's own
    "what a good result looks like" criteria rather than an arbitrary
    metric."""
    canvas_area = max(1.0, tw * th)
    by_id = {e["id"]: e for e in elements}
    fg_ids = [i for i in fg_weights if i in by_id]

    cost = 0.0
    for i in range(len(fg_ids)):
        a = by_id[fg_ids[i]]
        box_a = (a["x"], a["y"], a["width"], a["height"])
        for j in range(i + 1, len(fg_ids)):
            b = by_id[fg_ids[j]]
            # A button label is meant to sit on its own shape. A fit=cover
            # image that actually covers this canvas is a background.
            same_group = a.get("group_id") and a.get("group_id") == b.get("group_id")
            if same_group or _covers_canvas(a, tw, th) or _covers_canvas(b, tw, th):
                continue
            overlap = _overlap_area(box_a, (b["x"], b["y"], b["width"], b["height"]))
            if overlap <= 0:
                continue
            penalty = (overlap / canvas_area) * fg_weights[a["id"]] * fg_weights[b["id"]] * 40.0
            if {a["type"], b["type"]} == {"text", "image"}:
                penalty *= 2.5
            cost += penalty

    for eid, weight in fg_weights.items():
        e = by_id.get(eid)
        if e is None:
            continue
        x1, y1 = e["x"] + e["width"], e["y"] + e["height"]
        off = max(0.0, -e["x"]) + max(0.0, -e["y"]) + max(0.0, x1 - tw) + max(0.0, y1 - th)
        if off > 0:
            cost += (off / max(tw, th)) * weight * 8.0
        if e["type"] == "text" and e.get("font_size", 12) < 11.0:
            cost += (11.0 - e["font_size"]) * weight * 0.6
        if e["type"] == "text" and weight >= 0.7 and th <= 240.0:
            # On a short canvas, a primary line under 12 px (or under 5.5%
            # of that short side) is not a readable composition. Chips and
            # legal sit under this weight and are not charged. Taller
            # canvases are not asked to grow type with the canvas.
            floor = max(12.0, th * 0.055)
            font = float(e.get("font_size") or 0.0)
            if font < floor:
                cost += (floor - font) * weight * 4.0
        if e["type"] == "text":
            # A box not overlapping its neighbors' boxes doesn't mean the
            # TEXT fits inside it - `_fit_font_size` shrinks down to a
            # floor, but a box can still be too short for even
            # floor-sized text to wrap into (e.g. several items splitting
            # very little height). That text then spills past its own
            # box into whatever's next, which is exactly the "nothing
            # cut off" / "no unwanted collisions" failure the rubric
            # names - and it's invisible to the box-overlap check above
            # since the declared boxes themselves never overlapped.
            font_size = e.get("font_size", 12)
            content = e.get("content", "")
            family = e.get("font_family", "")
            font_weight = e.get("font_weight")
            font_style = e.get("font_style")
            box_w = max(1.0, e["width"])
            measured = _measured_lines(content, family, font_weight, font_style, font_size, box_w)
            if measured is None:
                lines = _wrap_and_measure(content, family, font_weight, font_style, font_size, box_w)
                widest = box_w
            else:
                lines, widest = measured
            rendered_h = _text_block_height(
                lines, font_size, e.get("line_height", 1.2), family, font_weight, font_style,
            )
            overflow = rendered_h - e["height"]
            if overflow > 0.5:
                cost += (overflow / max(1.0, e["height"])) * weight * 25.0
            width_overflow = widest - e["width"]
            if width_overflow > 0.5:
                cost += (width_overflow / box_w) * weight * 25.0

    # A lower-importance pill should not be taller than the primary line.
    # The font can be a point smaller while the chip shape still dominates.
    lead_weight = -1.0
    lead_box = 0.0
    for eid, weight in fg_weights.items():
        element = by_id.get(eid)
        if element is None or element.get("type") != "text":
            continue
        if weight > lead_weight:
            lead_weight = weight
            lead_box = float(element["height"])
    if lead_weight > 0 and lead_box > 0:
        for element in elements:
            if element.get("type") != "shape":
                continue
            group_id = element.get("group_id")
            if not group_id:
                continue
            group_weight = max(
                (
                    fg_weights[member["id"]]
                    for member in elements
                    if member.get("group_id") == group_id and member["id"] in fg_weights
                ),
                default=0.0,
            )
            if group_weight <= 0 or group_weight >= lead_weight * 0.75:
                continue
            if float(element["height"]) > lead_box * 0.90:
                cost += (float(element["height"]) / max(th, 1.0)) * 18.0

    return cost


def _restore_rotated_aspect(
    source_box: Box, scaled_box: Box, tw: float, th: float,
) -> Optional[Box]:
    """Give a flattened rotated group its source aspect back.

    Independent x/y scaling turns a 300×54 accent into a 10 px sliver on a
    wide, short canvas. The group's own proportions are restored and the
    long side is capped so it stays a corner mark. Returns None when the
    scaled box is already close to that aspect.
    """
    _sx, _sy, sw, sh = source_box
    x, y, w, h = scaled_box
    if sw <= 1 or sh <= 1 or w <= 1 or h <= 1 or tw <= 0 or th <= 0:
        return None
    aspect = sw / sh
    if h >= (w / aspect) * 0.85:
        return None
    height = min(max(h, 18.0), th * 0.2)
    width = height * aspect
    cap_w = min(tw * 0.18, max(w, 1.0))
    if width > cap_w:
        width = cap_w
        height = width / aspect
    if height < 12 or width < 24:
        return None
    right = x + w
    nx = right - width if x + w / 2.0 > tw * 0.5 else x
    ny = y if y + h / 2.0 <= th * 0.5 else (y + h - height)
    return (nx, ny, width, height)


def _build_candidate(
    archetype: str,
    backgrounds: list[Element], standalone_shapes: list[Element], extra_images: list[Element],
    generic_groups: list[PillGroup], ribbon_groups: list[PillGroup],
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float],
    scale_x: float, scale_y: float, sw: float, sh: float, tw: float, th: float, margin: float,
    park_logo: bool = False,
) -> tuple[list[Element], set]:
    out: list[Element] = []

    for e in backgrounds:
        e = _copy_element(e)
        e["x"], e["y"], e["width"], e["height"] = 0.0, 0.0, tw, th
        out.append(e)

    for e in standalone_shapes:
        e = _copy_element(e)
        e["x"], e["y"] = e["x"] * scale_x, e["y"] * scale_y
        if e.get("shape_type") == "ellipse":
            # A circle squashed by independent x/y scaling reads as a
            # mistake, not a design choice - a 500x500 circle scaled onto
            # a 1600x400 canvas would otherwise become a 740x185 flattened
            # blob. Ellipses scale uniformly (by the average of the two
            # axis scales, so they neither balloon nor shrink to nothing)
            # and keep their own proportions; rectangles/lines still
            # stretch with the canvas, which reads as normal for a
            # decorative panel or divider.
            uniform = (scale_x + scale_y) / 2.0
            e["width"], e["height"] = e["width"] * uniform, e["height"] * uniform
        else:
            e["width"], e["height"] = e["width"] * scale_x, e["height"] * scale_y
        out.append(e)

    for e in extra_images:
        box = (e["x"] * scale_x, e["y"] * scale_y, e["width"] * scale_x, e["height"] * scale_y)
        # Extra/uncategorized images are the lowest-priority passthrough
        # content (e.g. a small corner watermark) - cap their footprint so
        # they stay a modest accent on any canvas shape instead of scaling
        # up to a size that competes with real content. This also keeps
        # the renderer's "missing asset" fallback box (when a referenced
        # asset is genuinely absent) an unobtrusive corner mark rather
        # than a large, distracting placeholder.
        max_dim = min(tw, th) * 0.08
        if box[2] > max_dim or box[3] > max_dim:
            shrink = max_dim / max(box[2], box[3])
            new_w, new_h = box[2] * shrink, box[3] * shrink
            box = (box[0] + (box[2] - new_w) / 2, box[1] + (box[3] - new_h) / 2, new_w, new_h)
        out.append(_place_image(e, box))

    is_strip = archetype == "strip"
    allow_bleed_ids = {e["id"] for e in standalone_shapes if _is_out_of_bounds(e, sw, sh)}

    # See the module-level note on `_layout_strip` for why ribbons route
    # through the strip's own secondary band instead of this passthrough
    # when the archetype is "strip".
    passthrough_groups = generic_groups if is_strip else (ribbon_groups + generic_groups)
    if not is_strip:
        # A corner ribbon may keep bleeding only when its source already
        # does. Otherwise the rotated-bounds clamp pulls it back inside.
        # Exempting every ribbon let a wide, slightly tilted accent hang
        # off a short canvas even though the source was on the frame.
        for _, members in ribbon_groups:
            if any(_source_bleed(member, sw, sh) for member in members):
                allow_bleed_ids.update(member["id"] for member in members)
    for _, members in passthrough_groups:
        bbox = _group_bbox(members)
        new_box = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
        is_ribbon = any(abs(m.get("rotation", 0)) > 0.5 for m in members)
        if is_ribbon:
            restored = _restore_rotated_aspect(bbox, new_box, tw, th)
            if restored is not None:
                new_box = restored
            else:
                # Capped like a corner accent. A smaller ribbon is also an
                # easier fit for collision resolution. Do not apply this
                # cap after the aspect restore: that cap is what flattened
                # a readable accent back into a sliver.
                label_ratio = _group_label_height_ratio(members)
                max_dim = min(tw, th) * 0.12
                min_legible_h = 16.0 / label_ratio
                larger = max(new_box[2], new_box[3])
                shrink = min(1.0, max_dim / larger) if larger > 0 else 1.0
                if new_box[3] * shrink < min_legible_h:
                    shrink = max(shrink, min(1.0, min_legible_h / new_box[3])) if new_box[3] > 0 else shrink
                if shrink < 1.0:
                    nw, nh = new_box[2] * shrink, new_box[3] * shrink
                    new_box = (new_box[0] + (new_box[2] - nw) / 2, new_box[1] + (new_box[3] - nh) / 2, nw, nh)
        label = next((m for m in members if m["type"] == "text"), None)
        min_size = 10.0
        if label is not None:
            ratio = _group_label_height_ratio(members)
            if _text_min_height(label, new_box[2], min_size) > new_box[3] * ratio:
                min_size = _emergency_floor(label)
        out.extend(_apply_group_transform(members, bbox, new_box, min_size=min_size))

    if archetype == "strip":
        out += _layout_strip(
            logo, hero, headline, body_texts, legal, primary_cta, badges, ribbon_groups,
            importances, tw, th, margin, park_logo=park_logo,
        )
    elif archetype == "wide":
        out += _layout_wide(logo, hero, headline, body_texts, legal, primary_cta, badges, importances, tw, th, margin)
    elif archetype == "sparse":
        out += _layout_sparse(logo, headline, body_texts, legal, importances, tw, th)
    else:
        out += _layout_stack(logo, hero, headline, body_texts, legal, primary_cta, badges, importances, tw, th, margin)

    return out, allow_bleed_ids


# ---------------------------------------------------------------------------
# Collision resolution and grid alignment - two more of the rubric's own
# criteria ("no unwanted collisions", "things line up") applied directly as
# a cleanup pass on whichever candidate wins, rather than trusting layout
# math alone to always land clean. These are deliberately geometry-only
# (no change to what gets chosen) so they add robustness without adding
# risk to the archetype logic itself.
# ---------------------------------------------------------------------------


def _flow_units(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float],
) -> list[tuple[list[str], float]]:
    """The same message-bearing content as `_foreground_weights`, but kept
    as rigid units (all of a group's member ids together) rather than
    individual ids - so resolving a collision nudges a button's background
    and label as one piece instead of prying them apart."""
    units: list[tuple[list[str], float]] = []
    if logo:
        units.append(([logo["id"]], importances["logo"]))
    if hero:
        units.append(([hero["id"]], importances["hero"]))
    if headline:
        units.append(([headline["id"]], importances["headline"]))
    for i, bt in enumerate(body_texts):
        units.append(([bt["id"]], importances[f"body_{i}"]))
    if legal:
        units.append(([legal["id"]], importances["legal"]))
    if primary_cta:
        _, members = primary_cta
        units.append(([m["id"] for m in members], importances["cta"]))
    for i, (_, members) in enumerate(badges):
        units.append(([m["id"] for m in members], importances[f"badge_{i}"]))
    return units


def _unit_extent(members: list[Element]) -> Box:
    """A lone text line occupies its glyphs, not the empty rest of its box.

    A rotated accent can sit in that empty region. Groups and images still
    use the full painted box.
    """
    if len(members) == 1 and members[0].get("type") == "text":
        return _glyph_box(members[0])
    return _visual_bbox(members)


def _resolve_collisions(
    elements: list[Element], units: list[tuple[list[str], float]], tw: float, th: float, iterations: int = 16,
) -> list[Element]:
    """Pushes apart any two units whose boxes still overlap after layout,
    along whichever axis needs the smaller nudge, splitting the separation
    between them in inverse proportion to importance (the less important
    one gives way more). A few iterations handle chains of overlaps; it
    settles as soon as a pass makes no changes."""
    by_id = {e["id"]: e for e in elements}
    for _ in range(iterations):
        moved = False
        for i in range(len(units)):
            ids_a, wa = units[i]
            for j in range(i + 1, len(units)):
                ids_b, wb = units[j]
                ax0, ay0, aw, ah = _unit_extent([by_id[k] for k in ids_a])
                bx0, by0, bw, bh = _unit_extent([by_id[k] for k in ids_b])
                ox = min(ax0 + aw, bx0 + bw) - max(ax0, bx0)
                oy = min(ay0 + ah, by0 + bh) - max(ay0, by0)
                if ox <= 0 or oy <= 0:
                    continue
                total_w = wa + wb or 1.0
                a_share, b_share = wb / total_w, wa / total_w
                if ox < oy:
                    # +14 rather than +1: positions get snapped to a 4px
                    # grid right after this pass, and a too-precise
                    # separation can be nudged back into light contact by
                    # that rounding - extra headroom here has repeatedly
                    # been the cheapest fix for sub-pixel residual grazes.
                    push = ox + 10.0
                    direction = -1.0 if (ax0 + aw / 2) <= (bx0 + bw / 2) else 1.0
                    for k in ids_a:
                        el = by_id[k]
                        el["x"] = max(0.0, min(tw - el["width"], el["x"] + direction * push * a_share))
                    for k in ids_b:
                        el = by_id[k]
                        el["x"] = max(0.0, min(tw - el["width"], el["x"] - direction * push * b_share))
                else:
                    push = oy + 10.0
                    direction = -1.0 if (ay0 + ah / 2) <= (by0 + bh / 2) else 1.0
                    for k in ids_a:
                        el = by_id[k]
                        el["y"] = max(0.0, min(th - el["height"], el["y"] + direction * push * a_share))
                    for k in ids_b:
                        el = by_id[k]
                        el["y"] = max(0.0, min(th - el["height"], el["y"] - direction * push * b_share))
                moved = True
        if not moved:
            break
    return elements


def _snap_positions(elements: list[Element], tw: float, th: float, allow_bleed_ids: set, grid: float = 4.0) -> list[Element]:
    """Snaps every element's position (never its width/height, so this never
    touches text wrapping or an image's aspect ratio) to a small pixel
    grid - the same alignment discipline real design tools default to
    (Figma/Illustrator's pixel grid), so edges land in alignment instead of
    at arbitrary sub-pixel offsets."""
    out = []
    for e in elements:
        e = dict(e)
        x, y = round(e["x"] / grid) * grid, round(e["y"] / grid) * grid
        if e["id"] not in allow_bleed_ids:
            x = max(0.0, min(x, max(0.0, tw - e["width"])))
            y = max(0.0, min(y, max(0.0, th - e["height"])))
        e["x"], e["y"] = x, y
        out.append(e)
    return out


def _glyph_box(element: Element) -> tuple[float, float, float, float]:
    """The box the words actually occupy. A full-width line leaves the rest empty."""
    width = float(element["width"])
    if element.get("type") == "text":
        ink = _widest_line(
            str(element.get("content") or ""), element.get("font_family", ""),
            element.get("font_weight"), element.get("font_style"),
            float(element.get("font_size") or 1), width,
        )
        width = min(width, max(1.0, ink))
        align = str(element.get("text_align") or "left")
        if align == "center":
            x = float(element["x"]) + (float(element["width"]) - width) / 2.0
        elif align in {"right", "end"}:
            x = float(element["x"]) + float(element["width"]) - width
        else:
            x = float(element["x"])
    else:
        x = float(element["x"])
    return (x, float(element["y"]), width, float(element["height"]))


def _axis_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float], pad: float) -> bool:
    return not (
        left[0] + left[2] + pad <= right[0]
        or right[0] + right[2] + pad <= left[0]
        or left[1] + left[3] + pad <= right[1]
        or right[1] + right[3] + pad <= left[1]
    )


def _edge_pad(tw: float, th: float) -> float:
    """Inset from the canvas edge, as a fraction of the short side.

    A fixed 8 px margin is a large share of a small card and a hairline on
    a large one. 1.2% of the short side stays a small gap at every size,
    with a 4 px floor so two glyphs cannot sit on the same pixel.
    """
    return max(4.0, min(tw, th) * 0.012)


def _is_extreme_banner(tw: float, th: float) -> bool:
    """A canvas at least six times as wide as it is tall.

    That shape is one reading row, so the logo may park under the headline
    and the wide card is not a candidate. Whether a rotated accent can
    leave that row is a different measurement: the short side has to be
    tall enough for a readable label. Canvases that are not this wide
    still consider the wide card from 160 px.
    """
    return th > 1.0 and tw / th >= 6.0


def _has_room_for_a_corner(tw: float, th: float) -> bool:
    """True when 22% of the short side is at least a 28 px label.

    28 px is a readable accent, the same kind of glyph floor the engine
    already uses for a short label. A 90 px banner and a 120 px banner
    cannot spare that band. A 220 px canvas can, even when it is wider
    than 6:1. The search still caps the accent, so the open side stays open.
    """
    return th * 0.22 >= 28.0


def _full_canvas_backdrop(element: Element, tw: float, th: float) -> bool:
    """A shape or image that already covers the frame, so it is the ground.

    A hero, a logo, or a corner mark stays a foreground obstacle even when
    it is large. Size alone is not enough: the box has to start at the
    frame and reach the far edges.
    """
    if element.get("type") not in ("image", "shape"):
        return False
    x, y = float(element["x"]), float(element["y"])
    width, height = float(element["width"]), float(element["height"])
    return x <= tw * 0.05 and y <= th * 0.05 and x + width >= tw * 0.95 and y + height >= th * 0.95


def _painted_box(element: Element) -> tuple[float, float, float, float]:
    if abs(float(element.get("rotation") or 0.0)) > 0.5:
        return _visual_bbox([element])
    return (float(element["x"]), float(element["y"]), float(element["width"]), float(element["height"]))


def _foreground_obstacle(element: Element, tw: float, th: float) -> Optional[tuple[float, float, float, float]]:
    """Glyphs for text. Painted bounds for a visible shape or image.

    A full-canvas backdrop and a nearly invisible wash are not obstacles.
    """
    if element.get("type") == "text":
        return _glyph_box(element)
    if element.get("type") not in ("shape", "image"):
        return None
    if _full_canvas_backdrop(element, tw, th):
        return None
    if element.get("type") == "shape" and float(element.get("opacity", 1.0) or 1.0) < 0.25:
        return None
    return _painted_box(element)


def _relocate_offer_ribbons(design: dict, offer_ids: set[str], tw: float, th: float) -> dict:
    """Move a rotated discount group into open canvas, keeping its aspect.

    The caller chooses which labels may move. A discount uses this when
    the model ranked it primary or high. A rotated call to action uses
    it only when that group is the model's only high action. The role
    string is not changed here. A 6:1 banner is left alone:
    there is no corner, only a row. The accent may use at most 46% of the
    canvas width and 34% of its height, so the reading column remains.
    Those caps are fractions of the target, and the search samples the
    same fractions of the canvas at every size. The font has to grow by
    at least 35%: a fixed 8 px step is a new poster on a 12 px accent and
    a nudge on an 80 px one. Visible foreground shapes and images block a
    candidate. A backdrop that already covers the canvas does not.
    """
    if not _has_room_for_a_corner(tw, th) or not offer_ids:
        return design
    elements = [_copy_element(element) for element in design["elements"]]
    obstacles = []
    for element in elements:
        if element.get("id") in offer_ids or element.get("group_id") and any(
            other.get("id") in offer_ids and other.get("group_id") == element.get("group_id")
            for other in elements
        ):
            continue
        obstacle = _foreground_obstacle(element, tw, th)
        if obstacle is not None:
            obstacles.append(obstacle)
    moved = False
    for label in [element for element in elements if element.get("id") in offer_ids and element.get("type") == "text"]:
        group_id = label.get("group_id")
        if not group_id:
            continue
        members = [element for element in elements if element.get("group_id") == group_id]
        if not any(abs(float(member.get("rotation") or 0)) > 0.5 for member in members):
            continue
        old = _group_bbox(members)
        if old[2] <= 1 or old[3] <= 1:
            continue
        current_font = float(label.get("font_size") or 0)
        best_box = None
        best_font = current_font
        pad = _edge_pad(tw, th)
        # 35% of the accent's own size. The steps below are multiples of
        # the group's box, so a larger source accent searches larger boxes.
        for scale in (1.6, 2.0, 2.5, 3.0, 3.6, 4.2):
            new_w, new_h = old[2] * scale, old[3] * scale
            if new_w > tw * 0.46 or new_h > th * 0.34:
                continue
            for x_frac in (0.50, 0.62, 0.74, 0.86):
                for y_frac in (0.04, 0.16, 0.30, 0.46, 0.62):
                    x = min(tw - new_w - pad, max(pad, x_frac * tw))
                    y = min(th - new_h - pad, max(pad, y_frac * th))
                    trial = _apply_group_transform(members, old, (x, y, new_w, new_h), min_size=8.0)
                    visual = _visual_bbox(trial)
                    if visual[0] < 1 or visual[1] < 1 or visual[0] + visual[2] > tw - 1 or visual[1] + visual[3] > th - 1:
                        continue
                    if any(_axis_overlap(visual, obstacle, pad) for obstacle in obstacles):
                        continue
                    font = next((float(item.get("font_size") or 0) for item in trial if item.get("type") == "text"), 0)
                    if font >= current_font * 1.35 and font > best_font:
                        best_font = font
                        best_box = trial
        if best_box is None:
            continue
        replaced = {member["id"]: member for member in best_box}
        elements = [replaced.get(element["id"], element) for element in elements]
        moved = True
    if not moved:
        return design
    updated = dict(design)
    updated["elements"] = elements
    return updated


def resize(source: dict[str, Any], target_canvas: dict[str, Any], *, judgments: Optional[dict] = None) -> dict[str, Any]:
    """Resize `source` into `target_canvas`.

    `judgments` is an optional map of text-element id to a role, prominence,
    and must-read flag from the text-context analyzer. Omit it for the
    geometric baseline. Judgments never supply coordinates or font sizes.

    When judgments are passed, the engine also builds the baseline and keeps
    the judged layout only when it does not add a text collision or crush a
    line that was already readable. The extra pass is intentional: stage one
    has no time limit, and a readable design outranks a faster one.
    """
    token, ctx = layout_bridge.bind()
    try:
        if not judgments:
            result = _resize_locked(source, target_canvas, judgments=None)
            ctx.decision = "baseline"
            layout_bridge.publish(ctx)
            return result
        judged = _resize_locked(source, target_canvas, judgments=judgments)
        guidance = ctx.guidance
        override = None
        if getattr(guidance, "dropped_structural_slots", False):
            # The model left both source slots empty. A second candidate
            # puts those source elements back and is kept only when the
            # measured picture is clearer. The judgment itself is unchanged.
            retained = _resize_locked(
                source, target_canvas, judgments=judgments, retain_structural_slots=True,
            )
            retained_guidance = ctx.guidance
            wins, measured = _structural_candidate_wins(
                judged, retained, target_canvas,
                getattr(guidance, "heuristic_headline_id", None),
                getattr(guidance, "heuristic_legal_id", None),
            )
            if wins:
                judged = retained
                guidance = retained_guidance
                override = dict(retained_guidance.layout_override or {})
                override["measured"] = measured
            else:
                judged = _resize_locked(source, target_canvas, judgments=judgments)
                guidance = ctx.guidance
        ctx.layout_override = override
        offer_ids = set(getattr(guidance, "offer_ids", set()) or set())
        _clear_judgment_state()
        baseline = _resize_locked(source, target_canvas, judgments=None)
        if _keep_judged_layout(source, target_canvas, baseline, judged, guidance, judgments):
            ctx.decision = "judged"
            chosen = judged
            chosen_judgments: Optional[dict] = judgments
        else:
            ctx.decision = "baseline_kept"
            chosen = baseline
            chosen_judgments = None
            ctx.layout_override = None
        ctx.offer_moved = False
        if offer_ids and _has_room_for_a_corner(
            float(target_canvas.get("width") or 0), float(target_canvas.get("height") or 0),
        ):
            # The text-layout choice is already recorded. The offer move is
            # a separate candidate, and it is published only when the same
            # readability checks accept it and the painted result stays
            # clear of glyphs and foreground shapes.
            moved = _resize_locked(
                source, target_canvas, judgments=chosen_judgments, offer_ids=offer_ids,
                retain_structural_slots=ctx.layout_override is not None,
            )
            if _offer_geometry_changed(chosen, moved, offer_ids) and _offer_move_keeps_the_layout(
                source, target_canvas, chosen, moved, guidance, judgments,
            ):
                chosen = moved
                ctx.offer_moved = True
        ctx.accent_moved = False
        accent_ids = set(getattr(guidance, "solo_rotated_action_ids", set()) or set()) - offer_ids
        if (
            not ctx.offer_moved
            and accent_ids
            and _has_room_for_a_corner(
                float(target_canvas.get("width") or 0), float(target_canvas.get("height") or 0),
            )
        ):
            # Same corner search as a discount, for the one rotated group
            # the model already ranked a high action. The role is unchanged.
            moved = _resize_locked(
                source, target_canvas, judgments=chosen_judgments, offer_ids=accent_ids,
                retain_structural_slots=ctx.layout_override is not None,
            )
            if _offer_geometry_changed(chosen, moved, accent_ids) and _offer_move_keeps_the_layout(
                source, target_canvas, chosen, moved, guidance, judgments,
            ):
                chosen = moved
                ctx.accent_moved = True
        ctx.guidance = guidance
        layout_bridge.publish(ctx)
        return chosen
    finally:
        layout_bridge.unbind(token)


def _clear_judgment_state() -> None:
    ctx = layout_bridge.current()
    ctx.must_read_ids = set()
    ctx.protected_ids = set()
    ctx.promoted_ids = set()
    ctx.importance_by_id = {}


def _token_yields_the_row(
    eid: str, after: float, base_size: dict[str, float], judged_size: dict[str, float],
    judged: dict, essential: set[str],
) -> bool:
    """True when an unbreakable token shrank so another essential line could be read.

    The token has to stay one line and within half a pixel of the 12 px
    floor the rest of this gate uses. Some other essential line that was
    under 8 px has to end up at least 12 px. A normal sentence does not
    qualify, so a headline cannot use this to shrink itself.
    """
    element = next((item for item in judged.get("elements", []) if item.get("id") == eid), None)
    if element is None or element.get("type") != "text":
        return False
    if not _unbreakable(str(element.get("content") or "")):
        return False
    if after < 11.5:
        return False
    for other, before in base_size.items():
        if other == eid or other not in essential:
            continue
        if before < 8.0 and judged_size.get(other, before) >= 12.0:
            return True
    return False


def _offer_price_keeps_its_size(eid: str, judgments: Optional[dict]) -> bool:
    """A price the model ranked primary or high is part of the offer.

    It does not give up half its size so another line can grow. A medium,
    low, or minimal price can still yield, so a product-led design is not
    forced to keep every price large. No judgment means this is false and
    the older yield stays.
    """
    row = (judgments or {}).get(eid)
    if not isinstance(row, dict) or row.get("role") != "price":
        return False
    return row.get("prominence") in {"primary", "high"}


def _secondary_yields_to_a_larger_line(
    eid: str, before: float, after: float, base_size: dict[str, float],
    judged_size: dict[str, float], guidance: Any, essential: set[str],
    judgments: Optional[dict] = None,
) -> bool:
    """A non-headline line may shrink when another essential line grows.

    The baseline often gives the largest source font the lead slot. On a
    real resize that can leave a price or a badge bigger than the headline
    or the action. Yielding is allowed while the line stays at least 16 px
    and some other essential line grows by 4 px or more. Message lines keep
    at least half their baseline size; a badge may keep 35 percent. The
    headline itself never uses this exception. A price the model still
    ranked primary or high does not use it either.
    """
    if _offer_price_keeps_its_size(eid, judgments):
        return False
    headline = getattr(guidance, "headline", None)
    headline_id = headline.get("id") if isinstance(headline, dict) else None
    if headline_id is None or eid == headline_id:
        return False
    # A badge is a secondary accent even if the model calls its label a
    # headline. Its structural slot can yield more room to the real action.
    # Prices and other message lines keep the stricter half-size limit.
    min_fraction = 0.35 if getattr(guidance, "slots", {}).get(eid) == "badge" else 0.50
    if after < 16.0 or after < before * min_fraction:
        return False
    for other in essential:
        if other == eid:
            continue
        was = base_size.get(other)
        now = judged_size.get(other, was)
        if was is None or now is None:
            continue
        if now >= was + 4.0:
            return True
    return False


def _high_price_lost_the_lead(
    baseline: dict, judged: dict, guidance: Any, judgments: Optional[dict],
) -> bool:
    """True when a high or primary price led the baseline and the judged
    layout put another line above it.

    A low or medium price can still sit under a product name. A design
    whose baseline already led with that other line is left alone, so a
    product-led poster is not rewritten into a price poster.
    """
    headline = getattr(guidance, "headline", None)
    headline_id = headline.get("id") if isinstance(headline, dict) else None
    if not headline_id or not judgments:
        return False
    base_size = _font_sizes(baseline)
    judged_size = _font_sizes(judged)
    if headline_id not in base_size or headline_id not in judged_size:
        return False
    for eid, row in judgments.items():
        if not isinstance(row, dict) or row.get("role") != "price":
            continue
        if row.get("prominence") not in {"primary", "high"}:
            continue
        before_price = base_size.get(eid)
        after_price = judged_size.get(eid)
        if before_price is None or after_price is None:
            continue
        if before_price + 0.5 < base_size[headline_id]:
            continue
        if judged_size[headline_id] > after_price + 0.5:
            return True
    return False


def _offer_geometry_changed(before: dict, after: dict, offer_ids: set[str]) -> bool:
    """True when the discount group itself moved or changed size."""
    def geom(design: dict) -> dict[str, tuple]:
        elements = design["elements"]
        groups = {
            element.get("group_id")
            for element in elements
            if element.get("id") in offer_ids and element.get("group_id")
        }
        found = {}
        for element in elements:
            if element.get("id") not in offer_ids and element.get("group_id") not in groups:
                continue
            found[element["id"]] = tuple(
                round(float(element.get(key) or 0.0), 2)
                for key in ("x", "y", "width", "height", "font_size")
            )
        return found

    return geom(before) != geom(after)


def _offer_move_keeps_the_layout(
    source: dict, target_canvas: dict, before: dict, after: dict,
    guidance: Any, judgments: Optional[dict],
) -> bool:
    """True when the moved offer is safe to publish on top of `before`.

    Readability uses the same rules as the text-layout gate. Collision
    uses glyphs and visible foreground shapes, so an accent in the empty
    part of a wide text box is not treated as a hit. A full-canvas
    backdrop is not a hit either.
    """
    if not _keep_judged_layout(
        source, target_canvas, before, after, guidance, judgments, box_overlap=False,
    ):
        return False
    return not _visible_collision_got_worse(
        before, after, float(target_canvas["width"]), float(target_canvas["height"]),
    )


def _visible_collision_got_worse(before: dict, after: dict, tw: float, th: float) -> bool:
    prior = _visible_collision_map(before, tw, th)
    later = _visible_collision_map(after, tw, th)
    for pair, share in later.items():
        if share > prior.get(pair, 0.0) + 0.05:
            return True
    return False


def _visible_collision_map(design: dict, tw: float, th: float) -> dict[frozenset, float]:
    elements = design["elements"]
    found: dict[frozenset, float] = {}
    for left in elements:
        if left.get("type") != "text":
            continue
        left_box = _box_dict(_glyph_box(left))
        for right in elements:
            if right.get("id") == left.get("id"):
                continue
            if left.get("group_id") and left.get("group_id") == right.get("group_id"):
                continue
            obstacle = _foreground_obstacle(right, tw, th)
            if obstacle is None:
                continue
            right_box = _box_dict(obstacle)
            share = _overlap_fraction(left_box, right_box)
            if share > 0.05:
                pair = frozenset((left["id"], right["id"]))
                found[pair] = max(found.get(pair, 0.0), share)
    return found


def _box_dict(box: tuple[float, float, float, float]) -> dict[str, float]:
    return {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}


def _image_area(design: dict, tw: float, th: float) -> float:
    total = 0.0
    for element in design.get("elements", []):
        if element.get("type") != "image" or _covers_canvas(element, tw, th):
            continue
        total += float(element["width"]) * float(element["height"])
    return total


def _off_canvas_count(design: dict, tw: float, th: float) -> int:
    count = 0
    for element in design.get("elements", []):
        x, y = float(element["x"]), float(element["y"])
        w, h = float(element["width"]), float(element["height"])
        if x < -0.5 or y < -0.5 or x + w > tw + 0.5 or y + h > th + 0.5:
            count += 1
    return count


def _images_held(empty: dict, retained: dict, tw: float, th: float) -> bool:
    """A retained picture may not shrink a visible image.

    An image the empty layout already crushed can grow. A full-canvas
    backdrop is not one of these images.
    """
    floor = max(24.0, th * 0.20)
    retained_by = {element["id"]: element for element in retained.get("elements", [])}
    for element in empty.get("elements", []):
        if element.get("type") != "image" or _covers_canvas(element, tw, th):
            continue
        other = retained_by.get(element["id"])
        if other is None:
            return False
        before = float(element["height"])
        after = float(other["height"])
        if before < floor and after > before + 0.5:
            continue
        if after + 0.5 < before * 0.90:
            return False
    return True


def _structural_candidate_wins(
    empty: dict, retained: dict, target_canvas: dict,
    headline_id: Optional[str], legal_id: Optional[str],
) -> tuple[bool, dict[str, Any]]:
    """True when putting the source slots back measures better.

    The check is geometry: element identity, overlap, overflow, bleed,
    image height, and the font size of those two source lines. It does
    not read the words.
    """
    tw = float(target_canvas["width"])
    th = float(target_canvas["height"])
    measured: dict[str, Any] = {}
    if not headline_id or not legal_id:
        return False, measured
    if [element["id"] for element in empty.get("elements", [])] != [
        element["id"] for element in retained.get("elements", [])
    ]:
        return False, measured
    if _overlap_got_worse(empty, retained):
        return False, measured
    if _text_overflow_ids(retained) - _text_overflow_ids(empty):
        return False, measured
    if _off_canvas_count(retained, tw, th) > _off_canvas_count(empty, tw, th):
        return False, measured
    empty_fonts = _font_sizes(empty)
    retained_fonts = _font_sizes(retained)
    empty_max = max(empty_fonts.values()) if empty_fonts else 0.0
    retained_max = max(retained_fonts.values()) if retained_fonts else 0.0
    lead_before = empty_fonts.get(headline_id, 0.0)
    lead_after = retained_fonts.get(headline_id, 0.0)
    fine_before = empty_fonts.get(legal_id, 0.0)
    fine_after = retained_fonts.get(legal_id, 0.0)
    measured = {
        "headline_font": [round(lead_before, 1), round(lead_after, 1)],
        "legal_font": [round(fine_before, 1), round(fine_after, 1)],
        "image_area": [round(_image_area(empty, tw, th), 1), round(_image_area(retained, tw, th), 1)],
        "max_font": [round(empty_max, 1), round(retained_max, 1)],
    }
    if empty_max > 0 and retained_max < empty_max * 0.75:
        return False, measured
    if not _images_held(empty, retained, tw, th):
        return False, measured
    lead_up = lead_after >= lead_before + 4.0 and fine_after + 1.0 >= fine_before
    fine_up = fine_after >= fine_before + 1.0 and lead_after + 1.0 >= lead_before
    return lead_up or fine_up, measured


def _keep_judged_layout(
    source: dict, target_canvas: dict, baseline: dict, judged: dict,
    guidance: Any, judgments: Optional[dict], *, box_overlap: bool = True,
) -> bool:
    """True when the judged design is the one to return.

    A model reply is not enough. The judged layout is dropped when it adds
    a text collision, adds a text-box overflow, crushes a line that was
    already at least 16 px, or makes an essential line less readable than
    the baseline.     A same-size canvas keeps the baseline unless the judged
    design removes sub-8 px text, or grows an essential line by at least
    2 px without shrinking another essential line by more than 2 px and
    below 92 percent of its baseline size.
    """
    if box_overlap and _overlap_got_worse(baseline, judged):
        return False
    new_overflows = _text_overflow_ids(judged) - _text_overflow_ids(baseline)
    if new_overflows:
        return False
    base_size = _font_sizes(baseline)
    judged_size = _font_sizes(judged)
    # A baseline size that already spills out of its box is not a readable
    # size to protect. Shrinking it until the glyphs fit is the correction.
    base_overflow = _text_overflow_ids(baseline)
    essential = set(getattr(guidance, "essential_ids", set()) or set())
    for eid, before in base_size.items():
        after = judged_size.get(eid, before)
        if eid in base_overflow:
            continue
        if before >= 16.0 and after < 12.0 and not _token_yields_the_row(
            eid, after, base_size, judged_size, judged, essential,
        ):
            return False
        # A headline that is already large should not be cut down so a
        # secondary label can grow. A price or a badge can give up part of
        # its baseline size when the headline or the action actually grows
        # and the smaller line stays readable. An unbreakable token is the
        # other exception: it can only get smaller, and keeping it large is
        # what crushes the other line on a short row.
        if before >= 24.0 and after < before * 0.75 and not _token_yields_the_row(
            eid, after, base_size, judged_size, judged, essential,
        ) and not _secondary_yields_to_a_larger_line(
            eid, before, after, base_size, judged_size, guidance, essential, judgments,
        ):
            return False
    for eid in essential:
        before = base_size.get(eid)
        after = judged_size.get(eid, before)
        if before is None or after is None or eid in base_overflow:
            continue
        # A line that is already too small may lose a pixel when the
        # headline and action take the room. A real collapse, or a line
        # that was readable and stops being readable, is still rejected.
        if before < 8.0 and after + 1.5 < before:
            return False
        if before >= 8.0 and after < 8.0:
            return False
    # A small must-read disclosure, price, or date may not get smaller
    # than the baseline. A line that is already large can lose a pixel.
    for eid in set(getattr(guidance, "protected_ids", set()) or set()):
        before = base_size.get(eid)
        after = judged_size.get(eid, before)
        if before is None or after is None:
            continue
        if before < 12.0 and after + 0.4 < before:
            return False
    if _sub_eight_count(judged) > _sub_eight_count(baseline):
        # A crowded canvas cannot make every line large. Keep the judged
        # layout when a message line actually got bigger and the extra
        # small type is not one of those message lines. The checks above
        # already reject a new collision, a new clip, or an essential
        # line that fell under 8 px.
        grew = any(
            judged_size.get(eid, 0.0) >= base_size.get(eid, 0.0) + 1.0
            for eid in essential
        )
        if not grew:
            return False
    if _high_price_lost_the_lead(baseline, judged, guidance, judgments):
        return False
    if _nearly_same_canvas(source, target_canvas):
        # The canvas did not change shape, so a rearrangement has to earn
        # its place. Fewer unreadable lines qualifies. So does a clearer
        # essential line, as long as no other essential line gives up a
        # real amount of size. A flat judged layout stays on the baseline.
        if _sub_eight_count(judged) < _sub_eight_count(baseline):
            return True
        grew = False
        for eid in essential:
            before = base_size.get(eid)
            after = judged_size.get(eid, before)
            if before is None or after is None:
                continue
            if after >= before + 2.0:
                grew = True
            if after + 2.0 < before and after < before * 0.92:
                return False
        return grew
    return True


def _text_overflow_ids(design: dict) -> set[str]:
    """Text whose wrapped glyphs are taller than the box, or whose longest
    unbreakable line is wider than the box. The renderer does not clip."""
    found = set()
    for element in design["elements"]:
        if element.get("type") != "text":
            continue
        size = float(element.get("font_size") or 0)
        if size <= 0:
            continue
        width = float(element["width"])
        needed = _text_min_height(element, width, size)
        wide = _widest_line(
            element.get("content", ""), element.get("font_family", ""),
            element.get("font_weight"), element.get("font_style"), size, width,
        )
        if needed > float(element["height"]) + 1.0 or wide > width + 1.0:
            found.add(element["id"])
    return found


def _nearly_same_canvas(source: dict, target_canvas: dict) -> bool:
    sw = float(source["canvas"]["width"])
    sh = float(source["canvas"]["height"])
    tw = float(target_canvas["width"])
    th = float(target_canvas["height"])
    return abs(tw - sw) <= max(2.0, sw * 0.02) and abs(th - sh) <= max(2.0, sh * 0.02)


def _font_sizes(design: dict) -> dict[str, float]:
    return {
        element["id"]: float(element.get("font_size") or 0)
        for element in design["elements"]
        if element.get("type") == "text"
    }


def _sub_eight_count(design: dict) -> int:
    return sum(1 for size in _font_sizes(design).values() if size < 8.0)


def _overlap_got_worse(baseline: dict, judged: dict) -> bool:
    """A text overlap that appears, or one that covers noticeably more of
    the smaller element than the baseline already did.

    A wide box can cross a neighbor in the empty part past the last letter.
    That pair counts only when the words themselves also cover more of the
    neighbor than they did before.
    """
    before = _text_overlap_map(baseline)
    after = _text_overlap_map(judged)
    before_words = _word_overlap_map(baseline)
    after_words = _word_overlap_map(judged)
    for pair, share in after.items():
        if share <= before.get(pair, 0.0) + 0.05:
            continue
        if after_words.get(pair, 0.0) <= before_words.get(pair, 0.0) + 0.05:
            continue
        return True
    return False


def _word_overlap_map(design: dict) -> dict[frozenset, float]:
    """Overlap of the words, not the leftover box, against the other element's box."""
    elements = design["elements"]
    found: dict[frozenset, float] = {}
    for index, left in enumerate(elements):
        if left.get("type") != "text":
            continue
        left_box = _box_dict(_glyph_box(left))
        for other_index, right in enumerate(elements):
            if other_index == index:
                continue
            if left.get("group_id") and left.get("group_id") == right.get("group_id"):
                continue
            if right.get("type") not in ("text", "shape", "image"):
                continue
            if right.get("type") == "shape" and float(right.get("opacity", 1.0) or 1.0) < 0.25:
                continue
            right_box = _box_dict(_glyph_box(right)) if right.get("type") == "text" else right
            share = _overlap_fraction(left_box, right_box)
            if share > 0.05:
                pair = frozenset((left["id"], right["id"]))
                found[pair] = max(found.get(pair, 0.0), share)
    return found


def _text_overlap_map(design: dict) -> dict[frozenset, float]:
    elements = design["elements"]
    found: dict[frozenset, float] = {}
    for index, left in enumerate(elements):
        if left.get("type") != "text":
            continue
        for other_index, right in enumerate(elements):
            if other_index == index:
                continue
            if left.get("group_id") and left.get("group_id") == right.get("group_id"):
                continue
            if right.get("type") not in ("text", "shape", "image"):
                continue
            # A very faint shape is a wash behind the design, not a collision.
            if right.get("type") == "shape" and float(right.get("opacity", 1.0) or 1.0) < 0.25:
                continue
            share = _overlap_fraction(left, right)
            if share > 0.05:
                pair = frozenset((left["id"], right["id"]))
                found[pair] = max(found.get(pair, 0.0), share)
    return found


def _text_overlap_pairs(design: dict) -> set[frozenset]:
    """Text overlaps above 5% of the smaller box, ignoring a text sitting
    on the shape it was grouped with."""
    return set(_text_overlap_map(design))


def _overlap_fraction(left: dict, right: dict) -> float:
    ix = min(left["x"] + left["width"], right["x"] + right["width"]) - max(left["x"], right["x"])
    iy = min(left["y"] + left["height"], right["y"] + right["height"]) - max(left["y"], right["y"])
    if ix <= 0 or iy <= 0:
        return 0.0
    smaller = min(left["width"] * left["height"], right["width"] * right["height"])
    if smaller <= 0:
        return 0.0
    return (ix * iy) / smaller


def _resize_locked(
    source: dict[str, Any], target_canvas: dict[str, Any], judgments: Optional[dict] = None,
    offer_ids: Optional[set[str]] = None, *, retain_structural_slots: bool = False,
) -> dict[str, Any]:
    """See module docstring for the approach. Returns a design dict with the
    same shape as `source`, resized/reflowed - never stretched - into
    `target_canvas`."""
    elements = [_copy_element(e) for e in source["elements"]]
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    tw, th = float(target_canvas["width"]), float(target_canvas["height"])
    scale_x, scale_y = tw / sw, th / sh

    groups: dict[str, list[Element]] = {}
    ungrouped: list[Element] = []
    for e in elements:
        gid = e.get("group_id")
        (groups.setdefault(gid, []) if gid else ungrouped).append(e)

    # --- classify ungrouped elements ---
    backgrounds = [e for e in ungrouped if _covers_canvas(e, sw, sh)]
    rest = [e for e in ungrouped if e not in backgrounds]
    standalone_shapes = [e for e in rest if e["type"] == "shape"]
    images = [e for e in rest if e["type"] == "image"]
    texts = [e for e in rest if e["type"] == "text"]

    logo_like = [e for e in images if _looks_like_logo(e, sh)]
    logo = min(logo_like, key=lambda e: e["y"]) if logo_like else None
    other_images = [e for e in images if e is not logo]
    hero = max(other_images, key=lambda e: e["width"] * e["height"]) if other_images else None
    extra_images = [e for e in other_images if e is not hero]

    texts_sorted = sorted(texts, key=lambda e: e["font_size"], reverse=True)
    headline = texts_sorted[0] if texts_sorted else None
    legal = None
    body_texts: list[Element] = []
    if len(texts_sorted) > 1:
        rest_texts = texts_sorted[1:]
        smallest = rest_texts[-1]
        median_size = texts_sorted[len(texts_sorted) // 2]["font_size"]
        if smallest["font_size"] <= median_size * 0.65:
            legal = smallest
            rest_texts = rest_texts[:-1]
        body_texts = sorted(rest_texts, key=lambda e: e["y"])

    # --- classify groups ---
    pill_groups: list[tuple[str, list[Element]]] = []
    ribbon_groups: list[tuple[str, list[Element]]] = []
    generic_groups: list[tuple[str, list[Element]]] = []
    for gid, members in groups.items():
        has_rotation = any(abs(m.get("rotation", 0)) > 0.5 for m in members)
        has_shape = any(m["type"] == "shape" for m in members)
        has_text = any(m["type"] == "text" for m in members)
        if has_rotation:
            ribbon_groups.append((gid, members))
        elif has_shape and has_text:
            pill_groups.append((gid, members))
        else:
            generic_groups.append((gid, members))

    pill_groups.sort(key=lambda gm: _group_bbox(gm[1])[2] * _group_bbox(gm[1])[3], reverse=True)
    primary_cta = pill_groups[0] if pill_groups else None
    badges = pill_groups[1:]

    guidance = None
    if judgments:
        guidance = layout_bridge.apply_text_judgments(
            judgments, texts, pill_groups, ribbon_groups, headline, legal, primary_cta, badges,
            retain_structural_slots=retain_structural_slots,
        )
        headline = guidance.headline
        body_texts = guidance.body_texts
        legal = guidance.legal
        primary_cta = guidance.primary_cta
        badges = guidance.badges
        # The 8 px floor and the opacity lock apply to disclosures, prices,
        # dates, and a must-read call to action. A badge the model also
        # marked must-read keeps the engine's normal compression.
        readable = set(guidance.protected_ids) | set(guidance.essential_ids)
        # A restored source slot keeps the legibility floor. The model's
        # role string stays as the model wrote it.
        if guidance.layout_override:
            for eid in (
                guidance.layout_override.get("headline_id"),
                guidance.layout_override.get("legal_id"),
            ):
                if eid:
                    readable.add(eid)
        if guidance.primary_cta is not None:
            label = next((m for m in guidance.primary_cta[1] if m.get("type") == "text"), None)
            if label is not None and label["id"] in guidance.must_read_ids:
                readable.add(label["id"])
        ctx = layout_bridge.current()
        ctx.must_read_ids = readable
        ctx.protected_ids = set(guidance.protected_ids)
        ctx.promoted_ids = set(guidance.promoted_ids)
        ctx.importance_by_id = dict(guidance.importance_by_id)
        ctx.guidance = guidance
    else:
        layout_bridge.current().guidance = guidance

    importances = _compute_importances(logo, hero, headline, body_texts, legal, primary_cta, badges, sw, sh)
    if guidance is not None:
        importances = layout_bridge.overlay_importances(
            importances, guidance.importance_by_id, headline, body_texts, legal, primary_cta, badges,
        )
    margin = max(8.0, min(tw, th) * 0.045)
    ratio = tw / th

    # Build every genuinely-plausible archetype in full (see
    # `_candidate_archetypes` - usually just the one clear-cut choice, two
    # only near an actual boundary) and keep whichever scores best, instead
    # of letting a hard aspect-ratio cutoff decide blind.
    fg_weights = _foreground_weights(logo, hero, headline, body_texts, legal, primary_cta, badges, importances)
    best_elements: Optional[list[Element]] = None
    best_bleed_ids: set = set()
    best_name = ""
    best_score = float("inf")
    names = list(_candidate_archetypes(th, ratio))
    if _sparse_reading_fits(hero, badges, headline, body_texts, legal, primary_cta, tw, th):
        names.append("sparse")
    for name in names:
        cand_elements, cand_bleed_ids = _build_candidate(
            name, backgrounds, standalone_shapes, extra_images, generic_groups, ribbon_groups,
            logo, hero, headline, body_texts, legal, primary_cta, badges, importances,
            scale_x, scale_y, sw, sh, tw, th, margin,
        )
        score = _score_candidate(cand_elements, fg_weights, tw, th)
        if score < best_score:
            best_score, best_elements, best_bleed_ids, best_name = score, cand_elements, cand_bleed_ids, name

    if (
        best_name == "strip" and best_elements is not None and logo is not None
        and headline is not None and _is_extreme_banner(tw, th)
    ):
        parked, parked_bleed = _build_candidate(
            "strip", backgrounds, standalone_shapes, extra_images, generic_groups, ribbon_groups,
            logo, hero, headline, body_texts, legal, primary_cta, badges, importances,
            scale_x, scale_y, sw, sh, tw, th, margin, park_logo=True,
        )
        cta_label = None
        if primary_cta is not None:
            cta_label = next((member for member in primary_cta[1] if member.get("type") == "text"), None)
        badge_ids = set()
        for _, members in badges:
            label = next((member for member in members if member.get("type") == "text"), None)
            if label is not None:
                badge_ids.add(label["id"])
        if _parked_logo_clears_the_headline(
            best_elements, parked, headline["id"],
            legal["id"] if legal is not None else None,
            cta_label["id"] if cta_label is not None else None,
            logo["id"], hero["id"] if hero is not None else None,
            badge_ids, tw, th,
        ):
            best_elements, best_bleed_ids = parked, parked_bleed

    # Ribbons (outside the strip archetype, where they're already placed
    # in-flow) and uncategorized "extra" images are positioned by simple
    # proportional passthrough, completely independent of where the
    # chosen archetype reflowed everything else - at an aspect ratio far
    # from the source's, that can land them right on top of reflowed
    # content (e.g. a logo that moved, or legal text that's now
    # elsewhere). Including them here as low-importance units lets them
    # get pushed aside on contact instead of silently overlapping it.
    if best_elements is not None and offer_ids and best_name != "strip" and _has_room_for_a_corner(tw, th):
        # Place a high discount group before collision cleanup. The later
        # shrink only narrows a group in place, so a corner accent cannot
        # discover the open region once that pass has started.
        best_elements = _relocate_offer_ribbons(
            {"canvas": {"width": tw, "height": th}, "elements": best_elements},
            set(offer_ids), tw, th,
        )["elements"]
    units = _flow_units(logo, hero, headline, body_texts, legal, primary_cta, badges, importances)
    if best_name != "strip":
        units += [([m["id"] for m in members], layout_bridge.judged_importance(members, 0.15)) for _, members in ribbon_groups]
    units += [([e["id"]], 0.2) for e in extra_images]
    out_elements = _resolve_collisions(best_elements or [], units, tw, th)
    out_elements = _clamp_elements(out_elements, tw, th, best_bleed_ids)
    out_elements = _snap_positions(out_elements, tw, th, best_bleed_ids)
    # Grid-snapping can nudge two already-separated elements back into
    # light contact (each can move up to ~2px toward the other), and
    # clamping a just-resolved element back on-canvas can likewise graze
    # a neighbor it wasn't touching before - resolve and clamp don't
    # commute perfectly. Rather than tune a single push-buffer constant
    # to survive both in one shot (which empirically just moves the
    # residual collision to a different test case instead of removing
    # it), alternate the two until a pass finds nothing left to fix.
    for _ in range(5):
        before = {e["id"]: (e["x"], e["y"]) for e in out_elements}
        out_elements = _resolve_collisions(out_elements, units, tw, th, iterations=8)
        out_elements = _clamp_elements(out_elements, tw, th, best_bleed_ids)
        if all(before.get(e["id"]) == (e["x"], e["y"]) for e in out_elements):
            break

    # Headline/body/legal text sits directly on the background (unlike a
    # badge or CTA label, it isn't locked to a group's rigid transform),
    # so it's free to take a small, collision-safe local nudge toward a
    # visually calmer part of that background when it's landed somewhere
    # busy - directly serving the rubric's own "text never sits on a busy
    # image" criterion, using the real rendered pixels rather than a
    # guess about this particular asset.
    eligible_ids = set()
    if headline:
        eligible_ids.add(headline["id"])
    for bt in body_texts:
        eligible_ids.add(bt["id"])
    if legal:
        eligible_ids.add(legal["id"])
    if eligible_ids and backgrounds:
        bg_gray = _load_cover_background_gray(backgrounds[0].get("asset_id"), tw, th)
        if bg_gray is not None:
            # Deliberately broader than `units` (which excludes ribbons in
            # the strip archetype, since those are handled by its own
            # secondary-band placement rather than the general collision
            # resolver) - a candidate nudge here needs to avoid landing on
            # ANYTHING actually occupying canvas space, not just the
            # subset of elements the resolver itself tracks.
            exclude_ids = {e["id"] for e in backgrounds} | {e["id"] for e in standalone_shapes}
            other_boxes = {
                e["id"]: (e["x"], e["y"], e["width"], e["height"])
                for e in out_elements if e["id"] not in exclude_ids
            }
            out_elements = _avoid_busy_background(out_elements, eligible_ids, other_boxes, bg_gray, tw, th)
            out_elements = _clamp_elements(out_elements, tw, th, best_bleed_ids)

    for _, members in ribbon_groups:
        if any(member["id"] in best_bleed_ids for member in members):
            continue
        # Strip ribbons used to skip this pass. A rotated accent placed in
        # the row can still swing into a neighbor's glyphs. A group that
        # is already clear is left where it is.
        _shrink_tilted_group_clear_of_neighbors(out_elements, members, tw, th)
        out_elements = _clamp_elements(out_elements, tw, th, best_bleed_ids)

    _fit_units_on_canvas(out_elements, tw, th)

    out_by_id = {e["id"]: e for e in out_elements}
    ordered = [out_by_id[e["id"]] for e in source["elements"]]
    return {"canvas": {"width": target_canvas["width"], "height": target_canvas["height"]}, "elements": ordered}


def _shrink_tilted_group_clear_of_neighbors(
    elements: list[Element], members: list[Element], tw: float, th: float,
) -> None:
    """Narrow a tilted group that still covers a neighbor or leaves the frame.

    Height stays put, so the label's font is not crushed to buy width. The
    group keeps the canvas edge it is already nearest. A group that is
    already clear is left alone, which is what keeps a same-size accent
    where the source put it.
    """
    ids = {member["id"] for member in members}
    by_id = {element["id"]: element for element in elements}
    group = [by_id[eid] for eid in ids if eid in by_id]
    if len(group) < 2 or not any(abs(float(element.get("rotation") or 0)) > 0.5 for element in group):
        return

    def painted() -> Box:
        return _visual_bbox(group)

    def blocked() -> bool:
        box = painted()
        if box[0] < -0.5 or box[1] < -0.5 or box[0] + box[2] > tw + 0.5 or box[1] + box[3] > th + 0.5:
            return True
        painted_el = {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}
        for other in elements:
            if other["id"] in ids:
                continue
            if float(other["width"]) * float(other["height"]) > 0.85 * tw * th:
                continue
            if other.get("type") == "text":
                gx, gy, gw, gh = _glyph_box(other)
                neighbor = {"x": gx, "y": gy, "width": gw, "height": gh}
            else:
                neighbor = other
            if _overlap_fraction(painted_el, neighbor) > 0.05:
                return True
        return False

    if not blocked():
        return
    anchor_right = painted()[0] + painted()[2] / 2.0 >= tw / 2.0
    label = next((element for element in group if element.get("type") == "text"), None)
    for _ in range(12):
        if not blocked():
            break
        if label is not None:
            one_line = _widest_line(
                label.get("content", ""), label.get("font_family", ""),
                label.get("font_weight"), label.get("font_style"),
                float(label["font_size"]), 10000.0,
            )
            if float(label["width"]) * 0.82 < one_line + 4.0:
                break
        for element in group:
            width = float(element["width"])
            if width <= 36.0:
                return
            narrowed = max(36.0, width * 0.82)
            if anchor_right:
                element["x"] = float(element["x"]) + width - narrowed
            element["width"] = narrowed
        box = painted()
        dx = dy = 0.0
        if box[0] < 0:
            dx = -box[0]
        elif box[0] + box[2] > tw:
            dx = tw - (box[0] + box[2])
        if box[1] < 0:
            dy = -box[1]
        elif box[1] + box[3] > th:
            dy = th - (box[1] + box[3])
        if dx or dy:
            for element in group:
                element["x"] = float(element["x"]) + dx
                element["y"] = float(element["y"]) + dy


def _fit_units_on_canvas(elements: list[Element], tw: float, th: float) -> None:
    """Pull every unit fully onto the canvas without squashing it.

    Members of one group share a scale and a shift, so a button and its
    label stay together. Width and height scale by the same factor, so a
    circle stays a circle and an image keeps its aspect. A `fit: cover`
    image may still be cropped by the renderer inside that box. The asset
    file is not edited.
    """
    groups: dict[str, list[Element]] = {}
    order: list[str] = []
    for index, element in enumerate(elements):
        key = str(element.get("group_id") or f"\0{index}")
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(element)
    for key in order:
        _fit_one_unit_on_canvas(groups[key], tw, th)


def _unit_paint(members: list[Element]) -> tuple[float, float, float, float]:
    boxes = [
        _rotated_bbox(
            float(member["x"]), float(member["y"]),
            float(member["width"]), float(member["height"]),
            float(member.get("rotation") or 0),
        )
        for member in members
    ]
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _fit_one_unit_on_canvas(members: list[Element], tw: float, th: float) -> None:
    if tw <= 0 or th <= 0 or not members:
        return
    x0, y0, painted_w, painted_h = _unit_paint(members)
    if painted_w <= 0 or painted_h <= 0:
        return
    if x0 >= -0.49 and y0 >= -0.49 and x0 + painted_w <= tw + 0.49 and y0 + painted_h <= th + 0.49:
        return
    limit_w = max(1.0, tw - 0.5)
    limit_h = max(1.0, th - 0.5)
    scale = min(1.0, limit_w / painted_w, limit_h / painted_h)
    if scale < 0.999:
        for element in members:
            cx = float(element["x"]) + float(element["width"]) / 2.0
            cy = float(element["y"]) + float(element["height"]) / 2.0
            ncx = x0 + (cx - x0) * scale
            ncy = y0 + (cy - y0) * scale
            nw = float(element["width"]) * scale
            nh = float(element["height"]) * scale
            element["x"] = ncx - nw / 2.0
            element["y"] = ncy - nh / 2.0
            element["width"] = nw
            element["height"] = nh
            if element.get("type") == "text" and element.get("font_size"):
                element["font_size"] = max(1.0, float(element["font_size"]) * scale)
        x0, y0, painted_w, painted_h = _unit_paint(members)
    dx = dy = 0.0
    if x0 < 0:
        dx = -x0
    elif x0 + painted_w > tw:
        dx = tw - (x0 + painted_w)
    if y0 < 0:
        dy = -y0
    elif y0 + painted_h > th:
        dy = th - (y0 + painted_h)
    if dx or dy:
        for element in members:
            element["x"] = float(element["x"]) + dx
            element["y"] = float(element["y"]) + dy


def _clamp_elements(elements: list[Element], tw: float, th: float, allow_bleed_ids: set) -> list[Element]:
    out = []
    for e in elements:
        if e["id"] in allow_bleed_ids:
            out.append(e)
            continue
        e = dict(e)
        rotation = e.get("rotation", 0)
        if rotation:
            # Clamp by the rotated (true visual) extent, then apply the
            # same shift to the stored unrotated x/y - width/height stay
            # as they are (they're the unrotated storage box the renderer
            # rotates about its own center), only position moves, so a
            # rotated element can't visually stick off the canvas edge
            # even though that's invisible if you only look at its
            # unrotated box.
            rx, ry, rw, rh = _rotated_bbox(e["x"], e["y"], e["width"], e["height"], rotation)
            new_rx = min(max(rx, 0.0), max(0.0, tw - rw))
            new_ry = min(max(ry, 0.0), max(0.0, th - rh))
            e["x"] += new_rx - rx
            e["y"] += new_ry - ry
        else:
            w = min(e["width"], tw)
            h = min(e["height"], th)
            x = min(max(e["x"], 0.0), max(0.0, tw - w))
            y = min(max(e["y"], 0.0), max(0.0, th - h))
            e["x"], e["y"], e["width"], e["height"] = x, y, w, h
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Layout archetypes
# ---------------------------------------------------------------------------


def _badge_width(members: list[Element], height: float, available: float) -> float:
    """Keep a lone badge close to its source proportions when space permits.

    A badge does not become a full-width bar just because it owns a row. Its
    label may still ask for more width than the source aspect would provide.
    """
    source = _group_bbox(members)
    aspect_width = height * source[2] / max(source[3], 1.0)
    label = next((member for member in members if member.get("type") == "text"), None)
    label_width = 0.0
    if label is not None:
        label_width = _widest_line(
            str(label.get("content") or ""), str(label.get("font_family") or ""),
            label.get("font_weight"), label.get("font_style"),
            float(label.get("font_size") or 10.0), 10000.0,
        ) / 0.84
    return min(available, max(aspect_width, label_width, 10.0))


def _layout_stack(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float], tw: float, th: float, margin: float,
) -> list[Element]:
    """Vertical stack for portrait / near-square targets."""
    out: list[Element] = []
    avail_w = max(10.0, tw - 2 * margin)
    avail_h = max(10.0, th - 2 * margin)

    items: list[tuple[str, Any, float, float, float]] = []  # kind, payload, weight, min_px, natural_px
    if logo:
        imp = importances["logo"]
        items.append(("logo", logo, imp, 36, _natural_size("logo", imp, avail_w, avail_h)))
    if hero:
        imp = importances["hero"]
        items.append(("hero", hero, imp, 60, _natural_size("hero", imp, avail_w, avail_h)))
    if headline:
        imp = _space_weight(headline, importances["headline"])
        min_h = max(16.0, _text_min_height(headline, avail_w, _min_size_for(imp, element=headline)))
        items.append(("headline", headline, imp, min_h, _natural_size("headline", imp, avail_w, avail_h)))
    for i, bt in enumerate(body_texts):
        imp = _space_weight(bt, importances[f"body_{i}"])
        min_h = max(14.0, _text_min_height(bt, avail_w, _min_size_for(imp, element=bt)))
        items.append((f"body_{i}", bt, imp, min_h, _natural_size("body", imp, avail_w, avail_h)))
    if badges:
        n_b = len(badges)
        if n_b > 1 and avail_w / n_b < 70:
            # Too narrow for badges side by side (a 45px-wide pill can't
            # hold "AWARD WINNING 2026" without its label text spilling
            # out vertically and overlapping whatever's next) - the same
            # responsive-design move as a nav row becoming a nav column
            # on a narrow screen: each badge becomes its own full-width
            # vertical item instead of fighting the others for a sliver
            # of row width.
            for i, (gid, members) in enumerate(badges):
                bimp = importances[f"badge_{i}"]
                label = next((m for m in members if m["type"] == "text"), None)
                min_h = 22.0
                if label is not None:
                    ratio = _group_label_height_ratio(members)
                    min_h = max(min_h, _text_min_height(label, avail_w, _min_size_for(bimp, element=label)) / ratio)
                items.append((f"badge_{i}", (gid, members), bimp, min_h, _natural_size("badges", bimp, avail_w, avail_h)))
        else:
            badge_weight = sum(importances[f"badge_{i}"] for i in range(n_b)) / n_b
            items.append(("badges", badges, badge_weight, 30, _natural_size("badges", badge_weight, avail_w, avail_h)))
    if primary_cta:
        imp = importances["cta"]
        _, cta_members = primary_cta
        cta_label = next((m for m in cta_members if m["type"] == "text"), None)
        cta_w_est = min(avail_w, max(avail_w * 0.55, 180))
        min_h = 36.0
        if cta_label is not None:
            ratio = _group_label_height_ratio(cta_members)
            min_h = max(min_h, _text_min_height(cta_label, cta_w_est, _min_size_for(imp, element=cta_label)) / ratio)
        items.append(("cta", primary_cta, imp, min_h, _natural_size("cta", imp, avail_w, avail_h)))
    if legal:
        imp = _space_weight(legal, importances["legal"])
        min_h = max(10.0, _text_min_height(legal, avail_w, _min_size_for(imp, element=legal)))
        items.append(("legal", legal, imp, min_h, _natural_size("legal", imp, avail_w, avail_h)))
    if not items:
        return out

    sizes, gap, lead = _flow_sizes(
        [it[4] for it in items], [it[2] for it in items], [it[3] for it in items], avail_h,
    )
    y = margin + lead
    for (kind, payload, weight, needed, _), h in zip(items, sizes):
        box = (margin, y, avail_w, h)
        if kind in ("logo", "hero"):
            out.append(_place_image(payload, box))
        elif kind == "cta":
            _, members = payload
            cta_label = next((m for m in members if m["type"] == "text"), None)
            cta_w = min(avail_w, max(avail_w * 0.55, 180))
            cta_box = (margin + (avail_w - cta_w) / 2, y, cta_w, h)
            placed = _apply_group_transform(members, _group_bbox(members), cta_box, min_size=_adjusted_min_size(weight, needed, h, element=cta_label))
            out.extend(_deemphasize_if_underfunded(placed, needed, h))
        elif kind == "badges":
            n_b = len(payload)
            gap_b = max(4.0, avail_w * 0.02) if n_b > 1 else 0.0
            budget = max(10.0, avail_w - gap_b * (n_b - 1))
            # A short badge row used to cap each pill at 4.5 times its
            # height. On a 13 px row that left a long label in a 60 px box
            # and the font fell to about 5 px, with empty space beside the
            # row. The baseline path keeps that cap. A judged row uses the
            # full width and gives the longer label more of it.
            if layout_bridge.active_importance_by_id:
                if n_b == 1:
                    widths = [_badge_width(payload[0][1], h, budget)]
                else:
                    lengths = []
                    for _, members in payload:
                        label = next((m for m in members if m.get("type") == "text"), None)
                        lengths.append(max(len(str((label or {}).get("content") or "")), 1))
                    total = float(sum(lengths))
                    widths = [budget * length / total for length in lengths]
            else:
                bw = max(10.0, min(budget / n_b, h * 4.5))
                widths = [bw] * n_b
            bx = margin
            for bi, (_, members) in enumerate(payload):
                bimp = importances[f"badge_{bi}"]
                badge_label = next((m for m in members if m["type"] == "text"), None)
                placed = _apply_group_transform(
                    members, _group_bbox(members), (bx, y, widths[bi], h),
                    min_size=_adjusted_min_size(bimp, needed, h, element=badge_label),
                )
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
                bx += widths[bi] + gap_b
        elif kind.startswith("badge_"):
            # Unlike the side-by-side badge row above, a stacked badge
            # has the whole row to itself - nothing else is competing for
            # its width, so capping it to ~70% (a leftover from the
            # row-sharing case) only forces its label to wrap more than
            # it needs to for no reason.
            _, members = payload
            badge_label = next((m for m in members if m["type"] == "text"), None)
            badge_w = _badge_width(members, h, avail_w) if layout_bridge.active_importance_by_id else avail_w
            box_b = (margin, y, badge_w, h)
            placed = _apply_group_transform(members, _group_bbox(members), box_b, min_size=_adjusted_min_size(weight, needed, h, element=badge_label))
            out.extend(_deemphasize_if_underfunded(placed, needed, h))
        else:
            placed = [_place_text(payload, box, _adjusted_min_size(weight, needed, h, element=payload))]
            out.extend(_deemphasize_if_underfunded(placed, needed, h))
        y += h + gap
    return out


def _layout_wide(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float], tw: float, th: float, margin: float,
) -> list[Element]:
    """Two-column layout for wide banners/posters: hero image on the left,
    a top strip for logo/badges, and headline/body/CTA stacked on the
    right."""
    out: list[Element] = []
    has_top = bool(logo or badges)
    # 20% of the height on a tall card (an 80 px band on a 400 px card, the
    # same band the old 40 px floor produced there). On a short card the
    # 40 px floor made the chips taller than the headline, so the band is
    # 16% and nothing under that is forced up to 40.
    if not has_top:
        strip_h = 0.0
    elif th <= 220:
        strip_h = th * 0.16
    else:
        strip_h = max(th * 0.20, 40.0)
    x = margin

    if has_top:
        if logo:
            logo_w = min(tw * 0.26, strip_h * 2.8)
            out.append(_place_image(logo, (x, margin, logo_w, strip_h)))
            x += logo_w + margin
        if badges:
            remaining_w = max(10.0, tw - margin - x)
            n_b = len(badges)
            gap_b = margin if n_b > 1 else 0.0
            bw = max(10.0, min((remaining_w - gap_b * (n_b - 1)) / n_b, strip_h * 4.5))
            for bi, (_, members) in enumerate(badges):
                bimp = importances[f"badge_{bi}"]
                box = (x, margin, bw, strip_h)
                badge_label = next((m for m in members if m["type"] == "text"), None)
                out.extend(_apply_group_transform(members, _group_bbox(members), box, min_size=_min_size_for(bimp, element=badge_label)))
                x += bw + gap_b

    body_top = margin + strip_h + (margin if has_top else 0.0)
    body_h = max(10.0, th - body_top - margin)

    if hero:
        img_w = tw * 0.36
        out.append(_place_image(hero, (margin, body_top, img_w - margin, body_h)))
        text_x = margin + img_w
    else:
        text_x = margin
    text_w = max(10.0, tw - text_x - margin)

    sub_items: list[tuple[str, Any, float, float, float]] = []
    if headline:
        imp = importances["headline"]
        min_h = max(14.0, _text_min_height(headline, text_w, _min_size_for(imp, element=headline)))
        sub_items.append(("headline", headline, imp, min_h, _natural_size("headline", imp, text_w, body_h)))
    for i, bt in enumerate(body_texts):
        imp = importances[f"body_{i}"]
        min_h = max(12.0, _text_min_height(bt, text_w, _min_size_for(imp, element=bt)))
        sub_items.append((f"body_{i}", bt, imp, min_h, _natural_size("body", imp, text_w, body_h)))
    if primary_cta:
        imp = importances["cta"]
        _, cta_members = primary_cta
        cta_label = next((m for m in cta_members if m["type"] == "text"), None)
        cta_w_est = min(text_w, max(text_w * 0.7, 150))
        min_h = 30.0
        if cta_label is not None:
            ratio = _group_label_height_ratio(cta_members)
            min_h = max(min_h, _text_min_height(cta_label, cta_w_est, _min_size_for(imp, element=cta_label)) / ratio)
        sub_items.append(("cta", primary_cta, imp, min_h, _natural_size("cta", imp, text_w, body_h)))
    if legal:
        imp = importances["legal"]
        min_h = max(10.0, _text_min_height(legal, text_w, _min_size_for(imp, element=legal)))
        sub_items.append(("legal", legal, imp, min_h, _natural_size("legal", imp, text_w, body_h)))

    if sub_items:
        sizes, gap, lead = _flow_sizes(
            [it[4] for it in sub_items], [it[2] for it in sub_items], [it[3] for it in sub_items], body_h,
        )
        y = body_top + lead
        for (kind, payload, weight, needed, _), h in zip(sub_items, sizes):
            if kind == "cta":
                _, members = payload
                cta_label = next((m for m in members if m["type"] == "text"), None)
                cta_w = min(text_w, max(text_w * 0.7, 150))
                cta_box = (text_x, y, cta_w, h)
                placed = _apply_group_transform(members, _group_bbox(members), cta_box, min_size=_adjusted_min_size(weight, needed, h, element=cta_label))
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
            else:
                placed = [_place_text(payload, (text_x, y, text_w, h), _adjusted_min_size(weight, needed, h, element=payload))]
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
            y += h + gap
    return out


def _secondary_label(item: Any, is_group: bool) -> Optional[Element]:
    if is_group:
        return next((member for member in item[1] if member.get("type") == "text"), None)
    if item.get("type") == "text":
        return item
    return None


def _judgment_strip_widths(secondary: list, row_w: float) -> tuple[list[float], float]:
    """Give long copy the width an 8 px wrap needs; keep photos and ribbons narrow.

    Used only when text judgments are active. The baseline strip still uses
    `_distribute`.
    """
    count = len(secondary)
    gap = max(4.0, row_w * 0.012) if count > 1 else 0.0
    if gap * (count - 1) > row_w * 0.2:
        gap = (row_w * 0.2) / max(1, count - 1)
    budget = max(1.0, row_w - gap * (count - 1))
    plan: list[tuple[str, float]] = []
    for item, _imp, is_group in secondary:
        rotated = is_group and any(abs(member.get("rotation", 0)) > 0.5 for member in item[1])
        label = _secondary_label(item, is_group)
        content = str((label or {}).get("content") or "")
        if rotated:
            # Wide enough for a short ribbon label at a small size, narrow
            # enough that the tilt stays off the main row.
            plan.append(("fixed", 36.0))
        elif not is_group and item.get("type") == "image":
            plan.append(("fixed", min(48.0, budget * 0.08)))
        elif len(content) <= 24:
            # Wide enough for the words on one line at a modest size, and
            # never more than a sixth of the row. A 50 px slot wraps a
            # short badge onto two lines and the type collapses.
            one_line = max(36.0, min(budget * 0.16, max(len(content), 1) * 7.2))
            plan.append(("short", one_line))
        else:
            plan.append(("long", float(max(len(content), 1))))
    # One long line can be the supporting message. Further long lines stay
    # narrower so a legal paragraph does not take the same space as the
    # subhead on a short strip.
    seen_long = False
    for index, (kind, value) in enumerate(plan):
        if kind == "long" and seen_long:
            plan[index] = ("medium", min(value, budget * 0.18))
        elif kind == "long":
            seen_long = True
    reserved = sum(value for kind, value in plan if kind not in {"long"})
    long_total = sum(value for kind, value in plan if kind == "long")
    if reserved >= budget or long_total <= 0:
        scale = budget / reserved if reserved else 1.0
        return [value * scale if kind != "long" else budget / max(1, count) for kind, value in plan], gap
    remain = budget - reserved
    cap = budget * 0.36
    sizes = [0.0] * count
    long_widths = []
    for kind, value in plan:
        if kind == "long":
            long_widths.append(remain * value / long_total)
    leftover = 0.0
    long_index = 0
    for index, (kind, value) in enumerate(plan):
        if kind == "long":
            width = long_widths[long_index]
            long_index += 1
            if width > cap:
                leftover += width - cap
                width = cap
            sizes[index] = width
        else:
            sizes[index] = value
    # Spare width goes to the supporting long line, not to every badge.
    # A wider ribbon on a 90 px strip crosses the main row.
    long_indexes = [index for index, (kind, _value) in enumerate(plan) if kind == "long"]
    if leftover > 0 and long_indexes:
        sizes[long_indexes[0]] += leftover
    # On a crowded row, the supporting sentence keeps a real share. Short
    # badges can be narrower; a long paragraph cannot.
    if len(long_indexes) == 1 and count >= 4:
        floor_w = budget * 0.38
        index = long_indexes[0]
        if sizes[index] < floor_w:
            deficit = floor_w - sizes[index]
            # A rotated ribbon keeps its fixed width. Scaling it down makes
            # the tilted label unreadably small.
            scalable = [
                other for other in range(count)
                if other != index and plan[other][0] != "fixed"
            ]
            pool = sum(sizes[other] for other in scalable)
            if pool > deficit:
                scale = (pool - deficit) / pool
                for other in scalable:
                    sizes[other] *= scale
                sizes[index] = floor_w
    return sizes, gap


def _strip_slot_minimum(item: Any, is_group: bool, row_w: float) -> float:
    """Width floor for one strip secondary slot.

    With no judgments this is the baseline rule: a protected line reserves
    28% of the row, everything else can be narrow. With judgments, a long
    essential sentence gets that same 28% and a short badge gets a modest
    slot. A rotated ribbon stays narrow so its tilt does not cross the
    main row. The no-judgment path is unchanged.
    """
    member_ids = [m["id"] for m in item[1]] if is_group else [item.get("id")]
    if not layout_bridge.active_importance_by_id:
        if any(mid in layout_bridge.active_protected_ids for mid in member_ids if mid):
            return max(6.0, row_w * 0.28)
        return 6.0
    rotated = is_group and any(abs(m.get("rotation", 0)) > 0.5 for m in item[1])
    if rotated:
        return 6.0
    label = item if (not is_group and item.get("type") == "text") else None
    if is_group:
        label = next((m for m in item[1] if m.get("type") == "text"), None)
    content = str((label or {}).get("content") or "")
    if len(content) > 24:
        # Longer copy needs more of the row before an 8 px wrap fits in
        # the short secondary band. Capped so one sentence cannot take
        # the whole strip.
        share = min(0.42, 0.16 + len(content) / 700.0)
        return max(6.0, row_w * share)
    return min(row_w * 0.10, max(40.0, len(content) * 5.0))


def _strip_companions(body_texts: list[Element], importances: dict[str, float]) -> list[tuple[Element, float]]:
    """Up to two body lines the model ranked high, for the strip's main row.

    Baseline runs have an empty importance map, so this returns nothing and
    the strip stays the engine's original headline-plus-secondary split.
    """
    if not layout_bridge.active_importance_by_id:
        return []
    ranked = []
    for index, element in enumerate(body_texts):
        weight = float(importances.get(f"body_{index}", 0.0))
        if weight >= 0.75 and element.get("id") in layout_bridge.active_importance_by_id:
            ranked.append((weight, index, element))
    ranked.sort(reverse=True)
    return [(element, weight) for weight, _index, element in ranked[:2]]


def _font_of(elements: list[Element], eid: Optional[str]) -> Optional[float]:
    if eid is None:
        return None
    for element in elements:
        if element.get("id") == eid and element.get("type") == "text":
            return float(element.get("font_size") or 0)
    return None


def _parked_logo_clears_the_headline(
    current: list[Element], parked: list[Element], headline_id: str,
    legal_id: Optional[str], cta_id: Optional[str], logo_id: str, hero_id: Optional[str],
    badge_ids: set[str], tw: float, th: float,
) -> bool:
    """True when parking the logo makes the headline larger and leaves the
    action, the legal line, the classified pills, and the product alone.

    `badge_ids` are the text members of the pill groups that are not the
    main action. The element id is not read for the word badge. The size
    checks are fractions of this canvas or of the logo's own box, plus the
    engine's existing 8 px reading floor for a legal line that was already
    that large. That floor is a glyph size, not a fraction of one sample.
    """
    before = _font_of(current, headline_id)
    after = _font_of(parked, headline_id)
    # 8% of the headline, and at least 1.5 px, so a one-pixel fit change
    # does not pass on a large line or a tiny line.
    if before is None or after is None or after < before + max(1.5, before * 0.08):
        return False
    current_headline = next(element for element in current if element.get("id") == headline_id)
    parked_headline = next(element for element in parked if element.get("id") == headline_id)
    width_gain = float(parked_headline["width"]) - float(current_headline["width"])
    current_logo = next((element for element in current if element.get("id") == logo_id), None)
    # The headline has to receive most of the slot the logo left. Half of
    # that logo, and at least 2.5% of the canvas, rejects a pixel wiggle
    # on a wide banner and a real gain on a narrow one.
    logo_slot = float(current_logo["width"]) * 0.45 if current_logo is not None else 0.0
    if width_gain < max(logo_slot, tw * 0.025):
        return False
    for eid in badge_ids:
        was = _font_of(current, eid)
        now = _font_of(parked, eid)
        if was is None or now is None or now < was - max(0.5, was * 0.05):
            return False
    for eid in (cta_id, legal_id):
        was = _font_of(current, eid)
        now = _font_of(parked, eid)
        if was is None or now is None:
            continue
        if now < was - max(0.4, was * 0.05):
            return False
        # The rest of the engine already treats 8 px as the reading floor.
        if eid == legal_id and was >= 8.0 and now < 7.5:
            return False
    logo = next((element for element in parked if element.get("id") == logo_id), None)
    # 16% of the strip height. A fixed 16 px logo is readable on a 90 px
    # banner and a speck on a taller one, or impossible on a shorter one.
    if logo is None or float(logo.get("height") or 0) < th * 0.16:
        return False
    # Parking that keeps the headline but cuts the logo to a fraction of
    # the logo that already fit the row is a bad trade. 65% keeps a park
    # that only steps the logo down (a 27 px logo to 20 px) and rejects one
    # that drops it by half.
    if current_logo is not None and float(logo["height"]) < float(current_logo["height"]) * 0.65:
        return False
    if hero_id is not None:
        was = next((element for element in current if element.get("id") == hero_id), None)
        now = next((element for element in parked if element.get("id") == hero_id), None)
        if was is None or now is None:
            return False
        if abs(float(was["width"]) - float(now["width"])) > max(1.0, float(was["width"]) * 0.01):
            return False
    current_doc = {"elements": current}
    parked_doc = {"elements": parked}
    if _text_overlap_pairs(parked_doc) - _text_overlap_pairs(current_doc):
        return False
    return True


def _sparse_reading_fits(
    hero: Optional[Element], badges: list[PillGroup], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    tw: float, th: float,
) -> bool:
    """A wide canvas with a handful of lines and no product or pills.

    The row layout then stretches the top line across the empty width and
    leaves the rest unreadably small. This candidate is a left column sized
    to the measured words. A hero, a pill row, or a button is a different
    design and stays on the row or the wide card.
    """
    if hero is not None or badges or primary_cta is not None:
        return False
    count = (1 if headline is not None else 0) + len(body_texts) + (1 if legal is not None else 0)
    if count < 2 or count > 5:
        return False
    return th >= 150.0 and tw / max(th, 1.0) >= 2.5


def _layout_sparse(
    logo: Optional[Element], headline: Optional[Element], body_texts: list[Element],
    legal: Optional[Element], importances: dict[str, float], tw: float, th: float,
) -> list[Element]:
    """Left column for a sparse wide offer. The right side stays empty.

    Line size follows importance. The column is only as wide as the words
    at that size, capped so a long name cannot become a bar. Heights are
    the line itself, not the leftover canvas, and the stack stops around
    80% of the height so the lower area stays open for a rotated accent.
    """
    pad = _edge_pad(tw, th)
    items: list[tuple[Element, float]] = []
    if headline is not None:
        items.append((headline, float(importances.get("headline", 0.9))))
    for index, element in enumerate(body_texts):
        items.append((element, float(importances.get(f"body_{index}", 0.4))))
    if legal is not None:
        items.append((legal, float(importances.get("legal", 0.3))))
    if not items:
        return []
    top = max(weight for _, weight in items)
    lead_size = min(56.0, max(18.0, th * 0.26))
    column_cap = min(tw * 0.42, max(160.0, lead_size * 8.0))

    def sizes_for(lead: float) -> list[tuple[Element, float, float]]:
        planned: list[tuple[Element, float, float]] = []
        for element, weight in items:
            share = weight / max(top, 0.01)
            size = lead * max(0.40, min(1.0, share))
            if weight >= top * 0.75:
                size = max(size, lead * 0.62)
            planned.append((element, weight, size))
        return planned

    planned = sizes_for(lead_size)
    for _ in range(8):
        used = pad
        if logo is not None:
            used += th * 0.18 + pad
        for _element, _weight, size in planned:
            used += max(size * 1.4, 16.0) + pad
        if used <= th * 0.82:
            break
        lead_size *= 0.88
        planned = sizes_for(lead_size)

    out: list[Element] = []
    y = pad
    if logo is not None:
        aspect = float(logo["width"]) / max(float(logo["height"]), 1.0)
        logo_h = min(th * 0.18, column_cap / max(aspect, 0.1))
        logo_w = min(column_cap * 0.7, logo_h * aspect)
        out.append(_place_image(logo, (pad, y, logo_w, logo_h)))
        y += logo_h + pad
    for element, _weight, size in planned:
        ink = _widest_line(
            str(element.get("content") or ""), element.get("font_family", ""),
            element.get("font_weight"), element.get("font_style"), size, column_cap,
        )
        width = min(column_cap, max(size * 2.4, ink + pad * 2.0))
        height = max(size * 1.4, 16.0)
        if y + height > th - pad:
            height = max(16.0, th - pad - y)
        out.append(_place_text(element, (pad, y, width, height), min_size=max(8.0, size * 0.8)))
        y += height + pad
    return out


def _layout_strip(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], ribbon_groups: list[PillGroup], importances: dict[str, float],
    tw: float, th: float, margin: float, park_logo: bool = False,
) -> list[Element]:
    """Single-row layout for ultra-thin strips (e.g. a 728x90 leaderboard).

    Logo, headline, and the call to action stay on the tall row. The
    product image is not given that row: at this height it would be an
    unrecognizable sliver, and the width is worth more on the message.
    Every other element stays present. On a judged crowded strip, ungrouped
    must-read sentences get their own band and badges sit in a shorter
    band under it. Other strips keep one importance-sized secondary band.
    Opacity on that secondary band can drop, but never to zero, and a
    must-read line keeps its source opacity."""
    out: list[Element] = []
    pad = max(3.0, th * 0.08)
    row_w = max(10.0, tw - 2 * pad)
    # Rotated ribbons aren't part of the classified roles `importances`
    # covers, so they get a flat low-but-nonzero weight here - they're
    # decorative accents, not message-bearing content.
    # A price or subhead the model marks high stays on the main row. The
    # secondary band is too short to keep that line readable, and crushing
    # it is the failure the rubric cares about.
    companions = _strip_companions(body_texts, importances)
    companion_ids = {element["id"] for element, _weight in companions}
    secondary: list[tuple[Any, float, bool]] = (
        [(bg, importances[f"badge_{i}"], True) for i, bg in enumerate(badges)]
        # A rotated ribbon cannot safely claim a large share of a 90 px
        # strip's secondary band: its rotation extends into the main row.
        # Keep the baseline ribbon share here even when the text model calls
        # its label a headline or CTA. The model still guides ribbons in
        # layouts with enough height.
        + [(rb, 0.15, True) for rb in ribbon_groups]
        + ([(hero, importances["hero"], False)] if hero else [])
        + [(bt, importances[f"body_{i}"], False) for i, bt in enumerate(body_texts) if bt["id"] not in companion_ids]
        + ([(legal, importances["legal"], False)] if legal else [])
    )
    secondary.sort(key=lambda t: t[1], reverse=True)
    # A judged layout keeps the content image in its own column. The text
    # uses the width that remains, so the photo is not squeezed into the
    # short badge band. The no-judgment path does not take this branch.
    hero_span = 0.0
    row_before_hero = row_w
    if hero is not None and layout_bridge.active_importance_by_id and float(hero.get("height") or 0) > 0:
        aspect = float(hero["width"]) / float(hero["height"])
        column_h = max(8.0, th - 2 * pad)
        # The column uses the strip height, but not so much of a short
        # banner that the headline loses the width it needs.
        hero_span = min(column_h * aspect, row_w * 0.22, max(32.0, th * 0.52))
        if hero_span >= 16.0:
            secondary = [item for item in secondary if item[0] is not hero]
            row_w = max(10.0, row_w - hero_span - pad)
        else:
            hero_span = 0.0
    origin = pad + (hero_span + pad if hero_span else 0.0)
    has_secondary = bool(secondary)

    avail_h = max(6.0, th - 2 * pad)

    # Width shares don't depend on the primary/secondary height split (they
    # only depend on row_w), so they can be computed up front - which lets
    # the split itself be informed by what the secondary band actually
    # needs, instead of a fixed importance-based guess that has no idea
    # whether it's handing a 6-item row enough height or not.
    # Companions already occupy the main row (a price beside a product
    # name). Those strips keep the importance split so the main row stays
    # tall. Strips whose long copy is stuck in the secondary band use the
    # reading-width split instead.
    if layout_bridge.active_importance_by_id and not companions:
        slot_sizes, sgap = _judgment_strip_widths(secondary, row_w) if has_secondary else ([], 0.0)
    else:
        slot_minimums = [_strip_slot_minimum(item, is_group, row_w) for item, _imp, is_group in secondary]
        slot_sizes, sgap = (
            _distribute(list(zip((imp for _, imp, _ in secondary), slot_minimums)), row_w) if has_secondary else ([], 0.0)
        )
    reading: list[tuple[Any, float, bool]] = []
    decor: list[tuple[Any, float, bool]] = []
    read_widths: list[float] = []
    reading_h = 0.0
    decor_h = 0.0
    band_gap = 0.0
    inner_gap = 0.0
    secondary_true_need = 0.0
    for (item, imp, is_group), sw_ in zip(secondary, slot_sizes):
        label = item if (not is_group and item.get("type") == "text") else None
        ratio = 1.0
        if is_group:
            label = next((m for m in item[1] if m["type"] == "text"), None)
            if label is not None:
                ratio = _group_label_height_ratio(item[1])
        if label is not None:
            needed = _text_min_height(label, sw_, _min_size_for(imp, scale=0.5, floor=5.0, element=label)) / ratio
            secondary_true_need = max(secondary_true_need, needed)

    if has_secondary and avail_h >= 40:
        # A rotated element in the secondary band (a ribbon) has a true
        # visual extent taller than its stored height, which can eat a
        # couple of px into a too-thin gap and graze the primary row above
        # it - a slightly larger gap than a non-rotated layout would need
        # gives that overhang somewhere to go.
        gap = max(5.0, th * 0.04)
        # The secondary band gets what it actually needs first (capped so
        # it can't swallow the whole row either) - it's the cheaper ask in
        # absolute pixels, same reasoning as `_priority_allocate`'s water
        # filling. The headline/CTA primary row gets whatever's left, with
        # its own content-aware growth below still able to claim a little
        # more if there's genuine slack.
        secondary_h = max(8.0, min(secondary_true_need, avail_h * 0.62))
        # A crowded short strip keeps the secondary band to about a third
        # of the height. The headline and action stay the large row. Forcing
        # every badge, legal line, and ribbon up to the same size makes the
        # banner a wall of type. The baseline path does not take this cap.
        crowded = len(secondary) >= 4 and th <= 120
        judged_strip = bool(layout_bridge.active_importance_by_id) and not companions
        if judged_strip and crowded:
            for item in secondary:
                payload, _imp, is_group = item
                if (not is_group) and payload.get("type") == "text" and _is_must_read(payload):
                    reading.append(item)
                else:
                    decor.append(item)
        if reading:
            # Must-read sentences do not share one short row with every badge.
            # They get their own band, side by side, as tall as an 8 px wrap
            # needs, until the headline row would stop being the large row.
            lengths = [max(len(str(payload.get("content") or "")), 1) for payload, _imp, _is_group in reading]
            total_chars = float(sum(lengths))
            inner_gap = 4.0 if len(reading) > 1 else 0.0
            budget = max(10.0, row_w - inner_gap * (len(reading) - 1))
            read_widths = [budget * length / total_chars for length in lengths]
            need = 0.0
            for (payload, _imp, _is_group), width in zip(reading, read_widths):
                need = max(need, _text_min_height(payload, width, 8.0))
            reading_h = min(max(need, 14.0), avail_h * 0.36)
            decor_h = min(15.0, avail_h * 0.18) if decor else 0.0
            band_gap = 3.0 if decor else 0.0
            spare = avail_h - gap - band_gap
            if reading_h + decor_h > spare * 0.58:
                scale = (spare * 0.58) / max(1.0, reading_h + decor_h)
                reading_h *= scale
                decor_h *= scale
            primary_h = max(10.0, avail_h - gap - reading_h - band_gap - decor_h)
            if decor:
                slot_sizes, sgap = _judgment_strip_widths(decor, row_w)
            secondary_h = reading_h + band_gap + decor_h
        elif judged_strip and crowded:
            secondary_h = min(secondary_h, max(18.0, avail_h * 0.30))
            primary_h = max(10.0, avail_h - secondary_h - gap)
        elif judged_strip and any(is_group for _item, _imp, is_group in secondary):
            # A pill needs a taller band than a single line of body copy.
            # Take that height only when the main row still keeps half the strip.
            pill_h = min(24.0, avail_h * 0.32)
            if avail_h - pill_h - gap >= avail_h * 0.5:
                secondary_h = max(secondary_h, pill_h)
            primary_h = max(10.0, avail_h - secondary_h - gap)
        else:
            primary_h = max(10.0, avail_h - secondary_h - gap)
    else:
        primary_h = avail_h
        gap = 0.0
        secondary_h = 0.0

    y0 = pad
    x = origin
    # A compact logo can leave the headline row on a very short banner and
    # sit in the space under the reading copy. The caller keeps this only
    # when the headline actually gets larger and the action and legal line
    # do not.
    logo_parked = None
    if park_logo and logo is not None and has_secondary and float(logo.get("height") or 0) > 1:
        read_bottom = y0 + primary_h + gap + (reading_h if reading else 0.0)
        logo_top = read_bottom + 2.0
        room_h = (th - 2.0) - logo_top
        aspect = float(logo["width"]) / float(logo["height"])
        logo_h = room_h
        logo_w = logo_h * aspect
        cap_w = row_w * 0.20
        if logo_w > cap_w:
            logo_w = cap_w
            logo_h = logo_w / aspect if aspect else logo_h
        if logo_h >= 16.0 and logo_w >= 28.0 and logo_top + logo_h <= th - 1.0:
            logo_parked = (origin, logo_top, logo_w, logo_h)

    if hero_span and hero is not None:
        out.append(_place_image(hero, (pad, pad, hero_span, max(8.0, th - 2 * pad))))

    if logo and logo_parked is None:
        logo_w = min(primary_h * 2.6, row_w * 0.22)
        out.append(_place_image(logo, (x, y0, logo_w, primary_h)))
        x += logo_w + pad

    cta_w = 0.0
    if primary_cta:
        # primary_h * 5 (not 3) - the tighter multiplier was forcing the
        # button narrower than its own text needs even when row_w had
        # plenty of room to spare, which is what pushed "Shop the Summer
        # Line" to wrap to 2 lines in a box sized for 1.
        cta_w = min(row_before_hero * 0.24, max(primary_h * 5.0, 70.0))
        # The product column already took its width from this row. The
        # button keeps the share it had before that column, unless the
        # remaining row cannot hold it.
        if hero_span:
            cta_w = min(cta_w, max(70.0, row_w * 0.40))
        # A single unbreakable token does not fit a normal button width.
        # Give it the width one 12 px line needs, and leave the other
        # primary text enough for its own 12 px line. A phrase that
        # already fits the button is not widened.
        cta_label_early = next((m for m in primary_cta[1] if m.get("type") == "text"), None)
        if cta_label_early is not None and _unbreakable(str(cta_label_early.get("content") or "")):
            one_line = _widest_line(
                cta_label_early["content"], cta_label_early.get("font_family", ""),
                cta_label_early.get("font_weight"), cta_label_early.get("font_style"),
                12.0, 10000.0,
            )
            reserve = 0.0
            if headline is not None:
                reserve = _widest_line(
                    str(headline.get("content") or ""), headline.get("font_family", ""),
                    headline.get("font_weight"), headline.get("font_style"),
                    12.0, 10000.0,
                ) + pad
            room = max(cta_w, row_w - reserve - pad)
            if one_line + 8.0 > cta_w:
                cta_w = min(max(cta_w, one_line + 8.0), room)

    headline_right = origin + row_w - (cta_w + pad if primary_cta else 0.0)
    span = max(10.0, headline_right - x)
    row_slots: list[tuple[Element, float]] = []
    if headline:
        row_slots.append((headline, float(importances.get("headline", 0.9))))
    row_slots.extend(companions)
    if row_slots:
        gap_px = pad if len(row_slots) > 1 else 0.0
        usable = max(10.0, span - gap_px * (len(row_slots) - 1))
        if headline and len(row_slots) == 2:
            widths = [usable * 0.62, usable * 0.38]
        elif headline and len(row_slots) == 3:
            widths = [usable * 0.50, usable * 0.25, usable * 0.25]
        else:
            widths = [usable / len(row_slots)] * len(row_slots)
        cx = x
        for (element, weight), slot_w in zip(row_slots, widths):
            h_min = _min_size_for(weight, scale=0.65, floor=7.5, element=element)
            # primary_h already accounts for the secondary band's real need
            # (computed above); if the line still wants more than that, it
            # shrinks via _fit_font_size's own search rather than clawing
            # space back from a secondary band that was sized to protect.
            if _text_min_height(element, slot_w, h_min) > primary_h:
                h_min = _emergency_floor(element)
            out.append(_place_text(element, (cx, y0, slot_w, primary_h), min_size=h_min))
            cx += slot_w + gap_px

    if primary_cta:
        _, members = primary_cta
        cta_x = origin + row_w - cta_w
        cta_label = next((m for m in members if m["type"] == "text"), None)
        c_min = _min_size_for(importances["cta"], scale=0.6, floor=7.0, element=cta_label)
        cta_ratio = _group_label_height_ratio(members) if cta_label is not None else 1.0
        if cta_label is not None and _text_min_height(cta_label, cta_w, c_min) > primary_h * cta_ratio:
            c_min = _emergency_floor(cta_label)
        out.extend(_apply_group_transform(members, _group_bbox(members), (cta_x, y0, cta_w, primary_h), min_size=c_min))

    if has_secondary:
        # `slot_sizes`/`sgap` (the importance-weighted width split) were
        # already computed above, to size `secondary_h` itself - reused
        # here rather than recomputed.
        if reading:
            rx = origin
            ry = y0 + primary_h + gap
            for (payload, imp, _is_group), width in zip(reading, read_widths):
                min_size = _min_size_for(imp, scale=0.5, floor=5.0, element=payload)
                if _text_min_height(payload, width, min_size) > reading_h:
                    min_size = _emergency_floor(payload)
                out.append(_place_text(payload, (rx, ry, width, reading_h), min_size=min_size))
                rx += width + inner_gap
            place_items = decor
            if decor_h > 0:
                slot_y, slot_h, scale, floor = ry + reading_h + band_gap, decor_h, 0.5, 5.0
            else:
                place_items = []
                slot_y, slot_h, scale, floor = ry, 0.0, 0.5, 5.0
        elif secondary_h > 0:
            place_items = secondary
            slot_y, slot_h, scale, floor = y0 + primary_h + gap, secondary_h, 0.5, 5.0
        else:
            place_items = secondary
            slot_h = max(2.0, th * 0.025)
            slot_y, scale, floor = th - slot_h, 0.4, 4.0
        logo_reserve = 0.0
        if logo_parked is not None:
            out.append(_place_image(logo, logo_parked))
            logo_reserve = logo_parked[2] + pad
            if slot_sizes and row_w > logo_reserve + 10.0:
                width_scale = (row_w - logo_reserve) / row_w
                slot_sizes = [width * width_scale for width in slot_sizes]
        sx = origin + logo_reserve
        for (item, imp, is_group), sw_ in zip(place_items, slot_sizes):
            box = (sx, slot_y, sw_, slot_h)
            label_for_floor = item if (not is_group and item.get("type") == "text") else None
            if is_group:
                label_for_floor = next((m for m in item[1] if m["type"] == "text"), None)
            base_min = _min_size_for(imp, scale=scale, floor=floor, element=label_for_floor)
            min_size = base_min
            # The row/column split above sizes this slot by importance
            # share, not by what its actual text needs - check at the
            # width it actually got, and shrink further (same idea as
            # `_adjusted_min_size`) rather than let it overflow into
            # whatever's next.
            label_for_check = item if (not is_group and item["type"] == "text") else None
            label_ratio = 1.0
            if is_group:
                label_for_check = next((m for m in item[1] if m["type"] == "text"), None)
                if label_for_check is not None:
                    label_ratio = _group_label_height_ratio(item[1])
            if label_for_check is not None and _text_min_height(label_for_check, sw_, base_min) > slot_h * label_ratio:
                min_size = _emergency_floor(label_for_check)
            # Opacity used to be dimmed here. The entrypoint copies source
            # opacity back before it writes JSON or renders the PNG, so that
            # dimming never reached the published design. The source value
            # is left as the placer copied it.
            if is_group:
                _, members = item
                placed = _apply_group_transform(members, _group_bbox(members), box, min_size=min_size)
            elif item["type"] == "image":
                placed = [_place_image(item, box)]
            else:
                placed = [_place_text(item, box, min_size=min_size)]
            out.extend(placed)
            sx += sw_ + sgap
    return out
