"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.

---

Approach (see README.md for the full write-up): a deterministic,
model-free reflow engine, not a scale/stretch transform.

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
    and never into a new collision with other foreground content."""
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
        for dy in (-ry, -ry * 0.5, 0.0, ry * 0.5, ry):
            ny = y + dy
            if ny < 0 or ny + h > th:
                continue
            nx = 0.0
            while nx + w <= tw:
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
                nx += step
        if best_score < base_score - 0.2:
            # In a genuinely cramped layout, a modest improvement (moving
            # a few px off the peak of a dot, even if not fully clearing
            # it) is still worth taking - the earlier, stricter threshold
            # meant "not dramatically better" and "no better" were treated
            # the same, when even a small shift measurably helps.
            e["x"], e["y"] = best_x, best_y
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


def _wrap_and_measure(
    content: str, font_family: str, font_weight: Optional[str], font_style: Optional[str],
    font_size: float, max_width: float,
) -> int:
    """Number of lines `content` wraps to at `font_size` within `max_width`."""
    manager = _get_font_manager()
    if manager is not None:
        try:
            from renderer.converter.wrap import char_widths, wrap_text

            weight = _normalize_weight(font_weight)
            style = _font_style_for(font_weight, font_style)
            widths = char_widths(manager, font_family, weight, style, font_size, content)
            wrapped = wrap_text(content, widths, max_width)
            return wrapped.count("\n") + 1
        except Exception:
            pass

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
    for _ in range(40):
        lines = _wrap_and_measure(content, font_family, font_weight, font_style, size, box_w)
        block_h = _text_block_height(lines, size, line_height, font_family, font_weight, font_style)
        if block_h <= box_h or size <= min_size:
            break
        size = max(min_size, size * 0.92)
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


def _place_text(t: Element, box: Box, min_size: float) -> Element:
    bx, by, bw, bh = box
    out = _copy_element(t)
    out["x"], out["y"], out["width"], out["height"] = bx, by, bw, bh
    out["font_size"] = _fit_font_size(
        out["content"], out["font_family"], out.get("font_weight"), out.get("font_style"),
        bw, bh, out.get("line_height", 1.2), out["font_size"], min_size=min_size,
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
                e["font_size"] * min(sx, sy), min_size=min_size,
            )
        out.append(e)
    return out


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


def _min_size_for(importance: float, scale: float = 1.0, floor: float = 6.0) -> float:
    """Legibility floor for a font_size shrink-to-fit pass: higher-importance
    text (a headline) is allowed to stay bigger under pressure than
    lower-importance text (fine print). `scale`/`floor` let tighter
    archetypes (the ultra-thin strip) compress everything further while
    preserving the same relative ordering."""
    importance = max(0.0, min(1.0, importance))
    return max(floor, round((6.0 + 11.0 * importance) * scale, 1))


def _adjusted_min_size(importance: float, needed_h: float, granted_h: float, scale: float = 1.0) -> float:
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
    base = _min_size_for(importance, scale=scale)
    if needed_h <= 0 or granted_h >= needed_h:
        return base
    return 5.0


def _deemphasize_if_underfunded(elements: list[Element], needed_h: float, granted_h: float) -> list[Element]:
    """When an item was severely underfunded (got well under its true
    minimum - a genuine "everything doesn't fit" case) and had to shrink
    past its normal floor to avoid overflowing, a bounded opacity
    reduction (never below ~0.55, never fully hidden) signals that this
    is deliberately de-prioritized content rather than a layout mistake -
    the same visual-hierarchy move already used for the strip archetype's
    overflow band, applied consistently wherever content ends up this
    compressed."""
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
    return (
        e["type"] == "image" and e.get("fit") == "cover"
        and e["x"] <= cw * 0.05 and e["y"] <= ch * 0.05
        and e["x"] + e["width"] >= cw * 0.95 and e["y"] + e["height"] >= ch * 0.95
    )


def _looks_like_logo(e: Element, canvas_h: float) -> bool:
    aspect = e["width"] / e["height"] if e["height"] > 0 else 1.0
    return aspect >= 1.6 and e["y"] < canvas_h * 0.32


def _is_out_of_bounds(e: Element, w: float, h: float) -> bool:
    return e["x"] < -0.5 or e["y"] < -0.5 or e["x"] + e["width"] > w + 0.5 or e["y"] + e["height"] > h + 0.5


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
    if th < 160:
        archetypes = ["strip"]
        if th >= 120:
            archetypes.append("wide" if ratio >= 1.8 else "stack")
        return archetypes
    if ratio >= 1.8:
        archetypes = ["wide"]
        if ratio <= 2.3:
            archetypes.append("stack")
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
            lines = _wrap_and_measure(
                e.get("content", ""), e.get("font_family", ""), e.get("font_weight"),
                e.get("font_style"), font_size, max(1.0, e["width"]),
            )
            rendered_h = lines * font_size * e.get("line_height", 1.2)
            overflow = rendered_h - e["height"]
            if overflow > 0:
                cost += (overflow / max(1.0, e["height"])) * weight * 25.0

    return cost


def _build_candidate(
    archetype: str,
    backgrounds: list[Element], standalone_shapes: list[Element], extra_images: list[Element],
    generic_groups: list[PillGroup], ribbon_groups: list[PillGroup],
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], importances: dict[str, float],
    scale_x: float, scale_y: float, sw: float, sh: float, tw: float, th: float, margin: float,
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
        # A corner ribbon is capped to a small, legible size (below) and
        # collision resolution actively tries to keep it clear of real
        # content first - bleed only becomes visible in the genuinely
        # cramped layouts resolution can't fully clear otherwise, not as
        # a default treatment applied regardless of whether it's needed.
        for _, members in ribbon_groups:
            allow_bleed_ids.update(m["id"] for m in members)
    for _, members in passthrough_groups:
        bbox = _group_bbox(members)
        new_box = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
        is_ribbon = any(abs(m.get("rotation", 0)) > 0.5 for m in members)
        if is_ribbon:
            # Capped like a corner accent (see extra_images above) - a
            # smaller ribbon is also an easier fit for collision
            # resolution to find it clean space for, rather than needing
            # a large push (or an edge bleed) to clear busier content
            # like a logo/badge row.
            label_ratio = _group_label_height_ratio(members)
            max_dim = min(tw, th) * 0.12
            min_legible_h = 16.0 / label_ratio  # the GROUP height needed for the LABEL (a fraction of it) to reach min_legible_h
            larger = max(new_box[2], new_box[3])
            shrink = min(1.0, max_dim / larger) if larger > 0 else 1.0
            if new_box[3] * shrink < min_legible_h:
                # A ribbon can already be very short after extreme
                # vertical compression (e.g. a tall source reflowed into
                # a short, wide target) before this cap even applies -
                # shrinking it further to honor the width cap would crush
                # an already-thin label past legibility. Prioritize
                # height over strictly honoring the cap: use whichever of
                # "shrink to the width cap" or "shrink only as far as
                # keeps the height legible" shrinks less.
                shrink = max(shrink, min(1.0, min_legible_h / new_box[3])) if new_box[3] > 0 else shrink
            if shrink < 1.0:
                nw, nh = new_box[2] * shrink, new_box[3] * shrink
                new_box = (new_box[0] + (new_box[2] - nw) / 2, new_box[1] + (new_box[3] - nh) / 2, nw, nh)
        label = next((m for m in members if m["type"] == "text"), None)
        min_size = 10.0
        if label is not None:
            ratio = _group_label_height_ratio(members)
            if _text_min_height(label, new_box[2], min_size) > new_box[3] * ratio:
                min_size = 5.0  # let _fit_font_size's own search find the best fit down to the floor
        out.extend(_apply_group_transform(members, bbox, new_box, min_size=min_size))

    if archetype == "strip":
        out += _layout_strip(logo, hero, headline, body_texts, legal, primary_cta, badges, ribbon_groups, importances, tw, th, margin)
    elif archetype == "wide":
        out += _layout_wide(logo, hero, headline, body_texts, legal, primary_cta, badges, importances, tw, th, margin)
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
                ax0, ay0, aw, ah = _visual_bbox([by_id[k] for k in ids_a])
                bx0, by0, bw, bh = _visual_bbox([by_id[k] for k in ids_b])
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


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
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

    importances = _compute_importances(logo, hero, headline, body_texts, legal, primary_cta, badges, sw, sh)
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
    for name in _candidate_archetypes(th, ratio):
        cand_elements, cand_bleed_ids = _build_candidate(
            name, backgrounds, standalone_shapes, extra_images, generic_groups, ribbon_groups,
            logo, hero, headline, body_texts, legal, primary_cta, badges, importances,
            scale_x, scale_y, sw, sh, tw, th, margin,
        )
        score = _score_candidate(cand_elements, fg_weights, tw, th)
        if score < best_score:
            best_score, best_elements, best_bleed_ids, best_name = score, cand_elements, cand_bleed_ids, name

    # Ribbons (outside the strip archetype, where they're already placed
    # in-flow) and uncategorized "extra" images are positioned by simple
    # proportional passthrough, completely independent of where the
    # chosen archetype reflowed everything else - at an aspect ratio far
    # from the source's, that can land them right on top of reflowed
    # content (e.g. a logo that moved, or legal text that's now
    # elsewhere). Including them here as low-importance units lets them
    # get pushed aside on contact instead of silently overlapping it.
    units = _flow_units(logo, hero, headline, body_texts, legal, primary_cta, badges, importances)
    if best_name != "strip":
        units += [([m["id"] for m in members], 0.15) for _, members in ribbon_groups]
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

    out_by_id = {e["id"]: e for e in out_elements}
    ordered = [out_by_id[e["id"]] for e in source["elements"]]
    return {"canvas": {"width": target_canvas["width"], "height": target_canvas["height"]}, "elements": ordered}


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
        imp = importances["headline"]
        min_h = max(16.0, _text_min_height(headline, avail_w, _min_size_for(imp)))
        items.append(("headline", headline, imp, min_h, _natural_size("headline", imp, avail_w, avail_h)))
    for i, bt in enumerate(body_texts):
        imp = importances[f"body_{i}"]
        min_h = max(14.0, _text_min_height(bt, avail_w, _min_size_for(imp)))
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
                    min_h = max(min_h, _text_min_height(label, avail_w, _min_size_for(bimp)) / ratio)
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
            min_h = max(min_h, _text_min_height(cta_label, cta_w_est, _min_size_for(imp)) / ratio)
        items.append(("cta", primary_cta, imp, min_h, _natural_size("cta", imp, avail_w, avail_h)))
    if legal:
        imp = importances["legal"]
        min_h = max(10.0, _text_min_height(legal, avail_w, _min_size_for(imp)))
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
            cta_w = min(avail_w, max(avail_w * 0.55, 180))
            cta_box = (margin + (avail_w - cta_w) / 2, y, cta_w, h)
            placed = _apply_group_transform(members, _group_bbox(members), cta_box, min_size=_adjusted_min_size(weight, needed, h))
            out.extend(_deemphasize_if_underfunded(placed, needed, h))
        elif kind == "badges":
            n_b = len(payload)
            gap_b = max(4.0, avail_w * 0.02) if n_b > 1 else 0.0
            bw = max(10.0, min((avail_w - gap_b * (n_b - 1)) / n_b, h * 4.5))
            bx = margin
            for bi, (_, members) in enumerate(payload):
                bimp = importances[f"badge_{bi}"]
                placed = _apply_group_transform(members, _group_bbox(members), (bx, y, bw, h), min_size=_adjusted_min_size(bimp, needed, h))
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
                bx += bw + gap_b
        elif kind.startswith("badge_"):
            # Unlike the side-by-side badge row above, a stacked badge
            # has the whole row to itself - nothing else is competing for
            # its width, so capping it to ~70% (a leftover from the
            # row-sharing case) only forces its label to wrap more than
            # it needs to for no reason.
            _, members = payload
            box_b = (margin, y, avail_w, h)
            placed = _apply_group_transform(members, _group_bbox(members), box_b, min_size=_adjusted_min_size(weight, needed, h))
            out.extend(_deemphasize_if_underfunded(placed, needed, h))
        else:
            placed = [_place_text(payload, box, _adjusted_min_size(weight, needed, h))]
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
    strip_h = max(th * 0.20, 40.0) if has_top else 0.0
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
                out.extend(_apply_group_transform(members, _group_bbox(members), box, min_size=_min_size_for(bimp)))
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
        min_h = max(14.0, _text_min_height(headline, text_w, _min_size_for(imp)))
        sub_items.append(("headline", headline, imp, min_h, _natural_size("headline", imp, text_w, body_h)))
    for i, bt in enumerate(body_texts):
        imp = importances[f"body_{i}"]
        min_h = max(12.0, _text_min_height(bt, text_w, _min_size_for(imp)))
        sub_items.append((f"body_{i}", bt, imp, min_h, _natural_size("body", imp, text_w, body_h)))
    if primary_cta:
        imp = importances["cta"]
        _, cta_members = primary_cta
        cta_label = next((m for m in cta_members if m["type"] == "text"), None)
        cta_w_est = min(text_w, max(text_w * 0.7, 150))
        min_h = 30.0
        if cta_label is not None:
            ratio = _group_label_height_ratio(cta_members)
            min_h = max(min_h, _text_min_height(cta_label, cta_w_est, _min_size_for(imp)) / ratio)
        sub_items.append(("cta", primary_cta, imp, min_h, _natural_size("cta", imp, text_w, body_h)))
    if legal:
        imp = importances["legal"]
        min_h = max(10.0, _text_min_height(legal, text_w, _min_size_for(imp)))
        sub_items.append(("legal", legal, imp, min_h, _natural_size("legal", imp, text_w, body_h)))

    if sub_items:
        sizes, gap, lead = _flow_sizes(
            [it[4] for it in sub_items], [it[2] for it in sub_items], [it[3] for it in sub_items], body_h,
        )
        y = body_top + lead
        for (kind, payload, weight, needed, _), h in zip(sub_items, sizes):
            if kind == "cta":
                _, members = payload
                cta_w = min(text_w, max(text_w * 0.7, 150))
                cta_box = (text_x, y, cta_w, h)
                placed = _apply_group_transform(members, _group_bbox(members), cta_box, min_size=_adjusted_min_size(weight, needed, h))
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
            else:
                placed = [_place_text(payload, (text_x, y, text_w, h), _adjusted_min_size(weight, needed, h))]
                out.extend(_deemphasize_if_underfunded(placed, needed, h))
            y += h + gap
    return out


def _layout_strip(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], ribbon_groups: list[PillGroup], importances: dict[str, float],
    tw: float, th: float, margin: float,
) -> list[Element]:
    """Single-row layout for ultra-thin strips (e.g. a 728x90 leaderboard).
    Logo / headline / CTA - the three things a leaderboard ad actually
    needs to communicate - get the row, with its primary/secondary split
    sized by how much the headline itself needs rather than a fixed
    ratio. The product/hero image is deliberately NOT given primary-row
    space here: at this height it would render as an unrecognizable
    sliver anyway, so that width is worth more going to the headline/CTA
    that carry the message. Everything that doesn't fit (hero, badges,
    body copy, legal text) is a genuine "impossible trade-off" at this
    height - every element stays present (ids preserved, nothing hidden
    via zero opacity) in an importance-sorted, importance-sized secondary
    band, with a modest and bounded opacity de-emphasis (never below
    ~0.6, still clearly visible) so it reads as deliberate hierarchy
    rather than clutter competing with the primary row."""
    out: list[Element] = []
    pad = max(3.0, th * 0.08)
    row_w = max(10.0, tw - 2 * pad)
    # Rotated ribbons aren't part of the classified roles `importances`
    # covers, so they get a flat low-but-nonzero weight here - they're
    # decorative accents, not message-bearing content.
    secondary: list[tuple[Any, float, bool]] = (
        [(bg, importances[f"badge_{i}"], True) for i, bg in enumerate(badges)]
        + [(rb, 0.15, True) for rb in ribbon_groups]
        + ([(hero, importances["hero"], False)] if hero else [])
        + [(bt, importances[f"body_{i}"], False) for i, bt in enumerate(body_texts)]
        + ([(legal, importances["legal"], False)] if legal else [])
    )
    secondary.sort(key=lambda t: t[1], reverse=True)
    has_secondary = bool(secondary)

    avail_h = max(6.0, th - 2 * pad)

    # Width shares don't depend on the primary/secondary height split (they
    # only depend on row_w), so they can be computed up front - which lets
    # the split itself be informed by what the secondary band actually
    # needs, instead of a fixed importance-based guess that has no idea
    # whether it's handing a 6-item row enough height or not.
    slot_sizes, sgap = _distribute([(imp, 6.0) for _, imp, _ in secondary], row_w) if has_secondary else ([], 0.0)
    secondary_true_need = 0.0
    for (item, imp, is_group), sw_ in zip(secondary, slot_sizes):
        label = item if (not is_group and item.get("type") == "text") else None
        ratio = 1.0
        if is_group:
            label = next((m for m in item[1] if m["type"] == "text"), None)
            if label is not None:
                ratio = _group_label_height_ratio(item[1])
        if label is not None:
            needed = _text_min_height(label, sw_, _min_size_for(imp, scale=0.5, floor=5.0)) / ratio
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
        primary_h = max(10.0, avail_h - secondary_h - gap)
    else:
        primary_h = avail_h
        gap = 0.0
        secondary_h = 0.0

    y0 = pad
    x = pad

    if logo:
        logo_w = min(primary_h * 2.6, row_w * 0.22)
        out.append(_place_image(logo, (x, y0, logo_w, primary_h)))
        x += logo_w + pad

    cta_w = 0.0
    if primary_cta:
        # primary_h * 5 (not 3) - the tighter multiplier was forcing the
        # button narrower than its own text needs even when row_w had
        # plenty of room to spare, which is what pushed "Shop the Summer
        # Line" to wrap to 2 lines in a box sized for 1.
        cta_w = min(row_w * 0.24, max(primary_h * 5.0, 70.0))

    headline_right = pad + row_w - (cta_w + pad if primary_cta else 0.0)
    headline_w = max(10.0, headline_right - x)
    if headline:
        h_min = _min_size_for(importances["headline"], scale=0.7, floor=7.5)
        # primary_h already accounts for the secondary band's real need
        # (computed above); if headline still wants more than that, it
        # shrinks via _fit_font_size's own search rather than clawing
        # space back from a secondary band that was sized to protect.
        if _text_min_height(headline, headline_w, h_min) > primary_h:
            h_min = 5.0
        out.append(_place_text(headline, (x, y0, headline_w, primary_h), min_size=h_min))

    if primary_cta:
        _, members = primary_cta
        cta_x = pad + row_w - cta_w
        c_min = _min_size_for(importances["cta"], scale=0.6, floor=7.0)
        cta_label = next((m for m in members if m["type"] == "text"), None)
        cta_ratio = _group_label_height_ratio(members) if cta_label is not None else 1.0
        if cta_label is not None and _text_min_height(cta_label, cta_w, c_min) > primary_h * cta_ratio:
            c_min = 5.0
        out.extend(_apply_group_transform(members, _group_bbox(members), (cta_x, y0, cta_w, primary_h), min_size=c_min))

    if has_secondary:
        # `slot_sizes`/`sgap` (the importance-weighted width split) were
        # already computed above, to size `secondary_h` itself - reused
        # here rather than recomputed.
        if secondary_h > 0:
            slot_y, slot_h, scale, floor = y0 + primary_h + gap, secondary_h, 0.5, 5.0
        else:
            slot_h = max(2.0, th * 0.025)
            slot_y, scale, floor = th - slot_h, 0.4, 4.0
        sx = pad
        for (item, imp, is_group), sw_ in zip(secondary, slot_sizes):
            box = (sx, slot_y, sw_, slot_h)
            base_min = _min_size_for(imp, scale=scale, floor=floor)
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
                min_size = 5.0  # let _fit_font_size's own search find the best fit down to the floor
            # Pushed lower again (was 0.45-0.80): a saturated brand color
            # (a badge sharing the CTA's exact orange) stays eye-catching
            # even dimmed to ~60%, which an independent review read as
            # still competing with the CTA for attention rather than
            # reading as clearly secondary. Lower still reads as
            # deliberate hierarchy, not an error - zero-geometry-risk,
            # same as the first pass at this.
            opacity_mult = 0.32 + 0.28 * max(0.0, min(1.0, imp))
            if is_group:
                _, members = item
                placed = _apply_group_transform(members, _group_bbox(members), box, min_size=min_size)
            elif item["type"] == "image":
                placed = [_place_image(item, box)]
            else:
                placed = [_place_text(item, box, min_size=min_size)]
            for pe in placed:
                # `opacity_mult` targets a final on-screen visibility, not a
                # flat multiplier - a label whose own `color` already bakes
                # in reduced alpha (fine-print text styled at 60% by the
                # source design, unlike every other label here which is
                # fully opaque) would otherwise get dimmed twice: once by
                # its own color, again by this multiplier, landing far
                # fainter than an equally-"secondary" item that started
                # fully opaque. Dividing out the color's own alpha first
                # makes every secondary item land at the *same* effective
                # visibility regardless of how its source color was
                # authored, instead of punishing pre-dimmed text for being
                # pre-dimmed.
                if is_group and pe.get("type") == "text":
                    # A badge/ribbon's own "secondary" signal already comes
                    # from its pill background being dimmed - dimming the
                    # label on top of that too compounds into washed-out,
                    # low-contrast text (a muted fill *and* faded white
                    # text), which reads as broken/blurry rather than as
                    # deliberate hierarchy. Keep the label itself crisp;
                    # the pill it sits on is what carries the emphasis cue.
                    compensated = 1.0
                elif pe.get("type") == "shape" and pe.get("fill_color"):
                    # The same `opacity_mult` lands very differently
                    # depending on how dark the fill already is: a near-
                    # black navy blended at the same alpha as a bright
                    # orange loses almost all contrast against a dark
                    # canvas (sampled in practice: a navy pill dimmed this
                    # way came out within a few RGB values of the
                    # background right next to it - visually just gone,
                    # not "subtle"). Boost opacity for dark fills only,
                    # just enough to keep fill_luminance * opacity above a
                    # floor - bright fills (already well above that floor
                    # at the normal `opacity_mult`) are untouched.
                    try:
                        from renderer.converter.colorutil import parse_rgba

                        r, g, b, _a = parse_rgba(pe["fill_color"])
                        luminance = (r + g + b) / (3 * 255.0)
                    except Exception:
                        luminance = 1.0
                    visibility_floor = 0.22
                    compensated = min(1.0, max(opacity_mult, visibility_floor / max(0.05, luminance)))
                else:
                    base_alpha = 1.0
                    if pe.get("type") == "text" and pe.get("color"):
                        try:
                            from renderer.converter.colorutil import parse_rgba

                            base_alpha = max(0.05, parse_rgba(pe["color"])[3])
                        except Exception:
                            base_alpha = 1.0
                    compensated = min(1.0, opacity_mult / base_alpha)
                pe["opacity"] = round(max(0.0, min(1.0, pe.get("opacity", 1.0))) * compensated, 3)
            out.extend(placed)
            sx += sw_ + sgap
    return out
