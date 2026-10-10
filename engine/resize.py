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
2. Archetype selection - the target canvas's aspect ratio (and, for very
   short canvases, its absolute height) picks one of three layout
   templates: a vertical stack (portrait/near-square), a two-column
   horizontal layout (wide banners/posters), or a single-row layout
   (ultra-thin strips like a 728x90 leaderboard).
3. Each archetype allocates space to the classified roles by priority
   weight, then places each element (or, for a group, the whole group as
   one rigid unit so a button's background and label never drift apart)
   into its slot: images are aspect-preserving "contain"-fit (except
   `fit:cover`, which is expected to crop), and text's font_size is
   shrunk/grown to fit using the exact font metrics the real renderer
   uses (falling back to a rough estimate if that's unavailable).
4. Purely decorative elements (ungrouped shapes, rotated accent groups)
   are passed through a simple proportional transform instead of being
   forced into the flow - they're meant to bleed/overlap by design.
5. A final clamp guarantees nothing but an already-intentionally-bleeding
   decorative element ends up outside the target canvas.
"""

from typing import Any, Optional

Element = dict[str, Any]
Box = tuple[float, float, float, float]  # x, y, width, height

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
_font_manager = None
_font_manager_failed = False


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
        if lines * size * line_height <= box_h or size <= min_size:
            break
        size = max(min_size, size * 0.92)
    return round(size, 1)


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
    (never below its share) when it doesn't. Returns (sizes, gap)."""
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

    # --- bleed allowance: decorative elements already off the source
    # canvas keep bleeding proportionally; nothing else should end up
    # off-canvas. ---
    allow_bleed_ids = {e["id"] for e in standalone_shapes if _is_out_of_bounds(e, sw, sh)}

    out_elements: list[Element] = []

    for e in backgrounds:
        e = _copy_element(e)
        e["x"], e["y"], e["width"], e["height"] = 0.0, 0.0, tw, th
        out_elements.append(e)

    for e in standalone_shapes:
        e = _copy_element(e)
        e["x"], e["y"] = e["x"] * scale_x, e["y"] * scale_y
        e["width"], e["height"] = e["width"] * scale_x, e["height"] * scale_y
        out_elements.append(e)

    for e in extra_images:
        box = (e["x"] * scale_x, e["y"] * scale_y, e["width"] * scale_x, e["height"] * scale_y)
        out_elements.append(_place_image(e, box))

    margin = max(8.0, min(tw, th) * 0.045)
    ratio = tw / th
    is_strip = th < 160

    # A decorative ribbon's "near its original corner" placement only makes
    # sense when there's enough canvas for a corner to mean something. In an
    # ultra-thin strip, a simple proportional passthrough would almost
    # certainly land it on top of the primary row (logo/headline/CTA), so
    # it's folded into the strip's deprioritized secondary band instead -
    # same treatment as badges, just still carrying its rotation.
    passthrough_groups = generic_groups if is_strip else (ribbon_groups + generic_groups)
    if not is_strip:
        for _, members in ribbon_groups:
            if any(_is_out_of_bounds(m, sw, sh) for m in members):
                allow_bleed_ids.update(m["id"] for m in members)
    for _, members in passthrough_groups:
        bbox = _group_bbox(members)
        new_box = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
        out_elements.extend(_apply_group_transform(members, bbox, new_box))

    if is_strip:
        out_elements += _layout_strip(logo, hero, headline, body_texts, legal, primary_cta, badges, ribbon_groups, tw, th, margin)
    elif ratio >= 1.8:
        out_elements += _layout_wide(logo, hero, headline, body_texts, legal, primary_cta, badges, tw, th, margin)
    else:
        out_elements += _layout_stack(logo, hero, headline, body_texts, legal, primary_cta, badges, tw, th, margin)

    out_elements = _clamp_elements(out_elements, tw, th, allow_bleed_ids)

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

PillGroup = tuple[str, list[Element]]


def _layout_stack(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], tw: float, th: float, margin: float,
) -> list[Element]:
    """Vertical stack for portrait / near-square targets."""
    out: list[Element] = []
    avail_w = max(10.0, tw - 2 * margin)
    avail_h = max(10.0, th - 2 * margin)

    items: list[tuple[str, Any, float, float]] = []
    if logo:
        items.append(("logo", logo, 0.10, 36))
    if hero:
        items.append(("hero", hero, 0.34, 60))
    if headline:
        items.append(("headline", headline, 0.16, 30))
    for bt in body_texts:
        items.append(("body", bt, 0.09, 18))
    if badges:
        items.append(("badges", badges, 0.09, 30))
    if primary_cta:
        items.append(("cta", primary_cta, 0.12, 36))
    if legal:
        items.append(("legal", legal, 0.05, 12))
    if not items:
        return out

    sizes, gap = _distribute([(w, mn) for _, _, w, mn in items], avail_h)
    y = margin
    for (kind, payload, _, _), h in zip(items, sizes):
        box = (margin, y, avail_w, h)
        if kind in ("logo", "hero"):
            out.append(_place_image(payload, box))
        elif kind == "cta":
            _, members = payload
            cta_w = min(avail_w, max(avail_w * 0.55, 180))
            out.extend(_apply_group_transform(members, _group_bbox(members), (margin + (avail_w - cta_w) / 2, y, cta_w, h)))
        elif kind == "badges":
            n_b = len(payload)
            gap_b = max(4.0, avail_w * 0.02) if n_b > 1 else 0.0
            bw = max(10.0, min((avail_w - gap_b * (n_b - 1)) / n_b, h * 4.5))
            bx = margin
            for _, members in payload:
                out.extend(_apply_group_transform(members, _group_bbox(members), (bx, y, bw, h)))
                bx += bw + gap_b
        else:
            min_size = 8.0 if kind == "legal" else 10.0
            out.append(_place_text(payload, box, min_size))
        y += h + gap
    return out


def _layout_wide(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], tw: float, th: float, margin: float,
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
            for _, members in badges:
                out.extend(_apply_group_transform(members, _group_bbox(members), (x, margin, bw, strip_h)))
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

    sub_items: list[tuple[str, Any, float, float]] = []
    if headline:
        sub_items.append(("headline", headline, 0.40, 26))
    for bt in body_texts:
        sub_items.append(("body", bt, 0.22, 16))
    if primary_cta:
        sub_items.append(("cta", primary_cta, 0.30, 30))
    if legal:
        sub_items.append(("legal", legal, 0.10, 12))

    if sub_items:
        sizes, gap = _distribute([(w, mn) for _, _, w, mn in sub_items], body_h)
        y = body_top
        for (kind, payload, _, _), h in zip(sub_items, sizes):
            if kind == "cta":
                _, members = payload
                cta_w = min(text_w, max(text_w * 0.7, 150))
                out.extend(_apply_group_transform(members, _group_bbox(members), (text_x, y, cta_w, h)))
            else:
                min_size = 8.0 if kind == "legal" else 10.0
                out.append(_place_text(payload, (text_x, y, text_w, h), min_size))
            y += h + gap
    return out


def _layout_strip(
    logo: Optional[Element], hero: Optional[Element], headline: Optional[Element],
    body_texts: list[Element], legal: Optional[Element], primary_cta: Optional[PillGroup],
    badges: list[PillGroup], ribbon_groups: list[PillGroup], tw: float, th: float, margin: float,
) -> list[Element]:
    """Single-row layout for ultra-thin strips (e.g. a 728x90 leaderboard).
    Logo / headline / CTA get the row; anything that doesn't fit (badges,
    body copy, legal text) is a genuine "impossible trade-off" at this
    height - every element stays present (ids preserved, nothing hidden
    via zero opacity) but is compressed into a thin secondary band, or a
    minimal sliver along the bottom edge if there's truly no room for one."""
    out: list[Element] = []
    pad = max(3.0, th * 0.08)
    row_w = max(10.0, tw - 2 * pad)
    secondary_items: list[Any] = list(badges) + list(ribbon_groups) + list(body_texts) + ([legal] if legal else [])
    has_secondary = bool(secondary_items)

    avail_h = max(6.0, th - 2 * pad)
    if has_secondary and avail_h >= 46:
        primary_h = round(avail_h * 0.62)
        gap = max(2.0, th * 0.03)
        secondary_h = max(8.0, avail_h - primary_h - gap)
    else:
        primary_h = avail_h
        gap = 0.0
        secondary_h = 0.0

    y0 = pad
    x = pad

    if logo:
        logo_w = min(primary_h * 2.6, row_w * 0.24)
        out.append(_place_image(logo, (x, y0, logo_w, primary_h)))
        x += logo_w + pad

    if hero:
        thumb_w = min(primary_h, row_w * 0.14)
        out.append(_place_image(hero, (x, y0, thumb_w, primary_h)))
        x += thumb_w + pad

    cta_w = 0.0
    if primary_cta:
        cta_w = min(row_w * 0.26, max(primary_h * 3.2, 60.0))

    headline_right = pad + row_w - (cta_w + pad if primary_cta else 0.0)
    headline_w = max(10.0, headline_right - x)
    if headline:
        out.append(_place_text(headline, (x, y0, headline_w, primary_h), min_size=9.0))

    if primary_cta:
        _, members = primary_cta
        cta_x = pad + row_w - cta_w
        out.extend(_apply_group_transform(members, _group_bbox(members), (cta_x, y0, cta_w, primary_h), min_size=8.0))

    if has_secondary:
        n = len(secondary_items)
        slot_w = max(6.0, row_w / n)
        sx = pad
        if secondary_h > 0:
            slot_y, slot_h, min_size = y0 + primary_h + gap, secondary_h, 6.0
        else:
            slot_h = max(2.0, th * 0.025)
            slot_y, min_size = th - slot_h, 5.0
        for item in secondary_items:
            box = (sx, slot_y, max(6.0, slot_w - 3), slot_h)
            if isinstance(item, tuple):
                _, members = item
                out.extend(_apply_group_transform(members, _group_bbox(members), box, min_size=min_size))
            else:
                out.append(_place_text(item, box, min_size=min_size))
            sx += slot_w
    return out
