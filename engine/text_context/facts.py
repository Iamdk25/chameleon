"""Design context shown to the text model.

Everything here is read straight from the source JSON: text content and
styling, geometry, groups, and the shapes/images around each text. Element
ids and group ids are replaced by neutral aliases, because the sample files
use ids such as "headline" or "legal_text" that would hand the model the
answer. No role is inferred in this module.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def build_context(
    source: dict[str, Any], target_canvas: dict[str, Any], extra_context: Optional[str] = None,
) -> dict[str, Any]:
    """Return the prompt text plus the alias <-> id maps.

    `extra_context` is the optional hook for a future context provider (for
    example a permitted vision model's description of the source render). It
    is passed through verbatim and labelled as external; nothing in this
    iteration supplies it.
    """
    elements = list(source["elements"])
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    tw, th = float(target_canvas["width"]), float(target_canvas["height"])

    reading = sorted(elements, key=lambda e: (round(float(e["y"]) / 8.0), float(e["x"])))
    counters = {"text": 0, "image": 0, "shape": 0}
    prefix = {"text": "T", "image": "I", "shape": "S"}
    alias_of: dict[str, str] = {}
    for e in reading:
        kind = e.get("type", "shape")
        counters[kind] = counters.get(kind, 0) + 1
        alias_of[e["id"]] = f"{prefix.get(kind, 'E')}{counters[kind]}"
    group_alias: dict[str, str] = {}
    for e in reading:
        gid = e.get("group_id")
        if gid and gid not in group_alias:
            group_alias[gid] = f"G{len(group_alias) + 1}"

    texts = [e for e in reading if e.get("type") == "text"]
    sizes = sorted({float(t.get("font_size", 0)) for t in texts}, reverse=True)
    size_rank = {s: i + 1 for i, s in enumerate(sizes)}

    lines = [
        f"SOURCE canvas: {_n(sw)}x{_n(sh)} px.",
        f"TARGET canvas: {_n(tw)}x{_n(th)} px. {_target_shape(sw, sh, tw, th)}",
        "Coordinates are source pixels; smaller y is higher on the canvas.",
        "Image pixels are NOT available. For images you only know the asset filename and geometry.",
        "",
        "ELEMENTS (reading order, top to bottom):",
    ]
    for e in reading:
        lines.append(_describe(e, alias_of, group_alias, size_rank, len(sizes), sw, sh))
    groups = _group_lines(reading, alias_of, group_alias)
    if groups:
        lines += ["", "GROUPS (members move together):"] + groups
    lines += ["", "NEIGHBOURS OF EACH TEXT:"]
    for t in texts:
        lines.append(f"{alias_of[t['id']]}: " + _neighbours(t, elements, alias_of, sw, sh))
    if extra_context:
        lines += ["", "EXTERNAL CONTEXT (from an optional provider, not from the JSON):", extra_context.strip()]

    text_aliases = [alias_of[t["id"]] for t in texts]
    return {
        "prompt": "\n".join(lines),
        "alias_of": alias_of,
        "id_of": {a: i for i, a in alias_of.items()},
        "text_aliases": text_aliases,
        "all_aliases": [alias_of[e["id"]] for e in reading],
    }


def _n(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _target_shape(sw: float, sh: float, tw: float, th: float) -> str:
    ratio = tw / th if th else 0.0
    if ratio >= 4:
        shape = "an extremely wide, short strip"
    elif ratio >= 1.8:
        shape = "a wide banner"
    elif ratio >= 1.15:
        shape = "landscape"
    elif ratio > 0.87:
        shape = "roughly square"
    elif ratio > 0.6:
        shape = "portrait"
    else:
        shape = "a tall, narrow portrait"
    same = "It is the SAME size as the source." if (sw, sh) == (tw, th) else ""
    fit_scale = min(tw / sw, th / sh)
    return (
        f"Width/height = {ratio:.2f} ({shape}). Fitting the whole source uniformly would scale it by "
        f"{fit_scale:.3f}. {same}"
    ).strip()


def _color(value: Any) -> str:
    return str(value) if value else "-"


def _box(e: dict[str, Any], sw: float, sh: float) -> str:
    x, y, w, h = (float(e[k]) for k in ("x", "y", "width", "height"))
    area = 100.0 * max(0.0, w) * max(0.0, h) / max(1.0, sw * sh)
    out = f"box x={_n(x)} y={_n(y)} w={_n(w)} h={_n(h)} ({area:.1f}% of canvas)"
    if x < -0.5 or y < -0.5 or x + w > sw + 0.5 or y + h > sh + 0.5:
        out += ", extends past the canvas edge"
    return out


def _describe(
    e: dict[str, Any], alias_of: dict[str, str], group_alias: dict[str, str],
    size_rank: dict[float, int], n_sizes: int, sw: float, sh: float,
) -> str:
    a = alias_of[e["id"]]
    kind = e.get("type")
    parts = [a, kind]
    if kind == "text":
        size = float(e.get("font_size", 0))
        parts.append(json.dumps(e.get("content", ""), ensure_ascii=False))
        style = [e.get("font_family", "?"), f"{_n(size)}px (size rank {size_rank.get(size, 0)} of {n_sizes})"]
        if e.get("font_weight") and e.get("font_weight") != "regular":
            style.append(str(e["font_weight"]))
        if e.get("font_style") == "italic":
            style.append("italic")
        if e.get("capitalize"):
            style.append("rendered uppercase")
        if e.get("letter_spacing"):
            style.append(f"letter_spacing {e['letter_spacing']}")
        parts.append("font " + " ".join(style))
        parts.append(f"color {_color(e.get('color'))}")
        if e.get("text_align"):
            parts.append(f"align {e['text_align']}")
        if e.get("style_runs"):
            parts.append(f"{len(e['style_runs'])} styled runs")
    elif kind == "image":
        parts.append(f"asset file {json.dumps(e.get('asset_id', ''))}")
        parts.append(f"fit {e.get('fit', 'contain')}")
        if _covers(e, sw, sh):
            parts.append("covers the whole canvas")
    else:
        parts.append(str(e.get("shape_type", "shape")))
        parts.append(f"fill {_color(e.get('fill_color'))}")
        if e.get("corner_radius"):
            parts.append(f"corner_radius {e['corner_radius']}")
    parts.append(_box(e, sw, sh))
    if e.get("rotation"):
        parts.append(f"rotated {e['rotation']} deg")
    if e.get("opacity") is not None and float(e.get("opacity", 1)) < 1:
        parts.append(f"opacity {e['opacity']}")
    parts.append(f"z {e.get('z_index', 0)}")
    if e.get("group_id"):
        parts.append(f"group {group_alias[e['group_id']]}")
    return " | ".join(str(p) for p in parts)


def _covers(e: dict[str, Any], sw: float, sh: float) -> bool:
    """Same idea as the layout background check: fit=cover and real coverage."""
    return (
        e.get("type") == "image" and e.get("fit") == "cover"
        and float(e["x"]) <= sw * 0.05 and float(e["y"]) <= sh * 0.05
        and float(e["x"]) + float(e["width"]) >= sw * 0.95
        and float(e["y"]) + float(e["height"]) >= sh * 0.95
    )


def _group_lines(reading: list[dict[str, Any]], alias_of: dict[str, str], group_alias: dict[str, str]) -> list[str]:
    members: dict[str, list[dict[str, Any]]] = {}
    for e in reading:
        if e.get("group_id"):
            members.setdefault(e["group_id"], []).append(e)
    out = []
    for gid, ms in members.items():
        desc = [alias_of[m["id"]] for m in ms]
        shapes = [m for m in ms if m.get("type") == "shape"]
        texts = [m for m in ms if m.get("type") == "text"]
        note = ""
        if shapes and texts:
            s, t = shapes[0], texts[0]
            inside = (
                float(t["x"]) >= float(s["x"]) - 1 and float(t["y"]) >= float(s["y"]) - 1
                and float(t["x"]) + float(t["width"]) <= float(s["x"]) + float(s["width"]) + 1
                and float(t["y"]) + float(t["height"]) <= float(s["y"]) + float(s["height"]) + 1
            )
            aspect = float(s["width"]) / max(1.0, float(s["height"]))
            note = (
                f" -- text {alias_of[t['id']]} {'sits inside' if inside else 'overlaps'} "
                f"{s.get('shape_type')} {alias_of[s['id']]} (shape width/height {aspect:.1f})"
            )
        out.append(f"{group_alias[gid]}: {', '.join(desc)}{note}")
    return out


def _neighbours(
    t: dict[str, Any], elements: list[dict[str, Any]], alias_of: dict[str, str], sw: float, sh: float,
) -> str:
    tx, ty, tw_, th_ = (float(t[k]) for k in ("x", "y", "width", "height"))
    cx, cy = tx + tw_ / 2, ty + th_ / 2
    cands = []
    for o in elements:
        if o is t or (o.get("type") == "image" and _covers(o, sw, sh)):
            continue
        if t.get("group_id") and o.get("group_id") == t.get("group_id"):
            continue
        ox, oy, ow, oh = (float(o[k]) for k in ("x", "y", "width", "height"))
        gap_x = max(0.0, max(ox - (tx + tw_), tx - (ox + ow)))
        gap_y = max(0.0, max(oy - (ty + th_), ty - (oy + oh)))
        dist = (gap_x ** 2 + gap_y ** 2) ** 0.5
        if gap_x == 0 and gap_y == 0:
            rel = "overlaps"
        else:
            ocx, ocy = ox + ow / 2, oy + oh / 2
            if gap_y >= gap_x:
                rel = "below" if ocy > cy else "above"
            else:
                rel = "right of it" if ocx > cx else "left of it"
        cands.append((dist, alias_of[o["id"]], rel, int(round(dist)), o))
    cands.sort(key=lambda c: (c[0], c[1]))
    parts = []
    for _, alias, rel, gap, o in cands[:3]:
        what = o.get("type")
        if what == "text":
            what = f'text "{_short(o.get("content", ""))}"'
        elif what == "image":
            what = f"image {o.get('asset_id', '')}"
        else:
            what = f"{o.get('shape_type', 'shape')}"
        parts.append(f"{alias} ({what}) {rel}, gap {gap}px")
    return "; ".join(parts) if parts else "-"


def _short(text: str, n: int = 32) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"
