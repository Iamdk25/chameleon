"""Structural checks and visual diagnostics for one engine run.

Three kinds of finding, kept separate on purpose:

- `contract`: the exact JSON contract (ids, count, schema, protected
  fields, input untouched, run provenance). A failure here invalidates the
  run or the output.
- `rule_questions`: changes the schema accepts but the challenge text
  does not clearly permit (FAQ Q1 names position, size, and text font
  size). Reported, never silently accepted or rejected.
- `diagnostics`: geometry and legibility measurements for human review.
  Thresholds here are review flags chosen by us, not official rules, and
  a clean diagnostic list is not evidence of a good design.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

GEOMETRY_FIELDS = ("x", "y", "width", "height")
TEXT_SIZE_FIELD = "font_size"
PROTECTED_FIELDS = (
    "type", "content", "font_family", "font_weight", "font_style",
    "asset_id", "fit", "group_id", "shape_type",
)
RULE_QUESTION_FIELDS = ("opacity", "rotation", "z_index")
FIELD_DEFAULTS = {"opacity": 1, "rotation": 0, "group_id": None}

SMALL_TEXT_PX = 10
TINY_TEXT_PX = 8
OVERLAP_MIN_FRACTION = 0.05
EPS = 0.5

Box = tuple[float, float, float, float]


def rotated_aabb(x: float, y: float, w: float, h: float, rotation: float, cx: Optional[float] = None,
                 cy: Optional[float] = None) -> Box:
    """Axis-aligned bounds of rect (x, y, w, h) rotated clockwise by
    `rotation` degrees about (cx, cy), defaulting to the rect's center."""
    if cx is None:
        cx = x + w / 2.0
    if cy is None:
        cy = y + h / 2.0
    if not rotation:
        return (x, y, w, h)
    t = math.radians(rotation)
    c, s = math.cos(t), math.sin(t)
    xs, ys = [], []
    for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        dx, dy = px - cx, py - cy
        xs.append(cx + dx * c - dy * s)
        ys.append(cy + dx * s + dy * c)
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def element_box_aabb(el: dict) -> Box:
    return rotated_aabb(float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"]),
                        float(el.get("rotation", 0) or 0))


def text_ink_aabb(el: dict, metric: dict) -> Box:
    """Rendered text extent: the renderer draws lines from the box's top
    left without clipping, then rotates about the box center."""
    x0, y0, x1, y1 = metric["local_extent"]
    bx, by, bw, bh = float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"])
    return rotated_aabb(bx + x0, by + y0, x1 - x0, y1 - y0, float(el.get("rotation", 0) or 0),
                        cx=bx + bw / 2.0, cy=by + bh / 2.0)


def contained_rect(el: dict, native: Optional[dict]) -> Box:
    x, y, w, h = float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"])
    if not native or el.get("fit", "contain") != "contain" or w <= 0 or h <= 0:
        return (x, y, w, h)
    scale = min(w / native["width"], h / native["height"])
    dw, dh = native["width"] * scale, native["height"] * scale
    return (x + (w - dw) / 2.0, y + (h - dh) / 2.0, dw, dh)


def visual_aabb(el: dict, text_metrics: dict, asset_native: dict) -> Box:
    if el["type"] == "text" and el["id"] in text_metrics:
        return text_ink_aabb(el, text_metrics[el["id"]])
    if el["type"] == "image":
        r = contained_rect(el, asset_native.get(el.get("asset_id")))
        return rotated_aabb(*r, float(el.get("rotation", 0) or 0),
                            cx=float(el["x"]) + float(el["width"]) / 2.0,
                            cy=float(el["y"]) + float(el["height"]) / 2.0)
    return element_box_aabb(el)


def outside_canvas(box: Box, cw: float, ch: float) -> dict:
    x, y, w, h = box
    return {
        "left": max(0.0, -x), "top": max(0.0, -y),
        "right": max(0.0, x + w - cw), "bottom": max(0.0, y + h - ch),
    }


def is_outside(box: Box, cw: float, ch: float) -> bool:
    return any(v > EPS for v in outside_canvas(box, cw, ch).values())


def is_background(el: dict, cw: float, ch: float) -> bool:
    if el.get("type") != "image":
        return False
    x, y, w, h = float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"])
    return x <= cw * 0.05 and y <= ch * 0.05 and x + w >= cw * 0.95 and y + h >= ch * 0.95


def intersection_area(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def kendall_tau_b(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    return None if denom == 0 else (concordant - discordant) / denom


def _value(el: dict, key: str) -> Any:
    return el.get(key, FIELD_DEFAULTS.get(key))


# ---------------------------------------------------------------------------


def check_contract(source: dict, target: dict, output: dict, schema: dict) -> list[str]:
    failures: list[str] = []
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        for err in sorted(validator.iter_errors(output), key=lambda e: list(e.path)):
            failures.append(f"schema: {'/'.join(map(str, err.path)) or '<root>'}: {err.message}")
    except ImportError:
        failures.append("schema: jsonschema is not installed, schema not checked")

    if output.get("canvas") != {"width": target["width"], "height": target["height"]}:
        failures.append(f"canvas: output canvas {output.get('canvas')} != target {target}")

    src_ids = [e["id"] for e in source["elements"]]
    out_ids = [e.get("id") for e in output.get("elements", [])]
    if len(out_ids) != len(src_ids):
        failures.append(f"ids: element count {len(out_ids)} != source {len(src_ids)}")
    if len(set(out_ids)) != len(out_ids):
        failures.append("ids: duplicate ids in output")
    missing, extra = set(src_ids) - set(out_ids), set(out_ids) - set(src_ids)
    if missing or extra:
        failures.append(f"ids: missing {sorted(missing)}, unexpected {sorted(map(str, extra))}")

    out_by_id = {e.get("id"): e for e in output.get("elements", [])}
    for s in source["elements"]:
        o = out_by_id.get(s["id"])
        if o is None:
            continue
        for key in PROTECTED_FIELDS:
            if _value(s, key) != _value(o, key):
                failures.append(f"protected: {s['id']}.{key} changed {_value(s, key)!r} -> {_value(o, key)!r}")
    return failures


def changed_fields(source: dict, output: dict) -> dict:
    out_by_id = {e.get("id"): e for e in output.get("elements", [])}
    report: dict[str, dict] = {}
    for s in source["elements"]:
        o = out_by_id.get(s["id"])
        if o is None:
            continue
        entry: dict[str, Any] = {}
        for key in sorted(set(s) | set(o)):
            sv, ov = _value(s, key), _value(o, key)
            if sv == ov:
                continue
            if key in GEOMETRY_FIELDS or (key == TEXT_SIZE_FIELD and s.get("type") == "text"):
                kind = "faq_q1_allowed"
            elif key in RULE_QUESTION_FIELDS:
                kind = "rule_question"
            elif key in PROTECTED_FIELDS:
                kind = "protected"
            else:
                kind = "other"
            entry[key] = {"from": sv, "to": ov, "kind": kind}
        if entry:
            report[s["id"]] = entry
    return report


def rule_questions(changes: dict) -> list[str]:
    found = []
    for eid, fields in changes.items():
        for key, info in fields.items():
            if info["kind"] in ("rule_question", "other"):
                found.append(f"{eid}.{key}: {info['from']!r} -> {info['to']!r}")
    return found


def bounds_report(source: dict, output: dict, src_metrics: dict, out_metrics: dict) -> dict:
    scw, sch = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    ocw, och = float(output["canvas"]["width"]), float(output["canvas"]["height"])
    src_by_id = {e["id"]: e for e in source["elements"]}
    starter, carried, new, text_clipped_new, text_clipped_carried = [], [], [], [], []
    for o in output["elements"]:
        s = src_by_id.get(o["id"])
        x1, y1 = o["x"] + o["width"], o["y"] + o["height"]
        if o["x"] < -0.5 or o["y"] < -0.5 or x1 > ocw + 0.5 or y1 > och + 0.5:
            starter.append(o["id"])
        out_box = element_box_aabb(o)
        if is_outside(out_box, ocw, och):
            src_out = s is not None and is_outside(element_box_aabb(s), scw, sch)
            item = {"id": o["id"], "outside": outside_canvas(out_box, ocw, och)}
            (carried if src_out else new).append(item)
        if o["type"] == "text" and o["id"] in out_metrics:
            ink = text_ink_aabb(o, out_metrics[o["id"]])
            if is_outside(ink, ocw, och):
                src_clip = (s is not None and s["id"] in src_metrics
                            and is_outside(text_ink_aabb(s, src_metrics[s["id"]]), scw, sch))
                item = {"id": o["id"], "outside": outside_canvas(ink, ocw, och)}
                (text_clipped_carried if src_clip else text_clipped_new).append(item)
    return {
        "starter_self_check_off_canvas": starter,
        "rotation_aware_off_canvas_carried_source_bleed": carried,
        "rotation_aware_off_canvas_new": new,
        "text_clipped_by_canvas_new": text_clipped_new,
        "text_clipped_by_canvas_carried": text_clipped_carried,
    }


def text_fit_report(source: dict, output: dict, src_metrics: dict, out_metrics: dict) -> dict:
    src_types = {e["id"]: e for e in source["elements"]}
    items = {}
    for o in output["elements"]:
        if o["type"] != "text" or o["id"] not in out_metrics:
            continue
        m = out_metrics[o["id"]]
        sm = src_metrics.get(o["id"])
        w_over = m["max_line_width"] - float(o["width"])
        h_over = m["block_height"] - float(o["height"])
        src_w_over = (sm["max_line_width"] - float(src_types[o["id"]]["width"])) if sm else None
        src_h_over = (sm["block_height"] - float(src_types[o["id"]]["height"])) if sm else None
        items[o["id"]] = {
            "font_size": m["font_size"], "drawn_px": m["drawn_px"], "lines": m["lines"],
            "width_overflow_px": round(w_over, 2), "height_overflow_px": round(h_over, 2),
            "source_width_overflow_px": None if src_w_over is None else round(src_w_over, 2),
            "source_height_overflow_px": None if src_h_over is None else round(src_h_over, 2),
            "overflows_box": w_over > EPS or h_over > EPS,
            "source_overflowed_box": bool(sm) and (src_w_over > EPS or src_h_over > EPS),
        }
    return items


def overlap_report(source: dict, output: dict, src_metrics: dict, out_metrics: dict, asset_native: dict) -> dict:
    def pairs(design: dict, metrics: dict) -> dict:
        cw, ch = float(design["canvas"]["width"]), float(design["canvas"]["height"])
        els = [e for e in design["elements"] if not is_background(e, cw, ch)]
        boxes = {e["id"]: visual_aabb(e, metrics, asset_native) for e in els}
        found = {}
        for i, a in enumerate(els):
            for b in els[i + 1:]:
                if a.get("group_id") and a.get("group_id") == b.get("group_id"):
                    continue
                inter = intersection_area(boxes[a["id"]], boxes[b["id"]])
                smaller = min(boxes[a["id"]][2] * boxes[a["id"]][3], boxes[b["id"]][2] * boxes[b["id"]][3])
                if smaller > 0 and inter / smaller >= OVERLAP_MIN_FRACTION:
                    key = tuple(sorted((a["id"], b["id"])))
                    found[key] = {"pair": list(key), "types": sorted((a["type"], b["type"])),
                                  "fraction_of_smaller": round(inter / smaller, 3)}
        return found

    src_pairs = pairs(source, src_metrics)
    out_pairs = pairs(output, out_metrics)
    new = [v for k, v in out_pairs.items() if k not in src_pairs]
    carried = [v for k, v in out_pairs.items() if k in src_pairs]
    return {"new": new, "carried_from_source": carried,
            "new_text_involved": [v for v in new if "text" in v["types"]]}


def group_report(output: dict, out_metrics: dict) -> list[dict]:
    """Text in a group with a shape should still sit inside that shape."""
    groups: dict[str, list[dict]] = {}
    for e in output["elements"]:
        if e.get("group_id"):
            groups.setdefault(e["group_id"], []).append(e)
    problems = []
    for gid, members in groups.items():
        shapes = [m for m in members if m["type"] == "shape"]
        texts = [m for m in members if m["type"] == "text" and m["id"] in out_metrics]
        for t in texts:
            ink = text_ink_aabb(t, out_metrics[t["id"]])
            for s in shapes:
                sb = element_box_aabb(s)
                spill = {
                    "left": max(0.0, sb[0] - ink[0]), "top": max(0.0, sb[1] - ink[1]),
                    "right": max(0.0, ink[0] + ink[2] - sb[0] - sb[2]),
                    "bottom": max(0.0, ink[1] + ink[3] - sb[1] - sb[3]),
                }
                if any(v > 1.0 for v in spill.values()):
                    problems.append({"group": gid, "text": t["id"], "shape": s["id"],
                                     "spill_px": {k: round(v, 1) for k, v in spill.items()}})
    return problems


def image_report(source: dict, output: dict, asset_native: dict) -> list[dict]:
    src_by_id = {e["id"]: e for e in source["elements"]}
    rows = []
    for o in output["elements"]:
        if o["type"] != "image":
            continue
        s = src_by_id[o["id"]]
        native = asset_native.get(o.get("asset_id"))
        fit = o.get("fit", "contain")
        s_aspect = s["width"] / s["height"] if s["height"] else None
        o_aspect = o["width"] / o["height"] if o["height"] else None
        row = {"id": o["id"], "asset_id": o.get("asset_id"), "fit": fit,
               "asset_present": native is not None,
               "box_aspect_source": s_aspect, "box_aspect_output": o_aspect}
        if native and o_aspect:
            n_aspect = native["width"] / native["height"]
            if fit == "fill":
                row["pixel_distortion"] = round(o_aspect / n_aspect - 1.0, 4)
            else:
                row["pixel_distortion"] = 0.0
            if fit == "cover" and s_aspect:
                def visible(aspect: float) -> float:
                    return min(aspect / n_aspect, n_aspect / aspect)
                row["cover_visible_fraction_source"] = round(visible(s_aspect), 3)
                row["cover_visible_fraction_output"] = round(visible(o_aspect), 3)
        rows.append(row)
    return rows


def legibility_report(source: dict, out_metrics: dict, src_metrics: dict) -> dict:
    rows = {}
    for eid, m in out_metrics.items():
        sm = src_metrics.get(eid)
        rows[eid] = {
            "font_size": m["font_size"], "drawn_px": m["drawn_px"],
            "source_font_size": sm["font_size"] if sm else None,
            "size_ratio": round(m["font_size"] / sm["font_size"], 3) if sm and sm["font_size"] else None,
            "below_small_px": m["drawn_px"] < SMALL_TEXT_PX,
            "below_tiny_px": m["drawn_px"] < TINY_TEXT_PX,
        }
    drawn = [r["drawn_px"] for r in rows.values()]
    return {"per_text": rows, "min_drawn_px": min(drawn) if drawn else None,
            "count_below_small": sum(r["below_small_px"] for r in rows.values()),
            "count_below_tiny": sum(r["below_tiny_px"] for r in rows.values()),
            "thresholds_px": {"small": SMALL_TEXT_PX, "tiny": TINY_TEXT_PX}}


def hierarchy_report(source: dict, output: dict) -> dict:
    """Font-size order among texts, source vs output. A size inversion is
    only a prompt for review: a deliberate re-ranking can be correct."""
    out_by_id = {e["id"]: e for e in output["elements"]}
    texts = [e for e in source["elements"] if e["type"] == "text" and e["id"] in out_by_id]
    s_sizes = [float(e["font_size"]) for e in texts]
    o_sizes = [float(out_by_id[e["id"]]["font_size"]) for e in texts]
    inversions = []
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if s_sizes[i] >= s_sizes[j] * 1.1 and o_sizes[i] < o_sizes[j]:
                inversions.append({"larger_in_source": a["id"], "larger_in_output": b["id"],
                                   "source": [s_sizes[i], s_sizes[j]], "output": [o_sizes[i], o_sizes[j]]})
    def spread(v: list[float]) -> Optional[float]:
        return round(max(v) / min(v), 3) if v and min(v) > 0 else None
    return {"kendall_tau_b_font_size": kendall_tau_b(s_sizes, o_sizes),
            "size_inversions": inversions,
            "largest_over_smallest_source": spread(s_sizes),
            "largest_over_smallest_output": spread(o_sizes)}


def provenance_failures(result: dict, checkout: Path, source: dict) -> list[str]:
    """Prove the run used the intended checkout's engine, renderer, fonts,
    and assets. Both engines silently degrade when fonts fail to load."""
    root = str(checkout.resolve())
    kit = str((checkout / "brand_kit").resolve())
    failures = []
    for key in ("engine_module", "renderer_module", "engine_fonts_dir"):
        val = result.get(key)
        if not val or not str(val).startswith(root + "/"):
            failures.append(f"provenance: {key}={val!r} is not inside {root}")
    if result.get("engine_assets_dir") and not str(result["engine_assets_dir"]).startswith(root + "/"):
        failures.append(f"provenance: engine_assets_dir={result['engine_assets_dir']!r} is not inside {root}")
    if not result.get("font_manager_loaded"):
        failures.append("provenance: engine font manager did not load (engine fell back to estimates)")
    brand_opens = [o for o in result.get("opens", [])
                   if o["path"].lower().endswith((".ttf", ".otf", ".png", ".jpg", ".jpeg"))
                   and not o["path"].startswith(str(Path(result.get("out_dir", "/nonexistent")).resolve()))]
    for o in brand_opens:
        if not o["path"].startswith(kit + "/"):
            failures.append(f"provenance: {o['phase']} opened {o['path']} outside {kit}")
    for phase in ("resize", "render_output"):
        if not any(o["phase"] == phase and o["path"].endswith(".ttf") for o in brand_opens):
            failures.append(f"provenance: no brand font file was opened during {phase}")
    present = [a for a, n in (result.get("asset_native") or {}).items() if n]
    for asset in present:
        if not any(o["phase"] == "render_output" and o["path"] == f"{kit}/assets/{asset}" for o in brand_opens):
            failures.append(f"provenance: renderer never opened brand asset {asset}")
    if result.get("engine_assets_dir"):
        cw, ch = float(source["canvas"]["width"]), float(source["canvas"]["height"])
        bg_assets = {e.get("asset_id") for e in source["elements"]
                     if is_background(e, cw, ch) and e.get("fit") == "cover"} & set(present)
        for asset in bg_assets:
            if not any(o["phase"] == "resize" and o["path"] == f"{kit}/assets/{asset}" for o in brand_opens):
                failures.append(f"provenance: engine never opened background asset {asset} during resize")
    return failures


def run_all(task: dict, output: Optional[dict], result: dict, schema: dict, checkout: Path) -> dict:
    source, target = task["source"], task["target_canvas"]
    report: dict[str, Any] = {"contract": [], "provenance": provenance_failures(result, checkout, source)}
    if not result.get("input_file_unchanged", False):
        report["contract"].append("input: task file bytes changed during the run")
    if result.get("input_mutated_in_memory"):
        report["contract"].append("input: resize() mutated the source dict it was given")
    if result.get("status") != "ok":
        report["contract"].append(f"run: status {result.get('status')}")
    if output is None:
        report["contract"].append("run: no output produced")
        return report
    report["contract"] += check_contract(source, target, output, schema)
    sm = result.get("source_text_metrics", {})
    om = result.get("output_text_metrics", {})
    native = result.get("asset_native", {})
    report["changed_fields"] = changed_fields(source, output)
    report["rule_questions"] = rule_questions(report["changed_fields"])
    report["bounds"] = bounds_report(source, output, sm, om)
    report["text_fit"] = text_fit_report(source, output, sm, om)
    report["overlaps"] = overlap_report(source, output, sm, om, native)
    report["groups"] = group_report(output, om)
    report["images"] = image_report(source, output, native)
    report["legibility"] = legibility_report(source, om, sm)
    report["hierarchy"] = hierarchy_report(source, output)
    report["render_warnings"] = result.get("render_warnings", [])
    return report
