"""Structural checks for a stage-one design JSON.

FAQ Q1 allows updated position, size, and text font size. Everything else
that the schema stores as part of the design is locked: element type, copy,
font family, color, asset id, fit, shape fill, grouping, and the other
brand fields. A change to one of those is a hard failure.

Opacity, rotation, and z-index are still organizer questions. They are
reported, and the entrypoint keeps the source values rather than publishing
a change. Painted bounds outside the target canvas are a hard failure.
The brief checks the output for nothing off canvas, and no exception for
source bleed has been granted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "chameleon_design_schema_v0.9.json"

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}

# FAQ Q1: position, size, and optional text font size.
_LAYOUT_FIELDS = {"x", "y", "width", "height"}
_TEXT_LAYOUT_FIELDS = {"font_size"}
# Unresolved. Reported, and not treated as a permitted edit.
_UNRESOLVED_FIELDS = ("opacity", "rotation", "z_index")


def contract_failures(
    source: dict[str, Any],
    output: dict[str, Any],
    target_canvas: dict[str, Any],
    assets_dir: Optional[Path] = None,
) -> list[str]:
    problems: list[str] = []
    if "canvas" not in output or "elements" not in output:
        return ["output is missing top-level 'canvas' or 'elements'"]

    canvas = output["canvas"]
    if float(canvas.get("width", -1)) != float(target_canvas["width"]) or float(canvas.get("height", -1)) != float(target_canvas["height"]):
        problems.append(
            f"canvas is {canvas.get('width')}x{canvas.get('height')}, "
            f"target is {target_canvas['width']}x{target_canvas['height']}"
        )

    src_elements = source["elements"]
    out_elements = output["elements"]
    src_ids = [element["id"] for element in src_elements]
    out_ids = [element.get("id") for element in out_elements]
    if len(out_elements) != len(src_elements):
        problems.append(f"element count {len(out_elements)} != source {len(src_elements)}")
    if out_ids != src_ids:
        problems.append(f"element id order changed: source {src_ids} output {out_ids}")

    src_by_id = {element["id"]: element for element in src_elements}
    for element in out_elements:
        eid = element.get("id")
        for field in REQUIRED_COMMON:
            if field not in element:
                problems.append(f"element {eid!r} missing {field}")
        for field in REQUIRED_BY_TYPE.get(element.get("type"), []):
            if field not in element:
                problems.append(f"element {eid!r} missing {field}")
        if all(key in element for key in ("x", "y", "width", "height")):
            problems.extend(_geometry_problems(element, float(canvas.get("width") or 0), float(canvas.get("height") or 0)))
        source_element = src_by_id.get(eid)
        if source_element is None:
            continue
        problems.extend(_locked_field_changes(source_element, element))
        if element.get("type") == "image" and source_element.get("type") == "image" and _ratio_broken(source_element, element, assets_dir):
            problems.append(f"element {eid!r} image proportions changed")
    return problems


def _locked_field_changes(source_element: dict[str, Any], element: dict[str, Any]) -> list[str]:
    """Any field FAQ Q1 does not name, except the unresolved opacity/rotation/z-index set."""
    eid = source_element.get("id")
    problems: list[str] = []
    if element.get("type") != source_element.get("type"):
        problems.append(
            f"element {eid!r} type changed from {source_element.get('type')!r} to {element.get('type')!r}"
        )
    skip = set(_LAYOUT_FIELDS) | set(_UNRESOLVED_FIELDS)
    if source_element.get("type") == "text" and element.get("type") == "text":
        skip |= _TEXT_LAYOUT_FIELDS
    for field in sorted((set(source_element) | set(element)) - skip):
        if field == "type":
            continue
        if source_element.get(field) != element.get(field):
            label = {"content": "copy", "font_family": "font family", "asset_id": "asset id"}.get(field, field)
            problems.append(f"element {eid!r} {label} changed")
    return problems


def preserve_unresolved_fields(source: dict[str, Any], output: dict[str, Any]) -> list[str]:
    """Copy opacity, rotation, and z-index back from the source.

    The engine may have edited them. Whether that is allowed is unanswered,
    so the published design keeps the source values and the caller can say so.
    """
    notes: list[str] = []
    if "elements" not in output:
        return notes
    src_by_id = {element["id"]: element for element in source.get("elements", [])}
    for element in output["elements"]:
        source_element = src_by_id.get(element.get("id"))
        if source_element is None:
            continue
        for field in _UNRESOLVED_FIELDS:
            if field in source_element:
                if element.get(field) != source_element.get(field):
                    notes.append(
                        f"preserved source {field} on {element.get('id')} "
                        f"({element.get(field, 'absent')} -> {source_element.get(field)}); "
                        "whether a change is allowed is unanswered"
                    )
                element[field] = source_element[field]
            elif field in element:
                notes.append(
                    f"removed {field} on {element.get('id')} so the published element matches the source; "
                    "whether a change is allowed is unanswered"
                )
                element.pop(field, None)
    return notes


def unresolved_warnings(source: dict[str, Any], output: dict[str, Any], assets_dir: Optional[Path] = None) -> list[str]:
    """Opacity, rotation, and z-index are still organizer questions. Missing
    asset files are reported here too. Off-canvas geometry is a hard failure
    in `contract_failures`, not a warning."""
    warnings: list[str] = []
    if "elements" not in output or "canvas" not in output:
        return warnings
    src_by_id = {element["id"]: element for element in source["elements"]}
    for element in output["elements"]:
        eid = element.get("id")
        source_element = src_by_id.get(eid)
        if source_element is not None:
            for field in _UNRESOLVED_FIELDS:
                if element.get(field) != source_element.get(field):
                    warnings.append(
                        f"{field} changed on {eid} ({source_element.get(field, 'absent')} -> {element.get(field, 'absent')}); "
                        "whether that change is allowed is unanswered"
                    )
        if element.get("type") == "image" and assets_dir is not None:
            asset_id = element.get("asset_id")
            if asset_id and not (assets_dir / asset_id).is_file():
                warnings.append(
                    f"{eid} references missing asset {asset_id}; the Level 3 watermark is the sample that does this on purpose"
                )
    return warnings


def _geometry_problems(element: dict[str, Any], canvas_w: float, canvas_h: float) -> list[str]:
    eid = element.get("id")
    problems: list[str] = []
    numbers = []
    for key in ("x", "y", "width", "height"):
        value = element.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            problems.append(f"element {eid!r} {key} is not a finite number")
            continue
        numbers.append(float(value))
    if len(numbers) != 4:
        return problems
    x, y, width, height = numbers
    if width <= 0 or height <= 0:
        problems.append(f"element {eid!r} has non-positive geometry")
        return problems
    if _box_outside(x, y, width, height, canvas_w, canvas_h):
        problems.append(f"element {eid!r} stored box is off-canvas")
    px, py, pw, ph = _painted_box(x, y, width, height, float(element.get("rotation") or 0))
    if _box_outside(px, py, pw, ph, canvas_w, canvas_h):
        problems.append(f"element {eid!r} painted bounds are off-canvas")
    return problems


def _box_outside(x: float, y: float, width: float, height: float, canvas_w: float, canvas_h: float) -> bool:
    return x < -0.5 or y < -0.5 or x + width > canvas_w + 0.5 or y + height > canvas_h + 0.5


def _painted_box(x: float, y: float, width: float, height: float, rotation: float) -> tuple[float, float, float, float]:
    if not rotation:
        return x, y, width, height
    cx, cy = x + width / 2.0, y + height / 2.0
    rad = math.radians(rotation)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    painted_w = width * cos_a + height * sin_a
    painted_h = width * sin_a + height * cos_a
    return cx - painted_w / 2.0, cy - painted_h / 2.0, painted_w, painted_h


def _outside(element: dict[str, Any], width: float, height: float) -> bool:
    if not all(key in element for key in ("x", "y", "width", "height")) or width <= 0 or height <= 0:
        return False
    return (
        float(element["x"]) < -0.5
        or float(element["y"]) < -0.5
        or float(element["x"]) + float(element["width"]) > width + 0.5
        or float(element["y"]) + float(element["height"]) > height + 0.5
    )


def adobe_schema_failures(document: dict[str, Any], schema_path: Optional[Path] = None) -> list[str]:
    """Problems that make `document` fail the supplied Adobe design schema.

    This is the file Adobe ships, not a second copy of its rules. A failure
    here is a reason to publish nothing. It is not Adobe's hidden score.
    """
    path = schema_path or _SCHEMA_PATH
    schema = json.loads(path.read_text())
    return _schema_errors(document, schema, schema, "$")


def _schema_errors(instance: Any, node: dict[str, Any], root: dict[str, Any], where: str) -> list[str]:
    if "$ref" in node:
        return _schema_errors(instance, _resolve_ref(root, node["$ref"]), root, where)
    problems: list[str] = []
    if "type" in node and not _type_ok(instance, node["type"]):
        problems.append(f"{where} is {type(instance).__name__}, schema type is {node['type']}")
        return problems
    if "enum" in node and instance not in node["enum"]:
        problems.append(f"{where} value {instance!r} is not one of {node['enum']}")
    if "const" in node and instance != node["const"]:
        problems.append(f"{where} must be {node['const']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in node and instance < node["minimum"]:
            problems.append(f"{where} is {instance}, below the schema minimum {node['minimum']}")
        if "maximum" in node and instance > node["maximum"]:
            problems.append(f"{where} is {instance}, above the schema maximum {node['maximum']}")
    if isinstance(instance, dict):
        required = node.get("required") or []
        for key in required:
            if key not in instance:
                problems.append(f"{where} is missing required {key}")
        properties = node.get("properties") or {}
        if node.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            for key in extra:
                problems.append(f"{where} has property {key} which the schema does not allow")
        for key, value in instance.items():
            if key in properties:
                problems.extend(_schema_errors(value, properties[key], root, f"{where}.{key}"))
    if isinstance(instance, list) and "items" in node:
        if "minItems" in node and len(instance) < node["minItems"]:
            problems.append(f"{where} has {len(instance)} items, schema minimum is {node['minItems']}")
        for index, item in enumerate(instance):
            problems.extend(_schema_errors(item, node["items"], root, f"{where}[{index}]"))
    for branch in node.get("allOf") or []:
        problems.extend(_schema_errors(instance, branch, root, where))
    if "if" in node and not _schema_errors(instance, node["if"], root, where) and "then" in node:
        problems.extend(_schema_errors(instance, node["then"], root, where))
    return problems


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported schema ref {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _type_ok(instance: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_ok(instance, item) for item in expected)
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return False


def _ratio_broken(source_element: dict[str, Any], output_element: dict[str, Any], assets_dir: Optional[Path]) -> bool:
    if source_element.get("fit") == "cover":
        return False
    asset_id = source_element.get("asset_id")
    if assets_dir is not None and asset_id and not (assets_dir / asset_id).is_file():
        return False
    src_h = float(source_element.get("height") or 0)
    out_h = float(output_element.get("height") or 0)
    if src_h <= 0 or out_h <= 0:
        return False
    src_ratio = float(source_element["width"]) / src_h
    out_ratio = float(output_element["width"]) / out_h
    if src_ratio <= 0:
        return False
    return abs(out_ratio - src_ratio) / src_ratio > 0.02
