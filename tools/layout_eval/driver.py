"""Run one engine checkout on one task, inside that checkout.

Invoked as a script by `run.py` with the engine checkout as the working
directory. Both engines resolve fonts (and the newer one, background
assets) from `Path(__file__)`, and both silently fall back to estimates
when that lookup fails, so this driver imports `engine.resize` from the
checkout itself and records every file opened while resizing and
rendering. `run.py` rejects a run whose fonts or assets came from
anywhere else.

Stdlib plus the engine's own dependencies only: this file must not
import the harness package, because the checkout is first on sys.path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

_OPENS: list[dict] = []
_PHASE = ["startup"]


def _audit(event: str, args: tuple) -> None:
    if event != "open" or not args:
        return
    target = args[0]
    if isinstance(target, int):
        return
    try:
        path = os.fsdecode(target)
    except TypeError:
        return
    _OPENS.append({"phase": _PHASE[0], "path": os.path.abspath(path)})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_metrics(design: dict, fonts_dir: str) -> dict:
    """Rendered text extents computed with the renderer's own wrap and
    skia measurement. Glyphs are drawn at int(font_size) while wrap
    widths are measured at the float size (renderer/converter/convert.py,
    cr_renderer/text_utils.py), so both are recorded."""
    import skia  # type: ignore
    from renderer.converter import convert as conv  # type: ignore
    from renderer.converter.wrap import char_widths, wrap_text  # type: ignore

    fm = conv._get_renderer(fonts_dir).font_manager
    out = {}
    for el in design["elements"]:
        if el.get("type") != "text":
            continue
        content = str(el.get("content", ""))
        family = el.get("font_family", "Montserrat")
        size = float(el.get("font_size", 24))
        weight = conv._normalize_weight(el.get("font_weight"))
        is_bold = weight in conv._BOLD_WEIGHTS
        is_italic = conv._is_italic(el.get("font_style") or el.get("italic"))
        style = conv._weight_style_to_font_style(is_bold, is_italic)
        widths = char_widths(fm, family, weight, style, size, content)
        box_w = float(el.get("width", 1.0))
        wrapped = wrap_text(content, widths, box_w)
        lines = wrapped.split("\n")
        drawn_size = int(size)
        font_weight = "bold" if is_bold else "regular"
        typeface = skia.Typeface.MakeFromData(fm.lookup(family, font_weight, style))
        font = skia.Font(typeface, drawn_size)
        spacing = float(el.get("letter_spacing", 0.0))
        align = el.get("text_align", "left")
        line_widths = []
        offsets = []
        for line in lines:
            text = line.upper() if el.get("capitalize") else line
            glyphs = font.textToGlyphs(text)
            lw = sum(font.getWidths(glyphs)) + spacing * max(0, len(glyphs) - 1)
            line_widths.append(lw)
            if align == "center":
                offsets.append((box_w - lw) / 2.0)
            elif align == "right":
                offsets.append(box_w - lw)
            else:
                offsets.append(0.0)
        metrics = font.getMetrics()
        line_height = float(el.get("line_height", 1.2))
        block_h = (len(lines) - 1) * size * line_height + (metrics.fDescent - metrics.fAscent)
        x0 = min(offsets) if offsets else 0.0
        x1 = max(o + w for o, w in zip(offsets, line_widths)) if offsets else 0.0
        out[el["id"]] = {
            "font_size": size,
            "drawn_px": drawn_size,
            "lines": len(lines),
            "max_line_width": max(line_widths) if line_widths else 0.0,
            "block_height": block_h,
            "local_extent": [x0, 0.0, x1, block_h],
            "box": [float(el["x"]), float(el["y"]), box_w, float(el.get("height", 1.0))],
            "style_runs_ignored": bool(el.get("style_runs")),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkout = Path(args.checkout).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.addaudithook(_audit)

    result: dict = {"status": "ok", "checkout": str(checkout), "out_dir": str(out_dir), "errors": []}
    case_path = Path(args.case).resolve()
    raw_before = case_path.read_bytes()
    task = json.loads(raw_before)
    source, target = task["source"], task["target_canvas"]
    pristine = copy.deepcopy(source)

    _PHASE[0] = "import"
    import engine.resize as engine_resize  # type: ignore

    result["engine_module"] = str(Path(engine_resize.__file__).resolve())
    result["engine_sha256"] = _sha256(Path(engine_resize.__file__))
    fonts_dir = Path(getattr(engine_resize, "_FONTS_DIR", checkout / "brand_kit" / "fonts")).resolve()
    assets_attr = getattr(engine_resize, "_ASSETS_DIR", None)
    result["engine_fonts_dir"] = str(fonts_dir)
    result["engine_assets_dir"] = str(Path(assets_attr).resolve()) if assets_attr else None

    _PHASE[0] = "resize"
    output = None
    t0 = time.perf_counter()
    try:
        output = engine_resize.resize(source, target)
    except Exception:
        result["status"] = "engine_error"
        result["errors"].append(traceback.format_exc())
    result["resize_seconds"] = time.perf_counter() - t0
    result["font_manager_loaded"] = (
        getattr(engine_resize, "_font_manager", None) is not None
        and not getattr(engine_resize, "_font_manager_failed", False)
    )
    result["input_mutated_in_memory"] = source != pristine

    assets_dir = checkout / "brand_kit" / "assets"
    if output is not None:
        _PHASE[0] = "write"
        (out_dir / "output.json").write_text(json.dumps(output, indent=2) + "\n")
        _PHASE[0] = "render_output"
        try:
            from renderer.converter.convert import render_design_to_png  # type: ignore
            import renderer.converter.convert as conv_mod  # type: ignore

            result["renderer_module"] = str(Path(conv_mod.__file__).resolve())
            result["render_warnings"] = render_design_to_png(output, assets_dir, out_dir / "output.png")
        except Exception:
            result["status"] = "render_error"
            result["errors"].append(traceback.format_exc())

    _PHASE[0] = "render_source"
    try:
        from renderer.converter.convert import render_design_to_png  # type: ignore

        result["source_render_warnings"] = render_design_to_png(pristine, assets_dir, out_dir / "source.png")
    except Exception:
        result["errors"].append(traceback.format_exc())

    _PHASE[0] = "metrics"
    try:
        result["source_text_metrics"] = _text_metrics(pristine, str(fonts_dir))
        if output is not None:
            result["output_text_metrics"] = _text_metrics(output, str(fonts_dir))
        from PIL import Image

        native = {}
        for el in pristine["elements"]:
            aid = el.get("asset_id")
            if el.get("type") == "image" and aid and aid not in native:
                p = assets_dir / aid
                if p.is_file():
                    with Image.open(p) as im:
                        native[aid] = {"width": im.width, "height": im.height, "sha256": _sha256(p)}
                else:
                    native[aid] = None
        result["asset_native"] = native
    except Exception:
        result["errors"].append(traceback.format_exc())

    result["input_file_unchanged"] = case_path.read_bytes() == raw_before
    _PHASE[0] = "done"
    code_suffixes = (".py", ".pyc", ".so", ".pth")
    result["opens"] = [
        o for o in _OPENS
        if o["phase"] in ("import", "resize", "render_output", "render_source")
        and not o["path"].endswith(code_suffixes)
        and not os.path.isdir(o["path"])
    ]
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
