"""
Orchestrator — you shouldn't need to edit this file. It:
  1. Reads every *.json task file from /app/input (source + target_canvas)
  2. Calls your resize() from resize.py
  3. Runs a quick self-check (structural only — NOT the official scoring;
     see RULES.md for what the judge actually checks)
  4. Writes <task>_output.json and <task>_output.png to /app/output

Mirrors the same input/ -> output/ convention as the earlier India-program
challenges: drop task files in, get one JSON + one PNG out per task, no
manual steps in between.
"""

import json
import os
import sys
import time
from pathlib import Path

from engine.contract import (
    adobe_schema_failures,
    contract_failures,
    preserve_unresolved_fields,
    unresolved_warnings,
)
from engine.resize import resize
from engine.text_context import layout_bridge
from engine.text_context.analyzer import analyze, judgments_for_resize
from engine.text_context.entry import ModelRequiredError, select_backend

# Relative to the current working directory, which is /app both inside the
# Docker image (WORKDIR /app) and when run locally from this repo's root -
# so this works identically in both "python -m engine.main" (no Docker) and
# the containerized entrypoint.
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("brand_kit/assets")

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}


def self_check(source: dict, output: dict) -> list[str]:
    """Quick, non-authoritative sanity check so obvious mistakes are caught
    locally before you submit. This is NOT the official scorer — see
    RULES.md for what actually gates and grades your submission."""
    problems = []

    if "canvas" not in output or "elements" not in output:
        return ["output is missing top-level 'canvas' or 'elements'"]

    src_ids = {el["id"] for el in source["elements"]}
    out_ids = {el.get("id") for el in output["elements"]}
    if src_ids != out_ids:
        problems.append(
            f"element id mismatch — missing {sorted(src_ids - out_ids)}, "
            f"unexpected {sorted(out_ids - src_ids)}"
        )

    cw, ch = output["canvas"].get("width"), output["canvas"].get("height")
    for el in output["elements"]:
        for field in REQUIRED_COMMON:
            if field not in el:
                problems.append(f"element '{el.get('id')}': missing '{field}'")
                continue
        for field in REQUIRED_BY_TYPE.get(el.get("type"), []):
            if field not in el:
                problems.append(f"element '{el.get('id')}' (type={el.get('type')}): missing '{field}'")
        if cw and ch and all(k in el for k in ("x", "y", "width", "height")):
            x1, y1 = el["x"] + el["width"], el["y"] + el["height"]
            if el["x"] < -0.5 or el["y"] < -0.5 or x1 > cw + 0.5 or y1 > ch + 0.5:
                problems.append(f"element '{el.get('id')}' is off-canvas ({cw}x{ch})")

    return problems


class RendererUnavailable(RuntimeError):
    """The real renderer could not be imported. A placeholder PNG is not a result."""


def render_to_png(design: dict, assets_dir: Path, png_path: Path) -> None:
    """Render with the vendored cr-renderer. A missing renderer is a failure.

    This is different from a design that references an asset file the brand
    kit does not contain. The Level 3 holiday watermark is that second case:
    the renderer still runs and draws its own missing-asset mark.
    """
    try:
        from renderer.converter.convert import render_design_to_png  # type: ignore
    except ImportError as exc:
        raise RendererUnavailable(
            "cr-renderer is not importable, so no submission PNG was written. "
            "Install requirements.txt. A labeled-box placeholder is not a render."
        ) from exc
    render_design_to_png(design, assets_dir, str(png_path))


def _artifact_paths(stem: str) -> dict[str, Path]:
    return {
        "json": OUTPUT_DIR / f"{stem}_output.json",
        "png": OUTPUT_DIR / f"{stem}_output.png",
        "sidecar": OUTPUT_DIR / f"{stem}_sidecar.json",
    }


def discard_task_artifacts(stem: str) -> None:
    """Remove a previous run's files so a failure cannot be read as a result."""
    for path in _artifact_paths(stem).values():
        path.unlink(missing_ok=True)
    for path in OUTPUT_DIR.glob(f"{stem}_output.*.partial"):
        path.unlink(missing_ok=True)
    sidecar_partial = OUTPUT_DIR / f"{stem}_sidecar.json.partial"
    sidecar_partial.unlink(missing_ok=True)


def _model_required() -> bool:
    return os.environ.get("CHAMELEON_TEXT_MODE", "auto").strip().lower() == "on"


def _use_model_cache() -> bool:
    return os.environ.get("CHAMELEON_TEXT_NO_CACHE", "").strip().lower() not in {"1", "true", "yes"}


def _status_label(sidecar: dict) -> str:
    status = str(sidecar.get("status") or "no-model")
    if status == "fallback":
        reason = sidecar.get("fallback_reason") or "no model judgment was used"
        return f"fallback: {reason}"
    if status == "no-model":
        label = "no-model"
    elif status == "cached_model":
        label = "cached_model"
    elif status == "model":
        label = "model"
    else:
        label = status
    decision = sidecar.get("layout_decision")
    if decision:
        label = f"{label}; layout {decision}"
        if sidecar.get("offer_moved"):
            label = f"{label}; offer moved"
        if sidecar.get("accent_moved"):
            label = f"{label}; accent moved"
        if sidecar.get("layout_override"):
            return f"{label}; structural slots retained"
        return label
    return label


def _publish(stem: str, output: dict, sidecar: dict) -> float:
    """Render first. Publish JSON, PNG, and sidecar only after that succeeds.

    Returns the render time in seconds. Layout and model time are recorded
    by the caller before this runs.
    """
    paths = _artifact_paths(stem)
    partial_png = OUTPUT_DIR / f"{stem}_output.png.partial"
    partial_json = OUTPUT_DIR / f"{stem}_output.json.partial"
    partial_side = OUTPUT_DIR / f"{stem}_sidecar.json.partial"
    try:
        t_render = time.perf_counter()
        render_to_png(output, ASSETS_DIR, partial_png)
        render_s = time.perf_counter() - t_render
        sidecar.setdefault("phase_s", {})["render"] = round(render_s, 3)
        partial_json.write_text(json.dumps(output, indent=2) + "\n")
        partial_side.write_text(json.dumps(sidecar, indent=2) + "\n")
        partial_json.replace(paths["json"])
        partial_side.replace(paths["sidecar"])
        partial_png.replace(paths["png"])
        return render_s
    except Exception:
        partial_png.unlink(missing_ok=True)
        partial_json.unlink(missing_ok=True)
        partial_side.unlink(missing_ok=True)
        discard_task_artifacts(stem)
        raise


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_files = sorted(INPUT_DIR.glob("*.json"))
    if not task_files:
        print(f"No task files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    # Clear results for every current task before backend selection. A missing
    # or ambiguous required model must not leave a previous run looking valid.
    for task_path in task_files:
        discard_task_artifacts(task_path.stem)

    try:
        backend, backend_kind = select_backend()
    except ModelRequiredError as exc:
        print(f"text-context: {exc}", file=sys.stderr)
        return 2

    required = _model_required()
    exit_code = 0
    for task_path in task_files:
        stem = task_path.stem
        try:
            task = json.loads(task_path.read_text())
        except json.JSONDecodeError as exc:
            exit_code = 1 if exit_code == 0 else exit_code
            print(f"[{task_path.name}] TASK JSON FAILED: {exc}", file=sys.stderr)
            continue
        source, target_canvas = task["source"], task["target_canvas"]
        judgments = None
        sidecar = {"status": "no-model", "backend": backend_kind, "model_judgment": False}
        if backend is not None:
            cache_dir = os.environ.get("CHAMELEON_TEXT_CACHE_DIR")
            analyze_kwargs = {"use_cache": _use_model_cache()}
            if cache_dir:
                analyze_kwargs["cache_dir"] = Path(cache_dir)
            sidecar = analyze(source, target_canvas, backend, **analyze_kwargs)
            sidecar["backend"] = backend_kind
            judgments = judgments_for_resize(sidecar)
            if sidecar.get("status") in {"model", "cached_model"} and not judgments:
                sidecar["status"] = "fallback"
                sidecar["fallback_reason"] = "model status had no usable judgment for every text element"
                sidecar["judgments"] = {}
            sidecar["model_judgment"] = sidecar.get("status") in {"model", "cached_model"} and bool(judgments)
        if required and sidecar.get("status") == "fallback":
            discard_task_artifacts(stem)
            exit_code = 2
            reason = sidecar.get("fallback_reason") or "inference failed"
            errors = [item.get("error") for item in sidecar.get("attempts") or [] if item.get("error")]
            if errors:
                reason = f"{reason}: {errors[-1]}"
            print(
                f"[{task_path.name}] MODEL REQUIRED FAILED: {reason}. "
                "No JSON or PNG was published. A fallback layout is not a successful model result.",
                file=sys.stderr,
            )
            continue
        t_layout = time.perf_counter()
        output = resize(source, target_canvas, judgments=judgments)
        layout_s = time.perf_counter() - t_layout
        preserved = preserve_unresolved_fields(source, output)
        failures = contract_failures(source, output, target_canvas, ASSETS_DIR)
        failures.extend(adobe_schema_failures(output))
        warnings = unresolved_warnings(source, output, ASSETS_DIR)
        warnings.extend(preserved)
        tiny = [
            f"{element['id']} drawn {float(element['font_size']):.1f}px"
            for element in output["elements"]
            if element.get("type") == "text" and float(element.get("font_size") or 0) < 8
        ]
        if tiny:
            warnings.append(
                "text under 8 px remains (" + ", ".join(tiny) + "); "
                "Adobe has not set a minimum size, so this is a reported trade-off, not a hidden pass"
            )
        if failures:
            discard_task_artifacts(stem)
            exit_code = 2 if exit_code == 2 else 1
            print(
                f"[{task_path.name}] CONTRACT FAIL after layout {layout_s:.2f}s. "
                "No JSON or PNG was published.",
                file=sys.stderr,
            )
            for problem in failures:
                print(f"    contract: {problem}", file=sys.stderr)
            continue
        sidecar["layout_decision"] = layout_bridge.last_decision
        sidecar["offer_moved"] = bool(layout_bridge.last_offer_moved)
        sidecar["accent_moved"] = bool(layout_bridge.last_accent_moved)
        sidecar["layout_override"] = layout_bridge.last_layout_override
        sidecar["preserved_unresolved"] = preserved
        sidecar["unresolved_warnings"] = warnings
        attempt = (sidecar.get("attempts") or [{}])[-1]
        stats = attempt.get("stats") or {}
        sidecar["phase_s"] = {
            "hash": stats.get("hash_s"),
            "load": stats.get("load_s"),
            "generation": stats.get("wall_s"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "output_tokens": stats.get("output_tokens"),
            "layout": round(layout_s, 3),
        }
        try:
            render_s = _publish(stem, output, sidecar)
        except RendererUnavailable as exc:
            exit_code = 2 if exit_code == 2 else 1
            print(f"[{task_path.name}] RENDER FAILED: {exc}", file=sys.stderr)
            continue

        label = _status_label(sidecar)
        print(
            f"[{task_path.name}] {label} "
            f"layout {layout_s:.2f}s render {render_s:.2f}s "
            f"-> {stem}_output.json, {stem}_output.png, {stem}_sidecar.json"
        )
        for problem in warnings:
            print(f"    unresolved: {problem}")

    if backend is None:
        print("text-context: no-model path (no packaged GGUF, Ollama not requested)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
