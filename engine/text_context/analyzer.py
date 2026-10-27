"""One whole-design text judgment per (source design, target canvas).

`analyze()` returns a sidecar dict. Its `status` says where the judgments
came from:

- "model": a fresh, validated reply from the local model.
- "cached_model": a validated model reply loaded from the cache.
- "fallback": no usable model reply; `judgments` is empty and the layout
  engine must run its own baseline classification. This is never labelled
  as a model judgment.
- "no-text": the design contains no text, so there is nothing for the model
  to judge. The layout engine may still resize its other elements.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from engine.text_context import facts, prompts, schema
from engine.text_context.backends import Backend, BackendError

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "text_context"
MAX_ATTEMPTS = 3


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(source: dict, target_canvas: dict, identity: dict, extra_context: Optional[str]) -> str:
    return _sha(_canonical({
        "source": source,
        "target": {"width": target_canvas["width"], "height": target_canvas["height"]},
        "model": identity.get("model"),
        "digest": identity.get("digest"),
        "options": identity.get("options"),
        "prompt": prompts.PROMPT_VERSION,
        "system_sha": _sha(prompts.SYSTEM),
        "schema": schema.SCHEMA_VERSION,
        "consistency": "role-must-read-v1",
        "extra_context": extra_context,
    }))


def _judgment_usable(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        row.get("role") in schema.ROLES
        and row.get("prominence") in schema.PROMINENCE
        and isinstance(row.get("must_read"), bool)
    )


def cached_sidecar_usable(
    cached: Any,
    *,
    source: dict[str, Any],
    text_ids: list[str],
    identity: dict[str, Any],
) -> tuple[bool, str]:
    """A cache hit needs the same design, prompt, and one judgment per text.

    The filename is not enough. An empty or corrupt file used to be relabeled
    `cached_model`, and a required-model run then published the baseline.
    """
    if not isinstance(cached, dict):
        return False, "cache file is not a JSON object"
    if cached.get("status") != "model":
        return False, f"cache status is {cached.get('status')!r}, not a validated model reply"
    if cached.get("schema_version") != schema.SCHEMA_VERSION or cached.get("prompt_version") != prompts.PROMPT_VERSION:
        return False, "cache was written by a different prompt or schema"
    if cached.get("source_sha256") != _sha(_canonical(source)):
        return False, "cache source checksum does not match this design"
    model = cached.get("model") if isinstance(cached.get("model"), dict) else {}
    if identity.get("digest") and model.get("digest") != identity.get("digest"):
        return False, "cache model digest does not match the backend"
    if cached.get("text_ids") != text_ids:
        return False, "cache text ids do not match this design"
    judgments = cached.get("judgments")
    if not isinstance(judgments, dict) or set(judgments) != set(text_ids) or len(judgments) != len(text_ids):
        return False, "cache does not have exactly one judgment for every current text id"
    by_id = {element["id"]: element for element in source.get("elements", [])}
    for eid in text_ids:
        row = judgments.get(eid)
        if not _judgment_usable(row):
            return False, f"cache judgment for {eid} is not usable"
        if row.get("content") != by_id.get(eid, {}).get("content", ""):
            return False, f"cache copy for {eid} does not match the design"
    return True, ""


def _write_cache(cache_path: Path, sidecar: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    temporary.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, cache_path)


def analyze(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    backend: Optional[Backend],
    *,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    extra_context: Optional[str] = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    started = time.perf_counter()
    ctx = facts.build_context(source, target_canvas, extra_context)
    text_ids = [ctx["id_of"][a] for a in ctx["text_aliases"]]
    sidecar: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "status": "fallback",
        "fallback_reason": None,
        "model": None,
        "target_canvas": {"width": target_canvas["width"], "height": target_canvas["height"]},
        "source_sha256": _sha(_canonical(source)),
        "text_ids": text_ids,
        "aliases": ctx["id_of"],
        "attempts": [],
        "warnings": [],
        "judgments": {},
    }
    if not text_ids:
        sidecar["status"] = "no-text"
        sidecar["latency_s"] = round(time.perf_counter() - started, 3)
        return sidecar
    if backend is None:
        sidecar["fallback_reason"] = "no local model backend configured"
        sidecar["latency_s"] = round(time.perf_counter() - started, 3)
        return sidecar

    try:
        identity = backend.identity()
    except BackendError as exc:
        sidecar["fallback_reason"] = f"model unavailable: {exc}"
        sidecar["latency_s"] = round(time.perf_counter() - started, 3)
        return sidecar
    sidecar["model"] = identity
    key = cache_key(source, target_canvas, identity, extra_context)
    sidecar["cache_key"] = key
    cache_path = (cache_dir / f"{key}.json") if cache_dir else None
    if use_cache and cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sidecar["warnings"].append(f"cached judgment was unusable ({exc}); recomputing")
            cache_path.unlink(missing_ok=True)
            cached = None
        if cached is not None:
            usable, reason = cached_sidecar_usable(cached, source=source, text_ids=text_ids, identity=identity)
            if usable:
                _repair_cached_judgments(cached)
                cached["status"] = "cached_model"
                cached["cache_hit_latency_s"] = round(time.perf_counter() - started, 3)
                return cached
            sidecar["warnings"].append(f"cached judgment was unusable ({reason}); recomputing")
            cache_path.unlink(missing_ok=True)

    response_schema = schema.response_schema(ctx["text_aliases"], ctx["all_aliases"])
    user = prompts.user_message(ctx["prompt"], ctx["text_aliases"])
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        record: dict[str, Any] = {"attempt": attempt}
        try:
            if attempt > 1 and hasattr(backend, "options"):
                backend.options["seed"] = int(identity["options"].get("seed", 0)) + attempt - 1
            raw, stats = backend.generate(prompts.SYSTEM, user + feedback, response_schema)
            record["stats"] = stats
            record["raw"] = raw
            if stats.get("done_reason") == "length":
                raise schema.JudgmentError("reply hit the output token limit")
            payload = schema.parse_reply(raw)
            by_alias, warnings = schema.validate(payload, ctx["text_aliases"], ctx["all_aliases"])
        except (schema.JudgmentError, BackendError) as exc:
            record["ok"] = False
            record["error"] = str(exc)
            sidecar["attempts"].append(record)
            feedback = f"\n\nYour previous reply was rejected: {exc}. Reply again with valid JSON."
            if isinstance(exc, BackendError) and "unreachable" in str(exc):
                break
            continue
        finally:
            if hasattr(backend, "options") and identity.get("options"):
                backend.options["seed"] = identity["options"].get("seed")
        record["ok"] = True
        sidecar["attempts"].append(record)
        sidecar["warnings"] = list(sidecar.get("warnings") or []) + list(warnings)
        by_id = {}
        for alias, j in by_alias.items():
            eid = ctx["id_of"][alias]
            el = next(e for e in source["elements"] if e["id"] == eid)
            by_id[eid] = dict(j, alias=alias, content=el.get("content", ""),
                              related=[{"id": ctx["id_of"][r["id"]], "relation": r["relation"]} for r in j["related"]])
        sidecar["judgments"] = by_id
        sidecar["status"] = "model"
        sidecar["latency_s"] = round(time.perf_counter() - started, 3)
        if cache_path:
            _write_cache(cache_path, sidecar)
        return sidecar

    sidecar["fallback_reason"] = f"no valid model reply after {len(sidecar['attempts'])} attempt(s)"
    sidecar["latency_s"] = round(time.perf_counter() - started, 3)
    return sidecar


def _repair_cached_judgments(cached: dict[str, Any]) -> None:
    """A cache file from before a consistency rule still has to obey it."""
    notes: list[str] = []
    judgments = cached.get("judgments")
    if not isinstance(judgments, dict):
        return
    for eid, row in judgments.items():
        if not isinstance(row, dict) or not isinstance(row.get("must_read"), bool):
            continue
        fixed, note = schema.repair_must_read(str(row.get("role") or ""), row["must_read"])
        if note:
            row["must_read"] = fixed
            notes.append(f"{eid}: {note}")
    if notes:
        cached.setdefault("warnings", [])
        if isinstance(cached["warnings"], list):
            cached["warnings"].extend(notes)
        cached["judgment_corrections"] = notes


def judgments_for_resize(sidecar: dict[str, Any]) -> Optional[dict[str, dict[str, Any]]]:
    """The mapping resize() accepts, or None when the engine must use its
    baseline (a fallback is never passed on as if it were a judgment)."""
    if sidecar.get("status") not in ("model", "cached_model"):
        return None
    judgments = sidecar.get("judgments")
    if not isinstance(judgments, dict) or not judgments:
        return None
    text_ids = sidecar.get("text_ids")
    if isinstance(text_ids, list) and set(judgments) != set(text_ids):
        return None
    if any(not _judgment_usable(row) for row in judgments.values()):
        return None
    return judgments


def default_backend() -> Backend:
    from engine.text_context.backends import OllamaBackend

    return OllamaBackend(model=os.environ.get("CHAMELEON_TEXT_MODEL"))
