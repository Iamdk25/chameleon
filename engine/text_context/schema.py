"""Judgment vocabulary, the JSON schema sent to the model, and reply validation.

Validation is mechanical: it checks structure, enums, and element ids. It
never assigns a role or a prominence itself.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = "text-judgment-v1"

ROLES = (
    "headline",
    "subheadline",
    "product_name",
    "brand_name",
    "price",
    "discount",
    "date_time",
    "call_to_action",
    "body",
    "caption",
    "badge",
    "disclosure",
    "other",
)

# Ordered from most to least visually dominant on the requested target.
PROMINENCE = ("primary", "high", "medium", "low", "minimal")

RELATIONS = ("supports", "describes", "labels", "price_of", "condition_of", "pairs_with")

UNCERTAINTY = ("low", "medium", "high")


MUST_READ_TRUE = frozenset({"disclosure", "price", "date_time", "call_to_action"})
MUST_READ_FALSE = frozenset({"badge"})


def repair_must_read(role: str, must_read: bool) -> tuple[bool, str]:
    """Apply the prompt's role and must_read rule. Returns (value, note).

    The note is empty when the model was already consistent. This does not
    look at the words. The model still chooses the role.
    """
    if role in MUST_READ_TRUE and must_read is not True:
        return True, f"role {role} requires must_read true"
    if role in MUST_READ_FALSE and must_read is not False:
        return False, f"role {role} is decorative, so must_read is false"
    return must_read, ""


class JudgmentError(ValueError):
    """The model reply cannot be used as a judgment for this design."""


def response_schema(text_aliases: list[str], all_aliases: list[str]) -> dict[str, Any]:
    """JSON schema for one whole-design reply. Ollama turns this into a
    grammar, so ids outside the design cannot be generated at all."""
    judgment = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": text_aliases},
            "role": {"type": "string", "enum": list(ROLES)},
            "prominence": {"type": "string", "enum": list(PROMINENCE)},
            "must_read": {"type": "boolean"},
            "related": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": all_aliases},
                        "relation": {"type": "string", "enum": list(RELATIONS)},
                    },
                    "required": ["id", "relation"],
                },
            },
            "uncertainty": {"type": "string", "enum": list(UNCERTAINTY)},
            "why": {"type": "string"},
        },
        "required": ["id", "role", "prominence", "must_read", "related", "uncertainty", "why"],
    }
    return {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": len(text_aliases),
                "maxItems": len(text_aliases),
                "items": judgment,
            }
        },
        "required": ["judgments"],
    }


def parse_reply(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgmentError(f"reply is not JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise JudgmentError("reply must be a JSON object")
    return value


def validate(
    payload: dict[str, Any], text_aliases: list[str], all_aliases: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return (alias -> judgment, warnings). Raises JudgmentError if the reply
    does not have exactly one well-formed judgment per text alias."""
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        raise JudgmentError("missing 'judgments' list")
    warnings: list[str] = []
    found: dict[str, dict[str, Any]] = {}
    known = set(all_aliases)
    for row in rows:
        if not isinstance(row, dict):
            raise JudgmentError("each judgment must be an object")
        alias = row.get("id")
        if alias not in text_aliases:
            raise JudgmentError(f"judgment for unknown or non-text id {alias!r}")
        if alias in found:
            raise JudgmentError(f"duplicate judgment for {alias}")
        role = _enum(row, "role", ROLES)
        prominence = _enum(row, "prominence", PROMINENCE)
        uncertainty = _enum(row, "uncertainty", UNCERTAINTY)
        must_read = row.get("must_read")
        if not isinstance(must_read, bool):
            raise JudgmentError(f"{alias}.must_read must be true or false")
        must_read, note = repair_must_read(role, must_read)
        if note:
            warnings.append(f"{alias}: {note}")
        related_in = row.get("related", [])
        if not isinstance(related_in, list):
            raise JudgmentError(f"{alias}.related must be a list")
        related = []
        for rel in related_in:
            if not isinstance(rel, dict):
                raise JudgmentError(f"{alias}.related entries must be objects")
            target = rel.get("id")
            if target not in known:
                raise JudgmentError(f"{alias} relates to unknown id {target!r}")
            relation = _enum(rel, "relation", RELATIONS, where=f"{alias}.related")
            if target == alias:
                warnings.append(f"{alias} related to itself; dropped")
                continue
            related.append({"id": target, "relation": relation})
        why = row.get("why", "")
        found[alias] = {
            "role": role,
            "prominence": prominence,
            "must_read": must_read,
            "related": related,
            "uncertainty": uncertainty,
            "why": why if isinstance(why, str) else "",
        }
    missing = [a for a in text_aliases if a not in found]
    if missing:
        raise JudgmentError("no judgment for " + ", ".join(missing))
    primaries = [a for a, j in found.items() if j["prominence"] == "primary"]
    if len(primaries) > 2:
        warnings.append(f"{len(primaries)} texts marked primary: {', '.join(primaries)}")
    return found, warnings


def _enum(row: dict[str, Any], key: str, allowed: tuple[str, ...], where: str = "") -> str:
    value = row.get(key)
    if not isinstance(value, str) or value not in allowed:
        label = f"{where}.{key}" if where else f"{row.get('id')}.{key}"
        raise JudgmentError(f"{label}={value!r} is not one of {', '.join(allowed)}")
    return value
