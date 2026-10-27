"""Turn validated text judgments into the slots and priorities resize() uses.

The model chooses a role, a prominence, and whether a line must stay
readable. This module maps those fields onto the engine's existing slots
(one headline, body lines, one disclosure, one call to action, badges).
It does not measure fonts or pick coordinates, and it does not rewrite
the model's role. A second action can still sit in the badge slot; that
is a layout choice, not a change to the judgment.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.text_context.schema import PROMINENCE, ROLES

Element = dict[str, Any]
PillGroup = tuple[str, list[Element]]

# Visual weight passed to the engine's existing importance math.
# Kept separate from must_read: a disclosure can be "low" here and still
# be protected by the legibility floor.
PROMINENCE_IMPORTANCE = {
    "primary": 0.95,
    "high": 0.78,
    "medium": 0.52,
    "low": 0.30,
    "minimal": 0.16,
}

# Roles that must stay readable even when they are not the focal line.
_PROTECTED_ROLES = ("disclosure", "price", "date_time")

_PROM_RANK = {name: index for index, name in enumerate(PROMINENCE)}
_TEXT_CTA = "__text_cta__"

# Roles that carry the message. A badge can be lively without being one of
# these; the 8 px floor is reserved for the lines a reader has to understand.
_ESSENTIAL_ROLES = (
    "headline", "subheadline", "product_name", "price",
    "call_to_action", "disclosure", "date_time",
)


@dataclass
class BridgeContext:
    """Per-call layout state. A ContextVar so two resize() calls in
    different threads cannot overwrite each other's judgments."""

    must_read_ids: set[str] = field(default_factory=set)
    protected_ids: set[str] = field(default_factory=set)
    promoted_ids: set[str] = field(default_factory=set)
    importance_by_id: dict[str, float] = field(default_factory=dict)
    guidance: Optional["TextLayoutGuidance"] = None
    decision: str = "baseline"
    offer_moved: bool = False
    accent_moved: bool = False
    layout_override: Optional[dict[str, Any]] = None


_active: contextvars.ContextVar[BridgeContext] = contextvars.ContextVar("chameleon_text_bridge_active")
_published: contextvars.ContextVar[BridgeContext] = contextvars.ContextVar("chameleon_text_bridge_published")


def current() -> BridgeContext:
    try:
        return _active.get()
    except LookupError:
        ctx = BridgeContext()
        _active.set(ctx)
        return ctx


def bind() -> tuple[contextvars.Token, BridgeContext]:
    ctx = BridgeContext()
    return _active.set(ctx), ctx


def unbind(token: contextvars.Token) -> None:
    _active.reset(token)


def publish(ctx: BridgeContext) -> None:
    _published.set(ctx)


def published() -> BridgeContext:
    try:
        return _published.get()
    except LookupError:
        return BridgeContext()


def __getattr__(name: str) -> Any:
    """Reads used by resize() see the in-flight call. Reads after resize()
    returns see the published decision and guidance."""
    active_fields = {
        "active_must_read_ids": "must_read_ids",
        "active_protected_ids": "protected_ids",
        "active_promoted_ids": "promoted_ids",
        "active_importance_by_id": "importance_by_id",
    }
    if name in active_fields:
        return getattr(current(), active_fields[name])
    if name == "last_guidance":
        return published().guidance
    if name == "last_decision":
        return published().decision
    if name == "last_offer_moved":
        return published().offer_moved
    if name == "last_accent_moved":
        return published().accent_moved
    if name == "last_layout_override":
        return published().layout_override
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class TextLayoutGuidance:
    headline: Optional[Element]
    body_texts: list[Element]
    legal: Optional[Element]
    primary_cta: Optional[PillGroup]
    badges: list[PillGroup]
    must_read_ids: set[str] = field(default_factory=set)
    protected_ids: set[str] = field(default_factory=set)
    essential_ids: set[str] = field(default_factory=set)
    promoted_ids: set[str] = field(default_factory=set)
    importance_by_id: dict[str, float] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    offer_ids: set[str] = field(default_factory=set)
    dropped_structural_slots: bool = False
    heuristic_headline_id: Optional[str] = None
    heuristic_legal_id: Optional[str] = None
    layout_override: Optional[dict[str, Any]] = None
    solo_rotated_action_ids: set[str] = field(default_factory=set)

    def summary(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "must_read_ids": sorted(self.must_read_ids),
            "essential_ids": sorted(self.essential_ids),
            "importance_by_id": {k: self.importance_by_id[k] for k in sorted(self.importance_by_id)},
            "ignored": self.ignored,
        }


def apply_text_judgments(
    judgments: dict[str, dict[str, Any]],
    ungrouped_texts: list[Element],
    pill_groups: list[PillGroup],
    ribbon_groups: list[PillGroup],
    headline: Optional[Element],
    legal: Optional[Element],
    primary_cta: Optional[PillGroup],
    badges: list[PillGroup],
    *,
    retain_structural_slots: bool = False,
) -> TextLayoutGuidance:
    """Reassign engine slots from judgments. Grouped labels stay inside their
    group; only the choice of which group is the call to action can change.

    `retain_structural_slots` puts the source headline and disclosure elements
    back into those slots when the model roles left both empty. It does not
    rewrite the judgment. The caller records that choice separately.
    """
    by_id: dict[str, Element] = {e["id"]: e for e in ungrouped_texts}
    for _, members in list(pill_groups) + list(ribbon_groups):
        for member in members:
            if member.get("type") == "text":
                by_id[member["id"]] = member
    usable, ignored = _usable(judgments, by_id)

    heuristic_headline = headline
    heuristic_legal = legal
    headline = _pick_headline(usable, ungrouped_texts, by_id, headline)
    legal = _pick_legal(usable, ungrouped_texts, by_id, headline, legal)
    dropped = (
        heuristic_headline is not None
        and heuristic_legal is not None
        and heuristic_headline.get("id") != heuristic_legal.get("id")
        and headline is None
        and legal is None
    )
    override = None
    if retain_structural_slots and dropped:
        headline = heuristic_headline
        legal = heuristic_legal
        override = {
            "kind": "structural_slots",
            "headline_id": heuristic_headline["id"],
            "legal_id": heuristic_legal["id"],
            "model_role": {
                heuristic_headline["id"]: (usable.get(heuristic_headline["id"]) or {}).get("role"),
                heuristic_legal["id"]: (usable.get(heuristic_legal["id"]) or {}).get("role"),
            },
        }
    loose_cta = _pick_loose_cta(usable, ungrouped_texts, headline, legal)
    primary_cta, badges, loose_used = _pick_cta(usable, pill_groups, primary_cta, badges, loose_cta)

    consumed = {e["id"] for e in (headline, legal) if e is not None}
    if loose_used is not None:
        consumed.add(loose_used["id"])
    body = sorted((e for e in ungrouped_texts if e["id"] not in consumed), key=lambda e: float(e["y"]))

    slots: dict[str, str] = {}
    if headline is not None:
        slots[headline["id"]] = "headline"
    for element in body:
        slots[element["id"]] = "body"
    if legal is not None:
        slots[legal["id"]] = "disclosure"
    if primary_cta is not None:
        label = _label(primary_cta[1])
        if label is not None:
            slots[label["id"]] = "call_to_action"
    for _, members in badges:
        label = _label(members)
        if label is not None:
            slots[label["id"]] = "badge"
    for _, members in ribbon_groups:
        label = _label(members)
        if label is not None and label["id"] not in slots:
            slots[label["id"]] = "badge"

    return TextLayoutGuidance(
        headline=headline,
        body_texts=body,
        legal=legal,
        primary_cta=primary_cta,
        badges=badges,
        must_read_ids={eid for eid, row in usable.items() if row.get("must_read") is True},
        protected_ids={
            eid for eid, row in usable.items()
            if row.get("must_read") is True and row["role"] in _PROTECTED_ROLES
        },
        essential_ids={
            eid for eid, row in usable.items()
            if slots.get(eid) != "badge"
            and (
                row["role"] in _ESSENTIAL_ROLES
                or (row["role"] == "body" and _PROM_RANK[row["prominence"]] <= _PROM_RANK["medium"])
            )
        },
        promoted_ids=_promoted_ids(ungrouped_texts, headline, primary_cta),
        importance_by_id=_importance_for_slots(usable, slots, override),
        ignored=ignored,
        slots=slots,
        offer_ids={
            eid for eid, row in usable.items()
            if row.get("role") == "discount" and row.get("prominence") in {"primary", "high"}
        },
        dropped_structural_slots=dropped,
        heuristic_headline_id=heuristic_headline["id"] if heuristic_headline is not None else None,
        heuristic_legal_id=heuristic_legal["id"] if heuristic_legal is not None else None,
        layout_override=override,
        solo_rotated_action_ids=_solo_rotated_action_ids(
            usable, ribbon_groups, pill_groups, loose_used,
        ),
    )


def overlay_importances(
    importances: dict[str, float],
    importance_by_id: dict[str, float],
    headline: Optional[Element],
    body_texts: list[Element],
    legal: Optional[Element],
    primary_cta: Optional[PillGroup],
    badges: list[PillGroup],
) -> dict[str, float]:
    """Replace heuristic text priorities with the model's prominence.

    Image priorities (logo, hero) stay on the engine's area heuristic.
    A text with no usable judgment keeps the value already in `importances`.
    """
    updated = dict(importances)

    def put(key: str, element: Optional[Element]) -> None:
        if element is not None and element.get("id") in importance_by_id and key in updated:
            updated[key] = importance_by_id[element["id"]]

    put("headline", headline)
    for index, element in enumerate(body_texts):
        put(f"body_{index}", element)
    put("legal", legal)
    if primary_cta is not None:
        put("cta", _label(primary_cta[1]))
    for index, (_, members) in enumerate(badges):
        put(f"badge_{index}", _label(members))
    return updated


def judged_importance(members: list[Element], default: float) -> float:
    """Importance of a ribbon (or other group) from its label, else `default`."""
    label = _label(members)
    if label is None:
        return default
    return float(current().importance_by_id.get(label["id"], default))


def _usable(
    judgments: dict[str, dict[str, Any]], by_id: dict[str, Element],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    usable: dict[str, dict[str, Any]] = {}
    ignored: list[str] = []
    for eid, row in judgments.items():
        element = by_id.get(eid)
        if element is None:
            ignored.append(f"{eid}: no such text element")
            continue
        if row.get("content") != element.get("content"):
            ignored.append(f"{eid}: judgment content does not match the source text")
            continue
        if row.get("uncertainty") == "high":
            ignored.append(f"{eid}: uncertainty is high, so the heuristic is kept")
            continue
        if row.get("role") not in ROLES or row.get("prominence") not in PROMINENCE_IMPORTANCE:
            ignored.append(f"{eid}: role or prominence is missing")
            continue
        usable[eid] = row
    return usable, ignored


def _pick_headline(usable, ungrouped, by_id, heuristic):
    found = [e["id"] for e in ungrouped if usable.get(e["id"], {}).get("role") == "headline"]
    if not found:
        # No line was called the headline. A product name is the main line
        # when the model did not rank a price above it. A louder price stays
        # the offer: the product name remains supporting, and a tie still
        # lets the product name lead. This is the model's prominence, not a
        # rule that every price is the headline.
        names = [e["id"] for e in ungrouped if usable.get(e["id"], {}).get("role") == "product_name"]
        if names and not _price_outranks(usable, ungrouped, names):
            found = names
    if found:
        return by_id[max(found, key=lambda eid: _rank(usable[eid]))]
    if heuristic is not None and heuristic["id"] in usable and usable[heuristic["id"]]["role"] not in ("headline", "other"):
        return None
    return heuristic


def _pick_legal(usable, ungrouped, by_id, headline, heuristic):
    headline_id = headline["id"] if headline is not None else None
    found = [
        e["id"] for e in ungrouped
        if e["id"] != headline_id and usable.get(e["id"], {}).get("role") == "disclosure"
    ]
    if found:
        def rank(eid: str) -> tuple:
            row = usable[eid]
            return (1 if row.get("must_read") else 0, -_PROM_RANK[row["prominence"]])

        return by_id[max(found, key=rank)]
    if heuristic is not None and heuristic["id"] in usable and usable[heuristic["id"]]["role"] not in ("disclosure", "other"):
        return None
    return heuristic


def _pick_loose_cta(usable, ungrouped, headline, legal):
    skip = {e["id"] for e in (headline, legal) if e is not None}
    found = [
        e for e in ungrouped
        if e["id"] not in skip and usable.get(e["id"], {}).get("role") == "call_to_action"
    ]
    if not found:
        return None
    return max(found, key=lambda e: _rank(usable[e["id"]]))


def _pick_cta(usable, pill_groups, primary_cta, badges, loose_cta):
    """Returns (primary_cta, badges, loose element consumed as the CTA or None)."""
    chosen = []
    rest = []
    for group in pill_groups:
        label = _label(group[1])
        row = usable.get(label["id"]) if label is not None else None
        if row is not None and row["role"] == "call_to_action":
            chosen.append(group)
        else:
            rest.append(group)
    if chosen:
        # The largest pill is already the engine's button. When the model
        # also calls that pill a call to action, a smaller pill with a
        # louder prominence does not take the button. Both labels can be
        # misread as actions; the bigger grouped control stays the action.
        if primary_cta is not None and primary_cta in chosen:
            others = [group for group in pill_groups if group != primary_cta]
            return primary_cta, others, None
        chosen.sort(key=lambda group: _rank(usable[_label(group[1])["id"]]), reverse=True)
        return chosen[0], chosen[1:] + rest, None
    if loose_cta is not None and primary_cta is None:
        return (_TEXT_CTA, [loose_cta]), badges, loose_cta
    return primary_cta, badges, None


def _solo_rotated_action_ids(usable, ribbon_groups, pill_groups, loose_used) -> set[str]:
    """One rotated group the model ranked as the only high action.

    The role stays call_to_action. A badge, a second action, and a rotated
    line the model did not rank high or primary are left out. The layout
    then decides whether the canvas has room to draw that accent larger.
    """
    if loose_used is not None:
        return set()
    for _, members in pill_groups:
        label = _label(members)
        row = usable.get(label["id"]) if label is not None else None
        if row is not None and row.get("role") == "call_to_action":
            return set()
    found = []
    for _, members in ribbon_groups:
        label = _label(members)
        if label is None:
            continue
        row = usable.get(label["id"])
        if row is None or row.get("role") != "call_to_action":
            continue
        if row.get("prominence") not in {"primary", "high"}:
            continue
        found.append(label["id"])
    if len(found) != 1:
        return set()
    return {found[0]}


def _importance_for_slots(
    usable: dict[str, dict[str, Any]], slots: dict[str, str], override: Optional[dict[str, Any]],
) -> dict[str, float]:
    """Model prominence, except on slots the source structure kept.

    A restored headline or disclosure keeps the engine's own weight. The
    model's lower role is still stored on the judgment; it does not also
    shrink the line the source structure put back.
    """
    weights = _importance_by_id(usable, slots)
    if not override:
        return weights
    for eid in (override.get("headline_id"), override.get("legal_id")):
        if eid:
            weights.pop(eid, None)
    return weights


def _importance_by_id(usable: dict[str, dict[str, Any]], slots: dict[str, str]) -> dict[str, float]:
    """Prominence becomes the engine weight. A grouped label the model called
    a headline is not the headline slot, so it cannot outrank the real one.
    Ordinary badges stay at most medium so a loud label does not become the poster."""
    weights = {}
    for eid, row in usable.items():
        prominence = row["prominence"]
        slot = slots.get(eid)
        if slot == "badge" and row["role"] == "call_to_action":
            # The chosen button already has the call-to-action slot. A ribbon
            # or extra pill the model also calls a call to action stays a
            # badge. Rotation and grouping decided the slot; the words do not
            # move that accent into the main button.
            prominence = "low"
        elif slot == "badge" and row["role"] not in ("discount",):
            # Cap a loud badge at medium. A low or minimal badge stays quiet.
            # A discount keeps the model's prominence. A second action and a
            # misplaced headline are handled by the branches around this one.
            if _PROM_RANK[prominence] < _PROM_RANK["medium"]:
                prominence = "medium"
        # A grouped line the model calls the headline is still a badge or
        # ribbon. Leaving it at medium made that decoration grow into the
        # real headline. Keep it quiet.
        if row["role"] == "headline" and slot != "headline":
            prominence = "minimal"
        weights[eid] = PROMINENCE_IMPORTANCE[prominence]
    return weights


def _promoted_ids(ungrouped: list[Element], headline: Optional[Element], primary_cta: Optional[PillGroup]) -> set[str]:
    """Texts allowed to grow past their source font size: a line the model
    promoted into the headline or the action when it was not already the
    largest type. Already-dominant lines keep the engine's normal seed."""
    sizes = [float(element.get("font_size") or 0) for element in ungrouped]
    largest = max(sizes) if sizes else 0.0
    promoted = set()
    if headline is not None and float(headline.get("font_size") or 0) < largest - 0.1:
        promoted.add(headline["id"])
    if primary_cta is not None:
        label = _label(primary_cta[1])
        if label is not None and float(label.get("font_size") or 0) < largest - 0.1:
            promoted.add(label["id"])
    return promoted


def _price_outranks(usable, ungrouped, name_ids: list[str]) -> bool:
    """True when a price's prominence is strictly above every product name."""
    name_rank = min(_PROM_RANK[usable[eid]["prominence"]] for eid in name_ids)
    for element in ungrouped:
        row = usable.get(element["id"])
        if row is None or row.get("role") != "price":
            continue
        if _PROM_RANK[row["prominence"]] < name_rank:
            return True
    return False


def _rank(row: dict[str, Any]) -> tuple:
    """Higher is preferred. Primary outranks minimal. Low uncertainty
    breaks a tie. An equal tuple keeps the earlier candidate."""
    return (-_PROM_RANK[row["prominence"]], 1 if row.get("uncertainty") == "low" else 0)


def _label(members: list[Element]) -> Optional[Element]:
    return next((member for member in members if member.get("type") == "text"), None)
