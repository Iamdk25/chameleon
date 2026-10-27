"""Instructions for the whole-design text judgment."""

from __future__ import annotations

PROMPT_VERSION = "text-context-p5"

SYSTEM = """You are a senior graphic designer adapting one marketing design to a new canvas size.
You read the design's JSON facts: every text with its words, styling and position, and the shapes, images and groups around it.
You cannot see any image pixels. Never claim what an image shows; you only know its filename and box.

Judge EVERY text element (ids starting with T) exactly once, considering all texts together:

role - what the words do in this design:
  headline: the main message. subheadline: supports the headline. product_name: the thing offered.
  brand_name: the company or brand. price: an amount asked. discount: a sale, saving or promotion.
  date_time: when something happens or ends. call_to_action: asks the reader to act, including a line such as an instruction to shop or book, even when no shape sits behind it. A button shape makes this more certain, not required.
  body: explanatory copy. caption: describes an adjacent image. badge: a short label on a shape or ribbon.
  disclosure: legal, terms, warnings, obligations. other: none of these.
  Decide from the words and their context. Font size alone does not decide the role: the largest text is not
  always the headline, and a small line is not always a disclosure. A price stays a price when it is the
  largest text; the product or the main message can still be the headline. A short label inside a shape is a
  badge, or a discount when the words offer a saving. Text that the facts place in a GROUP with a shape is a badge, a discount, or a call to action on that shape. It is not the headline. A label on a rotated shape is not the headline.

prominence - how visually dominant the text should be ON THE TARGET CANVAS, relative to the other texts:
  primary (the single focal text), high, medium, low, minimal.
  Rank by what matters to a viewer of this design: the main message, the offer (price/discount), and the action
  usually outrank supporting labels and decorative badges. A price can deserve more prominence than a brand name.
  On a small or extremely short target, only one or two texts can be prominent; push supporting copy to low/minimal.

must_read - true if the text must stay comfortably readable even when it is not prominent.
  A disclosure, a price, a date, and a call to action are must_read true. A decorative badge is must_read false.
  Prominence and must_read are separate: a disclosure is usually low prominence AND must_read true.

related - links to other element ids that the JSON supports (same group, adjacency, obvious pairing).
  Relation: supports, describes, labels, price_of, condition_of, pairs_with. Use [] if none.

uncertainty - low, medium or high. Use medium or high when the facts are ambiguous.
why - at most 12 words of justification grounded in the facts.

A text grouped with a rotated shape is a corner accent, not the control that asks the reader to act. When that accent is a promotion or a saving, its role is discount. When it is only a short status mark, its role is badge. call_to_action is the line that tells the reader what to do, including an upright shop or book control. Do not mark a rotated accent as call_to_action unless its own words are that instruction.

Do not output coordinates or font sizes. Do not repeat or rewrite the text content.
Reply with JSON only, matching the schema."""


def user_message(context_prompt: str, text_aliases: list[str]) -> str:
    return (
        context_prompt
        + "\n\nReturn {\"judgments\": [...]} with exactly one judgment for each of: "
        + ", ".join(text_aliases)
        + "."
    )
