"""Whole-design text context analysis with a local, replaceable text model.

The analyzer judges each text element's role, prominence for the target
canvas, readability requirement, relations, and uncertainty in one model
call. `engine.resize.resize()` consumes those judgments; it still does all
measurement, fitting, positioning, and collision handling itself.
"""

from engine.text_context.analyzer import analyze, judgments_for_resize

__all__ = ["analyze", "judgments_for_resize"]
