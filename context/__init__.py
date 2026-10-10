"""Text-first context housekeeping for CHAMELEON v2.

PR0 only: keep an active context file from exceeding its line limit by
archiving the full text and writing a compact replacement first.
"""

from context.housekeeping import ACTIVE_LINE_LIMIT, ActiveContext, HousekeepingResult

__all__ = ["ACTIVE_LINE_LIMIT", "ActiveContext", "HousekeepingResult"]
