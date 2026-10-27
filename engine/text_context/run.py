"""Retired development command.

`python -m engine.text_context.run` used to write the JSON and the sidecar
before rendering. A render failure then left files that looked like a
finished result, and that path did not copy source opacity, rotation, and
z-index back the way `python -m engine.main` does.

Use the entrypoint instead:

    python -m engine.main
    CHAMELEON_USE_OLLAMA=1 python -m engine.main
    CHAMELEON_TEXT_MODE=on python -m engine.main

This module exits without writing a design JSON, a PNG, or a sidecar.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    del argv
    print(
        "engine.text_context.run is retired. It wrote JSON before the PNG "
        "and skipped the entrypoint's schema and field checks. "
        "Run `python -m engine.main`. "
        "Set CHAMELEON_USE_OLLAMA=1 for the local development model, "
        "or CHAMELEON_TEXT_MODE=on when a packaged model is required.",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
