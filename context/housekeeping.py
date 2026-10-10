"""Archive and compact an active context file before it exceeds its line limit.

The active file is plain text. Callers append through ``ActiveContext``
instead of writing the file themselves. When an append would make the file
longer than ``ACTIVE_LINE_LIMIT`` lines, the current bytes are copied into
``archive/`` and the active file is replaced with a shorter summary plus a
recent tail, and only then is the new text added.

The ceiling is inclusive: 1,000 lines does not exceed 1,000. The next write
that would produce 1,001 lines archives and compacts first, so the file on
disk never grows past the limit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_LINE_LIMIT = 1000


@dataclass(frozen=True)
class HousekeepingResult:
    archived: bool
    compacted: bool
    archive_paths: tuple[Path, ...]
    lines_before: int
    lines_after: int


class ActiveContext:
    """Filesystem-backed active context with a hard line ceiling."""

    def __init__(self, directory: Path | str, limit: int = ACTIVE_LINE_LIMIT) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.directory = Path(directory)
        self.limit = limit
        self.active_path = self.directory / "active.txt"
        self.archive_dir = self.directory / "archive"

    def line_count(self) -> int:
        return len(self._read_lines())

    def append(self, text: str) -> HousekeepingResult:
        """Append ``text``. Archive and compact first if the result would exceed the limit."""
        incoming = _block_lines(text)
        lines_before = self.line_count()
        if not incoming:
            return HousekeepingResult(False, False, (), lines_before, lines_before)

        archive_paths: list[Path] = []
        compacted = False

        if len(incoming) > self.limit:
            payload = "\n".join(incoming) + "\n"
            archived = self._archive_text(payload, "entry")
            archive_paths.append(archived)
            incoming = [
                f"[entry archived: {archived.name}; {len(incoming)} lines]"
            ]
            compacted = True

        current_lines = self._read_lines()
        if len(current_lines) + len(incoming) > self.limit:
            current_text = self._read_text()
            if current_text != "":
                archive_paths.append(self._archive_text(current_text, "active"))
            current_lines = self._compact_lines(current_lines, room_for=len(incoming))
            compacted = True

        if len(current_lines) + len(incoming) > self.limit:
            payload = "\n".join(incoming) + "\n"
            archived_entry = self._archive_text(payload, "entry")
            archive_paths.append(archived_entry)
            incoming = [
                f"[entry archived: {archived_entry.name}; {len(payload.splitlines())} lines]"
            ]
            compacted = True
            if len(current_lines) + len(incoming) > self.limit:
                current_lines = self._compact_lines(current_lines, room_for=len(incoming))

        new_lines = current_lines + incoming
        if len(new_lines) > self.limit:
            raise RuntimeError(
                "context housekeeping left the active file over the line limit"
            )
        self._write_lines(new_lines)
        return HousekeepingResult(
            archived=bool(archive_paths),
            compacted=compacted,
            archive_paths=tuple(archive_paths),
            lines_before=lines_before,
            lines_after=len(new_lines),
        )

    def housekeep(self) -> HousekeepingResult:
        """If the active file is already over the limit, archive it and compact in place."""
        lines = self._read_lines()
        before = len(lines)
        if before <= self.limit:
            return HousekeepingResult(False, False, (), before, before)
        archived = self._archive_text(self._read_text(), "active")
        compacted_lines = self._compact_lines(lines, room_for=0)
        self._write_lines(compacted_lines)
        return HousekeepingResult(
            archived=True,
            compacted=True,
            archive_paths=(archived,),
            lines_before=before,
            lines_after=len(compacted_lines),
        )

    def _compact_lines(self, lines: list[str], room_for: int) -> list[str]:
        budget = self.limit - room_for
        if budget < 1:
            return []
        if len(lines) <= budget:
            return list(lines)
        for recent_keep, summary_cap in (
            (160, 24),
            (80, 16),
            (40, 12),
            (16, 8),
            (0, 6),
            (0, 1),
        ):
            rendered = _render_compact(lines, recent_keep, summary_cap)
            if len(rendered) <= budget:
                return rendered
        return _render_compact(lines, 0, 1)[:budget]

    def _archive_text(self, text: str, prefix: str) -> Path:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.archive_dir / f"{prefix}-{stamp}.txt"
        seq = 1
        while path.exists():
            path = self.archive_dir / f"{prefix}-{stamp}-{seq}.txt"
            seq += 1
        path.write_text(text, encoding="utf-8")
        return path

    def _read_text(self) -> str:
        if not self.active_path.is_file():
            return ""
        return self.active_path.read_text(encoding="utf-8")

    def _read_lines(self) -> list[str]:
        text = self._read_text()
        if text == "":
            return []
        return text.splitlines()

    def _write_lines(self, lines: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        text = ("\n".join(lines) + "\n") if lines else ""
        temporary = self.active_path.with_name(self.active_path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.active_path)


def _block_lines(text: str) -> list[str]:
    lines = text.splitlines()
    while lines and lines[0] == "":
        del lines[0]
    while lines and lines[-1] == "":
        del lines[-1]
    return lines


def _summary_lines(older: list[str], cap: int) -> list[str]:
    if cap <= 0:
        return []
    head = f"- Earlier lines compacted: {len(older)} (full text archived)"
    if cap == 1 or not older:
        return [head]
    preferred: list[str] = []
    other: list[str] = []
    for line in older:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "- ", "* ")):
            preferred.append(stripped)
        else:
            other.append(stripped)
    picked: list[str] = []
    seen: set[str] = set()
    for source in (preferred, other):
        for stripped in source:
            bullet = "- " + stripped[:160]
            if bullet in seen:
                continue
            seen.add(bullet)
            picked.append(bullet)
            if len(picked) >= cap - 1:
                return [head] + picked
    return [head] + picked


def _render_compact(lines: list[str], recent_keep: int, summary_cap: int) -> list[str]:
    if recent_keep <= 0:
        recent: list[str] = []
        older = lines
    elif recent_keep >= len(lines):
        recent = list(lines)
        older = []
    else:
        recent = lines[-recent_keep:]
        older = lines[:-recent_keep]
    rendered = [
        "# CHAMELEON active context",
        f"# compacted: {len(older)} earlier lines folded; full copy archived",
        "## Earlier",
        *_summary_lines(older, summary_cap),
    ]
    if recent:
        rendered.extend(["", "## Recent", *recent])
    return rendered


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m context.housekeeping <context-directory>",
            file=sys.stderr,
        )
        return 2
    store = ActiveContext(args[0])
    result = store.housekeep()
    print(
        f"lines {result.lines_before} -> {result.lines_after} "
        f"archived={result.archived} compacted={result.compacted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
