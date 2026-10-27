"""Choose the text-model backend for the submission entrypoint.

`auto` uses a GGUF packaged inside the image when one is present. It does
not call a host Ollama server and it does not download weights. Ollama is
only used when CHAMELEON_TEXT_BACKEND=ollama or CHAMELEON_USE_OLLAMA=1,
which is a local development switch. CHAMELEON_TEXT_MODE=off is the
no-model path. CHAMELEON_TEXT_MODE=on fails if no packaged model exists,
instead of quietly pretending a judgment was made.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from engine.text_context.backends import Backend, LlamaCppBackend, OllamaBackend


class ModelRequiredError(RuntimeError):
    """The entrypoint was told to use a model and none is packaged."""


def select_backend() -> tuple[Optional[Backend], str]:
    mode = os.environ.get("CHAMELEON_TEXT_MODE", "auto").strip().lower()
    if mode in {"off", "0", "baseline", "none"}:
        return None, "no-model"
    packaged = _packaged_model_path()
    requested = os.environ.get("CHAMELEON_TEXT_BACKEND", "auto").strip().lower()
    if packaged and requested in {"auto", "llamacpp"}:
        return LlamaCppBackend(packaged), "packaged-gguf"
    if requested == "ollama" or os.environ.get("CHAMELEON_USE_OLLAMA") == "1":
        return OllamaBackend(), "ollama"
    if mode == "on":
        raise ModelRequiredError(
            "CHAMELEON_TEXT_MODE=on but no GGUF is available at CHAMELEON_MODEL_PATH or /app/models. "
            "Refusing to write a no-model result under a model-required run."
        )
    return None, "no-model"


def _packaged_model_path() -> Optional[str]:
    explicit = os.environ.get("CHAMELEON_MODEL_PATH", "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() else None
    directory = Path(os.environ.get("CHAMELEON_MODEL_DIR", "/app/models"))
    if not directory.is_dir():
        return None
    found = sorted(directory.glob("*.gguf"))
    if len(found) > 1:
        raise ModelRequiredError(
            f"Multiple GGUF files found in {directory}. Keep exactly one model in the image "
            "or select one with CHAMELEON_MODEL_PATH."
        )
    return str(found[0]) if found else None
