"""Local inference backends.

The model choice is provisional: Adobe has not published its approved model
list or size limit. Any backend only has to implement `generate()` and
`identity()`, so the model or runtime can be replaced without touching the
analyzer or the layout engine.

`OllamaBackend` talks to an Ollama server on the local machine
(127.0.0.1). It refuses non-loopback hosts so a misconfiguration cannot turn
this into a cloud call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import urlparse


class BackendError(RuntimeError):
    """The local model could not be reached or did not answer."""


class Backend(Protocol):
    def identity(self) -> dict[str, Any]:
        """Stable description of the model (name, digest, runtime options)."""

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return (raw reply text, timing/usage stats)."""


DEFAULT_MODEL = "llama3.2:latest"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class OllamaBackend:
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        *,
        seed: int = 7,
        num_ctx: int = 4096,
        num_predict: int = 1536,
        timeout_s: float = 120.0,
        keep_alive: str = "10m",
    ) -> None:
        self.model = model or os.environ.get("CHAMELEON_TEXT_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("CHAMELEON_OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        hostname = urlparse(self.host).hostname
        if hostname not in _LOOPBACK:
            raise BackendError(f"refusing non-local inference host {self.host!r}")
        self.options = {"temperature": 0, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self._identity: Optional[dict[str, Any]] = None

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib.request.Request(
            self.host + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise BackendError(f"ollama {path} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"ollama {path} unreachable at {self.host}: {exc}") from exc

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            show = self._post("/api/show", {"model": self.model}, timeout=15)
            details = show.get("details", {})
            digest = None
            try:
                with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as resp:
                    for m in json.loads(resp.read().decode()).get("models", []):
                        if m.get("name") == self.model or m.get("model") == self.model:
                            digest = m.get("digest")
            except (urllib.error.URLError, OSError):
                digest = None
            self._identity = {
                "runtime": "ollama",
                "model": self.model,
                "digest": digest,
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "capabilities": show.get("capabilities", []),
                "chat_template": bool(show.get("template")),
                "options": dict(self.options),
                "provisional": True,
            }
        return self._identity

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        needed = (len(system) + len(user)) // 4 + 256
        n_ctx = int(self.options["num_ctx"])
        if n_ctx < 1024 or needed > n_ctx:
            raise BackendError(
                f"this design needs about {needed} tokens and the context is {n_ctx}. "
                "Refusing to publish a truncated judgment."
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        if "thinking" in self.identity().get("capabilities", []):
            body["think"] = False
        started = time.perf_counter()
        structured = True
        try:
            reply = self._post("/api/chat", body, timeout=self.timeout_s)
        except BackendError as exc:
            if "HTTP" not in str(exc):
                raise
            structured = False
            body.pop("format", None)
            reply = self._post("/api/chat", body, timeout=self.timeout_s)
        wall = time.perf_counter() - started
        ns = 1e-9
        stats = {
            "wall_s": round(wall, 3),
            "load_s": round(reply.get("load_duration", 0) * ns, 3),
            "prompt_tokens": reply.get("prompt_eval_count"),
            "prompt_s": round(reply.get("prompt_eval_duration", 0) * ns, 3),
            "output_tokens": reply.get("eval_count"),
            "output_s": round(reply.get("eval_duration", 0) * ns, 3),
            "done_reason": reply.get("done_reason"),
            "n_ctx": n_ctx,
            "structured_json": structured,
            "chat_template": self.identity().get("chat_template"),
        }
        return (reply.get("message") or {}).get("content", ""), stats


class LlamaCppBackend:
    """In-process GGUF runtime. Weights must already be on disk.

    This is the container path: nothing is downloaded and nothing talks to
    a host Ollama server. llama-cpp-python is imported only when a model
    file is actually opened.
    """

    def __init__(self, model_path: str, *, n_ctx: int = 4096, n_predict: int = 1536, seed: int = 7) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_predict = n_predict
        self.seed = seed
        self._model = None
        self._identity: Optional[dict[str, Any]] = None
        self._load_s = 0.0
        self._hash_s = 0.0

    def identity(self) -> dict[str, Any]:
        if self._identity is not None:
            return self._identity
        path = Path(self.model_path)
        if not path.is_file():
            raise BackendError(f"packaged model file is missing: {self.model_path}")
        started = time.perf_counter()
        digest = _sha256_file(path)
        self._hash_s = time.perf_counter() - started
        self._identity = {
            "runtime": "llama-cpp-python",
            "model": path.name,
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "digest": digest,
            "n_ctx": self.n_ctx,
            "seed": self.seed,
            "options": {"temperature": 0, "seed": self.seed, "n_ctx": self.n_ctx, "n_predict": self.n_predict},
            "provisional": True,
            "adobe_approved": False,
        }
        return self._identity

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model = self._load()
        self._require_context(model, system, user)
        structured = True
        try:
            result = self._complete(model, system, user, schema)
        except BackendError:
            raise
        except Exception:
            structured = False
            try:
                result = self._complete(model, system, user, None)
            except Exception as exc:
                raise BackendError(f"packaged model inference failed: {exc}") from exc
        message = ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if not str(message).strip():
            raise BackendError("packaged model returned an empty reply")
        usage = result.get("usage") or {}
        choice = (result.get("choices") or [{}])[0]
        return message, {
            "wall_s": round(result.get("_wall_s", 0.0), 3),
            "load_s": round(self._load_s, 3),
            "hash_s": round(self._hash_s, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "done_reason": choice.get("finish_reason") or "llamacpp",
            "n_ctx": self._context_length(model),
            "structured_json": structured,
            "chat": True,
        }

    def _context_length(self, model: Any) -> int:
        reader = getattr(model, "n_ctx", None)
        if callable(reader):
            try:
                return int(reader())
            except (TypeError, ValueError):
                return self.n_ctx
        return self.n_ctx

    def _require_context(self, model: Any, system: str, user: str) -> None:
        n_ctx = self._context_length(model)
        needed = (len(system) + len(user)) // 4 + 256
        if n_ctx < 1024:
            raise BackendError(
                f"context length {n_ctx} is below 1024, which is too small for one whole-design judgment"
            )
        if needed > n_ctx:
            raise BackendError(
                f"this design needs about {needed} tokens and the model context is {n_ctx}. "
                "Refusing to publish a truncated judgment."
            )

    def _complete(self, model: Any, system: str, user: str, schema: Optional[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": self.n_predict,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}
        started = time.perf_counter()
        result = model.create_chat_completion(**kwargs)
        if isinstance(result, dict):
            result["_wall_s"] = time.perf_counter() - started
        return result

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise BackendError(
                "llama-cpp-python is not installed. The container must install it at build time "
                "when a GGUF is packaged."
            ) from exc
        started = time.perf_counter()
        try:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                seed=self.seed,
                n_gpu_layers=_gpu_layers(),
                verbose=False,
            )
            self._load_s = time.perf_counter() - started
        except Exception as exc:
            raise BackendError(f"packaged model could not be loaded: {exc}") from exc
        return self._model


def _gpu_layers() -> int:
    """Use the GPU when this build can offload. CPU-only builds stay on CPU.

    CHAMELEON_N_GPU_LAYERS overrides the choice. This does not download anything.
    """
    raw = os.environ.get("CHAMELEON_N_GPU_LAYERS", "").strip()
    if raw:
        return int(raw)
    try:
        from llama_cpp import llama_supports_gpu_offload
        return -1 if llama_supports_gpu_offload() else 0
    except Exception:
        return 0


def _sha256_file(path: Path) -> str:
    """Hash a GGUF in chunks. Do not read a multi-gigabyte file into one buffer.

    A sibling ``.sha256`` file is reused when the size and modification time
    still match, so a later run does not read the whole weight file again.
    The first run still hashes every byte.
    """
    stat = path.stat()
    stamp = f"{stat.st_size} {stat.st_mtime_ns}"
    cache = path.with_name(path.name + ".sha256")
    if cache.is_file():
        lines = cache.read_text().splitlines()
        if len(lines) >= 2 and lines[0] == stamp and len(lines[1]) == 64:
            return lines[1]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    try:
        cache.write_text(stamp + "\n" + value + "\n")
    except OSError:
        pass
    return value
