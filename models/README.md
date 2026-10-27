# Packaged text model

Put one `.gguf` file in this directory before `docker build` if the image
should run the text-context path. The file is not committed. Adobe has not
published the approved model list or size cap, so a file placed here is a
provisional local weight, not an approved submission model.

With no `.gguf`, `docker build` skips the llama-cpp compile. The image runs
the no-model engine and the sidecar says `no-model`. It does not call Ollama
and it does not download weights. `auto` does not fail in that case. Use
`CHAMELEON_TEXT_MODE=off` when the submission is intentionally the engine
alone. `CHAMELEON_TEXT_MODE=on` exits 2 and writes nothing if the file is
missing.

To package a model after Adobe names one and a size cap:

1. Place the approved file at `models/<name>.gguf`. Do not commit it.
   `models/*.gguf` and `models/*.sha256` stay gitignored. Record the sha256,
   byte size, and license in the top-level README. Run
   `python3 scripts/check_packaged_model.py` before building; it requires
   exactly one GGUF and prints the actual checksum and size. Use its
   `--expected-sha256` and `--max-bytes` options when Adobe provides those
   constraints. This check does not grant model approval. Do not upload the
   provisional Llama 3.2 file unless that decision explicitly allows it.
2. Build with Docker Buildx. The Dockerfile does not pin a platform:
   `docker buildx build --platform linux/amd64 --load -t my-chameleon-engine .`
   The legacy `docker build --platform` flag does not select the
   base image. The Dockerfile installs `requirements-text.txt` only when a
   `.gguf` is in the build context, then copies `engine/` after that so later
   layout edits do not recompile the runtime.
3. Run `docker run --rm --network none -e CHAMELEON_TEXT_MODE=on` with the
   image ENTRYPOINT. Confirm each sidecar says `model` and
   `model_judgment` is true. A `fallback` or `no-model` sidecar is not a
   successful model submission.

A file placed here before that decision is a provisional local weight, not an
approved submission model. Swap the file when the approved list exists.
The prompt and schema versions are part of the cache key, so a new model
does not reuse an old judgment.
