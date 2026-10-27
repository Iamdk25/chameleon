# Choose the image architecture with BuildKit, not with a platform pin here.
# `docker build --platform` on the legacy builder does not select the
# base image, and `FROM --platform=linux/amd64` fails when the local
# python:3.11-slim tag is only arm64. The working command is:
#   docker buildx build --platform linux/amd64 --load -t my-chameleon-engine .
FROM python:3.11-slim

WORKDIR /app

# System libs some renderer backends (e.g. skia) need at runtime, plus git
# (build-time only) so pip can install cr-renderer straight from its repo.
# Harmless to include even if your engine doesn't end up using them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# If you add dependencies for your own resize() approach, add them to
# requirements.txt above — everything needed must install at BUILD time.
# Stage-one scoring runs with --network none, so nothing at runtime can
# reach out to the internet (no pip installs, no API calls, no model
# downloads at run time — pull those in during the build instead).

COPY chameleon_design_schema_v0.9.json .
COPY brand_kit/ ./brand_kit/
COPY renderer/ ./renderer/
COPY requirements-text.txt .
COPY models/ ./models/

# A packaged GGUF is optional. When models/*.gguf is in the build context,
# the in-process runtime is compiled and installed now. `docker run --network none`
# cannot download it later. No host Ollama server is used. The compiler is
# removed after the install. GGML_NATIVE is off so the binary is not tied
# to the build machine's CPU. This layer stays cached when only engine
# code changes.
RUN if ls /app/models/*.gguf >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake && \
      CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir -r requirements-text.txt && \
      apt-get purge -y build-essential cmake && apt-get autoremove -y && \
      apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*; \
    fi

COPY engine/ ./engine/

# input/ and output/ are mounted as volumes at run time (see README) —
# not baked into the image.
RUN mkdir -p /app/input /app/output

# auto: use the packaged GGUF when one was copied, otherwise the no-model engine.
ENV CHAMELEON_TEXT_MODE=auto
ENV CHAMELEON_MODEL_DIR=/app/models

ENTRYPOINT ["python", "-m", "engine.main"]
