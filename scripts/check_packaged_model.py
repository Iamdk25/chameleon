"""Check the local GGUF selected for a model-required Stage 1 image.

This checks presence, identity, and an optional size limit. It cannot decide
whether Adobe permits a particular model or license.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models",
        help="directory copied into the Docker image (default: repository models/)",
    )
    parser.add_argument("--expected-sha256", help="known checksum for the intended weight")
    parser.add_argument("--max-bytes", type=int, help="Adobe's model-size cap, if supplied")
    args = parser.parse_args()

    files = sorted(args.model_dir.glob("*.gguf"))
    if len(files) != 1:
        print(f"ERROR: expected exactly one GGUF in {args.model_dir}; found {len(files)}.")
        print("Place the intended weight there before building a model-required image.")
        return 2
    path = files[0]
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            print(f"ERROR: {path.name} does not have a GGUF header.")
            return 2
        stream.seek(0)
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    size = path.stat().st_size
    print(f"MODEL_FILE={path.name}")
    print(f"MODEL_BYTES={size}")
    print(f"MODEL_SHA256={digest}")
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        print("ERROR: checksum differs from --expected-sha256.")
        return 2
    if args.max_bytes is not None and size > args.max_bytes:
        print(f"ERROR: model exceeds the supplied {args.max_bytes}-byte limit.")
        return 2
    print("Identity check passed. Adobe approval and license suitability remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
