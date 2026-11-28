"""Fetch the pinned QuCF reference inputs into a separate local cache.

The 32 original oracle/HDF5 files are external inputs, not redistributed by
this project or covered by its Apache-2.0/CC-BY-4.0 licences. Their upstream
paths, byte counts and SHA-256 digests are retained for reproducibility.
Run ``python -m Source.qucf_inputs`` before an offline reproduction.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.error
import urllib.request

REVISION = "f34f89c24eff4ab79d3ecdea72da3cec88dce94f"
UPSTREAM_PREFIX = "simulations/LCHS/KvN/"
RAW_ROOT = f"https://raw.githubusercontent.com/QuCF/QuCF/{REVISION}/"
from Source.data_locations import source_data_root

def manifest_path() -> Path:
    """Resolve the fixed upstream manifest only when the external data are used."""
    return source_data_root() / "Tables/Table_S11/LCHS_source_counts/upstream_input_manifest.json"


def entries() -> dict[str, dict]:
    """Read and validate the pinned 32-file upstream input manifest."""
    records = json.loads(manifest_path().read_text(encoding="utf-8"))
    result = {}
    for row in records:
        if not row["path"].startswith(UPSTREAM_PREFIX):
            continue
        relative = row["path"][len(UPSTREAM_PREFIX) :]
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not relative:
            raise ValueError(f"Unsafe upstream input path: {relative}")
        result[relative] = row
    if len(result) != 32:
        raise ValueError("The pinned QuCF input manifest must contain 32 unique files")
    return result


def cache_root() -> Path:
    """Resolve the revision-specific cache or the KVN_QUCF_CACHE override."""
    override = os.environ.get("KVN_QUCF_CACHE")
    root = (
        Path(override)
        if override
        else (Path.home() / ".cache" / "kvn_walsh" / "qucf" / REVISION / "KvN")
    )
    return root.expanduser().resolve()


def verify(path: Path, record: dict) -> None:
    """Require the expected file size and SHA-256 checksum."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing QuCF input: {path}; run python -m Source.qucf_inputs"
        )
    if path.stat().st_size != record["bytes"]:
        raise IOError(
            f"QuCF input size mismatch: {path}; remove this cached file and fetch again"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise IOError(
            f"QuCF input SHA-256 mismatch: {path}; remove this cached file and fetch again"
        )


def _fetch(path: Path, record: dict) -> None:
    """Download one pinned input, verify it, then atomically replace the cache entry."""
    url = RAW_ROOT + record["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".download-", dir=path.parent, delete=False
        ) as sink:
            temporary = Path(sink.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                # Read at most one byte beyond the expected size; reject
                # truncated responses, error pages and unexpected content.
                sink.write(response.read(record["bytes"] + 1))
        verify(temporary, record)
        os.replace(temporary, path)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Cannot fetch the pinned QuCF input {record['path']}. "
            "Check network access or populate KVN_QUCF_CACHE from that revision."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_inputs(*, download: bool = True) -> Path:
    """Verify pinned inputs; download missing files when download=True (the default)."""
    root = cache_root()
    fetched = 0
    for relative, record in entries().items():
        target = root / relative
        if not target.is_file() and download:
            _fetch(target, record)
            fetched += 1
        verify(target, record)
    if fetched:
        print(
            f"Fetched and verified {fetched} external QuCF inputs at revision {REVISION}.",
            file=sys.stderr,
        )
    return root


@lru_cache(maxsize=8)
def _ready(root: str) -> Path:
    # Key by cache location so changing KVN_QUCF_CACHE takes effect immediately.
    actual = ensure_inputs()
    if actual != Path(root):
        raise RuntimeError("QuCF cache setting changed during input resolution")
    return actual


def input_root() -> Path:
    """Resolve the cache after verifying its complete input set once per location."""
    return _ready(str(cache_root()))


def input_path(upstream_path: str) -> Path:
    """Resolve one manifest-listed upstream path in the verified cache."""
    if not upstream_path.startswith(UPSTREAM_PREFIX):
        raise KeyError(f"Not a pinned QuCF input: {upstream_path}")
    relative = upstream_path[len(UPSTREAM_PREFIX) :]
    if relative not in entries():
        raise KeyError(f"Not in the pinned QuCF manifest: {upstream_path}")
    return input_root() / relative


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all cached inputs without network access",
    )
    args = parser.parse_args()
    print(ensure_inputs(download=not args.check))
