"""Download the separately published Source_Data ZIP by its fixed Zenodo DOI.

Zenodo upload contract: one file named ``Source_Data.zip`` containing a
``Source_Data/`` directory and its ``manifest.csv``. The published source
data are never modified by this downloader or by normal figure plotting.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DOI = "10.5281/zenodo.24505927"
RECORD_ID = 24505927
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"


def cached_path() -> Path:
    override = os.environ.get("KVN_SOURCE_DATA_CACHE")
    root = Path(override) if override else Path.home() / ".cache" / "kvn_walsh" / f"zenodo_{RECORD_ID}"
    return root.expanduser().resolve() / "Source_Data"


def _file_entry() -> dict:
    try:
        with urllib.request.urlopen(RECORD_API, timeout=30) as response:
            record = json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise RuntimeError(
            f"Zenodo record {DOI} is unavailable or not yet public. "
            "Until publication, put Source_Data/ beside the cloned repository or set KVN_SOURCE_DATA."
        ) from exc
    files = record.get("files", [])
    exact = [item for item in files if item.get("key", "").lower() == "source_data.zip"]
    archives = [item for item in files if item.get("key", "").lower().endswith(".zip")]
    matches = exact if exact else archives
    if len(matches) != 1:
        raise RuntimeError(
            f"Zenodo {DOI} must provide Source_Data.zip (or exactly one ZIP); "
            f"files currently listed: {[item.get('key') for item in files]}"
        )
    entry = matches[0]
    url = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
    if not isinstance(url, str) or not url.startswith("https://zenodo.org/"):
        raise RuntimeError("Zenodo record has no valid HTTPS content link.")
    return entry


def _check_archive(archive: Path, entry: dict) -> None:
    declared_size = entry.get("size")
    if declared_size is not None and archive.stat().st_size != int(declared_size):
        raise IOError("Zenodo ZIP size mismatch")
    declared = entry.get("checksum", "")
    if not declared:
        return
    kind, separator, digest = declared.partition(":")
    if not separator:
        kind, digest = "md5", kind
    if kind not in {"md5", "sha256"}:
        raise ValueError(f"Unknown Zenodo checksum algorithm: {kind}")
    actual = hashlib.new(kind)
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest().lower() != digest.lower():
        raise IOError("Zenodo ZIP checksum mismatch")


def _extract(archive: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/"))
            if parts.is_absolute() or ".." in parts.parts or not parts.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unexpected archive symlink: {member.filename}")
            destination = target.joinpath(*parts.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    candidates = [target / "Source_Data", target]
    candidates.extend(p for p in target.iterdir() if p.is_dir())
    for candidate in candidates:
        if (candidate / "manifest.csv").is_file():
            return candidate
    raise RuntimeError("The Zenodo ZIP is missing Source_Data/manifest.csv")


def download_source_data() -> Path:
    cache = cached_path()
    if (cache / "manifest.csv").is_file():
        return cache
    entry = _file_entry()
    url = entry["links"].get("content") or entry["links"]["self"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zenodo-download-", dir=cache.parent) as tempdir:
        scratch = Path(tempdir)
        archive = scratch / "Source_Data.zip"
        with urllib.request.urlopen(url, timeout=120) as remote, archive.open("wb") as sink:
            shutil.copyfileobj(remote, sink, length=1024 * 1024)
        _check_archive(archive, entry)
        extracted = _extract(archive, scratch / "unpacked")
        if not (cache / "manifest.csv").is_file():
            if cache.exists():
                raise RuntimeError(f"Incomplete Source Data cache; inspect: {cache}")
            try:
                extracted.rename(cache)
            except FileExistsError:
                if not (cache / "manifest.csv").is_file():
                    raise
    print(f"Downloaded Source Data for {DOI}: {cache}")
    return cache
