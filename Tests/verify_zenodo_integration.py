"""Offline integration check for the DOI-addressed Source Data ZIP downloader."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from Source import zenodo_download


def main() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("Source_Data/manifest.csv", "published_path,original_path\n")
        package.writestr("Source_Data/Figures/Fig_3/plot_inputs/probe.csv", "x,y\n0,1\n")
    data = archive.getvalue()
    record = {
        "files": [{
            "key": "Source_Data.zip", "size": len(data),
            "checksum": "md5:" + hashlib.md5(data).hexdigest(),
            "links": {"self": "https://zenodo.org/api/records/24505927/files/Source_Data.zip/content"},
        }]
    }

    def urlopen(request, timeout=None):
        if request == zenodo_download.RECORD_API:
            return io.BytesIO(json.dumps(record).encode("utf-8"))
        if request == record["files"][0]["links"]["self"]:
            return io.BytesIO(data)
        raise AssertionError(f"Unexpected URL: {request}")

    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {"KVN_SOURCE_DATA_CACHE": directory}):
            with patch("urllib.request.urlopen", side_effect=urlopen):
                recovered = zenodo_download.download_source_data()
                assert recovered == Path(directory) / "Source_Data"
                assert (recovered / "manifest.csv").is_file()
                assert (recovered / "Figures/Fig_3/plot_inputs/probe.csv").read_text() == "x,y\n0,1\n"
                assert zenodo_download.download_source_data() == recovered
    print("PASS: fixed DOI record lookup, verified archive, extracted cache, no second download")


if __name__ == "__main__":
    main()
