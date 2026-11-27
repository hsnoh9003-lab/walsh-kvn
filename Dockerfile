# Python release and multi-platform base image are pinned. The resulting image
# digest also captures resolved Debian packages; record it when publishing.
FROM python:3.12.14-slim-trixie@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ libgomp1 inkscape fonts-dejavu-core fonts-liberation ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Codes
COPY requirements*.txt ./
COPY Source/CFD/force_replay/requirements.txt Source/CFD/force_replay/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements_reproduction.txt \
    && python -m pip check

# Resolve OS metadata before kernel threads start. This avoids the observed
# debugger-initialization hang under amd64 emulation on Apple Silicon.
# The standard kernel and debugger remain enabled; host settings are untouched.
RUN python - <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.prefix) / 'share/jupyter/kernels/python3/kernel.json'
spec = json.loads(path.read_text())
spec['argv'] = [sys.executable, '-c',
    'import platform; platform.platform(); from ipykernel.kernelapp import main; main()',
    '-f', '{connection_file}']
path.write_text(json.dumps(spec, indent=2) + '\n')
PY

RUN useradd --create-home --uid 1000 researcher
COPY --chown=researcher:researcher . .
RUN mkdir -p /workspace/Outputs && chown -R researcher:researcher /workspace
USER researcher
ENV KVN_SOURCE_DATA=/workspace/Source_Data \
    KVN_DEMO_OUTPUT=/workspace/Outputs/Demo \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
# Populate the font cache before a threaded notebook kernel invokes fontconfig.
RUN python -c "import matplotlib.pyplot"
EXPOSE 8888
CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--no-browser", "demo.ipynb"]
