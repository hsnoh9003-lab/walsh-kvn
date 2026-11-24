#!/usr/bin/env python3
"""Generate the Gidney cultivation Sinter statistics used by the RE inputs.

The published experiment used the Strilanc magic-state cultivation circuits:
end2end-inplace-distillation, Y basis, css gateset, uniform physical noise,
d1=3, r1=d1, r2=5, and d2=7,9,...,25. Sinter uses the desaturation
decoder. Shot targets below match the *observed per-distance counts* of the
released v100/SI103 Source Data; some distances received more shots than
others. p=5e-4 is an independent 10-billion-shot study.

This file owns circuit creation and raw Sinter collection only. The companion
build_tradeoff.py reconstructs the gap-filtered P_prep and e_prep estimates.
Neither script modifies published Source Data by default.

Prerequisite: the original Strilanc cultivation repository, with its
tools/make_circuits command, and its Cultiv/Sinter/StIM environment.

Examples (from Codes/):
  python -m Source.Gidney.generate_raw_stats
  python -m Source.Gidney.generate_raw_stats --cultiv-repo /path/to/cultivation \
      --p 5 --d2 7 --workers 32 --execute

Without --execute this prints the full collection plan and writes nothing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

PHYSICAL_ERROR_INDICES = tuple(range(2, 10))
D2_VALUES = tuple(range(7, 26, 2))
D1, R2, BASIS, GATESET, DECODER = 3, 5, "Y", "css", "desaturation"

# Published total shots by distance, d2 = 7, 9, ..., 25 (2026-11-24 release).
# Entries below are not uniform nominal p-specific targets. They reproduce
# the final per-distance acquisition plan, including selectively extended jobs.
SHOTS_BY_P = {
    2: (1252716851, 1170271539, 1119368499, 20000000000, 20000000000,
        20000000000, 20000000000, 20000000000, 1026447667, 1021695283),
    3: (800000000, 800000000, 800000000, 800000000, 10000000000,
        10000000000, 10000000000, 10000000000, 10000000000, 800000000),
    4: (800000000, 800000000, 800000000, 800000000, 4000000000,
        4000000000, 4000000000, 4000000000, 4000000000, 4000000000),
    5: (10000000000,) * 10,
    6: (800000000, 800000000, 800000000, 1200000000, 800000000,
        1200000000, 1200000000, 1200000000, 1200000000, 1200000000),
    7: (800000000, 800000000, 800000000, 1000000000, 1000000000,
        800000000, 1000000000, 1000000000, 1000000000, 1000000000),
    8: (800000000,) * 10,
    9: (800000000,) * 10,
}


def shots_for(p_index: int, d2: int) -> int:
    """Published cumulative shot total for a physical error and distance."""
    return SHOTS_BY_P[p_index][D2_VALUES.index(d2)]


def make_circuit_command(repo: Path, p_index: int, d2: int, out: Path) -> list[str]:
    """Original Cultiv end-to-end, inplace-distillation circuit configuration."""
    return [
        sys.executable, str(repo / "tools" / "make_circuits"),
        "--circuit_type", "end2end-inplace-distillation",
        "--noise_strength", str(p_index * 1e-4),
        "--gateset", GATESET, "--out_dir", str(out),
        "--basis", BASIS, "--d1", str(D1), "--d2", str(d2),
        "--r1", "d1", "--r2", str(R2),
    ]


def collect_command(
    sinter_executable: str, circuit: Path, stats_csv: Path, workers: int,
    p_index: int, d2: int,
) -> list[str]:
    """Collect raw shots with a resumable file and no max-errors stopping."""
    return [
        sinter_executable, "collect", "--metadata_func", "auto",
        "--circuits", str(circuit), "--decoders", DECODER,
        "--processes", str(workers),
        "--max_shots", str(shots_for(p_index, d2)),
        "--custom_decoders", "cultiv:sinter_samplers",
        "--save_resume_filepath", str(stats_csv),
    ]


def run_one(
    repo: Path, output_root: Path, p_index: int, d2: int, workers: int,
    sinter_executable: str,
) -> None:
    """Regenerate one circuit and collect its cumulative Sinter statistics."""
    job = output_root / f"p{p_index}e-4" / f"d2_{d2}"
    circuits = job / "circuits"
    circuits.mkdir(parents=True, exist_ok=True)
    subprocess.run(make_circuit_command(repo, p_index, d2, circuits), cwd=repo, check=True)
    stim_files = sorted(circuits.glob("*.stim"))
    if len(stim_files) != 1:
        raise RuntimeError(f"Expected exactly one Stim circuit for (p,d2)=({p_index},{d2}); found {stim_files}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    stats = job / "stats.csv"
    subprocess.run(
        collect_command(sinter_executable, stim_files[0], stats, workers, p_index, d2),
        cwd=repo, env=env, check=True,
    )
    print(f"Saved/resumed {stats}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cultiv-repo", type=Path,
                    help="Checkout of Strilanc magic-state cultivation, containing tools/make_circuits")
    ap.add_argument("--output-root", type=Path, default=Path("Build/Gidney/raw_stats"))
    ap.add_argument("--p", type=int, nargs="+", choices=PHYSICAL_ERROR_INDICES,
                    help="Physical error probabilities, in units of 1e-4 (default: show all)")
    ap.add_argument("--d2", type=int, nargs="+", choices=D2_VALUES,
                    help="Second-stage code distances (default: all)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--execute", action="store_true",
                    help="Actually run the very expensive Cultiv/Sinter shots")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")
    if args.execute and (args.cultiv_repo is None or args.p is None):
        ap.error("--execute requires both --cultiv-repo and an explicit --p selection")

    rates = tuple(dict.fromkeys(args.p or PHYSICAL_ERROR_INDICES))
    distances = tuple(dict.fromkeys(args.d2 or D2_VALUES))
    for p in rates:
        for d2 in distances:
            target = shots_for(p, d2)
            print(f"p={p}e-4  d2={d2:2d}  max_shots={target:,}", flush=True)
            if args.execute:
                repo = args.cultiv_repo.expanduser().resolve()
                if not (repo / "tools" / "make_circuits").is_file():
                    raise FileNotFoundError(repo / "tools" / "make_circuits")
                sinter_executable = shutil.which("sinter")
                if not sinter_executable:
                    raise RuntimeError("sinter is not on PATH; activate the Cultiv environment")
                run_one(repo, args.output_root.resolve(), p, d2, args.workers, sinter_executable)
    if not args.execute:
        print("PLAN ONLY: nothing executed. Supply --cultiv-repo, --p and --execute to collect.", flush=True)


if __name__ == "__main__":
    main()
