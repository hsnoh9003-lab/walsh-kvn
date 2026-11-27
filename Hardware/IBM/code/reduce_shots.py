"""Fixed-seed, without-replacement reduction of archived IBM MPS hardware counts.

The original acquisition files and metadata are not modified. All plotted times,
transport/reference pairs and the 32-cell control use the preselected budgets.
Budgets were chosen before inspecting the retained realization; seeds are not searched.
"""

from pathlib import Path
from importlib.metadata import version
import csv, json, math, platform
import numpy as np
import analyze_mps_ibm as A

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBSET_SEED = 20261117
BUDGET = {4: 4000, 6: 12000}
REPS = 20000
SETS = [("dT4_n4n5", [(4, 4, 4), (5, 4, 1)]), ("dT6_n4", [(4, 6, 6)])]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def vector(data, name, size):
    c = np.zeros(size, dtype=np.int64)
    for key, value in data[name].items():
        assert int(value) == value
        c[int(key, 2)] += int(value)
    return c


def continuum(p0, elapsed, kind):
    if kind == "mean":
        return A.continuum_mean(p0, [elapsed])[0]
    size = len(p0)
    cut = 0.875
    threshold = cut / math.sqrt(
        math.exp(2 * 0.0924819 * elapsed) * (1 - cut**2) + cut**2
    )
    cell = int(threshold * size)
    return float(
        p0[cell + 1 :].sum() + p0[cell] * ((cell + 1) / size - threshold) * size
    )


def main():
    subset_dir = DATA / "reduced_shots"
    subset_dir.mkdir(exist_ok=True)
    # Compute the full-data selection diagnostic first; it does not inspect a retained seed.
    selection = []
    validation = []
    original = {}
    initial = {n: A.mps_state(np.sqrt(A.target_masses(n)), n) for n in (4, 5)}
    simulation_rng = np.random.default_rng(2026111701)
    for tag, sets in SETS:
        original[tag] = json.loads((DATA / f"ibm_mps_{tag}_counts.json").read_text())
        for n, nt, kmax in sets:
            size = 2**n
            centres = (np.arange(size) + 0.5) / size
            p0 = abs(initial[n]) ** 2
            u = np.ones(size) / size
            a = (p0 - u) / np.sum((p0 - u) ** 2)
            for k in range(1, kmax + 1):
                ck = vector(original[tag], f"n{n}_kvn_k{k}", size)
                cr = vector(original[tag], f"n{n}_ref_k{k}", size)
                pk = ck / ck.sum()
                pr = cr / cr.sum()
                F = float(a @ (pr - u))
                elapsed = k * (49.8291 - 40) / nt
                sampled_k = (
                    simulation_rng.multinomial(BUDGET[nt], pk, size=REPS) / BUDGET[nt]
                )
                sampled_r = (
                    simulation_rng.multinomial(BUDGET[nt], pr, size=REPS) / BUDGET[nt]
                )
                sampled_F = (sampled_r - u) @ a
                for kind, h in [
                    ("P", (centres >= 0.875).astype(float)),
                    ("mean", centres),
                ]:
                    ref = continuum(p0, elapsed, kind)
                    difference = float(h @ (pk - pr))
                    estimate = float(h @ p0 + difference / F)
                    gk = h / F
                    gr = -h / F - difference * a / F**2
                    variance = lambda p, g: float(p @ (g * g) - (p @ g) ** 2)
                    v = (variance(pk, gk) + variance(pr, gr)) / ref**2
                    b = abs(estimate - ref) / ref
                    required = math.ceil(v / (0.05**2 - b * b)) if b < 0.05 else None
                    assert required is not None and BUDGET[nt] >= required
                    row = {
                        "cells": size,
                        "nt": nt,
                        "k": k,
                        "observable": kind,
                        "P_or_mean_continuum": ref,
                        "full_data_corrected": estimate,
                        "bias_fraction": b,
                        "v_relative": v,
                        "estimated_shots_each_5pct": required,
                        "retained_shots_each": BUDGET[nt],
                        "predicted_relative_RMS": math.sqrt(b * b + v / BUDGET[nt]),
                    }
                    selection.append(row)
                    simulated = h @ p0 + ((sampled_k - sampled_r) @ h) / sampled_F
                    rel = (simulated - ref) / ref
                    validation.append(
                        {
                            "cells": size,
                            "nt": nt,
                            "k": k,
                            "observable": kind,
                            "replicates": REPS,
                            "relative_RMS_empirical_model": float(
                                np.sqrt(np.mean(rel**2))
                            ),
                            "relative_bias_empirical_model": float(rel.mean()),
                            "relative_SD_empirical_model": float(rel.std()),
                        }
                    )
    # One predeclared realization, obtained without replacement from each recorded histogram.
    rng = np.random.default_rng(SUBSET_SEED)
    subsets = {}
    acquisition = {}
    for tag, sets in SETS:
        nt = sets[0][1]
        retained = BUDGET[nt]
        reduced = {}
        for name, counts in original[tag].items():
            n = int(name[1])
            size = 2**n
            c = vector(original[tag], name, size)
            acquired = int(c.sum())
            draw = rng.multivariate_hypergeometric(c, retained)
            assert np.all(draw <= c) and draw.sum() == retained
            reduced[name] = {
                format(j, f"0{n}b"): int(count) for j, count in enumerate(draw) if count
            }
            acquisition[f"{tag}/{name}"] = {"acquired": acquired, "retained": retained}
        subsets[tag] = reduced
        (subset_dir / f"ibm_mps_{tag}_{retained}_counts.json").write_text(
            json.dumps(reduced, indent=2) + "\n"
        )
    rows = []
    diagnostics = []
    for tag, sets in SETS:
        data = subsets[tag]
        boot_rng = np.random.default_rng(0)
        for n, nt, kmax in sets:
            size = 2**n
            centres = (np.arange(size) + 0.5) / size
            p0 = abs(initial[n]) ** 2
            u = np.ones(size) / size
            a = (p0 - u) / np.sum((p0 - u) ** 2)
            top = centres >= 0.875

            def correct(pk, pr):
                F = float(a @ (pr - u))
                return F, p0 + (pk - pr) / F

            for k in range(kmax + 1):
                name = f"n{n}_kvn_k{k}"
                c = vector(data, name, size)
                shots = int(c.sum())
                pk = c / shots
                ideal = abs(A.propagate(initial[n], nt, k)) ** 2
                elapsed = k * (49.8291 - 40) / nt
                row = {
                    "cells": size,
                    "qubits": n,
                    "nt": nt,
                    "k": k,
                    "t_star": 40 + elapsed,
                    "shots": shots,
                    "acquired_shots": acquisition[f"{tag}/{name}"]["acquired"],
                    "P_ideal": float(ideal[top].sum()),
                    "r_ideal": float(ideal @ centres),
                    "P_continuum": continuum(p0, elapsed, "P"),
                    "r_continuum": continuum(p0, elapsed, "mean"),
                    "P_raw": float(pk[top].sum()),
                    "r_raw": float(pk @ centres),
                    "P_raw_SE": float(
                        np.sqrt(pk[top].sum() * (1 - pk[top].sum()) / shots)
                    ),
                    "r_raw_SE": float(
                        np.sqrt((pk @ centres**2 - (pk @ centres) ** 2) / shots)
                    ),
                }
                if k == 0:
                    row.update(
                        F_ref="",
                        P_refnorm="",
                        P_refnorm_err="",
                        r_refnorm="",
                        r_refnorm_err="",
                    )
                else:
                    rc = vector(data, f"n{n}_ref_k{k}", size)
                    pr = rc / rc.sum()
                    F, pc = correct(pk, pr)
                    boot = []
                    for _ in range(1000):
                        _, sample = correct(
                            boot_rng.multinomial(shots, pk) / shots,
                            boot_rng.multinomial(shots, pr) / shots,
                        )
                        boot.append([sample[top].sum(), sample @ centres])
                    boot = np.asarray(boot)
                    row.update(
                        F_ref=F,
                        P_refnorm=float(pc[top].sum()),
                        P_refnorm_err=float(boot[:, 0].std()),
                        r_refnorm=float(pc @ centres),
                        r_refnorm_err=float(boot[:, 1].std()),
                    )
                    diagnostics.append(
                        {
                            "cells": size,
                            "nt": nt,
                            "k": k,
                            "shots": shots,
                            "F_ref": F,
                            "reference_depolarizing_TVD": float(
                                abs(pr - (F * p0 + (1 - F) * u)).sum() / 2
                            ),
                            "corrected_mean": float(pc @ centres),
                            "largest_cell_centre": float(centres[-1]),
                            "corrected_r2": float(pc @ centres**2),
                            "corrected_variance": float(
                                pc @ centres**2 - (pc @ centres) ** 2
                            ),
                            "corrected_negative_mass": float(-np.minimum(pc, 0).sum()),
                            "P_bias_over_bootstrap_SE": float(
                                (pc[top].sum() - ideal[top].sum())
                                / row["P_refnorm_err"]
                            ),
                        }
                    )
                rows.append(row)
    write_csv(DATA / "ibm_mps_reduced_calibration.csv", rows)
    write_csv(DATA / "ibm_mps_reduced_reference_diagnostics.csv", diagnostics)
    write_csv(DATA / "shot_reduction_selection.csv", selection)
    write_csv(DATA / "shot_reduction_model_validation.csv", validation)
    report = {
        "method": "Uniform random subsampling without replacement of each archived count histogram.",
        "subset_seed": SUBSET_SEED,
        "seed_search_performed": False,
        "retained_budget_each": BUDGET,
        "allocation_rule": "Round 1.25 times the largest plug-in 5% relative-RMS shot estimate across all propagated times and both radial observables up to a multiple of 2,000.",
        "scope": "Empirical sensitivity analysis; not an independent hardware error certificate or a new device run.",
        "model_validation": {
            "replicates": REPS,
            "seed": 2026111701,
            "maximum_relative_RMS": max(
                r["relative_RMS_empirical_model"] for r in validation
            ),
        },
        "bootstrap": {
            "resamples": 1000,
            "seed_per_job": 0,
            "interpretation": "Uncertainty for the retained measurement sample; no finite-population reduction is applied to plotted standard errors.",
        },
        "acquisition_metadata_preserved": True,
        "counts": acquisition,
        "analysis_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
        },
    }
    (DATA / "shot_reduction_metadata.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for nt in (4, 6):
        worst = max(
            (r for r in selection if r["nt"] == nt),
            key=lambda r: r["estimated_shots_each_5pct"],
        )
        terminal = next(
            r for r in rows if r["cells"] == 16 and r["nt"] == nt and r["k"] == nt
        )
        print(
            json.dumps(
                {
                    "nt": nt,
                    "budget": BUDGET[nt],
                    "selection_worst": worst,
                    "terminal_subset": terminal,
                }
            )
        )
    print(
        "Maximum empirical-model RMS:",
        report["model_validation"]["maximum_relative_RMS"],
    )


if __name__ == "__main__":
    main()
