"""Rebuild the published Re=100 basis from recovered velocity snapshots.

Run from the package root:
    python -m Source.POD_SINDy.replay_frozen_basis --out-dir ../Source_Data/Generated_Outputs/Replayed_POD

The saved basis is loaded only after construction, for verification. This does
not itself rerun the finite-element solver. The separate six-case CFD rerun
are separate from this snapshot-to-model reconstruction.
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
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.linalg import orthogonal_procrustes, subspace_angles
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# ROOT: code-package root; external inputs are resolved through Source_Data.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path, source_label
from .fit import fit
from .pod import fit_oscillatory_pair, project, closure as fit_closure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_for_verification(basis, coordinates, target):
    """Compare the independently constructed subspace, then fix its gauge."""
    # new/reference: rebuilt/saved spatial modes, with one mode per row.
    new = basis["Phi"].copy()
    reference = target["Phi"]
    # angles compare the two-dimensional subspaces independently of axis orientation.
    angles = subspace_angles(new[:2].T, reference[:2].T)
    assert float(np.max(angles)) < 1e-9
    # rotation aligns the oscillatory axes; sign aligns the third (shift) direction.
    rotation, _ = orthogonal_procrustes(new[:2].T, reference[:2].T)
    sign = 1.0 if new[2] @ reference[2] >= 0 else -1.0
    new[:2] = rotation.T @ new[:2]
    new[2] *= sign
    # Transform coordinates along with the basis so a*Phi remains unchanged.
    aligned = coordinates.copy()
    aligned[:, :2] = aligned[:, :2] @ rotation
    aligned[:, 2] *= sign
    # Neither rotation nor a shift-sign change alters the reconstructed field.
    transform = np.eye(3)
    transform[:2, :2] = rotation
    transform[2, 2] = sign
    np.testing.assert_allclose(new, transform.T @ basis["Phi"], atol=1e-12)
    np.testing.assert_allclose(transform @ new, basis["Phi"], atol=1e-12)
    # report: alignment parameters and the remaining differences from the saved basis.
    report = {
        "pair_principal_angles_radians": angles.tolist(),
        "pair_procrustes_rotation": rotation.tolist(),
        "shift_sign": sign,
        "pair_aligned_relative_residual": float(
            np.linalg.norm(new[:2] - reference[:2]) / np.linalg.norm(reference[:2])
        ),
        "shift_aligned_relative_residual": float(
            np.linalg.norm(new[2] - reference[2]) / np.linalg.norm(reference[2])
        ),
        "unaligned_modes_maximum_absolute_difference": float(
            np.max(np.abs(basis["Phi"] - reference))
        ),
    }
    return {**basis, "Phi": new}, aligned, report


def reproduce(out_dir):
    # source/data: external snapshot archive and its arrays.
    source = published_path("Data/CFD_ROM/Re100/data/source_snapshots.npz")
    with np.load(source) as f:
        data = {k: f[k] for k in f.files}
    # The recorded snapshot and mean windows are protocol inputs. Fixing the
    # mean window avoids changing it when the POD pair's arbitrary phase changes
    # between numerical-library implementations. No retained modes enter here.
    protocol_path = published_path("Data/CFD_ROM/Re100/data/model_configuration.json")
    protocol = json.loads(protocol_path.read_text())["learning"]
    # times/fields: saved times and the two-component velocity snapshots.
    times = data["times"]
    fields = data["fields"].astype(float)
    # lo/hi and pod_mask select the recorded POD fitting interval.
    lo, hi = protocol["POD_interval"]
    pod_mask = (times >= lo) & (times <= hi)
    # pair: oscillatory POD modes and their coordinate scaling.
    pair = fit_oscillatory_pair(
        times[pod_mask], fields[pod_mask], scale_mode="training_rms"
    )

    # bounds: recorded mean-window endpoints; tt includes those endpoints and interior samples.
    bounds = np.asarray(protocol["mean_cycle_bounds"])
    tt = np.r_[bounds[0], times[(times > bounds[0]) & (times < bounds[-1])], bounds[-1]]
    # Mean field: integral u(t) dt / (t_end-t_start), with linear endpoint interpolation.
    mean = trapezoid(interp1d(times, fields, axis=0)(tt), tt, axis=0) / (tt[-1] - tt[0])
    # base: flattened steady field; delta: mean-minus-steady field.
    base = data["steady"].astype(float).ravel()
    delta = mean.ravel() - base
    # Shift direction: (I-Q*Q.T)*delta, normalized to unit norm; Q=orthonormal POD modes.
    shift = delta - pair["pod_modes"] @ (pair["pod_modes"].T @ delta)
    shift /= np.linalg.norm(shift)
    # basis: steady reference and three spatial modes; coordinates: projected CFD trajectory.
    basis = {"reference": base, "Phi": np.vstack([pair["pair"].T, shift])}
    coordinates = project(basis, fields)
    metadata = {
        "POD_interval": [lo, hi],
        "recorded_mean_cycle_bounds": bounds.tolist(),
        "mean_window_source": source_label(protocol_path),
        "construction_uses_retained_modes": False,
    }

    # These retained arrays are comparison targets, not reconstruction inputs.
    target_path = published_path("Data/CFD_ROM/Re100/data/new_model_inputs.npz")
    with np.load(target_path) as f:
        target = {k: f[k] for k in f.files}
    # gauge records the axis/sign alignment used to compare coordinates and fitted coefficients.
    basis, coordinates, gauge = align_for_verification(basis, coordinates, target)
    # Test invariance with a deliberately rotated pair and reversed shift sign.
    angle = 0.37
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    changed = {"reference": basis["reference"], "Phi": basis["Phi"].copy()}
    changed["Phi"][:2] = rot.T @ changed["Phi"][:2]
    changed["Phi"][2] *= -1
    changed_coordinates = coordinates.copy()
    changed_coordinates[:, :2] = changed_coordinates[:, :2] @ rot
    changed_coordinates[:, 2] *= -1
    check, check_coordinates, _ = align_for_verification(
        changed, changed_coordinates, target
    )
    np.testing.assert_allclose(check["Phi"], basis["Phi"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(check_coordinates, coordinates, rtol=1e-10, atol=1e-11)

    # mask: coefficient-fitting interval; model: quadratic SINDy fit at threshold 0.03.
    mask = (times >= 10) & (times <= 80)
    model = fit(coordinates[mask], times[mask], 2, 0.03)
    # closure: fitted third-coordinate relation to the squared oscillation radius.
    closure = fit_closure(coordinates[mask])
    selected_path = published_path("Data/Model/candidate_K2.npz")
    with np.load(selected_path) as f:
        selected = {k: f[k] for k in f.files}
    # comparisons: reconstructed arrays and their saved verification targets.
    comparisons = {
        "steady_field": (basis["reference"], target["base"]),
        "spatial_modes": (basis["Phi"], target["Phi"]),
        "all_projected_coordinates": (coordinates, target["all_projected_coordinates"]),
        "training_coordinates": (coordinates[mask], target["training_coordinates"]),
        "training_times": (times[mask], target["training_times"]),
        "shift_closure": (closure, target["closure"]),
        "selected_physical_coefficients": (
            model["physical_C"],
            selected["physical_coefficients"],
        ),
        "selected_exponents": (model["p"], selected["powers"]),
    }
    # differences: maximum absolute error for each reconstruction check.
    differences = {}
    for name, (actual, expected) in comparisons.items():
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-11, err_msg=name
        )
        differences[name] = float(np.max(np.abs(actual - expected)))
    assert model["active_terms"] == 12

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "rebuilt_basis_and_model.npz",
        times=times,
        Phi=basis["Phi"],
        base=basis["reference"],
        all_projected_coordinates=coordinates,
        training_times=times[mask],
        training_coordinates=coordinates[mask],
        C=model["physical_C"],
        p=model["p"],
        closure=closure,
    )
    # report: source identity, reconstruction errors, axis alignment and numerical-library versions.
    report = {
        "status": "PASS",
        "input": source_label(source),
        "input_sha256": sha(source),
        "target_sha256": sha(target_path),
        "selected_model_sha256": sha(selected_path),
        "snapshots": len(times),
        "training_states": int(mask.sum()),
        "retained_coefficients": model["active_terms"],
        "maximum_absolute_differences": differences,
        "reconstructed_basis_metadata": metadata,
        "gauge_verification": gauge,
        "rotated_pair_and_shift_sign_control": "PASS",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "scope": "Recovered saved snapshots and recorded selection/mean windows to the POD subspace and shift mode. For coefficient checks, the pair is Procrustes-aligned and the shift sign is matched to the frozen axes. This command replays saved snapshots; the subsequent six-case transient CFD rerun and both-panel S1 numerical recovery are documented in Data/Physical_Forecast_Audit and Docs/S1_Physical_Forecast_Recovery.json.",
    }
    (out_dir / "replay_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=generated_outputs_root() / "Replayed_POD"
    )
    args = parser.parse_args()
    reproduce(args.out_dir)


if __name__ == "__main__":
    main()
