"""Verify the finite-shot field-error prediction used for Table S13.

Run from the repository root with::

    python -m Source.ROM.field_sampling_verification

Inputs are the archived circuit probabilities and joint observable values in
Source_Data/Supplementary_Data/Finite_Shot_Replay/sampling_inputs. This script
samples those probabilities; it does not rerun transport or submit hardware jobs.
It writes a summary CSV, the sampled moment/error arrays, and a validation JSON
to Generated_Outputs/Field_Sampling_Replay/.

For one shot, let v be the three-component observable vector and C its covariance.
The field metric H converts a moment error d into squared relative field error
d.T @ H @ d. For S independent shots the sample mean has covariance C/S, so its
expected mean squared error (MSE) is trace(H @ C)/S. The calculation below checks
this prediction against repeated multinomial samples. Its reference is the exact
finite-circuit mean: deterministic error against the continuum is not included.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



from pathlib import Path
import csv, json, hashlib, math
import numpy as np
import sys

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE))
from Source.data_locations import generated_outputs_root
from Source.data_locations import source_data_root

ROOT = generated_outputs_root() / "Field_Sampling_Replay"
INPUT = source_data_root() / "Supplementary_Data/Finite_Shot_Replay/sampling_inputs"
# REPEATS is the number of independently repeated sampling experiments, not the
# number of shots in one experiment. Keep the seed fixed for reproducible checks.
SEED = 20261107
REPEATS = 3000


def main():
    """Compare predicted and sampled MSE at four times and two shot counts."""
    ROOT.mkdir(parents=True, exist_ok=True)
    # h holds the noiseless joint circuit distribution at each saved step.
    # j holds the moment-to-modal-coordinate map M, the field Gram matrix G,
    # physical times, and one fixed final continuum-field normalization norm.
    h = np.load(INPUT / "direct_quantum_history.npz")
    j = np.load(INPUT / "joint_readout_inputs.npz")
    cfg = json.loads((INPUT / "configuration.json").read_text())
    reference = json.loads(
        (INPUT / "observable_diagnostics_reference.json").read_text()
    )["history"]
    # Each row of values is one possible shot's (r*s_X, r*s_Y, r**2), where
    # s_X and s_Y are the measured phase-qubit eigenvalues, each +1 or -1.
    # The common shot is essential: the three observables are correlated.
    omega = float(cfg["parameters"]["omega"])
    times = j["physical_times"]
    Mfinal = j["M"]
    G = j["G"]
    norm = float(j["norm"])
    values = h["joint_values"]
    rng = np.random.default_rng(SEED)
    rows = []
    replicates = {}
    norm_drift = []
    for step in [0, 11, 22, 33]:
        # Normalize only the tiny floating-point norm drift; report its size.
        p = h["joint_probabilities"][step].copy()
        norm_drift.append(abs(float(p.sum()) - 1))
        p /= p.sum()
        # Mfinal already contains the constant-rate phase at the final time.
        # Rotating by omega*(t - t_final) gives the field map at this saved time.
        angle = omega * (times[step] - times[-1])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        # mean = E[v], cov = E[v v.T] - E[v] E[v].T. Retain off-diagonal
        # covariances instead of treating the three readouts as separate samples.
        fieldmap = rot @ Mfinal
        mean = p @ values
        cov = (values.T * p) @ values - np.outer(mean, mean)
        # G weights modal errors by their reconstructed velocity-field norm.
        # H is that metric expressed in the measured-moment coordinates and
        # divided by the same fixed norm**2 at every time.
        H = fieldmap.T @ G @ fieldmap / norm**2
        variance = float(np.trace(H @ cov))
        # Check the analytical prediction against the archived 1,000-shot RMS
        # diagnostic; multiplication by 100 converts a relative error to percent.
        assert (
            abs(
                100 * np.sqrt(variance / 1000)
                - reference[step]["sampling_RMS_1000_pct"]
            )
            < 2e-8
        )
        for shots in [1000, 8000]:
            # A multinomial draw counts joint outcomes of `shots` measurements.
            # Each row is a fresh experiment, preserving within-shot correlations.
            counts = rng.multinomial(shots, p, size=REPEATS)
            moments = counts @ values / shots
            delta = moments - mean
            # Evaluate d.T @ H @ d for every replicate, then compare its mean
            # with trace(H @ cov)/shots. This compares mean SQUARED errors;
            # the reported RMS is the square root of the mean, not mean(sqrt).
            mse = np.einsum("bi,ij,bj->b", delta, H, delta)
            predicted = variance / shots
            empirical = float(mse.mean())
            se = float(mse.std(ddof=1) / np.sqrt(REPEATS))
            score = (empirical - predicted) / se
            # se estimates the standard error across REPEATS experiments.
            # The six-standard-error check is a numerical replay criterion;
            # it is not the paper's field-accuracy target or a per-shot bound.
            assert abs(score) < 6
            rows.append(
                dict(
                    step=step,
                    shots=shots,
                    repeats=REPEATS,
                    seed=SEED,
                    predicted_RMS_percent=100 * np.sqrt(predicted),
                    sampled_RMS_percent=100 * np.sqrt(empirical),
                    MSE_z_score=score,
                    reference="exact finite-circuit mean",
                    normalization="one fixed final continuum-field norm",
                    state="current MPS-prepared Walsh circuit",
                )
            )
            replicates[f"step{step}_shots{shots}_moments"] = moments
            replicates[f"step{step}_shots{shots}_MSE"] = mse
    # Save both the eight summary rows and all replicate arrays so readers can
    # inspect the sampling distribution rather than just the pass/fail result.
    with (ROOT / "Table_S13_sampling_verification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(ROOT / "sampling_MSE_replicates.npz", **replicates)
    # Record the input-file hashes and NumPy version alongside the fixed seed.
    qa = dict(
        status="PASS",
        seed=SEED,
        repeats=REPEATS,
        steps=[0, 11, 22, 33],
        shot_counts=[1000, 8000],
        rows=len(rows),
        maximum_abs_MSE_z_score=max(abs(x["MSE_z_score"]) for x in rows),
        expected_variances_match_current_saved_diagnostics=True,
        max_probability_norm_drift=max(norm_drift),
        new_dynamics_or_resource_estimates=False,
        numpy_version=np.__version__,
        input_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUT.iterdir()
        },
    )
    (ROOT / "SAMPLING_VERIFICATION_QA.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
