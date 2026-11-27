"""Cost a full-window QSVT exponential and reproduce its output on the wake task."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import argparse
import json
import math
import sys

import numpy as np
from scipy.special import jv

_RESEARCH_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_RESEARCH_ROOT))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
sys.path.insert(0, str(ROOT))

from benchmark.matched_output import DATA, output_metrics, radial_generators, read_npz
from outer.outer_qsvt import evaluate_success, outer_inventory

ALGORITHM_BUDGET = 1e-4
PROTECTED_BUDGET = 1e-4


def cosine_sine(alpha, time, tolerance):
    """Truncate the Jacobi–Anger series with separate even and odd tail bounds."""
    z = alpha * time
    cutoff = math.ceil(2 * abs(z)) + 16
    orders = np.arange(cutoff + 1)
    values = jv(orders, z)
    magnitudes = 2 * np.abs(values)
    parity = orders % 2
    tail_even = np.cumsum(np.where(parity == 0, magnitudes, 0)[::-1])[::-1]
    tail_odd = np.cumsum(np.where(parity == 1, magnitudes, 0)[::-1])[::-1]
    # Majorize the remaining Bessel terms by their factorial bound.
    log_term = (cutoff + 1) * math.log(abs(z) / 2) - math.lgamma(cutoff + 2)
    remainder = math.exp(log_term) * 2 / (1 - abs(z) / (2 * (cutoff + 2)))
    tail = np.maximum(tail_even, tail_odd) + max(remainder, 1e-20)
    possible = np.flatnonzero(6 * math.sqrt(2) * (tail + 1e-12) <= tolerance)
    if not len(possible) or possible[0] < 2:
        raise ValueError(
            "Expected a nontrivial interval with a cutoff that certifies the tolerance"
        )
    degree = int(possible[0] - 1)
    scale = 1 / (1 + tail[degree + 1] + 1e-12)
    coefs = 2 * (-1.0) ** (orders[: degree + 1] // 2) * values[: degree + 1] * scale
    coefs[0] /= 2
    cosine = np.where(parity[: degree + 1] == 0, coefs, 0)
    sine = np.where(parity[: degree + 1] == 1, coefs, 0)
    return degree, cosine, sine, float(6 * math.sqrt(2) * (tail[degree + 1] + 1e-12))


def data():
    archive = json.loads(
        published_path(
            "Comparisons/BE_KvN/inputs/archived_qsvt_inventory.json"
        ).read_text()
    )
    with ZipFile(
        published_path("Comparisons/BE_KvN/inputs/wake_task_inputs.zip")
    ) as source:
        config = json.loads(source.read(DATA + "new_qsvt_report.json"))
        state = read_npz(source, "qsvt_transport.npz")
        ref = read_npz(source, "unified_reference.npz")
    params = config["configuration"]
    time = params["mu"] * config["physical_duration"]
    phase_slope = params["gamma_normalized"] / params["mu"]
    return archive, state, ref, time, phase_slope


def metrics(probabilities, survival, state, ref):
    metric = ref["M"].T @ ref["G"] @ ref["M"] / float(ref["norm"]) ** 2
    return output_metrics(
        probabilities,
        state["values"],
        ref["continuum_moments"][-1],
        metric,
        0.01,
        2 * PROTECTED_BUDGET / (survival - PROTECTED_BUDGET),
    )


def structural_outer_inventory(degree, projector_bits):
    """Count explicit one-clean projected phases and OAA reflections.

    Generic outer-QSP angles are charged without claiming phase synthesis or
    exact-angle cancellations unavailable from the certified polynomial.
    """
    return outer_inventory(degree, projector_bits)


def resource_record(circuit):
    archive, state, ref, time, phase_slope = data()
    alpha = circuit["normalization_alpha"]
    degree, cosine, sine, bound = cosine_sine(alpha, time, ALGORITHM_BUDGET)
    radius = state["r"]
    radial = 1j * radial_generators(radius)["exact_face"]
    phase_term = phase_slope * np.diag(radius**2)
    final = []
    errors = []
    for occupied in (0, 1, 1, 2):
        eigenvalues, eigenvectors = np.linalg.eigh(radial - occupied * phase_term)
        signal = eigenvalues / alpha
        response = evaluate_success(cosine, sine, signal)
        errors.append(
            float(np.max(np.abs(response - np.exp(-1j * time * eigenvalues))))
        )
        final.append(
            eigenvectors @ (response * (eigenvectors.conj().T @ state["initial"])) / 2
        )
    assert max(errors) <= bound <= ALGORITHM_BUDGET
    final = np.asarray(final)
    success = float(np.sum(np.abs(final) ** 2))
    assert (1 - bound) ** 2 <= success <= 1 + 1e-12
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    measurement = np.kron(h @ np.diag([1, -1j]), h) @ final
    probabilities = np.abs(measurement.ravel()) ** 2 / success
    output = metrics(probabilities, success, state, ref)
    protected_output = metrics(probabilities, (1 - bound) ** 2, state, ref)
    assert (
        protected_output["accepted_shots_for_field_target"]
        >= output["accepted_shots_for_field_target"]
    )

    overhead, scratch = structural_outer_inventory(degree, circuit["projector_qubits"])
    counts = Counter(
        RZ=overhead["rz"],
        native_T=overhead["t"],
        Clifford=overhead["cx"] + overhead["clifford_other"],
    )
    plain, ctrl = circuit["ordinary"], circuit["controlled"]
    counts.update(
        {
            key: overhead["oracle"] * plain[key]
            + overhead["oracle_controlled"] * ctrl[key]
            for key in counts
        }
    )
    for part in ("initial_counts", "terminal_counts"):
        counts.update({key: archive["counts_audit"][part][key] for key in counts})
    # Oracle widths in literal_joint_counts are standalone (work at the end).
    # In the complete QSVT circuit, that same restored clean work bit is placed
    # *after* the two outer selectors and is shared with their projected phases
    # and OAA reflections: no independent allocation for each subroutine.
    macro_width = circuit["system_qubits"] + circuit["projector_qubits"]
    assert circuit["macro_width"] == macro_width
    assert plain["logical_qubits"] == macro_width + 1
    assert ctrl["logical_qubits"] == macro_width + 2
    assert scratch == 1
    width = macro_width + 2 + scratch
    source_angle_error_bound = overhead["oracle"] * plain.get(
        "source_angle_error_bound", 0.0
    ) + overhead["oracle_controlled"] * ctrl.get("source_angle_error_bound", 0.0)
    return {
        "method": "Guseynov Corollary 1 reconstruction, joint full-window generator",
        "substeps": 1,
        "T_normalized": time,
        "normalization_alpha": alpha,
        "alpha_T": alpha * time,
        "outer_qsp_degree": degree,
        "signal_calls_per_attempt": overhead["oracle"] + overhead["oracle_controlled"],
        "controlled_signal_calls_per_attempt": overhead["oracle_controlled"],
        "logical_qubits": width,
        **{key + "_per_attempt": value for key, value in counts.items()},
        "T_per_attempt_at_108_T_per_RZ": counts["native_T"] + 108 * counts["RZ"],
        "final_noiseless_field_bias_percent": 100 * output["relative_field_bias"],
        "accepted_shots_at_1pct_with_protected_allowance": protected_output[
            "accepted_shots_for_field_target"
        ],
        "nominal_success_probability": success,
        "source_angle_error_bound": source_angle_error_bound,
        "conservative_success_lower_bound": (1 - bound) ** 2,
        "max_realized_exponential_error": max(errors),
        "full_interval_exponential_error_bound": bound,
        "inventory_scope": "All BE STMC gates, four-branch projector phases and OAA "
        "reflections use the shared one-clean KvN-LCHS basis compiler. "
        "Unfitted outer-QSP angles are charged conservatively as arbitrary "
        "controlled phases; exact-angle cancellations are not claimed. "
        "The numerical output uses the certified polynomial, but individual "
        "high-degree outer phase angles are not synthesized.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        type=Path,
        default=published_path(
            "Comparisons/BE_KvN/fullwindow/literal_joint_counts.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=generated_outputs_root() / "BE_KvN/fullwindow_resources.json",
    )
    args = parser.parse_args()
    result = resource_record(json.loads(args.block.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
