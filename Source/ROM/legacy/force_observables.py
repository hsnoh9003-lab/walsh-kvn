"""Read the fitted force maps on the noiseless finite Walsh–KvN ensemble."""
from pathlib import Path

import numpy as np
import pandas as pd
from Source.CFD.forces import read_fem_forces, read_force_coefficients


def force_curves(case: Path, force_data: Path, end_time: float = 60):
    fem = read_fem_forces(force_data, end_time=end_time)
    coefficients = read_force_coefficients(force_data)
    cd = coefficients.loc[coefficients["observable"] == "C_D"].set_index("term")["coefficient"]
    cl = coefficients.loc[coefficients["observable"] == "C_L"].set_index("term")["coefficient"]

    with np.load(case / "direct_quantum_history.npz") as q:
        times, r, state = q["times"], q["r"], q["direct_history"]
        readout_moments = q["joint_probabilities"] @ q["joint_values"]
    state_norm = np.linalg.norm(state, axis=(1, 2))
    np.testing.assert_allclose(state_norm, 1, atol=1e-12, rtol=0)
    state /= state_norm[:, None, None]
    with np.load(case / "unified_reference.npz") as ref:
        np.testing.assert_array_equal(times, ref["times"])
        rigid_phase = np.exp(1j * ref["known_phase"][:, None])

    # Phase-branch order is 00, 01, 10, 11. The moments include the mass of each radius cell.
    x_pauli = np.array([[0, 1], [1, 0]])
    y_pauli = np.array([[0, -1j], [1j, 0]])
    probability = np.sum(abs(state)**2, axis=1)
    x_probe = np.einsum("tij,ik,tkj->tj", state.conj(), np.kron(np.eye(2), x_pauli), state).real
    y_probe = np.einsum("tij,ik,tkj->tj", state.conj(), np.kron(y_pauli, np.eye(2)), state).real
    first_phase = x_probe + 1j * y_probe
    r2 = r**2
    np.testing.assert_allclose(np.c_[x_probe @ r, y_probe @ r, probability @ r2],
                               readout_moments, atol=1e-13, rtol=0)
    first_phase *= rigid_phase
    second_phase = 4 * state[:, 0, :].conj() * state[:, 3, :] * rigid_phase**2

    # Insert x=r cos(theta), y=r sin(theta) and the circuit closure z=r² into the maps.
    radial_drag = cd["1"] + (cd["z"] + (cd["x^2"] + cd["y^2"]) / 2) * r2 + cd["z^2"] * r2**2
    angular_drag = ((cd["x^2"] - cd["y^2"]) * second_phase.real + cd["xy"] * second_phase.imag) @ (r2 / 2)
    lift = first_phase.real @ (r * (cl["x"] + cl["xz"] * r2))
    lift += first_phase.imag @ (r * (cl["y"] + cl["yz"] * r2))
    walsh = pd.DataFrame({"step": np.arange(len(times)), "time": times,
                          "C_D": probability @ radial_drag + angular_drag, "C_L": lift})
    assert len(walsh) == 34 and np.isfinite(walsh[["C_D", "C_L"]]).to_numpy().all()
    return fem, walsh
