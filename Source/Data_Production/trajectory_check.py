"""Independently replay the four retained coupled-model trajectories.

check_all loads each stored initial complex state, reconstructs the declared
drift and pair steps, and compares its moment history with the stored run.
Temporary model outputs are redirected to scratch. The check tests transport
from the saved preparation, not hardware execution or a new preparation fit.
"""

import os
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp


def check_all(release, inputs, scratch):
    # The archived model code writes directories when imported.
    """Replay every supplied coupled trajectory and return numerical agreement records."""
    os.environ["POLYNOMIAL_VALIDATION_OUTPUT"] = str(scratch)
    # Resolve the installed code, irrespective of the checkout directory name.
    code_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code_root))
    from Source.KvN import polynomial_validation as model

    case = next(c for c in model.declared_cases() if c["name"] == "coupled_cubic_2d")
    checks = []
    for path in sorted(inputs.glob("*.npz")):
        with np.load(path) as run:
            times = run["times"]
            initial = run["initial_state"]
            size = int(np.sqrt(len(initial)))
            steps = len(times) - 1
            _, pairs = model.generator_and_pairs(case, size)
            _, states = model.pair_evolution(
                initial, pairs, 2, times[-1] / steps, steps
            )
            quantum = abs(states) ** 2 @ np.array(model.coordinates(size, 2)).T
            np.testing.assert_allclose(
                quantum, run["moment_history"][:, :2], atol=1e-12, rtol=1e-10
            )

            # New records store the quadrature order. Preserve the historical defaults.
            order = (
                int(run["reference_quadrature_order"])
                if "reference_quadrature_order" in run
                else (8 if path.stem.endswith("near_unstable") else 4)
            )
            nodes, weights = leggauss(order)
            points = ((np.arange(size)[:, None] + (nodes + 1) / 2) / size).ravel()
            initial_positions = np.array(
                np.meshgrid(points, points, indexing="ij")
            ).ravel()
            mass = abs(initial.reshape(size, size)) ** 2
            mass /= mass.sum()
            weight_x = (mass.sum(axis=0)[:, None] * weights / 2).ravel()
            weight_y = (mass.sum(axis=1)[:, None] * weights / 2).ravel()
            joint_weight = np.outer(weight_x, weight_y).ravel()

            def velocity(time, position):
                return np.asarray(model.drift(case, position.reshape(2, -1))).ravel()

            solution = solve_ivp(
                velocity,
                (times[0], times[-1]),
                initial_positions,
                t_eval=times,
                method="DOP853",
                rtol=4e-13,
                atol=4e-15,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            reference = (
                solution.y.reshape(2, -1, len(times)).transpose(2, 0, 1) @ joint_weight
            )
            np.testing.assert_allclose(
                reference,
                run["prepared_cell_reference_moments"][:, :2],
                atol=1e-12,
                rtol=1e-10,
            )
            checks.append(
                dict(
                    initial_condition=path.stem,
                    time_points=len(times),
                    reference_quadrature_order=order,
                    max_abs_circuit_mean_difference=float(
                        np.max(abs(quantum - run["moment_history"][:, :2]))
                    ),
                    max_abs_characteristic_mean_difference=float(
                        np.max(
                            abs(
                                reference
                                - run["prepared_cell_reference_moments"][:, :2]
                            )
                        )
                    ),
                )
            )
    return checks
