"""Compile and compare the native KvN kernels on the current platform.

Checks OpenMP versus serial matching and the native replay against small
reference operations. Running this on Linux validates Linux compilation only;
Windows support requires executing the same check on Windows. A C++ compiler
and the matching OpenMP runtime are required.

Run from Codes/: python -m Tests.windows_cpp_smoke
"""
from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from pathlib import Path

import numpy as np

from Source.data_locations import generated_outputs_root
from Source.KvN.three_coordinate_time_convergence import matching_kernel
from Source.KvN import native_replay, transport


def main() -> None:
    r = (np.arange(4) + 0.5) / 4
    z = (np.arange(4) + 0.5) / 4
    modes = np.arange(-2, 2)
    coefficients = np.zeros((10, 3), dtype=np.float64)
    coefficients[1, 0] = 0.31
    coefficients[2, 1] = -0.23
    coefficients[3, 2] = 0.11
    coefficients[6, 0] = 0.07
    coefficients[9, 2] = -0.04
    layers, tables = transport.rotation_tables(r, z, modes, coefficients, 0.05, 2)
    rng = np.random.default_rng(20261126)
    initial = rng.normal(size=64) + 1j * rng.normal(size=64)
    initial = np.ascontiguousarray(initial / np.linalg.norm(initial), dtype=np.complex128)

    serial = initial.copy()
    parallel = initial.copy()
    args = (layers, tables, 4, 4, 4, len(layers), tables.shape[1])
    transport.kernel()(serial, *args)
    output = generated_outputs_root() / "Build/Windows_CPP_Smoke"
    step, _ = matching_kernel(output, 2)
    step(parallel, *args)
    np.testing.assert_allclose(parallel, serial, atol=1e-13, rtol=1e-13)
    if np.max(np.abs(serial - initial)) < 1e-8:
        raise AssertionError("The test propagation was unexpectedly trivial")
    print("PASS: serial/parallel matching DLL/SO and OpenMP")

    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    angle = 0.12
    program = [("h", (0,), 0.0), ("cx", (0, 1), 0.0), ("t", (1,), 0.0), ("rz", (0,), angle)]
    actual = native_replay.run(state, program)
    expected = np.array([
        np.exp(-0.5j * angle), 0, 0,
        np.exp(1j * (np.pi / 4 + 0.5 * angle)),
    ], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    print("PASS: native Clifford/T/Rz replay DLL/SO")
    print("Windows native-kernel smoke test completed")


if __name__ == "__main__":
    main()
