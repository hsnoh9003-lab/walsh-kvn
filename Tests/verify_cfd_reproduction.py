"""Check reviewed CFD input handling and published/historical force readers."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from Source.CFD.snapshots import interpolate_snapshots
from Source.CFD.forces import read_fem_forces, read_force_coefficients
from Source.data_locations import REPO_ROOT, source_file


def main():
    times = np.array([0., 1., 2.])
    fields = np.array([[1., 3.], [3., 7.], [5., 11.]])
    actual, audit = interpolate_snapshots(times, fields, [0., .5, 1., 2.])
    np.testing.assert_array_equal(actual, [[1., 3.], [2., 5.], [3., 7.], [5., 11.]])
    assert audit[0] == dict(left=0., target=0., right=0., right_weight=0.)
    for outside in (-.1, 2.1):
        try:
            interpolate_snapshots(times, fields, [outside])
        except AssertionError:
            pass
        else:
            raise AssertionError("An out-of-range snapshot was accepted.")

    published = source_file("Figure/F4/data/force_output/fig4f_fem_forces.csv")
    archived = source_file("Cylinder_Forces_Re100_20261110/results/forces_cfd.csv")
    expected = pd.read_csv(published)
    for given in (None, published, published.parent, archived, archived.parent):
        pd.testing.assert_frame_equal(read_fem_forces(given).reset_index(drop=True), expected)
    assert len(read_force_coefficients()) == 10

    # The reviewed simulator must reject invalid settings before any disk writes.
    sys.path.insert(0, str(REPO_ROOT / "Source/CFD/fem"))
    from Source.CFD.fem import simulate
    invalid = []
    for key in ("dt", "end", "reynolds"):
        for value in (0., -1., float("nan"), float("inf")):
            settings = dict(dt=.02, end=.04, case="train", reynolds=100.)
            settings[key] = value
            invalid.append(settings)
    invalid += [dict(dt=.02, end=.01, case="train", reynolds=100.),
                dict(dt=.02, end=.04, case="unknown", reynolds=100.)]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with patch.object(simulate, "ROOT", root):
            for settings in invalid:
                try:
                    simulate.run("coarse", **settings)
                except ValueError:
                    pass
                else:
                    raise AssertionError(settings)
        assert not list(root.iterdir())
        for case in simulate.CASES:
            simulate.validate_inputs(.02, .04, case, 100.)
        missing = root / "missing"
        output = root / "must_not_be_created"
        command = [sys.executable, "-B", "-m", "Reproduction.reproduce_cfd_forces",
                   "--data-root", str(missing), "--output", str(output), "--end", ".04"]
        result = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(),
                                capture_output=True, text=True)
        assert result.returncode == 2 and "--data-root" in result.stderr
        assert not output.exists()
        result = subprocess.run(
            [sys.executable, "-B", str(REPO_ROOT / "Source/CFD/fem/run_fem.py"),
             "--reference-replay", "--data-root", str(missing)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 2 and "--data-root" in result.stderr
    print("PASS: snapshot endpoints/ranges; five force-reader routes; "
          "14 rejected settings, seven valid cases; missing-data early exits.")


if __name__ == "__main__":
    main()
