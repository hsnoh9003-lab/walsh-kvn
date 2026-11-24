# Original Re100 finite-element replay

The recovered archive `FEM_Code_20261101(2).zip` contains the original Re100
source and initial mesh/steady field. The current transient solver keeps its
numerical operations and adds an explicit `--input` path; its default input is
resolved from the shared local/Zenodo manifest. The bundled historical NPZ is
not read by default. Reynolds-parameterized steady and transient solvers are in
`../fem/`, with recorded timesteps for Re100, Re200 and Re300.

Run the exact original transient configuration from the package root:
`python Source/CFD/fem/run_fem.py --reference-replay`.
Run new complete solves: `python Source/CFD/fem/run_fem.py --reynolds 100 200 300`.
Install the optional dependencies in `../fem/requirements.txt` first.

These are explicitly specified Taylor--Hood finite-element benchmarks, not
OpenFOAM input decks. Neither command silently replaces the adopted frozen
CFD fields, POD modes or selected SINDy coefficients. All six original Re=100 transient cases were subsequently reintegrated from the retained steady field for the S1 audit; the steady-state solve itself was not rerun. The saved-snapshot
POD analysis is now independently replayed as described below.

## Recovered snapshot-to-POD replay

The original complete Re=100 sampled velocity series has been recovered and
is retained once in the DOI dataset at
`Shared_source_data/CFD_ROM/CFD_ROM/Re100/data/source_snapshots.npz`. Run:

```bash
python -m Source.POD_SINDy.replay_frozen_basis
```

It rebuilds the published spatial modes, projected coordinates, shift closure
and selected 12-coefficient model from saved snapshots, before loading the
retained basis for comparison. The POD subspace is checked by principal angles. The oscillatory pair is
Procrustes-aligned and the shift sign is matched before coordinate/coefficient
comparison; a deliberately rotated-pair and shift-sign control also passes.
The current coefficient difference is below 3e-14. This closes the saved-snapshot analysis gap;
the six-case S1 transient CFD rerun and both-panel numerical recovery were
recorded in the private internal audit archive. The public regression can be
run with `python -m Source.POD_SINDy.replay_frozen_basis` and the DOI data.
