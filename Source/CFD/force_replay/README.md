# Recovered original Re100 drag/lift generator

The four original Python files and requirements.txt are preserved without byte
changes from Cylinder_Forces_Re100_20261110.zip. The original portable solver is
now at `audit/cfd_force_replay_original.py`; the working `cfd_force_replay.py`
has comments, a shorter module docstring and formatting edits for readability.
Its equations, coefficients, variable names and execution order are unchanged.

`provenance.json` records the archived hashes and the working-copy hash separately.
The wrapper's `--verify` checks those hashes and compares the working solver's
executable AST against the original, ignoring only the module docstring.
`audit/` holds historical source, including original default input paths;
use the public wrapper rather than executing historical audit files.

From the research-code root:

```bash
python -m pip install -r Source/CFD/force_replay/requirements.txt
python -m Reproduction.reproduce_cfd_forces --verify
```

The wrapper supplies the external input and keeps generated outputs in Build.
The pressure and viscous surface integrations are in pressure_x/pressure_y,
viscous_x/viscous_y and force_components. Force on the body uses the outward
fluid-domain normal (into the cylinder), with integrand
`p*n - nu*(grad(u)+grad(u).T)*n`. The full-domain coefficients are twice this
force because rho=U=D=1. C_D is positive downstream and C_L is positive upward.
`pressure_valid` means a pressure solution was supplied; it is false for the
initial perturbed field and true for each computed time step.

The archived raw and processed force references are under
`Source_Data/Shared_source_data/CFD_ROM/FEM/force_reference/`. The original
velocity snapshot reference is reused from
`Shared_source_data/CFD_ROM/CFD_ROM/Re100/data/source_snapshots.npz`.

The detailed internal recovery report is retained outside the public GitHub
code in `Separated_Figure/Private_Archive/Docs/CFD_Drag_Lift_Source_Recovery_20261125.md`.
For public reproduction, use the entry point documented above, which resolves
inputs from the Zenodo DOI or an explicit local dataset directory.
