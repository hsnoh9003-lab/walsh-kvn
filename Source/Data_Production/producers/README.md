# Preserved scientific programs

These files retain the scientific implementations used by the selected replay
jobs. Their original Python files are preserved; current input resolution,
isolated execution and publication export are explained in the documented
wrappers one directory above.

Use `replay_history.py --help` for the jobs listed in `jobs.json`. The wrapper
checks the declared input hashes, copies the selected program into a temporary
directory, supplies its expected relative paths and collects only its declared
outputs. `retained_experiments.py` separately replays the retained budget and
preparation routines. Running these old programs in place can otherwise put
outputs beside their inputs.

| Directory | Scientific role | Reader entry point |
| --- | --- | --- |
| `details/` | Numerical field and sampling diagnostics | `../replay_history.py` |
| `forcefit/` | Fit drag/lift observables to supplied force references | `../replay_history.py`; this does not regenerate CFD surface forces |
| `physical/` | Retained physical-cost comparison | `../replay_history.py` |
| `pod/` | Project supplied flow snapshots and reconstruct the reduced basis | `../replay_history.py` |
| `q5/` | Quintic-grid refinement with the retained transport/preparation implementations | `../replay_history.py` |
| `sindy/` | Polynomial model fitting and degree/cost comparisons | `../replay_history.py` |
| `experiments/` | Relative moment budgets and matrix product state preparation | `../retained_experiments.py` |

The historical code and input schemas document the retained calculation. The
current plotting folder is `Figure_Plots/`; old figure labels inside these
programs are translated by the wrappers where required.
