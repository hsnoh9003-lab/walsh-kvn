# POD and SINDy modules

The current reproduction entry point is:

```bash
python -m Source.POD_SINDy.replay_frozen_basis
```

It imports the active modules directly:

- `pod.py`: reflection, symmetric/antisymmetric decomposition, oscillatory POD pair, modal projection, and zero-intercept shift closure.
- `fit.py`: the reflection-constrained, training-RMS-scaled SINDy fit.
- `identification.py`: the shared monomial library, time derivatives, and guarded ODE prediction.
- `evaluate_degree_cases.py`: additional-case and blocked-validation evaluation.

`legacy/` retains historical alternatives and standalone analysis helpers. These
are callable but are not dependencies of the current snapshot-to-model replay.
The historical mean-window selection in `legacy.original_rom_functions.make_basis`
is distinct from the recorded mean-window protocol in `replay_frozen_basis`.

| Former location | Implementation now |
| --- | --- |
| `recovered_original.original_rom_functions.project`, `.closure` | `pod` |
| `recovered_original.shift_pipeline.reflect`, `.decompose`, `.fit_oscillatory_pair` | `pod` |
| Both copies of the polynomial/derivative/ODE helpers | `identification` |
| Generic unconstrained SINDy regression and equation-display helpers | `legacy.identification` |
| Historical basis construction, physical-model check, weighted fitting | `legacy.original_rom_functions` |
| Alternative shift construction and reduced-model diagnostics | `legacy.shift_pipeline` |

The three files under `recovered_original/` now only forward imports. They retain
old import paths for external collaborators; they contain no duplicate numerical
implementations. Historical names such as `identification.learn` also remain
available through lazy compatibility lookup, while new callers should import
these optional helpers from `legacy.identification` explicitly.

To check the layout, old import compatibility, and numerical input guards:

```bash
python -m Tests.verify_pod_module_layout
python -m Tests.verify_core_guards
```
