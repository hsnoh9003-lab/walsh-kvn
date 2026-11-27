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

Historical variants are retained in the separate internal archive and are not
importable production modules. The active snapshot-to-model replay above remains
the scientific reference for the public execution.

| Former location | Implementation now |
| --- | --- |
| POD projection and shift closure | `pod` |
| Reflection decomposition and oscillatory POD pair | `pod` |
| Polynomial, derivative and guarded ODE helpers | `identification` |
| Reflection-constrained fit | `fit` |

The release does not distribute compatibility import shims or alternative
historical implementations. Original producer programs that are necessary to
replay publication CSVs are in `Source/Data_Production/producers/`.

Additional implementation-regression tests are maintained outside the public
release; the public end-to-end workflow is documented in
`../../docs/REPRODUCTION_GUIDE.md`.
