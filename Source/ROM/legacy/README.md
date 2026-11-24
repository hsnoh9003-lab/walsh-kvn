# Retained standalone ROM helpers

These helpers are retained for external users and independent checks, outside
the current reproduction entry points:

- `force_observables.force_curves`: evaluates archived force maps on saved Walsh–KvN states.
- `model.raw_drift`: converts normalized model drift to physical-coordinate drift.

The former imports `Source.ROM.force_observables.force_curves` and
`Source.ROM.model.raw_drift` still resolve to these same implementations. Active
ROM/KvN calculations continue to use the main `model`, `ensemble`, and
`field_budget` modules.
