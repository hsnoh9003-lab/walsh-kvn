# Spatial and temporal refinement

The two rows show the radial wake and four polynomial drifts. The figure is 180 × 105 mm. From the package root, run `python -m Reproduction.run supply S03 --mode from-data`. The output is written to `Supply/S03/reproduced/`.

The six wake cell-count and step-count combinations are in `data/plot_values/wake_refinement.csv`. The density uses the archived 64-cell populations. `code/regenerate_refinement.py --native` replays the wake refinement. The two polynomial CSV files contain four resolutions for each of four drifts. `code/extract_polynomial_sweeps.py` selects these rows from the complete archived sweeps in `source_data/full_sweep/`.

Panel b groups the six measurements by spatial grid. Its rings mark (64 cells, 33 steps) and (128 cells, 66 steps), which share a step-duration-to-cell-width ratio. The 128-cell, 33-step control uses the same input and duration.

Panel d compares two references for the same final grid moments. Open markers use classical trajectories with the circuit-prepared cell masses spread uniformly over their respective cells. Filled markers use trajectories with the target continuous truncated Gaussian. Panel e uses the same prepared amplitudes and grid for the circuit and the matrix exponential, and minimizes their final Euclidean distance over one global phase.
