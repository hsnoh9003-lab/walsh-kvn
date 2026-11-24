# Coupled cubic ensemble means

The four panels track the mean coordinates of ensembles released in the interior, near the left boundary, near the stable point at (1, 0) and near the unstable point at (0, 1). The circuit uses 16 × 16 cells and 64 time steps up to time 1. Blue squares are shown at steps 0, 8, …, 64. The dashed curves average classical trajectories with the same initial cell probabilities. Arrows follow the drift in (S18a)–(S18b). Insets magnify the near-stable and near-unstable paths.

Run `python -m Reproduction.run supply S05 --mode plot` from the package root. The circuit and classical means are in `data/plot_values/coupled_mean_trajectories.csv`. The four panels span 180 × 53 mm and align with the input categories in Fig. S4.
