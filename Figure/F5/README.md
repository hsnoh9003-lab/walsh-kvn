# Figure 5 — nonlinear benchmarks and three-coordinate wake transport

Current manuscript v100 / SI v103. The plotter `code/plot.py` reads the published inputs from `Source_Data/Figures/Fig_5/plot_inputs/` using `Source.data_locations.figure_data('F5')`.

From the code root:

```bash
python -m Reproduction.run figure F5 --mode plot
python -m Reproduction.verify_uncapped_coupled_resources
python -m Reproduction.compile_one_clean_polynomial_benchmarks
```

Plot inputs include model-accuracy panels, scalar density tests, the coupled-cubic resource designs, wake field means, preparation checks and error budgets. Underlying RE selections are in `Source_Data/Shared_source_data/RE_common/RE/`; the transport input records are in `Source_Data/Shared_source_data/MPS_KvN_inputs/`. The current coupled benchmark uses one clean arithmetic work qubit. The fastest designs are checked without a physical-qubit cap; no obsolete v38-sized data mirrors in `Figure/F5/data/` are needed.
