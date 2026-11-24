# Figure 4 — wake reconstruction and force observables

The current Figure 4 data are under `Source_Data/Figures/Fig_4/plot_inputs/`, with shared fitted CFD/ROM arrays in `Source_Data/Shared_source_data/CFD_ROM/`. The polynomial validation, model comparison, field-error curves and 815-shot drag/lift data are referenced through `Source/data_locations.py`.

From the code root:

```bash
python -m Reproduction.run figure F4 --mode plot
python Figure/F4/code/rebuild_error_data.py
```

The only publication plotter is `code/plot.py`. The second command checks the retained selected model and can regenerate the panel-specific scientific source; it does not refit the model. Fitted force terms and finite-shot uncertainty are separately available in the Figure 4 Source Data folder. Figure numbers remain those of article v100.
