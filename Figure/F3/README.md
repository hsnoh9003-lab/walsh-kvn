# Figure 3 — error-corrected resource estimates

Current article Figure 3 (manuscript v100). The selected-design data are in `Source_Data/Figures/Fig_3/plot_inputs/` and the canonical RE records are in `Source_Data/Shared_source_data/RE_common/RE/`. The release retains the historical `v38_reference_selected.csv` filename for the current numeric table; the filename does not indicate an obsolete scientific model.

From the package root:

```bash
python -m Source.RE.verify
python -m Reproduction.generate_fig3_data
python -m Reproduction.run figure F3 --mode plot
```

For a diagnostic view beyond the publication clipping window, run `python Figure/F3/code/plot.py Build/F3_full_range --full-range`. The sole publication plotter is `code/plot.py` (supported by its `base.py` module). The RE fastest-design comparison uses a relative time tie tolerance of `1e-6`, with physical qubit count breaking ties.
