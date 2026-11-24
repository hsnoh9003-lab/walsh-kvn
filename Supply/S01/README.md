# Supplementary Figure S1 physical forecast audit

The short and extended forecast panels share a 0–22% vertical scale at a two-column width of 180 mm. The grouped bars compare the cubic forecast and three-mode projection for each case. The six Re=100 cases use the recovered initial
perturbations and the same frozen spatial modes and radial/phase model.

From the package root:

```bash
python -m Source.CFD.physical_audit
python -m Reproduction.run supply S01 --mode from-data
python Supply/S01/code/extract_data.py
```

The first command **calculates** both CSVs from projected FEM coordinates and
norms. It does not copy or fit the plotted error values. Its compact inputs are
extracted from real full-field FEM output and retain exactly the information
needed for the stated aggregate field-error metric. Their definitions and the
full FEM rerun command are in `Data/Physical_Forecast_Audit/README.md`.

The second command produces native PDF/SVG/PNG bars and text. The third exports
the computed short- and extended-window CSVs into `data/plot_values/`.
The [case settings](data/plot_values/case_settings.csv) give the initial streamfunction amplitude, centre and width for each of the six runs. Panel c lists the same six settings and the streamfunction used by the finite-element solver. Only A supplied the modes and fitted coefficients. B–F use that fixed model.

Both numerical panels were verified against fresh six-case FEM integrations.
The short-window CSV is byte-identical to the previously retained table.

The origin is the cylinder centre. Distances are in cylinder diameters. The streamwise position increases downstream, and the signed offset from the wake centreline is positive upwards. With $s=\min(1,\max(0,(x-0.7)/0.5))$, the common cutoff in panel c is $C(x)=s^3(10-15s+6s^2)$. The solver forms the velocity perturbation from $(\partial_y\delta\psi,-\partial_x\delta\psi)$ before enforcing its boundary and discrete divergence constraints.

[Figure S1 with its caption](pdf/Figure_S1_with_caption.pdf) explains how the common cutoff keeps the added velocity zero at the surface of the cylinder. The [artwork](pdf/Figure_S1.pdf) contains panels a–c.
