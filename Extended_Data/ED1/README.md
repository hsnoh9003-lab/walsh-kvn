# Extended Data Figure 1 full-acquisition analysis

Panels a,b show the four-step, 16-cell run at 16,000 shots per circuit. Panels c,d show the six-step, 16-cell run at 32,000 shots per circuit. The figure uses every shot from these circuits. The source table in `Hardware/IBM/data/ibm_mps_calibration_full_precision.csv` also records the separate one-step, 32-cell control. An identical copy is in `data/Extended_Data_Figure_1.csv`.

Caps extend past the symbols so that all nonzero error bars are visible at publication size. The plotting script also generates the publication Source Data and reference diagnostics under `Data/` from the full acquisition.

Measured points, mitigated values and 1,000-resample bootstrap errors use the full data. The estimator is Supplementary equation (S82). Each resample refits the reference contrast. The plotted bars exclude systematic model mismatch.

From the package root run `python -B Extended_Data/ED1/code/plot.py`. To replay full-count calibration and offline circuit checks first run `python -B Hardware/IBM/code/analyze_mps_ibm.py`. The original fixed-seed subset analysis is preserved separately for provenance.

The canvas remains 172.72 × 118.745 mm, with sans-serif 6–7 pt labels. `pdf/` contains the vector PDF, `artwork/` editable SVG and 300-dpi PNG, and `data/` the full-precision CSV.
