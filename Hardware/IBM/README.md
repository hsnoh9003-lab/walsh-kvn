# IBM MPS preparation and full-count KvN transport

The primary jobs use bond-two matrix product state preparation followed by nonlinear radial transport. Four qubits encode 16 cells; the five-qubit, 32-cell control performs one T/4 step. No 64-cell hardware transport is reported.

The caption of Extended Data Fig. 1 gives the propagation steps and shots per circuit for the 16-cell runs. The four-step job also included the 32-cell control. The complete [acquisition records](data/run_metadata.json) list all 12 and 13 circuits in the four- and six-step jobs, respectively, together with the device time for each job.

`data/ibm_mps_calibration_full_precision.csv` supplies the current figure. `code/analyze_mps_ibm.py` reproduces full-count estimates, 1,000-resample bootstrap errors, logical/device circuit checks and reference diagnostics. Calibration uses ideal-input value plus transport–reference difference divided by fitted contrast, as in Supplementary equation (S82). Each bootstrap refits the contrast. Bars exclude systematic noise-model mismatch and temporal correlations.

Historical fixed-seed subset data remain in `data/reduced_shots/` with selection and validation records. `code/reduce_shots.py` reproduces that archived analysis. The current figure reads the full-count CSV directly.

Matched noise references retain the input and logical address skeleton with zero transport angles. They differ from the continuum ROM reference for the protected field-error target. These hardware circuits have no phase readout or field reconstruction. Earlier exact StatePreparation and zero-noise-extrapolation jobs remain controls.

Offline reproduction from the package root:

    python -B Hardware/IBM/code/analyze_mps_ibm.py
    python -B Extended_Data/ED1/code/plot.py

Provider submission scripts are separate. The 64-cell input-family comparison is an offline calculation.
