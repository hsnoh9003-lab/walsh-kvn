# v100 / SI103 — public code and Zenodo data hand-off

Source Data DOI: **10.5281/zenodo.24505927**. This release does not claim that
the record is already public; the user must upload/publish `Source_Data.zip`.

## Exact release boundary

- `Codes/` — GitHub repository: source, comparisons, reproduction, IBM hardware,
  tests and figure plotters. `Reproduce_Figures.ipynb` has 12 independent
  figure-reproduction cells with the exact latest captions.
- `Source_Data/` — Zenodo dataset: 481 entries in each index and manifest,
  each with an independently checked SHA-256. All retained raw data and results
  have the same bytes as the input archive.
- `Separated_Figure/` — private: approved and freshly replayed figure artwork,
  four excluded image-only figure artworks (with their drawing inputs), current
  manuscript and translations, and internal Build/Docs records. Do not upload
  this directory to Zenodo or GitHub.

## Changed executable behavior

1. `Source/data_locations.py` first uses the sibling Source_Data directory or
   `KVN_SOURCE_DATA`, then the local cache, finally the fixed Zenodo DOI.
   `Source/zenodo_download.py` downloads the ZIP, verifies its declared digest,
   rejects unsafe archive members and extracts it into a persistent cache.
2. Plotting remains in the existing per-figure scripts; notebook cells call
   those scripts. There is no rewritten scientific plotting implementation.
3. `Reproduction.quick_check` and `Reproduction.reproduce_all` skip the four
   unpublished, image-only figures. Generated reports now go to `Build/`.
   A Figure 3 diagnostic comparison now obtains its CSV via the common
   Source Data resolver.
4. `.gitignore` excludes Build outputs, compiled libraries, notebooks'
   checkpoint files and rendered figure outputs from GitHub.
5. Historical `extract_data.py` entry points export exact figure inputs from
   the DOI dataset into the local `Build/Figure_Source_Exports/` tree. Old
   `Figure/*/data`, `Supply/*/data` and out-of-date `data_sources.json` mirrors
   were stored privately, not duplicated on GitHub. The plotting routines
   already read the public Source Data resolver and retain their implementations.

## Verification performed

- 481/481 source data SHA-256 digests matched the manifest.
- All 12 notebook figure code cells were executed successfully, including
  F1, F3–F5, S1, S3–S5, S8–S10 and Extended Data Fig. 1.
- `python -m Reproduction.quick_check --plots` succeeded, including 384
  resource-estimation design checks and current F3/F5/S8–S10 plot tests.
- Comparison workload verification passed (15 source-derived fields).
- Cultivation tradeoff verification passed for all eight physical error rates,
  including the 790-row 5e-4 experiment.
- `python -m Tests.verify_zenodo_integration` passed using a mocked public
  Zenodo record, ZIP and checksum. Actual remote DOI download cannot be
  verified until the dataset has been published.

## Supplementary Figure S7's cited Source Data

The currently supplied caption of **Supplementary Fig. S7** explicitly cites
`Source Data for Fig. S7`. Its numerical source-data summary and supporting baseline-policy table are
publicly retained at `Figures/Fig_S7/plot_inputs/Source_Data_Figure_S7.csv`
and `Figures/Fig_S7/plot_inputs/baseline_policy_optima.csv`, respectively.
Both files are indexed in the current manifest and source-data index.
Neither CSV is read directly by the SVG renderer; S7's author-drawn SVG and
figure source code remain in the private `Separated_Figure/` archive.

The Zenodo DOI is fixed here. The final GitHub repository URL has not been
provided, so no GitHub URL has been invented in the Source Data README.

## CFD/POD/ROM update merged on 2026-11-25

The supplied CFD changes were merged into the deduplicated release. The
481 source-data payloads, index, manifest, S7 supporting table and all existing
private files, including the manuscript and Supplementary Information, retain
their previous bytes. Generated outputs from the older incoming bundle were
not imported.

- Original force-generator files remain in `Source/CFD/force_replay/audit/`.
  The readable working copy has identical executable syntax to the archived
  solver; its original and working-copy hashes are verified separately.
- CFD inputs, POD reconstruction and ROM loading share the external data
  resolver. Explicit dataset changes clear cached paths. Fresh CFD audit
  inputs default to the run output directory.
- Active POD routines and historical alternatives are separated with old
  import paths retained. Derivative inputs, model names and the inner radial
  boundary in trajectory-exit diagnostics have regression checks.
- An optional bounded mesh/time-step sensitivity driver is included. It is
  not the original force generator and does not replace published results.

Validation for this merge:

- All 23 new path, input-guard, module-layout and refinement-statistics tests
  passed; the existing CFD reader/input tests also passed.
- `python -m Reproduction.quick_check --plots --pod-basis` passed all 17
  checks, including snapshot-to-POD reconstruction and drag/lift sampling.
- A fresh two-step CFD solve (dt=0.02, end=0.04) reproduced the archived force
  components exactly. The general and replay solvers matched its velocity
  states; the new refinement solver matched its forces and all saved state
  arrays, including pressure.
- The previously completed 4,500-step replay was rechecked: all three force
  CSVs matched byte for byte, and saved snapshots and final-state arrays
  matched exactly. This check reused that run rather than performing a second
  full solve.
- The DOI downloader test passed with a mocked record/archive; no live remote
  publication or download is asserted. The full mesh/time-step study was not
  run.

The private `CFD_Merge_Verification_20261125.json` records the merged paths and
these checks. No Source Data archive replacement is required for this code-only
update.
