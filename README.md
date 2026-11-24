# KvN research code — manuscript v100 / Supplementary Information v103

This is the **GitHub code** for the article, with current main-text numbering **Figures 1–5**, **Supplementary Figures S1–S10**, and **Extended Data Figure 1**. Numerical/plot inputs are published separately as a Zenodo Dataset at **[DOI: 10.5281/zenodo.24505927](https://doi.org/10.5281/zenodo.24505927)**. Figure 2 and Supplementary Figures S2, S6 and S7 are image-only and have no publicly distributed plotter or artwork. **S7's explicitly cited numerical Source Data CSV remains on Zenodo.**

## Source Data acquisition (automatic by DOI)

To work entirely offline, unpack the source dataset beside this code:

```text
publication/
  Codes/            # this GitHub repository
  Source_Data/      # separately downloaded Zenodo dataset
    manifest.csv
    Source_Data_Index.csv
```

Alternatively, set `KVN_SOURCE_DATA` to the absolute path of the extracted dataset. **If neither is supplied, the first data access queries Zenodo record 24505927 and downloads its `Source_Data.zip` automatically**, verifying Zenodo's checksum and caching it in `~/.cache/kvn_walsh/zenodo_24505927/Source_Data/`. Set `KVN_SOURCE_DATA_CACHE` to override the cache's parent directory. The DOI has been reserved, but automatic downloads work only **after the dataset is publicly published**. For the release upload, compress `Source_Data/` as one archive named exactly `Source_Data.zip` and attach it to that Zenodo record. Existing local copies are always preferred, and numerical inputs are never overwritten by ordinary plotting.

Automatic download is a fallback when no dataset directory with `manifest.csv`
is found. It does not repair individual files missing from an existing dataset.
An invalid explicit `KVN_SOURCE_DATA` path raises an error instead of silently
switching datasets.

To locate or download once:

```bash
python -m Source.data_locations --download
```

The `Source/data_locations.py` resolver looks up historical provenance paths in the downloaded `manifest.csv` without duplicating the numerical input tree inside GitHub. Generated audits go to `Build/` and are git-ignored.

### Shared CFD/POD/ROM input paths

Published CFD steady fields, POD training inputs, and ROM data use the same
`Source.data_locations` resolver and Zenodo manifest. Historical `Data/...`
strings passed to `source_file`/`published_path` are manifest keys, not directories
to recreate inside `Codes/`.

Both CFD launchers accept `--data-root`; this explicit override also clears any
previous resolver/manifest cache in the same Python process. For Python callers,
use `Source.data_locations.set_source_data_root(path)` when switching datasets.
POD fitting, model loading, and physical-audit imports defer data access until
it is actually needed.

`fem/simulate.py` and `replay_fem/cfd_benchmark.py` accept `--input` for a newly
generated steady-field NPZ. Without it they use the published Re100 coarse field.
Other Reynolds numbers or meshes require their own explicit steady input;
`run_fem.py --reynolds ...` generates and passes the matching input automatically.
The separately preserved force solver is still run through
`Reproduction.reproduce_cfd_forces`, which supplies its resolved input path.

## Single-notebook figure reproduction

Open **`Reproduce_Figures.ipynb`** from this `Codes/` directory. Each figure section includes the **verbatim v100/SI103 caption** and a code cell invoking the existing dedicated plotting script. The notebook covers **F1, F3–F5, S1, S3–S5, S8–S10 and Extended Data Fig. 1**. Fig. 2 and Figs. S2, S6, S7 have no public notebook cells because they are separately archived, image-only artwork. To run the notebook install Jupyter (for example `python -m pip install -r requirements_notebook.txt`); F1 SVG rendering additionally requires Inkscape.

## Reproduction commands

Run from `Codes/` with Python 3.11+ and the dependencies documented in the relevant source module:

```bash
# Quick RE, science, shot-count and selected plot checks
python -m Reproduction.quick_check --plots

# Current RE-selected records and Figure 3 inputs
python -m Source.RE.verify
python -m Reproduction.generate_fig3_data
python -m Reproduction.run figure F3 --mode plot

# Figure 4, Figure 5 and Supplementary Figures S8–S10
python -m Reproduction.run figure F4 --mode plot
python -m Reproduction.run figure F5 --mode plot
python -m Reproduction.run supply S08 --mode plot
python -m Reproduction.run supply S09 --mode plot
python -m Reproduction.run supply S10 --mode plot

# One-clean BE–KvN and LCHS–KvN input comparisons
python -m Reproduction.build_comparison_workloads
python Comparisons/KvN_LCHS/verify.py
python Comparisons/BE_KvN/fullwindow/fullwindow.py

# Table 1 and actual Gidney trial counts
python -m Reproduction.export_table1 --check
python -m Reproduction.verify_cultivation_shots
```

The LCHS source audit additionally requires `h5py`; source-circuit tooling may require `qiskit`, `stim` or other optional dependencies. The full CFD/MPS and RE parameter sweeps require substantially more time than the short checks above.

### Gidney cultivation source-data generation

The two source files in `Source/Gidney/` reproduce the input pipeline that
precedes the RE preparation-table selection:

```bash
# Inspect the actual 80 per-distance shot budgets without executing shots
python -m Source.Gidney.generate_raw_stats

# Acquire one raw Sinter configuration using the original Cultiv checkout
python -m Source.Gidney.generate_raw_stats --cultiv-repo /path/to/cultivation --p 5 --d2 7 --execute

# Rebuild P_prep--e_prep tradeoff from all per-distance Sinter stats (default: Build/)
python -m Source.Gidney.build_tradeoff --raw-root Build/Gidney/raw_stats

# Independently validate all eight published tradeoff CSVs without Sinter/Cultiv
python -m Source.Gidney.build_tradeoff --verify-published
```

The raw runner requires the original Strilanc cultivation repository, Cultiv,
Sinter and the desaturation decoder. The tradeoff builder also requires SciPy
for exact binomial intervals. For p=5e-4 the 10-billion-shot study uses
**1-dB gap thresholds**, while other rates use 5-dB thresholds. The two
files output only to `Build/Gidney/` by default and do not silently change
any published CSV or current RE result.

For a full 2–9 physical-error rate regeneration of the supplementary resource plot inputs, run `python -m Reproduction.generate_supplement_data --rates 2 3 4 5 6 7 8 9` to write results under `Build/`. The optional `--publish-figure-data` is **not appropriate for a read-only DOI cache**; it is intended only when the researcher has intentionally provided a writable staging copy through `KVN_SOURCE_DATA`. The Litinski model comparison has a separate producer, `Reproduction.generate_litinski_data`, followed by `Reproduction.update_litinski_s8b`.


## CFD drag and lift: recovered original solver

The original 2026-11-10 force-generating CFD solver is preserved under
`Source/CFD/force_replay/audit/`. The readable working copy is checked against
that original by source hashes and executable AST comparison. It integrates both pressure and viscous contributions
to **C_D(t) and C_L(t)** during each time step. Use the common data-root wrapper:

```bash
python -m pip install -r Source/CFD/force_replay/requirements.txt
python -m Reproduction.reproduce_cfd_forces --verify
```

The default is Re=100, dt=0.02 and final time 90 (4,500 steps).
Outputs go to `Build/CFD/force_replay/`: `forces.csv` includes the pressure-invalid
t=0 row, `forces_cfd.csv` contains 4,500 valid rows, and `fig4f_fem_forces.csv`
contains the 3,000 rows with time <=60 used in Figure 4. Published inputs are
never overwritten. The missing steady input and original force references are
now included in Source_Data; the existing velocity snapshots are reused.

`--data-root /path/to/Source_Data` overrides `KVN_SOURCE_DATA`. For an already
completed output directory, use `--export-only --verify --output /path/to/run`.
A short path/solver check is `--end 0.04 --output Build/CFD/smoke`; it does not
claim to reproduce the full published figure. Verification is strict and uses
the pinned dependencies in the original requirements file.

The reviewed general CFD files in `Source/CFD/fem/` retain their input validation
and readability changes. The `--reference-replay` launcher also uses the common
Source_Data location. The original force replay is a separate, fixed Re100 case.

```bash
python -m Tests.verify_cfd_reproduction
```

The optional mesh/time-step sensitivity driver is separate from the fixed
force replay:

```bash
python -m Source.CFD.refinement --output Build/CFD_Refinement
```

It compares three mesh levels and three time steps at Re=100, with an
eight-hour wall-time limit. Outputs include mean drag, root-mean-square lift
and shedding frequency, together with stationarity and refinement checks.
It does not change the published dataset. Adding this driver does not establish
mesh convergence: its full sensitivity study was not run during this merge.

Detailed internal numerical review records are held separately from this public GitHub package; the input provenance is recorded in Zenodo's `manifest.csv`.

## POD/SINDy module organization

Current POD construction is in `Source/POD_SINDy/pod.py`; the shared
identification helpers have one implementation. Historical alternatives are
retained in `Source/POD_SINDy/legacy/` and `Source/ROM/legacy/`. Old import paths
remain as compatibility forwards. See `Source/POD_SINDy/README.md`.

## Repository layout

- `Source/RE/`: code-distance/architecture search, Gidney preparation costs, MSF factory models, timing, gates, and RE verifications.
- `Source/KvN/`, `Source/ROM/`, `Source/CFD/`, `Source/POD_SINDy/`: transport, state preparation, drift fitting and continuum/finite-element comparisons.
- `Comparisons/BE_KvN/`, `Comparisons/KvN_LCHS/`: separate quantum-circuit comparisons, with their upstream oracle/phase input data in `Source_Data`.
- `Reproduction/`: numerical generators and unified entry points; special-purpose scripts formerly in `Data/` are in `Reproduction/auxiliary/`.
- `Figure/`, `Supply/`, `Extended_Data/`: figure-specific plotting code and exact current captions. Python plotters read the Zenodo `Source_Data/Figures/.../plot_inputs/`; rendered artwork is a generated output, not tracked on GitHub.
- `Hardware/`: IBM circuit-generation and hardware replay code. QASM/QPY circuit files are executable assets rather than duplicate plot tables.
- `Build/`: auto-created, git-ignored scratch output for numerical checks and compilation; not canonical source data.
- `Tests/`: reproducibility and numerical regression tests.
- `Reproduce_Figures.ipynb`: all publicly reproducible figures, one caption/code-cell pair per figure.

`Source_Data/Source_Data_Index.csv` and `Source_Data/manifest.csv` are the authoritative mappings from figures and table entries to inputs. Do not reconstruct a historical `Data/` mirror before running the current code; `Source/data_locations.py` already resolves the new paths.

This GitHub release excludes manuscript drafts, translations and internal reviews. They are preserved in the privately held `Separated_Figure/Private_Archive/` within the hand-off bundle, alongside the approved figure outputs.
