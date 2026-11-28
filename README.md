# Nonlinearity-preserving quantum simulation of flows in the early fault-tolerant regime

Code for **[Nonlinearity-preserving quantum simulation of flows in the early fault-tolerant regime](https://arxiv.org/abs/2611.23196)**. The study develops nonlinearity-preserving quantum simulation using Walsh-synthesized Koopman–von Neumann transport and evaluates its resource requirements in the early fault-tolerant regime. Finite-element cylinder-wake data provide the classical reference for validating the flow predictions. At fixed polynomial degree and number of coupled coordinates, the interior-layer rotation count scales polynomially in the logarithm of grid size; complete task costs also include time steps, preparation, sampling and error correction.

**Start with [demo.ipynb](demo.ipynb).** It walks through main-text Figures 1–5 and runs small examples on the supplied data. A CPU is sufficient; no GPU or quantum hardware account is needed.

This README contains the installation, usage and end-to-end reproduction instructions. [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) provides the executable sequence in a shorter form. Figure-specific `CAPTION.md` and `LABELS.md` files retain the panel descriptions and labels; [FEM implementation notes](Source/CFD/fem/README_KO.txt) document the solver setup in Korean.

| I want to… | Start here | Result |
| --- | --- | --- |
| Understand the method and run examples | [Install and run the demo](#install-and-run-the-demo) | Main figures, model fit, gate counts, resource and shot estimates |
| Check or redraw the published results | [Reproduce results](#reproduce-results) | Numerical checks and 12 public figure renders |
| Estimate resources for my circuit | [Use your own inputs](#use-your-own-inputs) | Physical qubits and time from RZ/native-T counts |
| Recalculate the publication data | [Full numerical regeneration](#full-numerical-regeneration) | A separate, regenerated working dataset |

[Environment and timing](#tested-environments-and-runtime) · [Scientific reference](#scientific-reference) · [Command reference](#command-reference) · [Citation and licences](#citation-and-licences)

## Install and run the demo

Run all commands from the directory containing this README and `Source/` (shown as `Codes/` below). Use Python 3.11 or 3.12; tested versions are 3.11.15 on macOS and 3.12.14 in the Linux container.

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_reproduction.txt
python -m pip check
```

This installs notebook, numerical, plotting and CFD packages with the direct-dependency versions in [requirements_verified.txt](requirements_verified.txt). Python and system tools must already be installed.

**In Cursor, select this project's** `.venv/bin/python` **as the notebook kernel** (`.venv\Scripts\python.exe` on Windows). Activating a terminal does not change an already selected notebook kernel. If a notebook reports a missing package, check its selected interpreter before reinstalling dependencies.

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`; the two `python -m pip` commands are unchanged. For the complete Linux toolchain, use the [container](#linux-container).

### 2. Obtain Source Data

**The Zenodo dataset is not yet published.** Use the supplied local `Source_Data` folder for this version. The planned public record is [Zenodo](https://doi.org/10.5281/zenodo.24505927). Arrange the local package as follows:

```text
publication/
├── Codes/
│   ├── Source/
│   ├── demo.ipynb
│   └── README.md
└── Source_Data/
    ├── manifest.csv
    ├── Source_Data_Index.csv
    ├── Figures/
    └── Shared_source_data/
```

For another location, set `KVN_SOURCE_DATA` to the extracted data directory. Its root must contain `manifest.csv`, without an extra nested `Source_Data` folder.

**The demo requires local data and does not download it.** Other entry points use [Source/data_locations.py](Source/data_locations.py), which prefers local data and can download `Source_Data.zip` from Zenodo if none are found. This fallback requires an accessible, published record and cannot replace the supplied local folder before publication. `python -m Source.data_locations --download` explicitly resolves/downloads the archive; `KVN_SOURCE_DATA_CACHE` overrides the default cache parent.

Keep the 987-payload dataset as the reference. The demo writes separate outputs; numerical regeneration uses a separate writable dataset.

### 3. Run the notebook

```bash
python -m notebook demo.ipynb
```

Run all cells in order. The notebook displays the original main-figure artwork and executes a SINDy refit, Walsh synthesis, restricted resource search, force-history comparison, MPS compression and field-shot calculation. Large CFD/quantum runs, hardware measurements and full resource searches are shown as archived results.

| Example | Expected default result |
| --- | --- |
| Quadratic SINDy refit | 12 coefficients; maximum difference from the archived fit below `1e-12` |
| Eight-entry linear Walsh angle table | Four nonzero Walsh coefficients |
| Restricted fast-block resource search | 559,953 physical qubits; about 1.62 h for 815 accepted samples |
| Ideal circuit force histories versus FEM | Relative RMS discrepancy: drag 0.193%, lift 8.813% |
| Late three-coordinate field readout | 100 accepted samples at a 1% field RMS target; 442 at 0.5% |

Allow **about 11 seconds on the tested Mac** to execute all 16 demo code cells with local data already available. This estimate comes from a recorded 10.678 s headless run with a fresh Matplotlib cache; see [measurement conditions](#tested-environments-and-runtime). Its outputs are `my_workload.json`, `resource_estimates.csv` and `measurement_budgets.csv` in `Source_Data/Generated_Outputs/Demo/`. Set `KVN_DEMO_OUTPUT` to choose another output directory. Input data are unchanged.

Smaller demo dataset and headless execution

From the complete local dataset, optionally copy only the demo inputs:

```bash
python -m Source.Data_Production.prepare_demo --source-data ../Source_Data \
  --output ../Demo_Source_Data
export KVN_SOURCE_DATA="$PWD/../Demo_Source_Data"
```

This creates a new directory with 11 numerical/configuration inputs and five figure previews, all checksum-verified and a subset manifest/index. It is sufficient for the demo, **not** for full figure reproduction. Previews, including the original Figure 2, reside in `Source_Data/Additional_Source_Data/Demo_Previews/`; they are not generated files inside `Codes/`.

To execute the notebook without opening a browser and retain rich plot outputs:

```bash
mkdir -p ../Source_Data/Generated_Outputs/Demo
python -m nbconvert --to notebook --execute demo.ipynb \
  --ExecutePreprocessor.timeout=180 \
  --output demo_executed.ipynb --output-dir ../Source_Data/Generated_Outputs/Demo
```



## Reproduce results

### Figures and short numerical checks

**Figure 1 requires Inkscape** to export its editable SVG master. The demo uses bundled previews and does not need Inkscape. On Debian/Ubuntu, install full-workflow system tools with `sudo apt-get install build-essential libgomp1 inkscape fonts-liberation`, or use the container. Native calculation kernels require GNU C++17/OpenMP; Apple's default `g++` is Clang.

Select the complete local dataset, especially if you previously selected the demo subset:

```bash
export KVN_SOURCE_DATA="$PWD/../Source_Data"
python -m Tests.windows_cpp_smoke
python -m Source.RE.export_table1 --check
python -m Figure_Plots.run all
```

The public smoke test checks native matching and replay kernels, Table 1 should report `PASS 7 rows`, and the figure runner redraws publication panels from deposited inputs. These commands do not run full CFD or a complete resource search. Independent regression/audit suites belong to the separate internal workspace and are not prerequisites for public reproduction.

The figure runner reads existing data and writes PDF/SVG/PNG files under the selected dataset's `Generated_Outputs/Figures/<figure>/`. For one figure use `python -m Figure_Plots.run F4`; for supplementary plots use `python -m Figure_Plots.run all --group supplementary`.  


| Figure entry | Content and numerical source |
| --- | --- |
| `F1` | Export of `Figures/Fig_1/editable_figure.svg`; prepared artwork, not a numerical recalculation |
| `F3` | Selected fault-tolerant designs; shared `RE_common/RE/` records |
| `F4` | Wake models, reconstructed fields and 815-shot force observables; `CFD_ROM/` and `Figures/Fig_4/` |
| `F5` | Scalar/coupled benchmarks, preparation and three-coordinate wake transport; `MPS_KvN_inputs/` |
| `S01` | Six-case physical forecast audit with frozen modes and model |
| `S03`, `S04`, `S05` | Grid/time refinement, scalar inputs and coupled ensemble trajectories |
| `S08`, `S09`, `S10` | Sampling/factory comparisons, native-T sensitivity and coherent capacity |
| `ED1` | Offline analysis of acquired IBM measurements |

These are the 12 public plotters. Figure 2 and Supplementary Figures S2, S6 and S7 are manually prepared illustrations; S7's cited numerical sources remain in the dataset. Captions and panel-specific definitions stay with each [Figure_Plots](Figure_Plots) directory. [Reproduce_Figures.ipynb](Reproduce_Figures.ipynb) is an existing figure notebook; the commands above are the direct plotting route.

To inspect the archived inputs for one figure, run `python -m Figure_Plots.common.export_published_figure_inputs F3`. Copies go to `Generated_Outputs/Figure_Source_Exports/F3/`.




### Full numerical regeneration

This workflow includes expensive force replay, four CFD refinement runs and resource-design searches. It starts from fixed scientific inputs and acquired measurements, not a new hardware acquisition. Use GNU C++17/OpenMP and Inkscape, plus the Python environment above.

```bash
python -m Source.qucf_inputs
python -m Source.qucf_inputs --check
python -m Source.Data_Production.prepare_minimal --reference-source-data ../Source_Data \
  --output ../Source_Data_Reproduced
python -m Source.Data_Production.end_to_end \
  --source-data ../Source_Data_Reproduced
```

The first command downloads 32 external QuCF oracle/HDF5 inputs at a pinned upstream revision; the second verifies their byte counts and SHA-256 hashes offline. They are separate from Zenodo. `KVN_QUCF_CACHE` overrides their cache directory. The full driver also fetches missing QuCF inputs before expensive calculations.

`prepare_minimal` requires a **new** output directory. It retains 40 fixed publication CSVs, 19 relocated input CSVs and required non-CSV inputs, including CFD snapshots, prepared states, configurations, acquired records, and fixed catalogue/protocol files under `Additional_Source_Data/Reproduction_Inputs/`. Generated publication CSVs are removed. The exact fixed-file selection is in [prepare_minimal.py](Source/Data_Production/prepare_minimal.py) and `Source_Data/Additional_Source_Data/Reproduction_Inputs/verified_outputs.json`.

The driver regenerates outputs in dependency order, assembles figure/table inputs, runs refinement and executes publication notebooks. Regenerated data and calculation/notebook reports remain in the **working** dataset, with transient reports under `Source_Data_Reproduced/Generated_Outputs/End_to_End/`. No code-path output directory is created. A missing prerequisite or unresolved producer stops the run. `--skip-cfd-refinement` and `--skip-notebooks` are partial development runs, not full reproduction.

For individual generators that write publication inputs, first select the **writable working copy**:

```bash
export KVN_SOURCE_DATA="$PWD/../Source_Data_Reproduced"
```

Do not publish regenerated files over the archived reference during ordinary reproduction. Numeric checks can pass even when floating-point serialization changes file hashes. A replacement deposit needs refreshed manifest hashes and its own release.

### CFD refinement

[CFD_Refinement.ipynb](CFD_Refinement.ipynb) compares **mean drag, drag fluctuation RMS, lift RMS and Strouhal number** over the last three complete shedding cycles within nondimensional time **60–90**. All four must meet **0.5%** for adjacent and original-to-finest comparisons, with the finer value in the denominator.

The original setup is `Re=100`, mesh factor `1.00`, `dt=0.020`. Added grid cases use factors `0.80` and `0.64` at `dt=0.020`; added time-step cases use `0.010` and `0.005` on the original mesh. All runs span time 0–90. Reanalysis of supplied histories gives largest original-to-finest differences of 0.4552% (mesh) and 0.08947% (time step); adjacent comparisons also pass.

The notebook analyzes existing histories and writes statistics/figures to `Generated_Outputs/CFD_Refinement_Analysis`; it does not launch CFD or overwrite the archived inputs. To calculate four new histories in a fresh directory:

```bash
python -m Source.CFD.refinement --output ../Source_Data/Generated/CFD_Refinement_New
export KVN_CFD_REFINEMENT_OUTPUT="$PWD/../Source_Data/Generated/CFD_Refinement_New"
python -m notebook CFD_Refinement.ipynb
```

The solver writes new histories to the specified `Source_Data/Generated` directory; the notebook reads them using `KVN_CFD_REFINEMENT_OUTPUT` and writes its analyses under `Generated_Outputs`. Use a writable working dataset for new CFD runs if the archive is mounted read-only. The runner has an eight-hour batch limit; interrupted cases restart from their initial state. Completed cases are reused only when code, protocol, inputs and environment signatures match. Do not edit provenance to bypass a mismatch.

The supplied run directory is `Source_Data/Generated/CFD_Refinement/`, including `report.json`, force histories and provenance. Reanalysis writes `refinement_errors.csv`, `four_quantity_statistics.csv`, `four_quantity_assessment.json` and plots to `Generated_Outputs/CFD_Refinement_Analysis`, while preserving the stored run directory.

## Use your own inputs

### Circuit inventory to physical resources

Copy the sample workload `../Source_Data/Additional_Source_Data/Example_Inputs/resource_workload.json`, edit the copy outside `Codes/`, and pass its path with `--workloads`. Do not edit the archived publication workload.

| JSON field | Meaning |
| --- | --- |
| `logical_qubits` | Positive integer circuit width, including auxiliary qubits |
| `RZ` | Nonnegative integer arbitrary-angle Z rotations per attempt |
| `native_T` | Nonnegative integer T/T-dagger count before RZ synthesis |
| `successful_shots` | Positive integer target number of accepted samples |
| `success_probability` | Circuit acceptance probability in `(0, 1]` |
| `source_angle_error_bound` | Existing angle-rounding error bound, default zero, below the implementation allowance |
| `is_wake` | Optional Boolean workload annotation; it does not certify a physical model |
| `workload_scope` | Optional description of the circuit and sampling task |

At least one of `RZ` and `native_T` must be positive. Native-T-only circuits (`RZ=0`) are supported; synthesis precision then has no effect on their T count. This T-state-driven timing model is not a standalone estimator for Clifford-only circuits. The optional Clifford count is metadata; core errors are modeled by architecture-dependent exposure rather than an independent error per listed Clifford.

Total T count is `native_T + RZ × T_per_RZ`. Do not include synthesized rotations in `native_T` a second time. Total expected task time is `seconds_per_attempt × successful_shots / success_probability`. Factory acceptance is already included in supply timing.

```bash
python -m Source.RE.run --workloads ../Source_Data/Additional_Source_Data/Example_Inputs/resource_workload.json --p 5 \
  --blocks fast --d-min 13 --d-max 21 --capped-only \
  --output ../Source_Data/Generated_Outputs/Custom_RE
```

This quick example searches only the fast block and core distances 13–21. It is not a global optimization certificate. For the standard search use:

```bash
python -m Source.RE.run --workloads ../Source_Data/Additional_Source_Data/Example_Inputs/resource_workload.json --p 5 \
  --output ../Source_Data/Generated_Outputs/Custom_RE_Full
```

`--p` is an integer index in units of `1e-4`; 5 means physical error probability `5e-4`. Included preparation tables cover indices 2–9. `--prep-dir` supplies compatible `prep_pN.csv` tables; available measured distances constrain the search. Tables with a `p_phys` column must match the requested rate. Do not change the error rate while silently reusing another rate's calibration. The search also requires the recorded preparation-footprint table in Source Data; `--prep-dir` does not replace that calibration. CLI help and argument parsing do not need the dataset, but actual resource evaluation does.

Standard initial search: compact/intermediate/fast data blocks, odd core distances 13–49, synthesis cost 45–200 T gates per RZ, and a one-million-qubit cap for capped results. `--q-max`, `--d-min`, `--d-max`, `--nt-min` and `--nt-max` control those limits. The subsequent uncapped-fastest search preserves the requested block families and minimum distance but can extend the upper core-distance bound; it remains bounded by the adopted factory/preparation model. If no design is feasible, the command raises an error; widen the search only with appropriate data and assumptions.

Outputs: `selected.csv` (minimum-qubit and fastest choices), `capped_selected.csv` (per-block capped selections), `audit.json`, and per-workload search records. In the historical CSV schema, `N_T` **means T gates per synthesized RZ**, while `T_count` **means total T gates per attempt**. `complete_task_seconds` includes all accepted samples and retries.

Hardware timing defaults are 1 μs per QEC round and 10 μs classical reaction, with a `1e-4` implementation allowance. They are model settings in `Source/RE/model.py`, not CLI workload fields. Changing `ModelConfig` for a fixed-design evaluation does not automatically change every stage of the uncapped search. Such changes need coordinated validation. `Source/RE/sensitivity.py` contains a fixed 4,803-RZ/815-shot comparison and is not the general user-input entry point.

### Desired field accuracy to measurement shots

`Source.ROM.field_budget.field_budget(readout, implementation_allowance, target_rms)` returns a conservative expected-RMS bound and the minimum accepted samples. The notebook demonstrates it with a saved terminal readout.

`readout` contains three one-dimensional `grid_centres` arrays, `outcome_mean` and `reference_outcome_mean` of length three, a 3×3 covariance, and a 3×3 field metric. With `polar=True`, coordinates are radius, angle and shift; moments use `(r cos(theta), r sin(theta), z)`. The metric maps coordinate errors to relative velocity-field error. Covariance and metric must be symmetric positive semidefinite. Supply mutually consistent normalized probabilities, moments, units and reference normalization.

Targets are fractions: `0.01` means 1%. If the bias bound already exhausts the target, `minimum_accepted_shots_under_rms_bound` is `None`; more shots cannot fix this. For a changed model or distribution, update the moments and metric, not just the target number.

### Other supported examples

- **SINDy:** `Source.POD_SINDy.fit.fit(raw, t, degree, threshold)` accepts finite `(n_times, 3)` real coordinates and uniformly spaced, increasing times (at least five for the default derivative), with nonzero training RMS in every coordinate. It assumes reflection parity `(-x, -y, z)` and fits in training-RMS coordinates. Its API supports compatible input arrays; the standalone module command instead verifies the published fit.
- **Walsh rotations:** `Source.KvN.sparse_walsh.controlled_ry(angles, controls, target)` accepts a one-dimensional table of `2**len(controls)` finite real angles. `one_clean_synthesis` expands the supported primitives and counts gates, rather than compiling an arbitrary external circuit.
- **MPS:** `Source.KvN.mps.compress(target, chi)` demonstrates real, normalized power-of-two amplitude vectors with supported power-of-two bond dimension. The CLI retains fixed publication case names. Complex-state preparation requires a separate implementation review.

## Linux container

The Dockerfile includes pinned Python dependencies, GNU C++17/OpenMP and Inkscape. Build the image locally; these instructions do not depend on a published registry image. The tested target is **linux/amd64**. Apple Silicon runs it under emulation; Triangle 20250106 did not install on linux/arm64, so native ARM Linux is not validated.

```bash
docker build --platform linux/amd64 -t walsh-kvn:local .
mkdir -p ../Demo_Outputs
docker run --rm -it --platform linux/amd64 \
  -p 127.0.0.1:8888:8888 \
  --mount "type=bind,src=$PWD/../Source_Data,dst=/workspace/Source_Data,readonly" \
  --mount "type=bind,src=$PWD/../Demo_Outputs,dst=/workspace/Outputs/Demo" \
  walsh-kvn:local
```

Open the localhost URL/token printed by Jupyter and select `demo.ipynb`. The container runs as a non-root user. For read-only data validation, replace the interactive launch with:

```bash
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD/../Source_Data,dst=/workspace/Source_Data,readonly" \
  --mount "type=bind,src=$PWD/../Demo_Outputs,dst=/workspace/Outputs/Demo" \
  walsh-kvn:local python -m Source.RE.export_table1 --check --output /workspace/Outputs/Demo/Table_1.csv
```

The launch above retains example outputs in the host directory `Demo_Outputs/` after container removal. In Jupyter, save a copy of the executed notebook in `/workspace/Outputs/Demo/demo_executed.ipynb` to retain it too. On Linux, the mounted output directory must be writable by the container user `researcher` (UID 1000). Figure previews come from the read-only Source Data mount. Full regeneration requires a separate writable working dataset mount.

Native-library caches are specific to their operating system, architecture and compiler. Use separate generated-output directories across hosts; the Docker build excludes host-compiled libraries.

The base Python image is pinned by digest; Debian tools are resolved at build time. A locally rebuilt image is not necessarily byte-identical to another build.

Native Windows compiler setup and short checks

Install 64-bit Python 3.11, [MSYS2](https://www.msys2.org/) and Inkscape. In an **MSYS2 UCRT64** terminal:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-libgomp
```

In PowerShell, from the code root (adjust installation paths if needed):

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;C:\Program Files\Inkscape\bin;' + $env:PATH
g++ --version
inkscape --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_reproduction.txt
python -m Tests.windows_cpp_smoke
python -m Source.RE.export_table1 --check
```

`KVN_CXX` selects another compatible compiler. Native loaders use `.dll` on Windows and register the MinGW runtime directory. The supplied Windows record covers kernels, short regressions and Table 1, not the complete workflow. WSL 2 can use the Linux procedure.

To attempt full regeneration after the short checks, fetch QuCF inputs as above, then use a new working directory:

```powershell
python -m Source.Data_Production.prepare_minimal --reference-source-data ..\Source_Data --output ..\Source_Data_Reproduced
python -m Source.Data_Production.end_to_end --source-data ..\Source_Data_Reproduced
```

Only completion of the full driver establishes a full run; passing short checks does not.



## Tested environments and runtime

The 28 November code-fix review passed 134 bounded tests and the short numerical checks. It additionally replayed nine deposited MPS preparations and checked seven polynomial operators, the Q5 phase-table producer and S8b checksum publication. These checks used Linux x86_64 and Python 3.12.14; they did not repeat the demo timing measurements below.

**Historical demo execution: about 11 seconds on the tested Apple M5 computer (16 GiB RAM).** This is the elapsed time to execute all 16 code cells of `demo.ipynb` using the headless `nbconvert` command above. Python/kernel startup, data loading, example calculations, plot generation and saving the executed notebook are included. Dependency installation, data download/copying, opening the browser and reading the notebook are excluded.

The recorded run took **10.678 s**, using Python 3.11.15 on macOS 26.3.1 arm64, one OpenBLAS/OMP/MKL thread, a fresh Matplotlib cache and a complete local `Source_Data` copy (about 93 MB). The demo reads its required inputs from that copy; it does not process or regenerate the whole archive. This is one observed run, not an average or a cross-machine guarantee. This timing predates the v2 module/data relocation and scientific-data updates as well as later prose/readability edits. V2 demo execution has been checked separately, but not timed again under these same measurement conditions.

| Environment | Recorded scope |
| --- | --- |
| Native macOS | Apple M5, 16 GiB, macOS 26.3.1, Python 3.11.15; demo and dependency installation timed in an earlier review; 120-test suite and short checks passed on the earlier snapshot |
| Linux amd64 container | Debian trixie, Python 3.12.14, under Apple Silicon emulation; v2 demo, 128-test suite and short checks passed using the existing validated dependency image. Earlier validation also covered the demo, all 12 plotters, Table 1, native/OpenMP kernels and a short CFD replay |
| Supplied original Linux record | AMD EPYC 9V45, Debian 13.7, Python 3.12.14, GCC 14.2.0, Inkscape 1.4; distinct from the local container tests |
| Supplied Windows record | Python 3.11 and MSYS2 UCRT64; native kernels, short checks and Table 1; full workflow not validated |

Installation and other recorded timings are separate from demo execution:

| Measurement | Time | Conditions |
| --- | --- | --- |
| Native Mac dependency installation | About 21 s (20.925 s recorded) | Earlier uncached pip installation; Python/system tools already present; excludes dataset download and 1.799 s venv creation. Download speed and package availability affect this time |
| CFD notebook, existing-history analysis | About 3 s (3.051 s recorded) | Earlier run; local data, reused Matplotlib cache; no CFD solve |
| Earlier native Mac quick check | 6.843 s | Includes temporary dataset copy; ten tasks and 48 Figure 3 rows |
| Earlier Linux amd64 container quick check | 14.837 s | Same checks; read-only archive, isolated working copy, Apple Silicon emulation; includes Docker process startup |
| Prior Linux amd64 container, all figures | 22.036 s | Earlier rendering run; local data and process startup included |
| Supplied Linux server quick check / all figures | 9.0 s / 20.0 s | Historical sum of timed stages; local inputs; one OpenMP/BLAS thread; excludes outer startup |

In the earlier plot validation, eleven native Mac plotters passed; Figure 1 required unavailable Inkscape. The container supplied Inkscape and passed all 12. Its short CFD run stopped at simulation time 0.04. The latest container checks validate current-code execution in the existing dependency environment, not a fresh installation. **Full end-to-end regeneration, complete CFD refinement and hardware acquisition are outside these timings.**

`Source_Data/Additional_Source_Data/Environment_Records/TESTED_ENVIRONMENT.json` preserves the earlier environment, installation and notebook measurements; its snapshot hashes are historical. The adjacent `RUNTIME_REFERENCE.json` preserves the supplied server stage timings. The internal verification suite is not required by the public reproduction workflow.

## Scientific reference




### Code and data map

| Code | Role | Source Data location |
| --- | --- | --- |
| `Source/CFD`, `Source/POD_SINDy`, `Source/ROM` | Flow reference, reduced model, field/force readout | `Shared_source_data/CFD_ROM/`; `Figures/Fig_4/` |
| `Source/KvN` | Walsh circuits, MPS preparation and transport | `Shared_source_data/MPS_KvN_inputs/`; `Figures/Fig_5/` |
| `Source/RE`, `Source/Gidney` | Resource search and preparation tradeoffs | `Shared_source_data/RE_common/`; `Shared_source_data/Gidney_cultivation/` |
| `Comparisons` | BE–KvN and LCHS–KvN comparisons | `Shared_source_data/RE_common/BE_inputs/`; `Tables/`; external QuCF cache |
| `Hardware/IBM` | Acquisition code and offline analysis | `Figures/Extended_Data_Fig_1/` |
| `Source/KvN`, `Source/ROM` | Finite-shot, radial and upwind comparisons | `Shared_source_data/Radial_refinement/`; `Supplementary_Data/Finite_Shot_Replay/` |
| `Source/Data_Production` | Numerical producers and end-to-end execution | Separate writable working dataset and its `Generated_Outputs/` |
| `Figure_Plots` | Publication rendering and source export | `Figures/`; output in the selected dataset's `Generated_Outputs/Figures/` |

`Source_Data_Index.csv` and `manifest.csv` are authoritative file mappings. Historical identifiers such as `Figure/F4/data/...`, `Data/...` and `Supply/...` are provenance keys resolved by `Source.data_locations`, not current code paths. Do not rename them mechanically or recreate a duplicate data tree.






### CFD, POD and force conventions

`python -m Source.POD_SINDy.replay_frozen_basis` rebuilds the modes, projected coordinates, zero-intercept shift closure and selected 12-coefficient model from saved snapshots. It compares POD subspaces by principal angles, aligns the oscillatory pair by Procrustes rotation and matches the shift sign before coefficient comparison. The recorded maximum coefficient difference is below `3e-14`.

Active POD/SINDy implementations are `pod.py` (reflection, decomposition, projection, closure), `fit.py` (reflection-constrained, training-RMS fit), `identification.py` (polynomials, derivatives, ODE prediction), and `evaluate_degree_cases.py` (additional-case/blocked validation). Retained historical programs are housed in the `Source/Data_Production/producers/` provenance subdirectories; the independent internal archive is excluded from the public execution path.

The original Re100 transient replay is `python Source/CFD/fem/run_fem.py --reference-replay`; parameterized solves use `python Source/CFD/fem/run_fem.py --reynolds 100 200 300`. These are Taylor–Hood FEM benchmarks. The retained steady field is an input; replay does not silently replace the frozen modes or model. The six-case S1 audit uses the same frozen model for A–F; only A supplied the modes and fit. Its compact projected states/norms retain the aggregate field-error information from full FEM output.

For both pressure and viscous forces, run:

```bash
python -m Source.CFD.reproduce_cfd_forces --verify
```

This fixed Re100 solve uses `dt=0.02`, ends at time 90 and writes to the selected dataset's `Generated_Outputs/CFD/force_replay/`. `forces.csv` includes the pressure-invalid initial row; `forces_cfd.csv` has 4,500 valid rows; `fig4f_fem_forces.csv` contains the 3,000 rows through time 60. `--data-root` overrides the dataset. `--export-only --verify --output PATH` checks a completed run; `--end 0.04 --output ../Source_Data/Generated_Outputs/CFD/smoke` is only a smoke check.

With the fluid-domain outward normal pointing into the cylinder, body-force traction is `p*n - nu*(grad(u)+grad(u).T)*n`. Since `rho=U=D=1`, coefficients are twice the integrated force. Drag is positive downstream and lift upward. The initial perturbed field has no pressure solution (`pressure_valid=False`). The wrapper checks archived/working hashes and executable AST equality, ignoring the module docstring; original programs remain in `Source/CFD/force_replay/audit/` for provenance.



### IBM measurements

ED1 uses 16-cell four-step runs at 16,000 shots/circuit and six-step runs at 32,000 shots/circuit. A separate 32-cell one-step control is archived; no 64-cell hardware transport is reported. `Source_Data/Figures/Extended_Data_Fig_1/hardware_inputs/run_metadata.json` records the 12/13 circuits and device times.

`python -m Figure_Plots.ED1.generate_data` exports diagnostics from the archived calibration records to `Generated_Outputs/Regenerated/ED1/`; it does not refit calibration or rerun bootstrapping. In the archived analysis, calibration is ideal-input value plus the transport–reference difference divided by fitted contrast (SI equation S82). Each of its 1,000 bootstrap resamples refits contrast. Error bars exclude systematic noise-model mismatch and temporal correlations. Matched zero-transport references differ from the continuum ROM reference; these experiments do not measure phase or reconstruct fields. Earlier exact-StatePreparation/ZNE controls and reduced-shot records remain separate. Plotting and diagnostics do not submit provider jobs.




### Circuit comparison boundaries

**BE–KvN:** the joint 64-cell radial/phase construction follows Guseynov et al., Corollary 1 (SI Note 13/Table S12). `compile_literal_joint.py` verifies 256 projected generator columns; `fullwindow.py` evaluates the archived full-window inventory. Reference counts per attempt are 25 logical qubits, 7,677,481 RZ, 511,970,388 native T and 732,189,482 Cliffords, with 813 accepted shots. High-degree outer-QSP angles are charged conservatively, not claimed as explicitly synthesized. Inputs/certificates reside in `RE_common/BE_inputs/` and `Tables/Table_S12/BE_fullwindow_inventory/`.

**LCHS–KvN:** the Novikau–Joseph comparison uses 128 cells, ten steps of 0.1 and seven initial position Hadamards. `Comparisons/KvN_LCHS/verify.py` reconstructs/counts source blocks and checks encoded matrices; it does not rerun the full 25-qubit evolution. The source histogram has 1,566 rows representing 108,018,555 occurrences; it cannot recover gate order. One-clean synthesis retains branch phases and checks work-qubit reset, producing 26 logical qubits, 307,305,994 RZ, 5,914,351,326 native T and 8,781,048,741 Cliffords (5,461,507,502 CNOTs). Global cancellation across primitives is outside its scope.

The saved success probability is `0.8762773928256311`; Table S11 includes retries for 815 accepted shots. At the reference fastest design, 126 T/RZ gives 44,634,906,570 total T/attempt. Earlier compiler/resource configurations and the 20-step tally are separate source checks.

Pinned sources: [QuCF f34f89c](https://github.com/QuCF/QuCF/tree/f34f89c24eff4ab79d3ecdea72da3cec88dce94f), [scripts-py 2727acd](https://github.com/QuCF/scripts-py/tree/2727acd977476ab2151a5b12e6765e0a219f8293), and [Novikau & Joseph, CPC 309, 109498 (2025)](https://doi.org/10.1016/j.cpc.2025.109498). The Source Data record `Tables/Table_S11/LCHS_source_counts/upstream_input_manifest.json` records 32 inputs plus two engine-source identifiers; its `source_verification.json` defines the audit boundary.

### Numerical targets and refinement scope

- **S03:** radial comparisons include 64 cells/33 steps and 128 cells/66 steps at matched step-duration/cell-width ratio, plus a 128-cell/33-step control. Polynomial moment plots distinguish prepared cell-mass and continuous-Gaussian references. State-vector comparisons use the same prepared input/grid and minimize over one global phase.
- **S04–S05:** scalar densities are cell probability divided by cell width; input centres/widths precede truncation. Four coupled ensembles use 32 × 32 cells and 64 steps to time 1. The classical reference uses matched initial probabilities and eight-point Gauss–Legendre quadrature per cell. The 8–256-step test makes 64 the first tested count with positive sampling budget for all four inputs at the 1% target and `1e-4` total-variation allowance.
- **Three-coordinate wake:** fixed signed input, grid, domain, model and continuum reference; only step count varies. Current selections are 33 steps/100 shots (late) and 264 steps/437 shots (growth). Both use second-order centred spatial differences. At physical error probability `5e-4`, the growth design takes 11.87 days on 809,048 physical qubits. The original 66/528-step resource table remains historical evidence. Selection compares tested candidates at eight physical-error rates, not a joint global optimization over grids and time steps.
- **Grid/step scaling:** uniform cell subdivision preserves prepared probabilities/phases; three-coordinate spatial tests refine radius/shift with 32 angular modes fixed. Polynomial reference evolution is an exact finite-grid matrix exponential. Wake step-doubling/extrapolation diagnostics are not rigorous exact-grid error bounds.
- **Radial alternative:** the main benchmark remains 33 steps. The separate eight-step alternative at physical error `5e-4` uses 1,074 shots, 0.464 h and 422,901 qubits; it does not replace canonical figures/Table 1.
- **Resource diagnostics:** fastest-design ties use relative time tolerance `1e-6`, then physical-qubit count. S9 uses core-distance ranges 19–31 (wake) and 19–29 (control); S10 compares all eight rates at both qubit caps. A shot budget, probability error, state-vector error and physical-resource estimate have different definitions.

## Command reference



### Public checks

Run `python -m Tests.windows_cpp_smoke` to check native C++ matching/replay kernels on the current host and `python -m Source.RE.export_table1 --check` to validate the current Table 1 export. `python -m Source.Data_Production.end_to_end --source-data ../Source_Data_Reproduced` is the actual production-data workflow. Development-only synthetic, regression and audit suites are maintained separately from the public code package; they are not reproduction dependencies.

The optional CSV comparison utility `python -m Source.Data_Production.verify` compares regenerated records with a separate archive using byte equality, parsed-cell equality or numerical tolerances. This does not independently establish physical validity. `source_data_rebuild.publish` replaces individual publication files atomically; resumed calculations must be compared to a reference before claiming numerical identity.



Individual numerical producers and supporting calculations

From `Codes/`, prefix each executable module below with `python -m` and append the shown arguments. Entries marked as imported helpers are not commands. Use `--help` only for entry points with an argument parser; fixed-task producers such as `Source.ROM.generate_fig4_error_data` execute immediately and do not support it. **Dataset-writing entries require the writable copy selected above.** Transient outputs go under the selected dataset's `Generated_Outputs/`; the linked implementations define detailed schemas.

Paths beginning `../Source_Data/` assume the sibling dataset. If `KVN_SOURCE_DATA` selects a different working dataset, replace that prefix with its path.

| Module / arguments | Result and scope |
| --- | --- |
| [`Comparisons.BE_KvN.apply_resource_results`](Comparisons/BE_KvN/apply_resource_results.py) | Merge a completed block-encoding resource sweep into the reference design tables. Requires all eight physical-error rates and matching circuit inventory. Use a writable dataset copy. |
| [`Comparisons.build_workloads`](Comparisons/build_workloads.py) | Construct circuit workloads for the comparison resource estimates. This constructs workloads; the hardware-design search is a separate Source.RE.run command. |
| [`Source.KvN.polynomial_benchmarks`](Source/KvN/polynomial_benchmarks.py) | Count and replay polynomial circuits using one reusable clean auxiliary qubit. The coupled replay checks all 64 transport steps against the archived amplitudes. |
| [`Figure_Plots.execute_notebook demo.ipynb --output ../Source_Data/Generated_Outputs/Demo_executed.ipynb`](Figure_Plots/execute_notebook.py) | Execute notebook code cells sequentially without launching a Jupyter server. Execution uses the same plotting and scientific functions as the notebook; the output notebook records text streams. |
| [`Source.RE.export_distillation_acceptance`](Source/RE/export_distillation_acceptance.py) | Calculate distillation-acceptance quantities for the selected resource designs. Use a writable dataset copy. This evaluates stored designs rather than repeating the design search. |
| [`Source.Data_Production.export_table_s1 --source-data ../Source_Data_Reproduced`](Source/Data_Production/export_table_s1.py) | Export full-precision model parameters underlying Table S1. Rounded printed table entries are not used to reconstruct full-precision parameters. |
| [`Figure_Plots.ED1.generate_data`](Figure_Plots/ED1/generate_data.py) | Calculate source diagnostics for Extended Data Figure 1. Runs offline, using acquired measurements. It does not submit jobs to a quantum provider. |
| [`Figure_Plots.F3.generate_data`](Figure_Plots/F3/generate_data.py) | Format selected resource designs into the numerical tables used by Figure 3. No resource optimization is performed; literature and hardware comparison inputs remain separate. |
| [`Source.ROM.generate_fig4_error_data`](Source/ROM/generate_fig4_error_data.py) | Recalculate the selected-model field-error curves for Figure 4e. The model is integrated and compared with the retained fields; circuit-bias and target rows are fixed inputs, not recomputed here. |
| [`Figure_Plots.S03.generate_data`](Figure_Plots/S03/generate_data.py) | Calculate radial spatial/time refinement for Supplementary Figure S3. --native additionally compiles and simulates the 128-cell, 33-step circuit. Use a writable dataset copy. |
| [`Source.KvN.generate_fig_s5_trajectories`](Source/KvN/generate_fig_s5_trajectories.py) | Regenerate the four coupled-model trajectories in Supplementary Figure S5. Checks terminal probabilities and reference moments against retained records before exporting trajectories. |
| [`Source.RE.generate_litinski_data --p 0.0005 --schemes A`](Source/RE/generate_litinski_data.py) | Search Litinski factory designs for the fixed nine-qubit workload. The objective is minimum physical-qubit count with runtime/error tie-breakers. Partial runs preserve other stored sweep conditions. |
| [`Source.RE.generate_nac_factory_catalogue --litinski-python /path/to/magicstates/Python --output ../Source_Data/Generated_Outputs/Litinski_catalogue.csv`](Source/RE/generate_nac_factory_catalogue.py) | Regenerate the numerical factory catalogue using the upstream Litinski implementation. The upstream distillation model is kept separate from hardware-layout corrections in Source.RE.Litinski. |
| [`Source.RE.generate_supplement_data --rates 2 3 4 5 6 7 8 9`](Source/RE/generate_supplement_data.py) | Generate resource-sensitivity tables for Supplementary Figures S8–S10. Rate integers are in units of 10^-4. --s8-only evaluates Fig. S8; publication requires a writable dataset copy. |
| [`Source.KvN.grid_step_scaling polynomial`](Source/KvN/grid_step_scaling.py) | Compare spatial and temporal refinement at fixed prepared input and duration. Polynomial tests use a finite-grid matrix exponential. Wake Richardson diagnostics are convergence estimates, not rigorous exact-grid error bounds. |
| [`Figure_Plots.plot_grid_step_scaling ../Source_Data/Generated_Outputs/Grid_Step_Refinement`](Figure_Plots/plot_grid_step_scaling.py) | Plot diagnostics from a completed grid/time-step refinement. This is an optional diagnostic view of a calculation, not a second publication-figure implementation. |
| [`Source.KvN.radial_step_comparison --output ../Source_Data/Generated_Outputs/Radial_Step_Comparison.csv`](Source/KvN/radial_step_comparison.py) | Compare the 16- and 33-step radial circuits at 64 cells. This compares the specified counts; it is not a search for a minimum feasible step count. |
| [`Source.KvN.radial_time_selection`](Source/KvN/radial_time_selection.py) | Evaluate centred radial time-step candidates and their complete task costs. Only tested counts are compared. The archived 33-step input supplies the state/model; it does not impose an upwind step criterion. |
| [`Source.Data_Production.rebuild_noncanonical --working-source-data ../Source_Data_Reproduced`](Source/Data_Production/rebuild_noncanonical.py) | Regenerate additional publication tables using the producer map. Used by end_to_end in multiple dependency passes. Raw Gidney cultivation records are retained experimental/calibration inputs. |
| [`Source.RE.reestimate_one_clean_three_coordinate --rates 2 3 4 5 6 7 8 9`](Source/RE/reestimate_one_clean_three_coordinate.py) | Reoptimize three-coordinate circuit resources with one-clean arithmetic. Exact index arithmetic changes the circuit inventory; this script does not refit the physical model or recompute the field budget. |
| [`Source.Data_Production.source_data_rebuild list`](Source/Data_Production/source_data_rebuild.py) | Manage the mapping from numerical producers to publication data files. bootstrap creates the catalogue; reproduce runs a producer; publish copies generated outputs; verify checks file sizes and SHA-256 digests against the manifest. |
| [`Source.RE.sweep_output (imported helper)`](Source/RE/sweep_output.py) | Merge partial parameter sweeps without discarding other stored conditions. Out-of-scope rows are rejected. Calculation and publication use a staging directory. |
| [`Source.RE.three_coordinate_resource_selection --workers 2`](Source/RE/three_coordinate_resource_selection.py) | Compile and reoptimize the tested three-coordinate time-step candidates. The analysis compares feasible tested candidates under the production resource model; it is not a continuous optimization over time step. |
| [`Source.KvN.three_coordinate_time_convergence late40 --threads 2`](Source/KvN/three_coordinate_time_convergence.py) | Test time-step convergence for the two three-coordinate wake intervals. Only the number of symmetric transport steps varies. Differences to a finer run are refinement diagnostics, not exact-grid error bounds. |
| [`Source.RE.update_litinski_s8b`](Source/RE/update_litinski_s8b.py) | Refresh the Litinski design curves used by Supplementary Figure S8b. The Silva model rows and plot labels are retained; only the two Litinski model rows are updated. |

Additional runs:

```bash
python -m Source.CFD.physical_audit
python -m Source.KvN.three_coordinate_time_convergence long298 --threads 6 --plot
python -m Source.KvN.grid_step_scaling late40 --threads 2
python -m Source.KvN.grid_step_scaling long298 --threads 6
python -m Source.Data_Production.source_data_rebuild publish grid_step_scaling --source-data /path/to/working/Source_Data --overwrite
python Comparisons/BE_KvN/fullwindow/compile_literal_joint.py
python Comparisons/BE_KvN/fullwindow/fullwindow.py
python Comparisons/KvN_LCHS/verify.py
python Comparisons/KvN_LCHS/verify_lchs_sine_shift.py
python Comparisons/KvN_LCHS/verify_lchs_sine_shift.py --qucf-dir "$(python -m Source.qucf_inputs)"
python Figure_Plots/F3/code/plot.py "${KVN_SOURCE_DATA:-../Source_Data}/Generated_Outputs/F3_full_range" --full-range
```

`physical_audit` calculates S1 errors from projected FEM coordinates/norms without refitting; its output goes to the selected writable dataset. The comparison workload builder does not run a resource search. `Comparisons.BE_KvN.apply_resource_results` needs all eight rates and matching inventory. `source_data_rebuild publish` and `--publish-figure-data` write the selected dataset; never target the read-only archive/cache.


Supporting comparisons use these entry points:

| Command | Scope |
| --- | --- |
| `python -m Source.ROM.field_sampling_verification` | Finite-shot covariance check against exact circuit means; excludes continuum bias |
| `python -m Source.KvN.radial_refinement_flow` | Same prepared density at 64/33 and 128/66; explicit gates and protected shots |
| `python -m Source.RE.radial_refinement_resources` | Resources from the deposited 128-cell inventory/budget |
| `python -m Source.KvN.grid_oscillation_diagnostics` | Total variation, moments and extrema versus characteristics |
| `python -m Source.KvN.q5_upwind_comparison SOURCE_CSV OUTPUT_DIR` | Exact-time centred/upwind quintic comparison; signed input versus positive-root control |
| `python -m Source.KvN.q5_phase_comparison` | Export the signed-MPS versus positive-root input-phase comparison from the two Q5 sweep tables. Independent stencil and rerun checks are in `Internal_Workspace/Validation/Checks/verify_q5_upwind.py` in the combined package. |
| `python -m Source.KvN.wake_upwind_refinement --grids 64 128` | Exact-time wake stencils and matched phase readout; not an LCHS compilation |

Here `SOURCE_CSV` is `Shared_source_data/Radial_refinement/Grid_Oscillation_Diagnostics/inputs/scalar_conservative_cell_masses.csv` within the dataset; `OUTPUT_DIR` is a separate chosen directory. Polynomial callers share `Source/KvN/polynomial_validation.py`, `Source/KvN/sparse_walsh.py` and `Source/KvN/wake_gate_core.py`; Gray parity reuse does not change binary cell addresses.

`polynomial_validation` contains the shared model and gate compiler; importing it does not read data or write outputs. Saved preparation gates are read from the deposited MPS JSON fits, or a bundle selected with `PQC_PREPARATION_ROOT`; both flat and `fits/` layouts are supported. A matching NPZ amplitude is compared only if it is supplied, and its absence is recorded rather than treated as a successful comparison. Independent transport checks are in `Internal_Workspace/Validation/Checks/verify_polynomial_transport.py` in the combined package. From that package root, run `python -m Internal_Workspace.Validation.Checks.verify_polynomial_transport --help` for `--preparation-root`, `--output` and optional `--provenance`. Its full sweep can be expensive; outputs default to the selected dataset's `Generated_Outputs/Polynomial_Validation/`.







### Fixed inputs for numerical reconstruction

The full-reproduction command above executes the scientific producers in their dependency order. Internal CSV audits are distinct from this public workflow and are not required steps of the reproduction guide.

`Source_Data/Additional_Source_Data/Reproduction_Inputs/verified_outputs.json` maps outputs to deposited files; the adjacent `columns.json` defines schemas and `jobs.json` records historical replay commands/input hashes. Groups cover `history-{sindy,physical,pod,forcefit,q5,details}`, `ibm`, `scientific`, `export`, `polynomial`, `records`, `projections`, `publication`, `retained_experiments` and `s8`.

`replay_history.py --help` lists retained jobs. Its wrapper checks hashes, stages original programs/relative inputs in a temporary directory and collects declared outputs. Use wrappers instead of running `Source/Data_Production/producers/` programs in place. `retained_experiments.py` handles preparation/budget replays. A force-map fit does not regenerate CFD surface forces.

Ordinary reproduction retains the 12 manually extracted, as-printed Supplementary Information table CSVs; it does not require the private manuscript document or its extraction utility.

Fixed inputs include literature/annotation rows, acquired IBM counts, eight raw cultivation tables and corresponding `prep_pN.csv` tables, fixed factory catalogues, snapshots/models, prepared amplitudes and acquisition protocols. The original 66/528-step resource table is retained only for comparison. [Source_Data/Additional_Source_Data/Reproduction_Inputs/catalogue](../Source_Data/Additional_Source_Data/Reproduction_Inputs/catalogue) holds static schemas and author-defined quantities, including Fig. 4e circuit-bias rows and S9 native-T inventories; it is not an output directory. `noncanonical` is a historical filename for additional-data regeneration, not a quality judgment; `one_clean` denotes one reusable zero-initialized auxiliary qubit.

### Optional upstream acquisition and factory calculations

Ordinary reproduction reuses acquired cultivation records. To inspect budgets or explicitly acquire new data:

```bash
python -m Source.Gidney.generate_raw_stats
python -m Source.Gidney.generate_raw_stats --cultiv-repo /path/to/cultivation --p 5 --d2 7 --execute
python -m Source.Gidney.build_tradeoff --raw-root "${KVN_SOURCE_DATA:-../Source_Data}/Generated_Outputs/Gidney/raw_stats"
python -m Source.Gidney.build_tradeoff --verify-published
```

The first command only lists all 80 budgets. Acquisition needs the original Strilanc Cultiv checkout, Sinter and the desaturation decoder; tradeoff rebuilding needs SciPy for exact binomial intervals. The `5e-4` study uses `10^10` shots per distance and 1-dB gap thresholds; other rates use 5 dB. Outputs default to `Generated_Outputs/Gidney/`.

Litinski regeneration needs a separate `litinski/magicstates/Python` checkout; `generate_nac_factory_catalogue` supplies its path explicitly. The retained official numerical catalogue has SHA-256 `db31af50c8433769ec8233e8fd02d52292b5d2df4fcf68a5d22594d4b461d969`; a new factory search must not silently replace it. Published resource exports evaluate selected records unless their command explicitly performs a new search.



Scientific input provenance and interpretation of retained results

These records explain which arrays, fits and reported values enter reconstruction. Archive names are historical identifiers; public runs resolve their extracted inputs through the manifest and wrappers.

| Input or calculation | Origin and use |
| --- | --- |
| Preserved historical programs | `jobs.json` records the six archive names, archive SHA-256 values, original working paths, commands and the digest of each input. Files in `Source/Data_Production/producers/` were copied from the retained source archives. `replay_history.py` stages the listed inputs and runs those programs in temporary directories. For the force fit, the original snapshots are supplied under both filenames expected by the archived program. |
| POD replay | The stored flow and mode arrays determine the projected coefficients. The old coefficient CSV supplies requested reporting times and a separate numerical check. |
| SINDy and field comparison | `SINDy_Candidate12_Comparison_20261113.zip`, `Candidate12/inputs/`, supplies `candidate_K2.npz`, `physical_model_inputs.npz` and `reported_model_scaling.npz`. The current physical-error file uses the candidate quadratic fit. The stored composite time-resolved error file uses the deployed quadratic fit for its single-flow interval. The current Fig. 4e renderer reads the separate `Figure4e_curves.csv`. |
| Quantum and reference arrays | The direct amplitudes and observation arrays come from the release's shared Re100 directory. The historical polar snapshot used for the readout comparison is in the existing `Shared_source_data/RE_common/BE_inputs/wake_task_inputs.zip`, member `Figure2/unified_run/cases/Re100/data/qsvt_transport.npz`. The configuration specifies the sampling seeds and the 815 direct and 814 polar shots used by the two archived comparisons. |
| Polynomial example parameters | The three scalar coefficient vectors and drift normalizations correspond to the model definitions in `Figure6_package.zip`, `input/protocol.json`. The two-dimensional drift is evaluated directly as specified by the model. |
| Fixed-shot polynomial inputs | `Figure5_Bias_Analysis.zip`, `Figure5_Bias_Analysis/input/` and its `grid32/` subdirectory, supply the eleven small extra endpoint files in `inputs/polynomial/`. The coupled interior endpoint already exists in the published Figure5_Bias_Analysis directory. The boundary Q5 input already exists in Grid_Oscillation_Diagnostics. The scalar boundary examples share the initial state. For Q2 and C3 the split-step transport is evaluated using the respective drift. The sample count, reference moments and ranges come from the archived `accuracy_protocol.json`. |
| Seven-model spatial and temporal sweeps | `KvN_v100_SI103_Full_Radial128_ForceSampling_Fixed.zip`, `KvN_v100_SI103/Supply/S03/source_data/full_sweep/`. Both full sweeps are retained. The Fig. S3 projection selects the four shown models and maps the C3i and Q5m labels to C3 and Q5. Hardware timing columns preserve the measured values from those sweeps. |
| Decomposition in Supplementary Data 4 | `Polynomial_KvN_PQC_validation_package_20261104.zip`, `polynomial_pqc_validation_20261104/prediction_error_decomposition.csv`, supplies the ideal cell-projection vector from the analytic target. The calculation uses the matched preparation from the separate seven-model sweeps. Subtracting transport and analytic cell-projection contributions from the total yields the fit contribution. |
| Supplementary Data 5 | The seven 64-step inventories come from the full time sweep. The working-qubit field retains the corrected one-clean-ancilla convention of the published resource records. |
| Input-specific gate inventory for Fig. 5c | `Figure5_v9_source_and_data.zip`, `Figure_source/work/KvN/Source_Data_CSV/Figure6/c_coupled_resources/workloads.csv`, records four different state preparations. Only the columns RZ, native_T and CX were extracted into `coupled_compiled_counts.json`. The updated width and step count come from the published coupled32 inventory. The shot fields come from the accuracy-budget table. This is an inventory projection. |
| Wake gate inventories | `Supplementary_S4_QSVT_MPS_update_20261107.zip` supplies `source_data/Table_S6_circuit_counts.csv` and the `circuit_inventories` object in `source/walsh_resource_summary.json`. Publication Table S2 supplies the operation counts for the rebuilt-parity CNOT sum. |
| Table S4 numerical source | `diagnostics_table.xml` contains the native table at zero-based index 28 in `word/document.xml` of `final/NC_KvN_Supplementary_47p_20261107.docx` from that same archive. Original Word SHA-256 is `346b97c89776f639aeaba6906b79b1bee8e3870c76c8570d82ab7db47e28888c`. Its 12 scalar rows retain the historical rounded numbers. The four coupled rows are recalculated from the revised 32 × 32 trajectories: preparation fidelity and RMS errors against the prepared-input and analytic Gaussian references, using the same reporting precision. |
| SI tables as printed | The 12 printed-table CSVs were exported from the Supplementary Information Word document (v103), which is not included in the public code release. These CSVs are provided in Source Data under `Tables/Table_S*/`. The optional `printed_tables.py --si DOCUMENT.docx --output DIRECTORY` command extracts selected tables from a document supplied by the reader. Supported equation structures become readable linear notation: grouped factors, subscripts/superscripts, fractions, roots, ceilings and floors. This notation retains implicit multiplication and printed rounding; it is not executable Python or a source of full-precision parameters. Unsupported structures raise a table/row/column error before any output is written. Standard numerical reproduction does not require that document. |
| Resource records | `records.py` reads the published source records named in each function. The S7 export uses eleven source tables. For the Table S10 scheme-B fastest design, the reported record is retained from the historical input JSON. The scheme-B minimum-qubit time uses the verified service period and native inventory. The resource exports do not run a new architecture optimization. |

The file named `Table_S8_current.csv` is a five-row projection of the native-T sensitivity sweep. One of these five rows also appears in the three-row printed Table S8. All three printed rows agree with the full sweep at their displayed precision. The audit records the two row selections separately.

The retained-experiment [provenance record](../Source_Data/Additional_Source_Data/Reproduction_Inputs/inputs/retained_experiments/provenance.json) records archive hashes and retained fields. Fig. 5 budgets replay the 16 original experiment files; full files and lossless subsets give the same shot budgets. Supplementary Data 3 refits nine targets with original seeds, executes their circuits and compares all four quality metrics at publication precision.

For S8a, `python -m Source.RE.generate_supplement_data --s8-only` derives the 1% field-error shot budget from the joint readout and `5e-4` preparation records, then selects minimum-qubit and fastest capped designs at each allowance. It can regenerate all eight plotted rows without the old output CSVs. Task seconds are accepted shots × seconds per attempt for this direct workload.

The S1 perturbation origin is the cylinder centre, lengths are cylinder diameters, downstream is positive x and upward is positive y. Its cutoff is `C=s^3(10-15s+6s^2)`, `s=clip((x-0.7)/0.5,0,1)`; velocity perturbation is `(∂yδψ, −∂xδψ)` before boundary/divergence enforcement. Case amplitudes, centres and widths are in `Figures/Fig_S1/plot_inputs/plot_values/case_settings.csv`; its caption describes the zero cylinder-surface perturbation.



## Citation and licences

Cite the [paper](https://arxiv.org/abs/2611.23196); [CITATION.cff](CITATION.cff) supplies citation metadata. The planned [Zenodo dataset record](https://doi.org/10.5281/zenodo.24505927) is pending publication. Its data DOI identifies the dataset, not a specific code version.

Original code and documentation use [Apache-2.0](LICENSE), with authorship/scope in [NOTICE](NOTICE). Author-generated Source Data are distributed separately under CC BY 4.0. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists upstream software and licences.

QuCF inputs are fetched from their pinned revision rather than redistributed. Gidney cultivation and Litinski regeneration depend on separate upstream software. Where supplied acquisition provenance lacks an exact upstream commit, the repository does not establish one; new acquisitions should record the revision used.
