# End-to-end reproduction

This guide uses the **same scientific modules and data paths as the published calculations**. Run commands from `Codes/`; there is no separate `Reproduction/` package. See [../README.md](../README.md) for installation, scientific conventions and individual experiments.

## Directory layout

```text
publication/
├── Codes/                 # Executable code, notebooks, licence and citation files
│   ├── Source/
│   ├── Comparisons/
│   ├── Figure_Plots/
│   ├── Tests/
│   └── docs/
└── Source_Data/           # Deposited source records and fixed inputs
    ├── manifest.csv
    ├── Additional_Source_Data/Reproduction_Inputs/
    ├── Figures/
    ├── Shared_source_data/
    └── Tables/
```

`Codes/` does not contain publication CSVs, fixed arrays, retained acquisition records, or generated numerical outputs. Scientific inputs are located through `Source.data_locations` and `Source.Data_Production.data_assets`. Default generated figures, intermediate results and native libraries go under the **selected** `Source_Data/Generated_Outputs/` directory.

## Inputs that must be retained

The entire deposited dataset is used to prepare a separate working copy. The preparation command identifies the required initial files; **do not manually guess which source data are disposable**.

- **40 fixed publication CSVs**: manually transcribed literature and reference values, author-defined annotations/definitions, eight Gidney cultivation raw observation tables, eight measured `prep_pN.csv` calibration tables, fixed factory reference catalogues, and the retained historical 66/528-step comparison. These are explicitly listed or computed in `Source/Data_Production/prepare_minimal.py`.
- **19 relocated input CSVs**: IBM acquisition/analysis inputs and retained reproduction checkpoints, formerly stored under `Codes/`. These are fixed inputs, separate from the 40 publication CSVs.
- **Non-CSV scientific inputs**: CFD snapshots, model parameters and configurations, prepared amplitudes/MPS records, acquired IBM count and circuit files, acquisition protocols, and other saved raw data used by the published generators.
- **Fixed reproduction assets**: `Additional_Source_Data/Reproduction_Inputs/` holds producer contracts, schema descriptions, recorded job/input hashes, numerical protocols and additional fixed experiment inputs. `Additional_Source_Data/Demo_Previews/` holds the five retained artwork previews used by `demo.ipynb` (including the manually prepared Figure 2).
- **External QuCF inputs**: 32 upstream oracle/HDF5 inputs are downloaded and checked separately at the pinned revision. They are **not** provided as files inside `Codes/` or the Source Data deposit.

These fixed inputs are **not reproducible from code alone**: they contain observations, literature transcription, upstream measurements or declared experimental inputs. All other manifest-listed publication CSVs are intended to be regenerated into the working dataset by the steps below.

## 1. Install

Use Python 3.11/3.12, GNU C++17 with OpenMP, Inkscape and the packages in `requirements_reproduction.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_reproduction.txt
python -m pip check
```

On Windows PowerShell, use `py -3.11 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. MSYS2 UCRT64 GCC, its OpenMP runtime and Inkscape must be on `PATH`; see the main README for the tested compiler setup. On Linux, the supplied `Dockerfile` provides the native tools. Docker notebook runs with a read-only input mount and a distinct writable output mount; complete regeneration requires a writable dataset mount.

## 2. Set the reference dataset and obtain external inputs

The supplied local `Source_Data/` folder must contain `manifest.csv` at its root. From `Codes/`:

```bash
export KVN_SOURCE_DATA="$PWD/../Source_Data"
python -m Source.qucf_inputs
python -m Source.qucf_inputs --check
```

The first command may download from the pinned upstream revision; the second verifies content hashes. In PowerShell, set `$env:KVN_SOURCE_DATA = (Resolve-Path ..\Source_Data).Path` instead of the `export` command.

## 3. Prepare an input-only working dataset

```bash
python -m Source.Data_Production.prepare_minimal \
  --reference-source-data ../Source_Data \
  --output ../Source_Data_Reproduced
```

The output directory **must not already exist**. The deposited reference remains unchanged. The working copy retains the fixed publication CSVs and non-CSV upstream evidence, and removes generated publication CSVs and stale CFD refinement state. It also retains the required fixed contracts and protocols under `Additional_Source_Data/Reproduction_Inputs/`.

## 4. Regenerate the publication dataset and figures

```bash
python -m Source.Data_Production.end_to_end \
  --source-data ../Source_Data_Reproduced
```

This single driver calls the original domain modules in dependency order to regenerate resource selections, other figure/table CSVs, all four full CFD refinement histories, CFD analysis and publication-figure notebooks. It verifies the inventory of manifest-listed CSVs; missing outputs fail the run. A successful complete run ends with `End-to-end publication reproduction completed`.

Full CFD refinement and resource searches are computationally expensive. The `--skip-cfd-refinement` and `--skip-notebooks` switches are **development-only partial runs**, not evidence of successful end-to-end reproduction. Hardware acquisitions and Gidney raw-cultivation reruns are not part of this workflow; their retained observations are fixed inputs.

To check Table 1 and redraw figures separately against a selected complete dataset:

```bash
export KVN_SOURCE_DATA="$PWD/../Source_Data_Reproduced"
python -m Source.RE.export_table1 --check
python -m Figure_Plots.run all
```

All generated publication CSVs are placed at their established `Source_Data` paths. Transient output files and executed notebooks are stored under `Source_Data_Reproduced/Generated_Outputs/`; figure renders under `Source_Data_Reproduced/Generated_Outputs/Figures/`. No `Codes/Generated_Outputs/` folder is created.

## 5. Verify scope

- The full driver verifies **presence** of the publication CSV inventory and performs the named numerical/table checks. For numerical equivalence against the deposited archive, use the optional CSV comparator with a separate output directory; differing floating-point serialization need not have identical SHA-256 hashes.
- Full CFD, time convergence, factory optimization and hardware acquisition have distinct costs and validation boundaries. The public `Tests.windows_cpp_smoke` is a *native-kernel compatibility test*, not a substitute for the complete driver.
- Figure 1 is an SVG artwork export, **not a regenerated numerical dataset**. Figure 2 and selected supplementary panels are prepared illustrations. Their fixed artwork and cited source records are retained separately from the computational plots.

For reference dataset provenance and licences see `Source_Data/manifest.csv`, [../NOTICE](../NOTICE), [../LICENSE](../LICENSE), [../CITATION.cff](../CITATION.cff) and [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
