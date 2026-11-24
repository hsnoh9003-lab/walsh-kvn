# Current numerical sources for the supplementary notes

The figures and tables use the separately published Zenodo dataset ([DOI: 10.5281/zenodo.24505927](https://doi.org/10.5281/zenodo.24505927)), not a local `Data/` mirror. A sibling `Source_Data/` tree or `KVN_SOURCE_DATA` override is supported, but optional: `python -m Source.data_locations --download` fetches the DOI's `Source_Data.zip` into a cache after publication. The index and path map are `Source_Data_Index.csv` and `manifest.csv` at the dataset root.

| Subject | Published folder | Producing or verifying code |
| --- | --- | --- |
| Wake CFD, POD–SINDy fits, force coefficients | `Source_Data/Shared_source_data/CFD_ROM/`; `Source_Data/Figures/Fig_4/` | `Source/POD_SINDy/`, `Source/CFD/`, `Source/ROM/`; `Figure/F4/code/plot.py` |
| Scalar-cell, upwind and 64/128-cell refinements | `Source_Data/Shared_source_data/Radial_refinement/`; `Source_Data/Figures/Fig_S3/` and `Fig_S4/` | `Reproduction/auxiliary/{q5_upwind,radial128,wake_upwind,grid_oscillation}/`; `Supply/S03/code/` |
| Three-coordinate preparation and transport | `Source_Data/Shared_source_data/MPS_KvN_inputs/`; `Source_Data/Figures/Fig_5/` | `Source/KvN/`; `Source/ROM/`; `Figure/F5/code/plot.py` |
| Independent finite-shot and four-basis force checks | `Source_Data/Supplementary_Data/Finite_Shot_Replay/`; `Source_Data/Figures/Fig_4/force_sampling/` | `Reproduction/auxiliary/field_sampling/`; `Source/ROM/force_sampling.py` |
| Error-corrected KvN resource model and Gidney trials | `Source_Data/Shared_source_data/RE_common/`; `Source_Data/Shared_source_data/Gidney_cultivation/` | `Source/RE/`; `Reproduction/verify_cultivation_shots.py` |
| BE–KvN and KvN–LCHS circuit source inputs | `Source_Data/Shared_source_data/RE_common/BE_inputs/` and `LCHS_source_inputs/`; `Source_Data/Tables/` | `Comparisons/BE_KvN/`; `Comparisons/KvN_LCHS/`; `Reproduction/build_comparison_workloads.py` |
| Supplementary resource Figures S8–S10 | `Source_Data/Figures/Fig_S8/`, `Fig_S9/`, `Fig_S10/` | `Reproduction/generate_supplement_data.py`; `Supply/S08/code/plot.py`, `Supply/S09/code/plot.py`, `Supply/S10/code/plot.py` |
| IBM experiment/Extended Data Figure 1 | `Source_Data/Figures/Extended_Data_Fig_1/` | `Hardware/IBM/code/`; `Extended_Data/ED1/code/plot.py` |

The published `Source_Data/` release also has `Tables/`, `Supplementary_Data/`, and `Additional_Source_Data/` for numbered source tables, explicitly cited supplementary datasets, and supplementary records not belonging to a single panel.

From the code root, use `python -m Reproduction.quick_check --plots` for short numerical and plotting checks; the full CFD/MPS and RE sweeps are separate high-cost computations. For supplementary factory sensitivities, regenerate all rates with `python -m Reproduction.generate_supplement_data --rates 2 3 4 5 6 7 8 9` into the local `Build/` directory, and rerun S8–S10 plotting. The DOI cache is not an authoring destination; `--publish-figure-data` is reserved for an explicitly writable local data staging copy.
