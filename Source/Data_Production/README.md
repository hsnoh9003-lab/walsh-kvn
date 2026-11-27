# CSV generators and numerical comparisons

These modules generate the CSV tables used by the publication from fixed scientific inputs. `Source.Data_Production.end_to_end` calls them in dependency order. Use that driver for a complete numerical rebuild; the commands below are for comparing a producer with an already-complete reference dataset.

Run from the repository root after installing the dependencies in the Reproduction Guide:

```bash
python Source/Data_Production/verify.py --source-data ../Source_Data --group scientific --output Generated_Outputs/CSV_Checks
```

Use `--group all` to check all public numerical groups. Each producer runs in a new temporary directory, so old outputs cannot satisfy a new comparison. The deposited dataset is read only for this check. The report contains source/output digests, comparison outcomes and maximum numerical differences. The output directory must be outside the reference dataset.

As-printed manuscript and literature tables are supplied in `Source_Data/Tables/` as fixed inputs. The unpublished manuscript and its extraction-only scripts are not part of the public code release.

| Producer group | Calculation or export |
| --- | --- |
| `history-sindy`, `history-physical` | Fit degree candidates and integrate trajectories for the physical-model comparison |
| `history-pod`, `history-forcefit` | Project CFD fields onto the modal basis and fit force observables to the FEM force history |
| `history-q5`, `history-details` | Propagate the quintic-drift refinements and calculate numerical diagnostics |
| `ibm` | Reanalyse acquired IBM counts and circuit records offline |
| `scientific`, `export`, `polynomial` | Calculate field/readout quantities and export polynomial drift, density and sampling tables |
| `records`, `projections`, `publication` | Assemble tables from circuit inventories, fixed records and model parameters |
| `retained_experiments` | Rerun shot-budget and state-preparation calculations; check coupled trajectories |
| `s8` | Recalculate shot counts and selected Fig. S8 designs from readout and calibration inputs |

The comparison checks the header, row order, row count and every cell. It first tests exact file equality, then parsed CSV equality, then numerical agreement within `1e-12 + 1e-10 * abs(reference)`. Categorical labels and bit strings must agree exactly. These tolerances compare serialized numerical outputs; they are not the paper's physical prediction-error target.

The JSON producer contracts and fixed arrays reside in `Source_Data/Additional_Source_Data/Reproduction_Inputs/`, not in the executable tree. The contracts describe output schemas, scientific model protocols and historical program inputs. The retained scientific programs under `producers/` are still required by some data-generation contracts; `replay_history.py` executes them with verified inputs in isolated temporary directories. The public execution entry point remains `Source.Data_Production.end_to_end`.
