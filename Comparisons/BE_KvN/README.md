# Matched BE–KvN circuit, one-clean reconstruction

This comparison implements the joint 64-cell radial-and-phase block encoding from Guseynov et al. Corollary 1, as documented in current SI Note 13 and Table S11. The coordinate loader and controlled signal blocks use the shared one-clean multi-control decomposition.

From the package root:

```bash
python Comparisons/BE_KvN/fullwindow/compile_literal_joint.py
python Comparisons/BE_KvN/fullwindow/fullwindow.py
python -m Reproduction.build_comparison_workloads
```

The first command verifies 256 projected generator columns and writes `Build/BE_KvN/literal_joint_counts.json`. The second uses the archived matching ledger (unless an alternative `--block` path is specified), evaluates the full-window polynomial and writes `Build/BE_KvN/fullwindow_resources.json`. The third reconstructs the RE workload independently of those output files and compares it with the current reference.

Reference values are 25 logical qubits, 7,677,481 arbitrary Rz rotations, 511,970,388 native T gates and 732,189,482 Cliffords per attempt. The matched field task uses 813 accepted shots. Individual high-degree outer-QSP phase angles are charged conservatively, not claimed as explicitly synthesized. The corresponding hardware searches are in `Source/RE/` and current selected records are in `Source_Data/Shared_source_data/RE_common/RE/v38_reference_selected.csv`.

The source numerical inputs reside in `Source_Data/Shared_source_data/RE_common/BE_inputs/`; published gate-count certificates and the archived outer inventory are in `Source_Data/Tables/Table_S11/BE_fullwindow_inventory/`. The code package does not retain duplicate copies.
