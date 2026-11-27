"""Create a separate dataset containing the fixed inputs needed for numerical regeneration.

Inputs
------
A complete deposited Source_Data tree supplied as --reference-source-data.

Outputs
-------
A new writable dataset at --output, without the generated publication CSVs.

Run from the repository root
----------------------------
python -m Source.Data_Production.prepare_minimal --reference-source-data ../Source_Data --output ../Source_Data_Reproduced

Scope
-----
The source dataset is not modified. Non-CSV upstream snapshots, configurations and acquired records are retained.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import json
from pathlib import Path
import shutil

from Source.Data_Production.source_data_rebuild import all_routes
from Source.Data_Production.data_assets import fixed_asset


HERE = Path(__file__).resolve().parent
FIXED_CSVS = {
    "Figures/Fig_3/plot_inputs/nisq_examples.csv",
    "Figures/Fig_3/plot_inputs/reference_points.csv",
    "Shared_source_data/RE_common/RE/gidney_circuit_footprints.csv",
    "Figures/Fig_4/explicit_source_data/Source_Data_Figure_4e_Definitions.csv",
    "Figures/Fig_1/plot_inputs/plot_values/PANEL_DATA_PROVENANCE.csv",
    "Shared_source_data/RE_common/RE/Litinski/not_as_costly_factory_candidate_catalog.csv",
    "Shared_source_data/Gidney_cultivation/audit/selected_design_changes.csv",
    "Shared_source_data/Gidney_cultivation/audit/shot_target_by_distance.csv",
    "Figures/Fig_4/plot_inputs/panel_e_markers.csv",
    "Shared_source_data/RE_common/Numerical_Details/reported_coupled_variance_ratios.csv",
    "Shared_source_data/RE_common/RE/Litinski/not_as_costly_official_factory_numerical.csv",
    # Frozen original 66/528-step comparison, not a current resource selection.
    "Additional_Source_Data/Three_Coordinate_Time_Convergence/baseline/resource_selected.csv",
}
FIXED_CSVS.update(
    f"Shared_source_data/Gidney_cultivation/raw/custom_p{p}e-4_d1_3_d2.csv"
    for p in range(2, 10)
)
FIXED_CSVS.update(
    f"Shared_source_data/RE_common/RE/prep_p{p}.csv" for p in range(2, 10)
)


def expected_fixed(source: Path) -> set[str]:
    """Resolve the immutable input files required by the regeneration registry."""
    contracts = json.loads(fixed_asset("verified_outputs.json", data=source).read_text())
    printed = {
        entry["source"] for entry in contracts if entry["group"] == "printed_tables"
    }
    with (source / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        relocated_inputs = {
            row["published_path"] for row in csv.DictReader(stream)
            if row["published_path"].endswith(".csv")
            and row.get("role") == "Fixed reproduction input"
        }
    return FIXED_CSVS | printed | relocated_inputs


def main() -> None:
    """Create a separate input-only working dataset without replacing the reference.

    The copied manifest still lists the complete archive, including outputs
    deliberately removed here. Non-CSV fixed inputs remain available to their
    producers; old refinement state is removed to prevent a false resume.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.reference_source_data.resolve()
    dest = args.output.resolve()
    if not (source / "manifest.csv").is_file():
        parser.error("Reference dataset has no manifest.csv")
    if dest.exists():
        parser.error("Output already exists; refusing to replace user data")
    if dest == source or dest.is_relative_to(source) or source.is_relative_to(dest):
        parser.error("Choose an output sibling, never a source parent or child")
    with (source / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        all_csv = {
            row["published_path"]
            for row in csv.DictReader(stream)
            if row["published_path"].endswith(".csv")
        }
    fixed = expected_fixed(source)
    if not fixed <= all_csv:
        raise RuntimeError(f"Fixed CSVs are missing from the manifest: {fixed-all_csv}")
    expected_c = {
        entry["source"]
        for entry in json.loads(
            fixed_asset("verified_outputs.json", data=source).read_text()
        )
        if entry["group"] != "printed_tables"
    }
    expected_b = {
        route.published_in_source_data
        for route in all_routes()
        if route.group != "gidney_trials"
    }
    generated = all_csv - fixed
    if not expected_c <= generated or not expected_b <= generated:
        raise RuntimeError("Published producer mappings overlap fixed inputs")
    print(f"Copying {source} to {dest} (reference remains unchanged)", flush=True)
    shutil.copytree(source, dest)
    for path in generated:
        (dest / path).unlink()
    # CFD refinement JSON summaries and final states are outputs, not upstream
    # evidence. In particular, a retained per-case summary would wrongly cause
    # the full solver to skip a case whose force CSV was removed above.
    refinement = dest / "Generated/CFD_Refinement"
    if refinement.is_dir():
        shutil.rmtree(refinement)
    remaining = {p for p in all_csv if (dest / p).is_file()}
    if remaining != fixed:
        raise RuntimeError("Cold-start input inventory was not constructed correctly")
    print(
        f"Cold-start input copy ready: {len(fixed)} retained CSVs, "
        f"{len(generated)} publication CSVs removed for regeneration",
        flush=True,
    )
    print("Non-CSV upstream scientific inputs were retained.", flush=True)


if __name__ == "__main__":
    main()
