"""Format selected resource designs into the numerical tables used by Figure 3.

Inputs
------
Reference resource-design selections in Source_Data.

Outputs
-------
Figure 3 plot-input CSVs in the selected writable dataset.

Run from the repository root
----------------------------
python -m Figure_Plots.F3.generate_data

Scope
-----
No resource optimization is performed; literature and hardware comparison inputs remain separate.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from pathlib import Path
import pandas as pd
from Source.data_locations import source_file, figure_data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = source_file("Data/RE/reference_selected.csv")
FIG = figure_data("F3")


def run():
    """Project the 48 selected designs into Figure 3 tables without rerunning the search."""
    df = pd.read_csv(SOURCE)
    df = df[
        df.model.isin(["direct", "generator", "nj_lchs"])
        & df.selection.isin(["minimum_qubits", "fastest_uncapped"])
    ].copy()
    assert len(df) == 48
    # Reference cache may leave the minimal-qubit per-shot field null.
    # It is exactly recoverable from the matched total runtime / shot count.
    per_shot = df.complete_task_seconds / df.successful_shots
    is_missing = df.seconds_per_accepted_shot.isna()
    df.loc[is_missing, "seconds_per_accepted_shot"] = per_shot.loc[is_missing]
    assert df.seconds_per_accepted_shot.notna().all()
    # Accepted shots differ from attempts for a probabilistic circuit. The
    # expected number of attempts divides accepted shots by success probability.
    df["expected_attempts"] = df.successful_shots / df.success_probability
    df["complete_task_hours"] = df.complete_task_seconds / 3600
    df["complete_task_days"] = df.complete_task_seconds / 86400
    df["complete_task_years"] = df.complete_task_seconds / (365.25 * 86400)
    FIG.mkdir(parents=True, exist_ok=True)
    df.sort_values(["model", "selection", "p_phys"]).to_csv(
        FIG / "panel_b_selected_designs.csv", index=False
    )
    df[df.p_phys.round(8) == 0.0005].sort_values(["model", "selection"]).to_csv(
        FIG / "panel_c_time.csv", index=False
    )
    df.sort_values(["model", "selection", "p_phys"]).to_csv(
        FIG / "architecture_levels.csv", index=False
    )
    print(
        "Wrote F3 selected-resource tables from authoritative current hardware selected.csv"
    )


if __name__ == "__main__":
    run()
