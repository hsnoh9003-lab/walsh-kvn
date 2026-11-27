"""Regenerate publication data in dependency order, then draw the figures.

Inputs
------
A writable Source_Data tree containing fixed inputs prepared by prepare_minimal; required external QuCF files.

Outputs
-------
Regenerated CSVs, transient calculations and notebook reports in the writable Source_Data tree.

Run from the repository root
----------------------------
python -m Source.Data_Production.end_to_end --source-data ../Source_Data_Reproduced

Scope
-----
The full run includes expensive CFD refinement. Skip flags support partial checks and do not establish full reproduction.
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
import os
from pathlib import Path
import shutil
import subprocess
import sys


CODES = Path(__file__).resolve().parents[2]
PUBLIC_SOURCE = (CODES.parent / "Source_Data").resolve()


def execute(
    args: list[str], source_data: Path, *, extra_env: dict[str, str] | None = None
) -> None:
    """Run one producer with the explicit working dataset; stop on failure."""
    environment = dict(
        os.environ,
        KVN_SOURCE_DATA=str(source_data),
        PYTHONPATH=str(CODES) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        MPLBACKEND="Agg",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if extra_env:
        environment.update(extra_env)
    print("RUN", sys.executable, " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=CODES, env=environment, check=True)


def copied_output(source: Path, destination: Path) -> None:
    """Require a generated file and copy it to its publication destination."""
    if not source.is_file():
        raise FileNotFoundError(f"Producer did not create {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify_csv_inventory(dataset: Path, *, allow_refinement: bool = False) -> None:
    """Require manifest-listed CSV paths, with six optional refinement outputs.

    This is a presence check. Numerical comparison and archive checksums are
    separate checks; an existing file alone does not certify its contents.
    """
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        expected = [
            row["published_path"]
            for row in csv.DictReader(handle)
            if row["published_path"].endswith(".csv")
        ]
    missing = [p for p in expected if not (dataset / p).is_file()]
    refinement_pending = 0
    if allow_refinement:
        # Only the expensive CFD refinement histories and their two derived
        # assessment tables may remain absent in a development smoke check.
        allowed = {
            "Generated/CFD_Refinement/" + name
            for name in (
                "mesh_0p8/forces.csv",
                "mesh_0p64/forces.csv",
                "dt_0p01/forces.csv",
                "dt_0p005/forces.csv",
                "four_quantity_statistics.csv",
                "refinement_errors.csv",
            )
        }
        refinement_pending = sum(p in allowed for p in missing)
        missing = [p for p in missing if p not in allowed]
    if missing:
        raise RuntimeError(
            f"{len(missing)} publication CSVs have not been generated:\n"
            + "\n".join(missing)
        )
    if allow_refinement:
        print(
            f"Non-refinement CSV inventory complete: "
            f"{len(expected)-refinement_pending}/{len(expected)} publication paths; "
            f"{refinement_pending} CFD refinement paths pending",
            flush=True,
        )
    else:
        print(
            f"Full CSV inventory complete: {len(expected)} publication paths",
            flush=True,
        )


def main() -> None:
    """Run producers in dependency order using an explicit writable working dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data",
        type=Path,
        required=True,
        help="Writable working Source_Data containing only fixed/raw inputs",
    )
    parser.add_argument(
        "--skip-cfd-refinement",
        action="store_true",
        help="Development smoke check only: skips four full CFD simulations",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Development smoke check only: skips publication notebook execution",
    )
    args = parser.parse_args()
    # Refinement statistics are produced by the analysis notebook. Reject an
    # incompatible partial-run request before acquisition or any calculation.
    if args.skip_notebooks and not args.skip_cfd_refinement:
        parser.error(
            "CFD refinement analysis requires its notebook; "
            "use --skip-cfd-refinement together with --skip-notebooks"
        )
    data = args.source_data.resolve()
    if data == PUBLIC_SOURCE:
        parser.error(
            "Use a separate working Source_Data directory, not the deposited reference"
        )
    if not (data / "manifest.csv").is_file():
        parser.error("Working Source_Data must contain its deposited manifest.csv")
    # Check external references before starting the expensive calculations.
    # QuCF originals are cached separately and are not part of the data deposit.
    from Source.qucf_inputs import ensure_inputs

    ensure_inputs()
    rates = [str(i) for i in range(2, 10)]
    resource = data / "Generated_Outputs/End_to_End/Resource_Search"

    # First calculate the shared resource designs. Every subsequent resource
    # plot/optimizer consumes these canonical published selections.
    execute(
        ["-m", "Source.RE.run", "--output", str(resource), "--canonical-export"], data
    )
    for name, published in [
        ("selected.csv", "reference_selected.csv"),
        ("capped_selected.csv", "reference_capped_selected.csv"),
    ]:
        copied_output(
            resource / name, data / "Shared_source_data/RE_common/RE" / published
        )

    # The native-T sweep grid is defined by the retained reproduction protocol,
    # so none of the regenerated S9 CSVs are needed to start this sweep.
    execute(
        [
            "-m",
            "Source.RE.generate_supplement_data",
            "--rates",
            *rates,
            "--publish-figure-data",
        ],
        data,
    )

    # Some producers require the recovered force fit; that fit in turn needs
    # the freshly simulated CFD forces. Execute the independent producers,
    # bridge the cross-family dependency, then finish the deferred groups.
    execute(
        [
            "-m",
            "Source.Data_Production.rebuild_noncanonical",
            "--working-source-data",
            str(data),
            "--partial",
        ],
        data,
    )
    execute(
        [
            "Source/Data_Production/rebuild.py",
            "--release-root",
            str(CODES.parent),
            "--working-source-data",
            str(data),
            "--groups",
            "history-forcefit",
            "history-physical",
            "history-q5",
            "history-sindy",
            "export",
            "polynomial",
        ],
        data,
    )
    execute(
        [
            "-m",
            "Source.Data_Production.rebuild_noncanonical",
            "--working-source-data",
            str(data),
            "--resume",
        ],
        data,
    )

    # Figure and table CSVs written directly to their canonical locations.
    execute(["-m", "Source.RE.update_litinski_s8b"], data)
    execute(["-m", "Source.RE.export_distillation_acceptance"], data)
    execute(["-m", "Figure_Plots.F3.generate_data"], data)
    execute(["Source/ROM/generate_fig4_error_data.py"], data)
    execute(["Figure_Plots/S03/generate_data.py"], data)
    execute(["-m", "Source.Data_Production.export_table_s1"], data)
    copied_output(
        data / "Shared_source_data/CFD_ROM/Physical_Forecast_Audit/short_window.csv",
        data / "Supplementary_Data/Supplementary_Data_1.csv",
    )

    execute(
        [
            "Source/Data_Production/rebuild.py",
            "--release-root",
            str(CODES.parent),
            "--working-source-data",
            str(data),
        ],
        data,
    )
    # Four refinement force histories and two statistical summaries are
    # intentionally produced only in the subsequent full-CFD stage.  Check
    # every other publication CSV now, then check all of them after CFD.
    verify_csv_inventory(data, allow_refinement=True)
    # Public reproduction uses numerical producers and the publication inventory.
    # The independent internal regression suite is not a release dependency.
    execute(["-m", "Source.RE.export_table1", "--check"], data)

    # These calculations are NOT light-weight. Default reproduction performs
    # all four force histories rather than using retained output CSVs.
    if not args.skip_cfd_refinement:
        output = data / "Generated/CFD_Refinement"
        execute(["-m", "Source.CFD.refinement", "--output", str(output)], data)
        execute(
            [
                "-m",
                "Figure_Plots.execute_notebook",
                "CFD_Refinement.ipynb",
                "--output",
                str(
                    data / "Generated_Outputs/End_to_End/CFD_Refinement_executed.ipynb"
                ),
            ],
            data,
            extra_env={"KVN_CFD_REFINEMENT_OUTPUT": str(output)},
        )
        # The analysis notebook keeps fresh results separate from its inputs.
        # Publish them to the paths listed in the working dataset's manifest.
        for name in (
            "four_quantity_statistics.csv",
            "refinement_errors.csv",
            "four_quantity_assessment.json",
        ):
            copied_output(data / "Generated_Outputs/CFD_Refinement_Analysis" / name,
                          output / name)

    if not args.skip_cfd_refinement:
        verify_csv_inventory(data)

    if not args.skip_notebooks:
        execute(
            [
                "-m",
                "Figure_Plots.execute_notebook",
                "Reproduce_Figures.ipynb",
                "--output",
                str(data / "Generated_Outputs/End_to_End/Figures_executed.ipynb"),
            ],
            data,
        )
    if not args.skip_cfd_refinement and not args.skip_notebooks:
        print("End-to-end publication reproduction completed", flush=True)
    else:
        print(
            "Development-only partial run: full end-to-end reproduction NOT certified",
            flush=True,
        )


if __name__ == "__main__":
    main()
