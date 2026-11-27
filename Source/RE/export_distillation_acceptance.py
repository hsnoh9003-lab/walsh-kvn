"""Calculate distillation-acceptance quantities for the selected resource designs.

Inputs
------
Selected design records and magic-state preparation/calibration tables in Source_Data.

Outputs
-------
Distillation-acceptance CSVs resolved through the working dataset manifest.

Run from the repository root
----------------------------
python -m Source.RE.export_distillation_acceptance

Scope
-----
Use a writable dataset copy. This evaluates stored designs rather than repeating the design search.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path, shared_data
from functools import lru_cache
import math
from pathlib import Path
from Source.RE.update_litinski_s8b import published_wake_designs
from Source.RE import model
from Source.RE.factory_sizing import interlevel_units
from Source.RE.preparation import load_workloads, make_config, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = shared_data("RE_common", "RE")


@lru_cache(None)
def configuration(p, block):
    """Load the cultivation calibration at p and the selected data-block geometry."""
    prep = DATA / f"prep_p{round(10000*p)}.csv"
    return make_config(p, prep, block, {"logical_qubits": 9})


def level_terms(
    series, name, selection, p, block, d_core, distances, units, native_t, is_wake
):
    """Propagate growth, distillation and interlevel-path errors through the selected factory chain."""
    cfg = configuration(p, block)
    factory = model.evaluate_factory_distances(p, d_core, distances, cfg, units)
    source_error = model.gidney_degree_table_row(distances[0], cfg)["e_prep"]
    previous_distance = distances[0]
    records = []
    for level, (distance, count) in enumerate(zip(distances, units), start=1):
        growth = model.growth_error(previous_distance, distance, p, cfg)
        e_in = model.combine_independent(source_error, growth)
        e_cliff = model.memory_error(distance, p, cfg)
        output, acceptance = model.distillation_level_metrics(e_in, p, distance, cfg)
        # The implemented acceptance approximation includes both input-state
        # rejection (15*e_in) and logical Clifford rejection (356*e_cliff).
        input_rejection = 15 * e_in
        clifford_rejection = 356 * e_cliff
        assert acceptance == factory["P_levels"][level - 1]
        full = 1 - input_rejection - clifford_rejection
        assert abs(acceptance - full) < 2e-16
        records.append(
            dict(
                series=series,
                model=name,
                selection=selection,
                p_phys=p,
                native_T=native_t,
                is_wake=is_wake,
                block=block,
                d_core=d_core,
                level=level,
                distance=distance,
                units=count,
                e_in=e_in,
                e_Cliff=e_cliff,
                rejection_15_e_in=input_rejection,
                rejection_356_e_Cliff=clifford_rejection,
                clifford_to_input_rejection_percent=100
                * clifford_rejection
                / input_rejection,
                clifford_rejection_included=True,
                acceptance_used=acceptance,
                acceptance_full_S45=full,
            )
        )
        path = model.mean_silva_path_patches(count, level)
        per_round = model.logical_error_per_round(p, distance, cfg)
        path_error = model.repeated_failure_probability(per_round, path * distance)
        source_error = model.combine_independent(output, path_error)
        previous_distance = distance
    delivered = model.combine_independent(
        source_error, model.growth_error(previous_distance, d_core, p, cfg)
    )
    assert math.isclose(delivered, factory["e_MSF"], rel_tol=1e-13, abs_tol=1e-30)
    return records


def litinski_116_to_12():
    """Calculate the rate ratio for the selected A Game designs in Fig. S8b."""
    records = []
    source = published_path(
        "Data/Regenerated/Litinski/litinski_nine_qubit_comparison.csv"
    )
    for r in published_wake_designs(read_csv(source)).values():
        if r["paper"] != "game" or r["protocol"] != "116-to-12_81tile":
            continue
        p_phys = float(r["p_phys"])
        distance = int(r["d_out"])
        # This layout takes raw T states with e_in = p_phys. We estimate a
        # logical Clifford fault with one memory cycle at the factory distance.
        e_in = p_phys
        e_cliff = model.memory_error(
            distance, p_phys, configuration(p_phys, r["core_style"])
        )
        acceptance = (1 - e_in) ** 116
        assert math.isclose(acceptance, float(r["p_success"]), rel_tol=2e-14)
        records.append(
            dict(
                protocol=r["protocol"],
                p_phys=p_phys,
                d_out=distance,
                input_states=116,
                output_states=12,
                e_in=e_in,
                e_Cliff=e_cliff,
                e_Cliff_over_e_in=e_cliff / e_in,
                acceptance_used=acceptance,
                input_rejection=1 - acceptance,
            )
        )
    if not records:
        raise ValueError(
            "No selected 116-to-12 designs found for the Fig. S8b comparison"
        )
    return sorted(records, key=lambda r: r["p_phys"])


# Decompose the acceptance model into input-error and Clifford-error terms
# for stored resource designs.
# This is an audit of selected configurations rather than a new optimization.
# Execution updates the corresponding RE source-data CSVs.
def main():
    """Export acceptance terms for the selected wake, refinement and native-T sensitivity designs."""
    works = load_workloads(DATA / "workloads.json")
    rows = []

    def group(value):
        return tuple(int(x) for x in str(value).split(":"))

    for series, path in [
        ("global_optimum", DATA / "reference_selected.csv"),
        ("capped_by_block", DATA / "reference_capped_selected.csv"),
        ("refined_wake", published_path("Data/Radial128/resources/selected.csv")),
    ]:
        for r in read_csv(path):
            name = r["model"]
            wake = name == "radial128" or works[name]["is_wake"]
            rows.extend(
                level_terms(
                    series,
                    name,
                    r["selection"],
                    float(r["p_phys"]),
                    r["block"],
                    int(r["d_core"]),
                    group(r["distances"]),
                    group(r["u_levels"]),
                    int(float(r["native_T"])),
                    wake,
                )
            )
    for r in read_csv(
        published_path("Supply/S09/data/plot_values/native_T_sensitivity.csv")
    ):
        native_t = int(r["native_T"])
        is_wake = native_t == works["direct"]["direct_T"]
        p = float(r["p_phys"])
        for col, policy in (("strict", "enforced_supply"), ("relaxed", "waiting")):
            block = r[f"{col}_block"]
            d_core = int(r[f"{col}_d_core"])
            levels = group(r[f"{col}_factory_distances"])
            cfg = configuration(p, block)
            chain = model.evaluate_factory_distances(p, d_core, levels, cfg)
            units = interlevel_units(
                p, d_core, chain, int(r[f"{col}_factory_units"]), cfg
            )
            rows.extend(
                level_terms(
                    "native_T_sweep",
                    "direct" if is_wake else "native_T_control",
                    policy,
                    p,
                    block,
                    d_core,
                    levels,
                    units,
                    native_t,
                    is_wake,
                )
            )
    write_csv(DATA / "distillation_acceptance_terms.csv", rows)
    litinski = litinski_116_to_12()
    write_csv(DATA / "Litinski/acceptance_116_to_12.csv", litinski)
    print(
        f"Wrote {len(rows):,} Silva level records and {len(litinski)} Litinski design records."
    )


if __name__ == "__main__":
    main()
