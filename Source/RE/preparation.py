"""Load circuit workloads and prepare measured magic-state input tables.

The compiled logical width is taken directly from each workload inventory.
"""

from __future__ import annotations
from pathlib import Path
import csv
import math
from decimal import Decimal, InvalidOperation
import numpy as np
from . import model


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# Convert measured Gidney magic-state preparation records into the
# distance-indexed input format used by the resource estimator.
#
# The raw-data field d2 denotes the cultivation output, or escape, distance.
# It becomes "degree" in the prepared table and corresponds to the first
# distillation distance d_1 in the resource model.
# It is distinct from both the raw cultivation d1 and a polynomial degree.
#
# A table that already contains degree, e_prep, and P_prep is returned directly.
# This early-return path does not filter or revalidate its p_phys values.
def make_prep_table(path, p_phys, out):
    """Select one measured preparation setting per output code distance."""
    rows = read_csv(path)
    if {"degree", "e_prep", "P_prep"}.issubset(rows[0]):
        return path
    required = {
        "p_phys",
        "d1",
        "d2",
        "r1",
        "r2",
        "gap_db",
        "shots",
        "discards",
        "kept",
        "logical_errors",
        "P_prep",
        "e_out_S",
        "e_out_S_95_lo",
        "e_out_S_95_hi",
        "e_out_T_est",
        "e_out_T_est_95_lo",
        "e_out_T_est_95_hi",
    }
    if not required.issubset(rows[0]):
        raise ValueError("Missing measured prep columns in " + str(path))
    # A supplied sweep can contain several physical error probabilities.
    rows = [row for row in rows if float(row["p_phys"]) == p_phys]
    if not rows:
        raise ValueError(f"No raw preparation records for p_phys={p_phys} in {path}")
    # Select one preparation setting for each output distance, ordered by:
    #   1. lowest estimated T-state error;
    #   2. highest preparation acceptance probability;
    #   3. lowest gap_db.
    # Selection uses the error point estimate rather than its confidence bound.
    best = {}
    for r in rows:
        d = int(float(r["d2"]))
        P_prep = float(r["P_prep"])
        e_prep = float(r["e_out_T_est"])
        if P_prep <= 0 or not np.isfinite(e_prep) or e_prep < 0:
            continue
        rank = (e_prep, -P_prep, float(r["gap_db"]))
        if d not in best or rank < best[d][0]:
            best[d] = (rank, r)
    chosen = []
    for d, (_, r) in sorted(best.items()):
        chosen.append(
            dict(
                degree=d,
                d2=d,
                p_phys=p_phys,
                cultivation_d1=int(float(r["d1"])),
                r1=r["r1"],
                r2=r["r2"],
                gap_db=r["gap_db"],
                shots=r["shots"],
                discards=r["discards"],
                kept=r["kept"],
                errors=r["logical_errors"],
                P_prep=r["P_prep"],
                expected_attempts=1 / float(r["P_prep"]),
                e_S_sim=r["e_out_S"],
                e_S_95_lo=r["e_out_S_95_lo"],
                e_S_95_hi=r["e_out_S_95_hi"],
                e_prep=r["e_out_T_est"],
                e_prep_95_lo=r["e_out_T_est_95_lo"],
                e_prep_95_hi=r["e_out_T_est_95_hi"],
                T_over_S_factor=2.0,
                selection_rule="min_estimated_T_error_then_max_P_prep",
            )
        )
    write_csv(out, chosen)
    return out


def make_config(p_phys, table, block, workload, *, d_min=13, d_max=49):
    """Build a resource configuration after checking any supplied table physical error rate."""
    rows = read_csv(table)
    if not rows:
        raise ValueError("Preparation table must not be empty")
    if "p_phys" in rows[0] and any(
        not math.isclose(float(row["p_phys"]), p_phys, rel_tol=1e-10, abs_tol=0)
        for row in rows
    ):
        raise ValueError(
            "Preparation table p_phys does not match the requested physical error probability"
        )
    cfg = model.ModelConfig(
        n_L=workload["logical_qubits"],
        data_block=block,
        p_phys=p_phys,
        d_min=d_min,
        d_max=d_max,
        prep_degree_table_path=str(Path(table).resolve()),
    )
    cfg.validate()
    return cfg


def load_workloads(path):
    """Load a workload JSON file and normalize its fields for resource estimation."""
    import json

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Workloads must be a nonempty JSON object keyed by circuit name"
        )
    clean = {}
    for name, r in raw.items():
        if not isinstance(r, dict):
            raise ValueError(f"Workload {name} must be a JSON object")

        def integer(key, minimum=0):
            value = r.get(key)
            try:
                number = Decimal(str(value))
            except InvalidOperation:
                raise ValueError(
                    f"{name}.{key} must be an integer >= {minimum}"
                ) from None
            if (
                not number.is_finite()
                or number != number.to_integral_value()
                or number < minimum
            ):
                raise ValueError(f"{name}.{key} must be an integer >= {minimum}")
            return int(number)

        logical_qubits = integer("logical_qubits", 1)
        rotations = integer("RZ")
        native_t = integer("native_T")
        shots = integer("successful_shots", 1)
        if rotations + native_t == 0:
            raise ValueError(
                f"{name}: this T-state resource model requires RZ or native_T > 0"
            )
        try:
            success = float(r["success_probability"])
            angle_error = float(r.get("source_angle_error_bound") or 0.0)
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"{name}: provide numeric success_probability and source_angle_error_bound"
            ) from None
        if not math.isfinite(success) or not 0 < success <= 1:
            raise ValueError(f"{name}.success_probability must be in (0, 1]")
        if not math.isfinite(angle_error) or not 0 <= angle_error < model.ETA:
            raise ValueError(
                f"{name}.source_angle_error_bound must be in [0, {model.ETA})"
            )
        is_wake = r.get(
            "is_wake", name in {"direct", "generator", "long298", "restart40"}
        )
        if not isinstance(is_wake, bool):
            raise ValueError(f"is_wake must be a Boolean for {name}")
        # RZ counts arbitrary-angle rotations per circuit attempt.
        # direct_T counts native T gates, excluding rotation-synthesis costs.
        # logical_qubits is the total logical width recorded by the circuit compiler.
        # shots is the required number of accepted samples.
        # success is the circuit acceptance probability, distinct from preparation
        # and distillation acceptance probabilities.
        # canonicalization_error_bound bounds the error from source-angle rounding.
        #
        # The Clifford inventory is retained as metadata. The current estimator
        # accounts for core faults through architecture-dependent exposure volumes
        # rather than adding an independent fault term for each listed Clifford gate.
        clean[name] = dict(
            logical_qubits=logical_qubits,
            is_wake=is_wake,
            RZ=rotations,
            direct_T=native_t,
            Clifford=integer("Clifford") if "Clifford" in r else 0,
            shots=shots,
            success=success,
            canonicalization_error_bound=angle_error,
            precision_min=45,
            precision_max=200,
            shot_basis="accepted samples",
            workload_scope=r.get(
                "workload_scope",
                (
                    "published nonlinear example"
                    if name == "nj_lchs"
                    else (
                        "same ensemble field-error task"
                        if name
                        in {
                            "direct",
                            "generator",
                            "2d_interior",
                            "long298",
                            "restart40",
                        }
                        else "user-specified circuit and accepted-sample target"
                    )
                ),
            ),
        )
        # Optional scientific protocol metadata; it never changes the cost model.
        for key in ("run", "steps"):
            if key in r:
                clean[name][key] = r[key]
    return clean


def design_key(row, capped=False):
    """Return the publication selection key used to align resource rows."""
    selection = row["selection"]
    if (
        not capped
        and row["model"] == "2d_interior"
        and selection == "fastest_under_cap"
    ):
        selection = "fastest_uncapped"
    return (
        row["model"],
        round(float(row["p_phys"]) * 10000),
        selection,
        row["block"] if capped else "",
    )
