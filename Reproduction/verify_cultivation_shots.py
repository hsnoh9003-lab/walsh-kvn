"""Check the measured cultivation inputs and the target shots per escape distance."""
from collections import defaultdict
from pathlib import Path

from Source.RE.preparation import read_csv
from Source.data_locations import shared_data

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DATA = shared_data('RE_common','RE')
RAW_DIRECTORY = shared_data('Gidney_cultivation','raw')
AUDIT_DIRECTORY = shared_data('Gidney_cultivation','audit')
# Keys are physical error probabilities in units of 10^-4.
TARGETS = {2: 20_000_000_000, 3: 10_000_000_000, 4: 4_000_000_000,
           5: 10_000_000_000, 6: 1_200_000_000, 7: 1_000_000_000,
           8: 800_000_000, 9: 800_000_000}


def setting(row):
    return round(float(row["p_phys"]) * 10_000), int(row["d2"])


def verify():
    groups = defaultdict(list)
    for rate in TARGETS:
        for row in read_csv(RAW_DIRECTORY / f'custom_p{rate}e-4_d1_3_d2.csv'):
            groups[setting(row)].append(row)
    expected = {(rate, d2) for rate in TARGETS for d2 in range(7, 26, 2)}
    assert set(groups) == expected
    audit = read_csv(AUDIT_DIRECTORY / 'shot_target_by_distance.csv')
    audit_by_setting = {setting(row): row for row in audit}
    assert len(audit) == len(expected) and set(audit_by_setting) == expected
    met = 0
    for rate, target in TARGETS.items():
        table = read_csv(RESOURCE_DATA / f"prep_p{rate}.csv")
        assert len(table) == 10 and {setting(row) for row in table} == {k for k in expected if k[0] == rate}
        for row in table:
            key = setting(row)
            record = audit_by_setting[key]
            shots = int(row["shots"])
            samples = groups[key]
            # All decoder thresholds at this setting share the same trials.
            assert {int(item["shots"]) for item in samples} == {shots}, key
            accepted = [item for item in samples if float(item["P_prep"]) > 0]
            best = min(accepted, key=lambda item: (float(item["e_out_T_est"]),
                       -float(item["P_prep"]), float(item["gap_db"])))
            for prep_col, raw_col in (("gap_db", "gap_db"), ("discards", "discards"),
                                     ("kept", "kept"), ("errors", "logical_errors"),
                                     ("P_prep", "P_prep"), ("e_prep", "e_out_T_est"),
                                     ("e_prep_95_lo", "e_out_T_est_95_lo"),
                                     ("e_prep_95_hi", "e_out_T_est_95_hi")):
                assert abs(float(row[prep_col])-float(best[raw_col])) < 1e-13, (key, prep_col)
            assert int(record["actual_shots"]) == shots and int(record["target_shots"]) == target, key
            assert record["target_met"] == str(shots >= target), key
            for column in ("gap_db", "P_prep", "e_prep"):
                assert float(record[column]) == float(row[column]), (key, column)
            met += int(shots >= target)
    print(f"Verified {len(audit)} preparation entries. Target reached at {met} of {len(audit)} settings.")


if __name__ == "__main__":
    verify()
