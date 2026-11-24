"""Check Figure 5c without a physical-qubit cap using the canonical RE search.

The existing capped solution supplies a feasible runtime incumbent only.
Source.RE.uncapped.find_fastest sets q_max to infinity and derives a finite
code-distance bound from that runtime. No field, circuit or shot input changes.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, math, time
import pandas as pd
from Source.RE import preparation as S, uncapped as U

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Build/One_Clean_Three_Coordinate/resources/uncapped_coupled_check'


def check_rate(ip):
    tick = time.monotonic()
    table = pd.read_csv(published_path('Data/RE/v38_reference_capped_selected.csv'))
    table = table[(table.model == '2d_interior') &
                  ((table.p_phys * 10000).round().astype(int) == ip)]
    options = table[table.selection == 'fastest_under_cap'].to_dict('records')
    incumbent = min(options, key=U.fastest_key)
    work = S.load_workloads(published_path('Data/RE/workloads_v38.json'))['2d_interior']
    fastest, audit = U.find_fastest('2d_interior', work, incumbent,
                                    published_path(f'Data/RE/prep_p{ip}.csv'))
    fastest.update(Clifford=work['Clifford'],
        complete_task_hours=fastest['complete_task_seconds'] / 3600,
        complete_task_days=fastest['complete_task_seconds'] / 86400,
        complete_task_years=fastest['complete_task_seconds'] / (86400 * 365.25))
    fields = ['physical_qubits', 'seconds_per_attempt', 'N_T', 'd_core',
              'block', 'distances', 'u_levels', 'T_count', 'successful_shots']
    same = all(str(fastest[k]) == str(incumbent[k]) if isinstance(fastest[k], str)
               else math.isclose(float(fastest[k]), float(incumbent[k]),
                                 rel_tol=1e-13, abs_tol=0)
               for k in fields)
    minimum = min(table[table.selection == 'minimum_qubits'].to_dict('records'),
                  key=lambda r: (r['physical_qubits'], r['seconds_per_attempt']))
    assert minimum['physical_qubits'] < 1_000_000
    audit.update(rate_index=ip, physical_qubit_cap_applied=False,
        capped_incumbent=incumbent, compared_fields=fields,
        same_fastest_design=same, elapsed_seconds=time.monotonic()-tick,
        minimum_qubits_unchanged_reason=(
            'The capped minimum lies below the cap; any newly admitted design '
            'has more qubits and cannot improve the minimum-qubit objective.'))
    OUT.mkdir(parents=True, exist_ok=True)
    S.write_csv(OUT / f'selected_p{ip}.csv', [fastest])
    (OUT / f'audit_p{ip}.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(rate=ip, same_fastest_design=same,
        physical_qubits=fastest['physical_qubits'],
        task_hours=fastest['complete_task_hours'],
        checked_geometries=audit['checked_geometries'],
        elapsed_seconds=round(audit['elapsed_seconds'], 2))), flush=True)
    return fastest, audit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=4)
    a=ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.processes) as pool:
        futures=[pool.submit(check_rate, ip) for ip in range(2,10)]
        results=[f.result() for f in as_completed(futures)]
    results.sort(key=lambda x: x[1]['rate_index'])
    S.write_csv(OUT/'selected.csv', [r for r,_ in results])
    summary={'physical_qubit_cap_applied':False,
        'all_eight_fastest_designs_unchanged':all(a['same_fastest_design'] for _,a in results),
        'rate_indices':[a['rate_index'] for _,a in results],
        'max_fastest_physical_qubits':max(r['physical_qubits'] for r,_ in results),
        'model':'2d_interior','successful_shots':674,
        'search':'Source.RE.uncapped.find_fastest with q_max=inf'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
