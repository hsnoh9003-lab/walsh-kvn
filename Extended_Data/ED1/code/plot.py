"""Reproduce Extended Data Figure 1 from archived IBM MPS data; no provider access."""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[3]))
from Source.data_locations import published_path
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DATA = published_path('Hardware/IBM/data/ibm_mps_calibration_full_precision.csv').parent
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 6.5,
    'axes.linewidth': .6, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'walsh-kvn-ed1',
})
BLUE, ORANGE, INK, GRAY = '#35658C', '#B95432', '#171717', '#888888'


def export_source_data(rows):
    # The continuum reference uses the prepared distribution and is independent of shots.
    sys.path.insert(0, str(ROOT/'Hardware/IBM/code'))
    from analyze_mps_ibm import mps_state, target_masses
    from reduce_shots import continuum

    initial = {}
    publication = []
    shots = {}
    for row in rows:
        n, nt, k = (int(row[x]) for x in ('qubits', 'nt', 'k'))
        if n not in initial:
            initial[n] = abs(mps_state(np.sqrt(target_masses(n)), n))**2
        published = dict(row)
        published['acquired_shots'] = row['shots']
        published['P_continuum'] = continuum(initial[n], k*(49.8291-40)/nt, 'P')
        publication.append(published)
        shots[(row['cells'], row['nt'], row['k'])] = row['shots']
    columns = list(rows[0])
    columns.insert(columns.index('shots') + 1, 'acquired_shots')
    columns.insert(columns.index('r_continuum'), 'P_continuum')
    destination = ROOT/'Build/Extended_Data/ED1'
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'Source_Data_Extended_Data_Figure_1.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(publication)
    with (DATA/'ibm_mps_reference_diagnostics.csv').open() as file:
        diagnostics = list(csv.DictReader(file))
    columns = list(diagnostics[0])
    columns.insert(columns.index('k') + 1, 'shots')
    for row in diagnostics:
        row['shots'] = shots[(row['cells'], row['nt'], row['k'])]
    with (destination/'IBM_Hardware_Reference_Diagnostics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-source-diagnostics', action='store_true',
                        help='Recompute source-data diagnostics into Build/; do not modify published Source_Data')
    args = parser.parse_args()
    source = DATA/'ibm_mps_calibration_full_precision.csv'
    rows = list(csv.DictReader(source.open()))
    fig, axes = plt.subplots(2, 2, figsize=(172.72/25.4, 118.745/25.4), sharex=True, sharey='col')
    fig.subplots_adjust(left=.095, right=.99, bottom=.10, top=.93, wspace=.24, hspace=.30)
    for row_index, nt in enumerate((4, 6)):
        series = [r for r in rows if int(r['nt']) == nt and int(r['cells']) == 16]
        col = lambda key: np.array([float(r[key]) for r in series])
        times = col('t_star')
        for col_index, (key, label) in enumerate((('P', r'$P(r\geq7/8)$'), ('r', r'$\langle r\rangle$'))):
            ax = axes[row_index, col_index]
            ax.plot(times, col(key+'_ideal'), color=INK, linewidth=.8, zorder=2)
            if key == 'r':
                ax.plot(times, col('r_continuum'), '--', color=GRAY, linewidth=.8, dashes=(4, 2), zorder=1)
                ax.axhline(31/32, color='#A9A9A9', linewidth=.55, linestyle=(0, (1.5, 2.5)), zorder=0)
                ax.text(40.2, 31/32+.006, '31/32', color='#777777', fontsize=6)
            # Keep the endpoints visible around small markers at publication size.
            # Cap width is in display points. All vertical lengths use the source errors.
            ax.errorbar(times, col(key+'_raw'), yerr=col(key+'_raw_SE'), fmt='none',
                        ecolor=BLUE, elinewidth=.68, capsize=3.1, capthick=.62, zorder=3,
                        gid=f'{row_index}-{key}-measured-errors')
            ax.plot(times, col(key+'_raw'), 'o', color=BLUE, markerfacecolor='white',
                    markersize=3.25, markeredgewidth=.72, zorder=4)
            # At k=0 only the preparation was measured; there is no paired reference.
            good = [r for r in series if int(r['k']) > 0]
            mitigated_t = [float(r['t_star']) for r in good]
            mitigated_y = [float(r[key+'_refnorm']) for r in good]
            ax.errorbar(mitigated_t, mitigated_y,
                        yerr=[float(r[key+'_refnorm_err']) for r in good], fmt='none',
                        ecolor=ORANGE, elinewidth=.72, capsize=3.1, capthick=.62, zorder=3)
            ax.plot(mitigated_t, mitigated_y, 's', color=ORANGE, markeredgecolor=ORANGE,
                    markersize=3.0, markeredgewidth=.45, zorder=4)
            ax.set_ylabel(label, labelpad=4)
            ax.set_xlim(39.7, 50.1); ax.set_xticks([40, 42, 44, 46, 48, 50])
            ax.tick_params(length=3, width=.6, labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.text(-.15, 1.065, 'abcd'[2*row_index+col_index], transform=ax.transAxes,
                    fontsize=7, fontweight='bold', va='bottom')
            if col_index == 0:
                ax.set_title(f'{nt} steps, {int(series[0]["shots"]):,} shots per circuit',
                             loc='left', pad=9, fontsize=6.5)
            if row_index == 1: ax.set_xlabel(r'$t^*$', labelpad=3)
            if key == 'P':
                ax.set_ylim(.25, 1.12); ax.set_yticks([.4, .6, .8, 1.])
            else:
                ax.set_ylim(.60, 1.065); ax.set_yticks([.6, .7, .8, .9, 1.])
    handles = [Line2D([], [], color=INK, lw=.8, label='Ideal circuit'),
               Line2D([], [], marker='o', color=BLUE, mfc='white', mew=.72, ms=3.25, lw=0, label='Measured'),
               Line2D([], [], marker='s', color=ORANGE, mec=ORANGE, mew=.45, ms=3.0, lw=0, label='Noise-mitigated'),
               Line2D([], [], color=GRAY, lw=.8, linestyle='--', label='Continuum radial model')]
    axes[0, 0].legend(handles=handles, loc='lower left', ncol=2,
                      frameon=False, fontsize=6.5, handlelength=1.8,
                      columnspacing=1.2, labelspacing=.45, borderaxespad=.7)
    for folder in ['artwork', 'pdf']:
        (HERE/folder).mkdir(exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(HERE/f'artwork/Extended_Data_Figure_1.{suffix}', bbox_inches=None, pad_inches=0,
                    metadata={'Date': None} if suffix == 'svg' else None)
    fig.savefig(HERE/'pdf/Extended_Data_Figure_1.pdf', bbox_inches=None, pad_inches=0,
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    if args.recompute_source_diagnostics:
        export_source_data(rows)
    print('Extended Data Figure 1: vector PDF/SVG and 300-dpi PNG from published Source Data.')


if __name__ == '__main__':
    main()
