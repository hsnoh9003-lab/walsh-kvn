from pathlib import Path
import csv
import sys
from reproduce_figure import CASES, FIGURE as ROOT
sys.path.insert(0, str(ROOT.parents[1]))
from Source.data_locations import figure_data

def extract(output):
    output.mkdir(exist_ok=True, parents=True)
    for source in sorted(figure_data('S03').glob('*_refinement.csv')):
        with source.open(newline='') as original, (output/source.name).open('w',newline='') as target:
            rows=csv.DictReader(original)
            writer=csv.DictWriter(target, rows.fieldnames)
            writer.writeheader()
            for row in rows:
                if row['case'] in CASES:
                    row['label']=CASES[row['case']]
                    writer.writerow(row)

if __name__=='__main__':
    extract(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parents[1]/'Build/Figure_Source_Exports/S03/sweeps')
