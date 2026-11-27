"""Portable replay of numerical-detail extraction (Python, NumPy and lxml)."""
from pathlib import Path
import csv, json, subprocess, sys
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'code/collect_details.py')],check=True)
out=root/'reproduced';out.mkdir(exist_ok=True)
for name,rows in json.loads((root/'tables.json').read_text()).items():
    with (out/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
print(out)
