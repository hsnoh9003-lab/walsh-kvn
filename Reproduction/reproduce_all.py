"""Batch replay for all publicly reproducible main/supplement figures.

Usage: python -m Reproduction.reproduce_all [--group figure|supply]
"""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED = {'Figure/F2', 'Supply/S02', 'Supply/S06', 'Supply/S07'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=['figure','supply','all'],default='all')
    args=p.parse_args()
    groups=[]
    if args.group in ('figure','all'):groups.append(('Figure','F',5))
    if args.group in ('supply','all'):groups.append(('Supply','S',10))
    summary=[]
    for group,prefix,last in groups:
        for i in range(1,last+1):
            name=f'{prefix}{i:02d}' if prefix=='S' else f'{prefix}{i}'
            item=f'{group}/{name}'
            if item in EXCLUDED:
                summary.append({'item': item, 'status': 'SKIPPED', 'reason': 'image-only; not publicly released'})
                print(item, 'SKIPPED (image-only)', flush=True)
                continue
            t=time.perf_counter()
            try:
                # Do not use PIPE here: PDF/SVG renderers may spawn children that
                # inherit stdout pipes, preventing the parent from reaching EOF.
                with tempfile.TemporaryFile(mode='w+t',encoding='utf-8') as log:
                    q=subprocess.run([sys.executable,'-m','Reproduction.run', 'figure' if prefix=='F' else 'supply',name,'--mode','plot'],cwd=ROOT,stdout=log,stderr=log,timeout=45)
                    log.seek(0);tail=log.read()[-1000:] if q.returncode else ''
                status='PASS' if q.returncode==0 else 'FAIL'
                summary.append({'item':item,'status':status,'exit_code':q.returncode,'elapsed_seconds':round(time.perf_counter()-t,2),'error_tail':tail})
            except subprocess.TimeoutExpired:
                status='TIMEOUT';summary.append({'item':item,'status':status,'elapsed_seconds':round(time.perf_counter()-t,2)})
            print(item,status,flush=True)
    report=ROOT/'Build/Figure_Reproduction_Results.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 1 if any(row['status'] not in {'PASS','SKIPPED'} for row in summary) else 0

if __name__=='__main__':sys.exit(main())
