"""Unified non-destructive presentation entrypoints; numerical generation is explicit."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def get_entry(kind,name,mode):
    unpublished = {('figure','F2'),('supply','S02'),('supply','S06'),('supply','S07')}
    if (kind,name) in unpublished:
        raise ValueError(f'{kind}/{name} is image-only and excluded from this GitHub code release')
    group={'figure':'Figure','supply':'Supply'}[kind]
    base=ROOT/group/name
    if not base.is_dir():raise FileNotFoundError(base)
    if mode=='plot':
        p=base/'code/plot.py'
        if not p.is_file():p=base/'code/reproduce_figure.py'
    elif mode=='from-data':
        # For existing plot-value workflows this is the explicit CSV-based entry.
        p=base/'code/plot.py'
        if not p.is_file():p=base/'code/reproduce_from_data.py'
        if not p.is_file():raise FileNotFoundError(f'No standalone from-data renderer for {kind}/{name}: {p}')
    else:raise ValueError(mode)
    if not p.is_file():raise FileNotFoundError(p)
    return p

def run(kind,name,mode):
    if mode=='generate-f3-data':
        if kind!='figure' or name!='F3':raise ValueError('Only Figure F3 generation registered')
        script=ROOT/'Reproduction/generate_fig3_data.py'
    else:script=get_entry(kind,name,mode)
    command=[sys.executable,str(script)]
    if mode in ('plot','from-data') and kind=='figure' and name=='F3' and script.name=='plot.py':
        command.extend([str(ROOT/'Figure/F3/reproduced'),'--clip-to-window'])
    elif mode in ('plot','from-data') and kind=='figure' and name=='F4' and script.name=='plot.py':
        command.extend(['--package-root',str(ROOT)])
    elif mode in ('plot','from-data') and script.name=='plot.py':
        command.append(str(ROOT/({'figure':'Figure','supply':'Supply'}[kind])/name/'reproduced'))
    subprocess.run(command,cwd=ROOT,check=True)
    if mode in ('plot','from-data'):
        sys.path.insert(0,str(ROOT/'T'))
        from ncs_style import normalize_folder
        key=name if kind=='figure' else 'S'+str(int(name[1:]))
        # These figures already apply their publication typography and page geometry.
        if key not in ['F4','F3','S1','S2','S3','S4','S5','S7','S8','S9','S10']:
            normalize_folder(ROOT/({'figure':'Figure','supply':'Supply'}[kind])/name/'reproduced',key)


def main():
    ap=argparse.ArgumentParser(description='Reproduce saved figure artwork or explicitly regenerate a verified table')
    ap.add_argument('kind',choices=['figure','supply'])
    ap.add_argument('name',help='F1..F5 or S01..S10')
    ap.add_argument('--mode',choices=['plot','from-data','generate-f3-data'],default='plot')
    args=ap.parse_args()
    run(args.kind,args.name,args.mode)
if __name__=='__main__':main()
