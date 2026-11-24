"""Render spatial and temporal refinement checks from the archived numerical results."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np

FIGURE = Path(__file__).resolve().parents[1]
PACKAGE = FIGURE.parents[1]
sys.path.insert(0, str(PACKAGE))
from Source.data_locations import figure_data, published_path
DATA = figure_data('S03')
CASES = {'quadratic':'Q2', 'cubic_interior':'C3', 'quintic_multistable':'Q5', 'coupled_cubic_2d':'2D C3'}
STYLES = [('#246a92','o','-'), ('#b45834','s','--'), ('#785493','D','-.'), ('#39816a','^',':')]
LEGEND = dict(frameon=False,fontsize=6.8,labelspacing=.15,handlelength=1.42,handletextpad=.37,borderaxespad=.29,borderpad=.2)

def table(filename):
    with (DATA/'plot_values'/filename).open(newline='') as f:
        return list(csv.DictReader(f))

def density(ax):
    with published_path('Data/Grid_Oscillation_Diagnostics/radial_cell_masses.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    def mass(step):
        s=sorted((r for r in rows if int(r['step'])==step),key=lambda r:int(r['cell']))
        assert len(s)==64
        return np.array([float(r['circuit_mass']) for r in s])
    initial,final=mass(0),mass(33)
    cfg=json.loads(published_path('Data/Radial128/inputs/configuration.json').read_text())
    g=np.exp(cfg['parameters']['mu']*(float(rows[-1]['time'])-float(rows[0]['time'])))
    edges=np.linspace(0,1,65)
    start=edges[:-1,None]+np.linspace(0,1,12)[None,:]/64
    den=1+start**2*(g**2-1)
    ax.stairs(64*initial,edges,fill=True,color='#dce1e4',label='Initial')
    ax.stairs(64*final,edges,color='#246a92',linewidth=.9,baseline=None,label='Circuit')
    ax.plot((g*start/np.sqrt(den)).ravel(),(64*initial[:,None]*den**1.5/g).ravel(),'--',color='#363f43',lw=.72,label='Continuum')
    ax.set(xlim=(0,1.018),ylim=(0,55),xlabel=r'Normalized radius, $r$',ylabel='Density',xticks=[0,.25,.5,.75,1],yticks=[0,10,20,30,40,50])
    ax.set_xticklabels(['0','0.25','0.5','0.75','1'])
    ax.legend(loc='upper left',**LEGEND)

def wake(b,c):
    rows=table('wake_refinement.csv')
    assert {(int(r['positive_cells']),int(r['steps'])) for r in rows}=={(x,t) for x in [64,128] for t in [33,66,132]} and len(rows)==6
    cell_handles=[]
    for cells,colour,marker in [(64,'#246a92','o'),(128,'#ad532e','^')]:
        series=sorted((r for r in rows if int(r['positive_cells'])==cells),key=lambda r:int(r['steps']))
        steps=[int(r['steps']) for r in series]
        biases=[float(r['bias_pct']) for r in series]
        spatial_limits={float(r['finite_generator_bias_pct']) for r in series}
        assert len(spatial_limits)==1
        b.axhline(spatial_limits.pop(),color=colour,ls=':',lw=.83,alpha=.88,zorder=1)
        line,=b.plot(steps,biases,marker=marker,color=colour,ms=3.75,lw=.95,label=f'{cells} cells',zorder=3)
        cell_handles.append(line)
        # On this fixed interval, Nt/Nx equal to the 64-cell benchmark fixes dt/h.
        index=steps.index(33*cells//64)
        b.plot(steps[index],biases[index],marker='o',ms=8,markerfacecolor='none',
               markeredgecolor='#25282a',markeredgewidth=.72,ls='none',zorder=4)
    exact=Line2D([],[],ls=':',color='#57636a',lw=.9,label='Spatial bias')
    matched=Line2D([],[],marker='o',ms=6,markerfacecolor='none',markeredgecolor='#25282a',
                   markeredgewidth=.72,ls='none',label='Cells & steps ×2')
    b.set_xscale('log',base=2)
    b.set(xlabel=r'Time steps, $N_t$',ylabel='Field bias (%)',xlim=(29,148),ylim=(.015,.157),
          yticks=[.02,.06,.10,.14])
    b.set_xticks([33,66,132],['33','66','132'])
    b.legend(handles=[cell_handles[0],exact,cell_handles[1],matched],
             loc='upper center',ncol=2,columnspacing=.49,**{**LEGEND, 'fontsize':6.5})
    temporal=sorted((r for r in rows if int(r['positive_cells'])==64),key=lambda r:int(r['steps']))
    err=np.array([float(r['time_error_pct']) for r in temporal])
    c.loglog([33,66,132],err,'o-',color='#246a92',ms=3.6,lw=.9)
    guide=np.array([40,107])
    c.loglog(guide,.70*err[0]*(33/guide)**2,':',color='#57636a',lw=.94,label=r'$N_t^{-2}$')
    c.set(xlabel=r'Time steps, $N_t$',ylabel='Trotterization error (%)',xlim=(30,145),ylim=(.0011,.067))
    c.set_xticks([33,66,132],['33','66','132'])
    c.legend(loc='upper right',**LEGEND)

def polynomial(d, e):
    spatial,time=table('spatial_refinement.csv'),table('time_refinement.csv')
    assert set(r['case'] for r in spatial)==set(r['case'] for r in time)==set(CASES)
    model_handles=[]
    for (key,label),(col,mark,ls) in zip(CASES.items(),STYLES):
        sg=sorted((r for r in spatial if r['case']==key),key=lambda r:int(r['cells_per_coordinate']))
        tg=sorted((r for r in time if r['case']==key),key=lambda r:int(r['steps']))
        assert len(sg)==len(tg)==4
        x=[int(r['cells_per_coordinate']) for r in sg]
        target=[float(r['moment_rms_total_analytic_error']) for r in sg]
        prepared=[float(r['moment_rms_transport_error']) for r in sg]
        steps=[int(r['steps']) for r in tg]
        state=[float(r['state_l2_splitting_error']) for r in tg]
        d.loglog(x,target,color=col,marker=mark,ls=ls,ms=3.1,mew=.5,lw=.8,zorder=3)
        d.loglog(x,prepared,color=col,marker=mark,ls='none',ms=5.3,mfc='none',mew=.75,zorder=4)
        e.loglog(steps,state,color=col,marker=mark,ls=ls,ms=3.5,lw=.9,mew=.7)
        model_handles.append(Line2D([],[],color=col,marker=mark,ls=ls,ms=3.5,lw=.8,label=label))
    d.set(xlim=(7,149),ylim=(.000035,.16),xlabel=r'Cells per coordinate, $N_x$',ylabel='Moment RMS error',yticks=[.0001,.001,.01,.1])
    model_legend=d.legend(handles=model_handles,loc='upper right',ncol=2,columnspacing=.72,**LEGEND)
    d.add_artist(model_legend)
    ref_handles=[Line2D([],[],marker='o',ls='none',ms=5.1,mfc='none',mec='#343434',mew=.75,label='Circuit cell probabilities'),
                 Line2D([],[],marker='o',ls='-',ms=3.3,color='#343434',lw=.8,label='Continuous Gaussian')]
    d.legend(handles=ref_handles,title='Classical reference',title_fontsize=6.7,loc='lower left',**LEGEND)
    e.set(xlim=(7,149),ylim=(.000055,.13),xlabel=r'Time steps, $N_t$',ylabel='State-vector distance')
    guide=np.array([17,107])
    e.loglog(guide,.075*(16/guide)**2,ls=':',color='#45494c',lw=.9)
    e.text(0.67,.66,r'$N_t^{-2}$',color='#45494c',transform=e.transAxes,fontsize=7.2)
    for a in (d,e):
        a.set_xticks([8,16,32,64,128],['8','16','32','64','128'])

def plot(out):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':8.5,'axes.linewidth':.6,
        'xtick.direction':'in','ytick.direction':'in','figure.dpi':150,'savefig.dpi':300,'pdf.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'kvn-refinement-s3'})
    fig=plt.figure(figsize=(180/25.4,105/25.4))
    grid=fig.add_gridspec(2,3)
    axs=[fig.add_subplot(grid[0,j]) for j in range(3)]
    axs.extend([fig.add_subplot(grid[1,:2]),fig.add_subplot(grid[1,2])])
    fig.subplots_adjust(left=.086,right=.986,bottom=.107,top=.946,wspace=.46,hspace=.61)
    density(axs[0]);wake(*axs[1:3]);polynomial(*axs[3:])
    for letter,ax in zip('abcde',axs):
        offset=-.014 if letter=='d' else -.035
        ax.text(offset,1.034,letter,weight='bold',fontsize=9.2,transform=ax.transAxes,va='bottom')
        ax.tick_params(labelsize=7.3,length=3,pad=2.2)
        ax.spines[['top','right']].set_visible(False)
        if ax.get_xscale()=='log':ax.xaxis.set_minor_locator(NullLocator())
        if ax.get_yscale()=='log':ax.yaxis.set_minor_formatter(NullFormatter())
    out.mkdir(exist_ok=True,parents=True)
    for ext,meta in [('svg',{'Date':None}),('pdf',{'CreationDate':None,'ModDate':None}),('png',None)]:
        fig.savefig(out/f'Figure_S3.{ext}',metadata=meta,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    plot(Path(sys.argv[1]) if len(sys.argv)>1 else FIGURE/'reproduced')
