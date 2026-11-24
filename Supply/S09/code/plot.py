"""Supply policy in minimum-qubit designs, from the native-T and distance sweeps."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd
if __package__:
    from . import policy_panels as policy
else:
    import policy_panels as policy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from Source.data_locations import figure_data
DATA = figure_data('S09') / 'plot_values'
REFERENCE = 0.0005
WAKE, CONTROL = 7392, 200000
STYLE = {'font.family':'serif','mathtext.fontset':'stix','font.size':8,
         'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
         'legend.fontsize':7,'axes.linewidth':.6,'xtick.direction':'in','ytick.direction':'in',
         'figure.dpi':150,'savefig.dpi':300,'pdf.fonttype':42,'svg.fonttype':'path',
         'svg.hashsalt':'kvn-merged-supply-s8','axes.unicode_minus':True}

def panel_label(fig, letter, x, y):
    fig.text(x, y, letter, ha='left', va='baseline', weight='bold', fontsize=10)

def heatmap(fig, data, selected, left, colour_left, metric):
    time = metric == 'runtime_change_pct'
    centres = np.sort(data.native_T.unique())
    middle = np.sqrt(centres[:-1] * centres[1:])
    edges = np.r_[centres[0]**2/middle[0], middle, centres[-1]**2/middle[-1]]
    rates = np.arange(2,10)
    values = data.pivot(index='rate_index',columns='native_T',values=metric).loc[rates,centres]
    assert np.isfinite(values.to_numpy()).all()
    colours = ['#276686','#f8f8f7','#ad4a37'] if time else ['#245b80','#709bb2','#f7f9fb']
    limits = (-25,25) if time else (-40,0)
    assert values.min().min()>=limits[0] and values.max().max()<=limits[1]
    cmap = LinearSegmentedColormap.from_list(metric,colours)
    norm = Normalize(*limits)
    ax = fig.add_axes([left,.685,.31,.222])
    cbax = fig.add_axes([colour_left,.685,.012,.222])
    mesh = ax.pcolormesh(edges,np.arange(1.5,10),values.to_numpy(),shading='flat',
                         cmap=cmap,norm=norm,edgecolors='face',linewidth=0,rasterized=True)
    ax.set(xscale='log',xlim=(edges[0],edges[-1]),ylim=(9.5,1.5),yticks=rates,xlabel='Native T count')
    ax.set_ylabel(r'Physical error ($10^{-4}$)',labelpad=1.5)
    ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)))
    ax.tick_params(which='major',length=2.4,pad=2)
    ax.tick_params(axis='x',which='minor',length=1.5)
    for y in np.arange(2.5,9):ax.axhline(y,color='#ffffff',lw=.45)
    ax.axvline(WAKE,color='#303030',lw=.75,ls=(0,(2,2)))
    for row in selected.itertuples():
        val=getattr(row,metric)
        ax.scatter(row.native_T,5,s=24,zorder=7,facecolor=cmap(norm(val)),edgecolor='#232323',linewidth=.85)
        pos=(1.7e4,3.65) if row.native_T==WAKE else (6.1e5,6.35)
        ax.annotate(f'{val:+.1f}%',(row.native_T,5),xytext=pos,ha='center',va='center',fontsize=6.8,
                    bbox={'boxstyle':'round,pad=.1','fc':'white','ec':'none','alpha':.92},
                    arrowprops={'arrowstyle':'-','lw':.5,'color':'#242424'},zorder=6)
    cb=fig.colorbar(mesh,cax=cbax,ticks=[-25,0,25] if time else [-40,-20,0])
    cb.ax.tick_params(labelsize=6.5,length=2,pad=1.5)
    cb.set_label('Time change (%)' if time else 'Qubit change (%)',fontsize=7,labelpad=1.5)
    return ax

def distance_pair(fig, curves, optimum, left, lo, hi):
    sel=np.isclose(curves.p_phys,REFERENCE) & curves.d_core.between(lo,hi)
    visible=curves.loc[sel].sort_values('d_core')
    ds=visible.d_core.to_list()
    assert ds==list(range(lo,hi+1,2))
    report={'core_distances':ds}
    for offset,metric,field in [(0,'qubits','q'),(.219,'runtime','t')]:
        ax=fig.add_axes([left+offset,.390,.141,.157])
        policy.draw_policy(ax,visible,optimum,metric)
        ax.set_xlabel('Core code distance',fontsize=6.7,labelpad=1.4)
        ax.tick_params(labelsize=6.3,pad=1.5)
        ax.yaxis.label.set_size(6.8)
        ax.yaxis.get_offset_text().set_fontsize(5.9)
        value=optimum.one_opt_Q if metric=='qubits' else optimum.one_opt_time
        ax.scatter(optimum.one_opt_d,value,s=25,marker='s',facecolor='white',
                   edgecolor=policy.BLUE,linewidth=1,zorder=10)
        vals=visible[['strict_'+field,'one_'+field]].to_numpy()
        bottom,top=np.nanmin(vals),np.nanmax(vals)
        padding=.1*(top-bottom)
        assert np.allclose(ax.get_ylim(),(bottom-padding,top+padding))
        assert np.allclose(ax.get_xlim(),(lo-.75,hi+.75))
        report[metric]={'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())}
    return report

def draw(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    scan=pd.read_csv(DATA/'native_T_sensitivity.csv')
    assert (scan.relaxed_factory_units==1).all()
    assert np.allclose(scan.runtime_change_pct,100*(scan.relaxed_time_s/scan.strict_time_s-1))
    assert np.allclose(scan.qubit_change_pct,100*(scan.relaxed_Q/scan.strict_Q-1))
    positive=scan.loc[scan.native_T>0].copy()
    positive['rate_index']=np.rint(positive.p_phys*1e4).astype(int)
    chosen=positive.loc[np.isclose(positive.p_phys,REFERENCE)&positive.native_T.isin([WAKE,CONTROL])].sort_values('native_T')
    assert chosen.native_T.to_list()==[WAKE,CONTROL]
    opts=pd.read_csv(DATA/'supply_optima.csv').sort_values('p_phys')
    wake=opts.loc[np.isclose(opts.p_phys,REFERENCE)].iloc[0]
    control=pd.read_csv(DATA/'control_distance_optima.csv').query('circuit_T == @CONTROL').iloc[0]
    for idx,opt in enumerate([wake,control]):
        assert np.isclose(opt.p_phys,REFERENCE)
        assert np.allclose(chosen.iloc[idx][['runtime_change_pct','qubit_change_pct']].to_numpy(dtype=float),
                           [opt.runtime_change_pct,opt.physical_qubit_change_pct])
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.05,5.63),facecolor='white')
        panel_label(fig,'a',.030,.919)
        panel_label(fig,'b',.535,.919)
        heatmap(fig,positive,chosen,.104,.431,'runtime_change_pct')
        heatmap(fig,positive,chosen,.608,.935,'qubit_change_pct')
        panel_label(fig,'c',.030,.555)
        panel_label(fig,'d',.535,.555)
        dc=distance_pair(fig,pd.read_csv(DATA/'supply_distances.csv'),wake,.104,19,31)
        dd=distance_pair(fig,pd.read_csv(DATA/'control_distance_curves.csv'),control,.608,19,29)
        panel_label(fig,'e',.030,.257)
        bars=fig.add_axes([.104,.083,.445,.157])
        policy.draw_error_bars(bars,opts,letter='')
        bars.set_xlabel(r'Physical error ($10^{-4}$)',labelpad=2)
        bars.set_xticks(np.arange(2,10)*1e-4,[str(i) for i in range(2,10)])
        bars.tick_params(labelsize=6.7)
        bars.legend(list(bars.containers),['Qubits','Time'],loc='lower right',
                    bbox_to_anchor=(1,1.04),ncol=2,fontsize=6.5,frameon=False,
                    handlelength=1.05,columnspacing=1.0,borderpad=0)
        common={'ms':4,'lw':.9}
        keys=[(Line2D([],[],color=policy.BLUE,marker='s',**common),'Waiting allowed'),
              (Line2D([],[],color=policy.RED,marker='o',**common),'Supply enforced'),
              (Line2D([],[],color='none',marker='s',ms=4,mfc='white',mec=policy.BLUE),'Minimum qubits with waiting'),
              (Line2D([],[],color='none',marker='*',ms=6.5,mfc=policy.STAR,mec='black'),'Minimum qubits with enforced supply'),
              (Line2D([],[],color='none',marker='D',ms=4,mfc='white',mec=policy.RED),'Multiple factory units')]
        fig.legend([x[0] for x in keys],[x[1] for x in keys],bbox_to_anchor=(.625,.172),
                   bbox_transform=fig.transFigure,loc='center left',fontsize=6.4,frameon=False,
                   handlelength=1.4,handletextpad=.6,borderaxespad=0,labelspacing=.45)
        fig.canvas.draw()
        for ax in fig.axes:
            b=ax.get_tightbbox(fig.canvas.get_renderer())
            assert b.x0>=-1 and b.x1<=fig.bbox.width+1 and b.y0>=-1 and b.y1<=fig.bbox.height+1,(ax.get_ylabel(),b)
        for ext,meta in [('svg',{'Date':None}),('pdf',{'CreationDate':None,'ModDate':None}),('png',None)]:
            fig.savefig(out/f'Figure_S9.{ext}',dpi=300,metadata=meta,facecolor='white')
        plt.close(fig)
    audit={'rows':len(scan),'positive_native_T_counts':positive.native_T.nunique(),
           'physical_error_probabilities':list(opts.p_phys),'colour_limits':{'a':[-25,25],'b':[-40,0]},
           'highlighted_rows':chosen[['p_phys','native_T','qubit_change_pct','runtime_change_pct']].to_dict('records'),
           'c':dc,'d':dd,'e_physical_error_rows':len(opts),
           'input_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ('native_T_sensitivity.csv','supply_distances.csv','supply_optima.csv','control_distance_curves.csv','control_distance_optima.csv')}}
    (ROOT/'S9_QA.json').write_text(json.dumps(audit,indent=2)+'\n')

def plot(out):
    draw(out)

if __name__=='__main__':
    plot(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'reproduced')
