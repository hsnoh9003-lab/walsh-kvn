"""Render current Supplementary Figure S10 directly from published Source Data."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pathlib import Path
from Source.data_locations import figure_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

BLUE, GREEN, ORANGE = '#285f86', '#528b78', '#b45239'
PURPLE, TEAL = '#7866a6', '#237c83'
BLOCKS = ['compact', 'intermediate', 'fast']
CAPS = [100_000,1_000_000]
PLABEL = r'Physical error probability ($10^{-4}$)'

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3,width=.6,pad=2.6)
    ax.tick_params(which='minor',length=1.5,width=.45)

def mark(fig, letter, ax):
    bbox=ax.get_position()
    fig.text(.014,bbox.y1+.006,letter,ha='left',va='bottom',fontsize=7.5,weight='bold')

def capacity(folder):
    data = pd.read_csv(folder/'capacity_selected_designs.csv')
    keys = ['block','p_phys','Q_budget']
    assert len(data)==48 and not data.duplicated(keys).any()
    fig = plt.figure(figsize=(180/25.4,126/25.4))
    gs = fig.add_gridspec(3,6,left=.105,right=.986,bottom=.105,top=.968,
                          hspace=.58,wspace=.32,height_ratios=[1.04,1.02,.91])
    reference_grid=gs[0,:].subgridspec(1,2,wspace=.31)
    bar_axes=[fig.add_subplot(reference_grid[0,j]) for j in range(2)]
    panels=np.array([[fig.add_subplot(gs[row+1,2*i:2*i+2]) for i in range(3)] for row in range(2)])
    for row in range(2):
        for i in range(1,3):
            panels[row,i].sharey(panels[row,0]);panels[row,i].tick_params(labelleft=False)
    baseline=data.loc[np.isclose(data.p_phys,.0005)]
    centres=np.arange(len(BLOCKS));bw=.32
    for ax,column,ylabel,scale in zip(bar_axes,['max_steps','service_period_s'],
            ['Maximum coherent steps','T-state consumption\n' r'interval, $\tau_T$ ($\mu$s)'],[1,1e6]):
        high=0.
        for k,cap in enumerate(CAPS):
            v=baseline[baseline.Q_budget.eq(cap)].set_index('block').loc[BLOCKS,column].to_numpy()*scale
            ax.bar(centres+(k-.5)*bw,v,width=bw,color=(BLUE,GREEN)[k],label=f'$10^{5+k}$-qubit cap',zorder=2)
            high=max(high,float(np.max(v)))
        ax.set_ylim(0,high*1.18)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        ax.set_xticks(centres,[name.title() for name in BLOCKS]);ax.set_ylabel(ylabel)
        style(ax);ax.tick_params(length=2.3,pad=2.)
    bar_axes[0].legend(loc='upper left',frameon=False,fontsize=6.2,handlelength=1.2,
                       handletextpad=.3,borderpad=.15,labelspacing=.18)
    key = [Line2D([],[],color=BLUE, marker='o',mfc='white',ms=4.1,mew=.9,ls='none',
                  label=r'$10^5$-qubit cap'),
           Line2D([],[],color=GREEN,marker='D',ms=3.55,mew=.4,ls='none',
                  label=r'$10^6$-qubit cap')]
    for col, block in enumerate(BLOCKS):
        upper, lower = panels[:,col]
        design = data[data.block.eq(block)].sort_values('p_phys')
        pairs = [design[design.Q_budget.eq(cap)] for cap in CAPS]
        x = np.arange(2,10)
        for subset in pairs:
            assert np.allclose(subset.p_phys*1e4,x)
        for ax in (upper,lower):
            ax.axvspan(4.70,5.30,color='#f0ede3',lw=0,zorder=0)
            ax.set_xlim(1.66,9.34)
            ax.set_xticks(x)
            style(ax)
            ax.tick_params(length=2.3,pad=2)
        upper.set_title(block.title(),fontsize=7.9,pad=3.0)
        upper.set_yscale('log');upper.set_ylim(12,720)
        upper.set_yticks([20,50,100,200,500],['20','50','100','200','500'])
        upper.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        upper.grid(axis='y',color='#dadfe1',lw=.4,zorder=0)
        upper.set_axisbelow(True)
        for rate,lo,hi in zip(x,pairs[0].max_steps,pairs[1].max_steps):
            upper.plot([rate-.13,rate+.13],[lo,hi],color='#a6afb5',lw=.72,zorder=1)
        lower.set_ylim(0,1.17)
        lower.set_yticks([0,.5,1],['0','0.5','1'])
        lower.axhline(1,color=GREEN,lw=.63,alpha=.55,zorder=1)
        for i, group in enumerate(pairs):
            color=(BLUE,GREEN)[i]
            opt=dict(color=color,marker=('o','D')[i],mfc=('white',GREEN)[i],
                     mec=color,ms=(4.05,3.55)[i],mew=(.86,.43)[i],ls='none',zorder=3)
            pos=x+(-.13,.13)[i]
            upper.plot(pos,group.max_steps,**opt)
            ratio=group.core_period_s.to_numpy()/group.service_period_s.to_numpy()
            if i==1: assert np.allclose(ratio,1)
            lower.plot(pos,ratio,**opt)
            if col==2:
                upper.annotate(str(int(group.max_steps.to_numpy()[3])),
                               (pos[3],group.max_steps.to_numpy()[3]),
                               xytext=((-3,-6) if i==0 else (2,5)),textcoords='offset points',
                               color=color,fontsize=6.3,ha='right' if i==0 else 'left',
                               va='top' if i==0 else 'bottom')
        upper.tick_params(labelbottom=False)
        lower.set_xticks(x)
    panels[0,0].set_ylabel('Maximum coherent steps')
    panels[1,0].set_ylabel('Nominal / actual interval')
    panels[0,0].legend(handles=key,loc='upper left',frameon=False,fontsize=6.15,
                       handletextpad=.3,borderpad=.15,labelspacing=.18)
    fig.supxlabel(PLABEL,x=.55,y=.024,fontsize=7.3)
    for letter,ax in [('a',bar_axes[0]),('b',panels[0,0]),('c',panels[1,0])]:mark(fig,letter,ax)
    return fig


def plot(output=None):
    root=Path(__file__).resolve().parents[1]
    out=Path(output) if output else root/'reproduced'
    out.mkdir(parents=True,exist_ok=True)
    options={'font.family':'serif','mathtext.fontset':'stix','font.size':6.4,'axes.labelsize':6.3,
             'xtick.labelsize':5.9,'ytick.labelsize':5.9,'axes.linewidth':.6,
             'xtick.direction':'in','ytick.direction':'in','figure.dpi':150,'savefig.dpi':400,
             'svg.fonttype':'path','pdf.fonttype':42,
             'svg.hashsalt':{8:'kvn-supplement-10',10:'kvn-supplement-9'}[10]}
    with plt.rc_context(options):
        plt.rcParams.update({"axes.labelsize":6.9,"xtick.labelsize":6.3,"ytick.labelsize":6.3})
        fig=capacity(figure_data('S10')/'plot_values')
        for ext in ['svg','pdf','png']:
            meta={'Date':None} if ext=='svg' else {'CreationDate':None} if ext=='pdf' else None
            fig.savefig(out/f'Figure_S10.{ext}',facecolor='white',metadata=meta)
        plt.close(fig)


if __name__ == '__main__':
    plot(sys.argv[1] if len(sys.argv)>1 else None)
