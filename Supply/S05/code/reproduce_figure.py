"""Draw the coupled cubic trajectories from the archived characteristic and circuit means."""
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
BLUE = '#28749c'


def trajectories(ax, rows):
    ax.plot(rows.characteristic_mean_x, rows.characteristic_mean_y, color='#242424',
            ls='--', lw=.76, dashes=(2,1.6), zorder=3)
    sampled = rows.iloc[::8]
    ax.plot(sampled.circuit_mean_x, sampled.circuit_mean_y, marker='s', ls='none',
            ms=2.5, color=BLUE, mew=.3, zorder=4)
    ax.plot([1],[0],ls='none',marker='o',ms=2.6,mew=.5,mfc='#222222',mec='#222222',clip_on=True,zorder=5)
    ax.plot([0],[1],ls='none',marker='o',ms=2.6,mew=.6,mfc='white',mec='#222222',clip_on=True,zorder=5)


def draw(out):
    from Source.data_locations import figure_data
    data = pd.read_csv(figure_data('S05')/'plot_values/coupled_mean_trajectories.csv')
    cases = [('interior','Interior'),('boundary_clipped','Near boundary'),
             ('near_stable','Near stable'),('near_unstable','Near unstable')]
    options = {'font.family':'serif','mathtext.fontset':'stix','font.size':7.5,
               'xtick.direction':'in','ytick.direction':'in','axes.linewidth':.5,
               'svg.fonttype':'path','pdf.fonttype':42,'svg.hashsalt':'kvn-si85-coupled'}
    with plt.rc_context(options):
        width_mm, height_mm = 180, 53
        fig = plt.figure(figsize=(width_mm/25.4,height_mm/25.4))
        # Match the four initial-condition column centres in the density figure above.
        centres = [.1924, .4255, .6587, .8921]
        panel_width = .188
        panel_height = panel_width*width_mm/height_mm
        axes = [fig.add_axes([centre-panel_width/2, .195, panel_width, panel_height])
                for centre in centres]
        # Equation S18, evaluated inside the common square domain.
        x,y = np.meshgrid(np.linspace(.14,.86,5),np.linspace(.14,.86,5))
        dx = x*(1-x)*(1+2*y)/3
        dy = y*(1-y)*(-2+x)/2
        for i,((key,title),ax) in enumerate(zip(cases,axes)):
            rows = data[data.id.eq('coupled_cubic_2d__'+key)].sort_values('t')
            assert len(rows)==65 and np.allclose(rows.t,np.arange(65)/64)
            ax.quiver(x,y,dx,dy,color='#b8c4cb',angles='xy',scale_units='xy',scale=2.0,
                     width=.0032,headwidth=3.2,headlength=3.6,pivot='mid',zorder=1)
            trajectories(ax, rows)
            if i < 2:
                ax.annotate(r'$t=0$', (rows.iloc[0].circuit_mean_x, rows.iloc[0].circuit_mean_y),
                            xytext=(2,3), textcoords='offset points', fontsize=6.0, color='#41484d')
            ax.set(xlim=(-.045,1.045),ylim=(-.045,1.045),aspect='equal',
                   xticks=[0,.5,1],yticks=[0,.5,1])
            ax.set_xticklabels(['0','0.5','1']);ax.set_yticklabels(['0','0.5','1'])
            ax.set_title(title,fontsize=7.6,pad=3.3)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=6.7,length=2.6,pad=1.6)
            if i>=2:
                # Put the magnified equilibrium in the open half of its host diagram.
                zoom = ax.inset_axes([.245,.39,.48,.48] if i==2 else [.43,.18,.48,.48])
                trajectories(zoom,rows)
                limits = [(.78,1.016),(-.016,.22)] if i==2 else [(-.016,.22),(.78,1.016)]
                zoom.set_xlim(limits[0]);zoom.set_ylim(limits[1]);zoom.set_aspect('equal')
                ticks=([.8,1],[0,.2]) if i==2 else ([0,.2],[.8,1])
                zoom.set_xticks(ticks[0],[f'{v:g}' for v in ticks[0]])
                zoom.set_yticks(ticks[1],[f'{v:g}' for v in ticks[1]])
                zoom.tick_params(labelsize=5.65,length=1.6,pad=.8)
                for spine in zoom.spines.values():
                    spine.set_color('#8796a0');spine.set_linewidth(.45)
        fig.text(.542,.04,r'$x$',ha='center',fontsize=7.6)
        fig.text(.041,.515,r'$y$',va='center',ha='center',fontsize=7.6)
        handles=[Line2D([],[],color='#222',ls='--',lw=.7,label='Reference'),
                 Line2D([],[],ls='none',color=BLUE,marker='s',ms=2.5,label='Circuit'),
                 Line2D([],[],ls='none',mec='#222',mfc='#222',marker='o',ms=2.8,label='Stable'),
                 Line2D([],[],ls='none',mec='#222',mfc='white',marker='o',ms=2.8,label='Unstable')]
        for ax, key in zip(axes[:2], (handles[:2],handles[2:])):
            ax.legend(handles=key,loc='upper right',frameon=True,facecolor='white',framealpha=1,
                      edgecolor='none',fontsize=6.05,handlelength=1.2,handletextpad=.33,
                      labelspacing=.15,borderpad=.16,borderaxespad=.1)
        out.mkdir(parents=True,exist_ok=True)
        for ext,metadata in [('pdf',{'CreationDate':None}),('svg',{'Date':None}),('png',None)]:
            fig.savefig(out/f'Figure_S5.{ext}',metadata=metadata,dpi=400,facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path,nargs='?',default=ROOT/'reproduced')
    draw(parser.parse_args().output)
