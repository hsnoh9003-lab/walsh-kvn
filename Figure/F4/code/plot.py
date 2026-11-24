"""Render Figure 4a–f from the retained field arrays and source CSVs.

No model fitting, numerical propagation, or digitization is performed here.
Panel b retains the model-validation frontier. Cross-case numerical results
remain in Supplementary Note 1 and their source CSVs, without a separate panel.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[3]))
from Source.data_locations import published_path, shared_data, figure_data, source_label
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import NullLocator, NullFormatter

import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--package-root',type=Path,default=Path.cwd())
_args=_parser.parse_args()
ROOT=_args.package_root.resolve()
HERE=ROOT/'Figure/F4'
PLOT=figure_data('F4')
OUTPUT=HERE/'reproduced'
BLUE='#2166ac'; RED='#b4655b'; DARK='#45535f'; ORANGE='#c64f31'

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def col(rows,key):return np.array([float(r[key]) for r in rows])

def panel(fig,label,x,y):
    fig.text(x,y,label,fontsize=7,fontweight='bold',ha='left',va='top')

def draw_degree(ax):
    source=shared_data('CFD_ROM', 'SINDy_Degree')
    candidates=read(source/'cost_accuracy_candidates.csv')
    colors={1:'#757575',2:BLUE,3:ORANGE,4:'#8c6bb1',5:'#2c8c73'}
    markers={1:'s',2:'o',3:'^',4:'D',5:'P'}
    positive=[]
    for degree in range(1,6):
        pts=[r for r in candidates if int(r['library_degree'])==degree and
             float(r['rotation_support_bound'])>0 and math.isfinite(float(r['validation_error_percent']))]
        positive.extend(pts)
        ax.scatter(col(pts,'rotation_support_bound'),col(pts,'validation_error_percent'),
            s=14,marker=markers[degree],edgecolor=colors[degree],facecolor='none',
            linewidth=.55,alpha=.8,label=str(degree),zorder=3)
        frontier_points=[r for r in pts if r['pareto_nondominated']=='True']
        if frontier_points:
            ax.scatter(col(frontier_points,'rotation_support_bound'),
                col(frontier_points,'validation_error_percent'),
                s=24,marker=markers[degree],color=colors[degree],zorder=4)
    frontier={int(r['rotation_support_bound']):float(r['validation_error_percent'])
              for r in positive if r['pareto_nondominated']=='True'}
    xs=sorted(frontier)
    ax.plot(xs,[frontier[x] for x in xs],color='#494949',lw=.65,zorder=1)
    ax.text(.10,.90,r'Grid-size scaling (fixed $K$)'+'\n'+
        r'$O((\log N_x)^K)$'+'\n'+
        r'$N_x$: cells per coordinate; $K$: drift degree',
        transform=ax.transAxes,ha='left',va='top',fontsize=5,linespacing=1.25)
    adopted=next(r for r in positive if int(r['library_degree'])==2 and float(r['threshold'])==.03)
    adopted_xy=(float(adopted['rotation_support_bound']),float(adopted['validation_error_percent']))
    ax.scatter(*adopted_xy,s=42,facecolors='none',edgecolors='#202020',linewidths=.7,zorder=5)
    ax.annotate('Adopted quadratic',adopted_xy,xytext=(0,14),textcoords='offset points',
        ha='center',fontsize=5,arrowprops={'arrowstyle':'-','color':'#404040',
        'linewidth':.6,'shrinkA':2,'shrinkB':4})
    ax.set(xscale='log',yscale='log',xlabel='Walsh rotations (upper bound)',ylabel='Validation error (%)')
    ax.legend(frameon=False,ncol=5,fontsize=5,title='Library degree',title_fontsize=5,
        handletextpad=.1,columnspacing=.3,loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.labelpad=1;ax.yaxis.labelpad=1
    ax.tick_params(length=2,pad=1.5)
    return {'validation_candidates':len(candidates),'positive_support_candidates':len(positive),
        'adopted_model':{'library_degree':2,'threshold':.03,'rotation_support_bound':226,
            'validation_error_percent':float(adopted['validation_error_percent'])},
        'cross_case_results':'Retained in Supplementary Note 1 and Source Data; former g panel omitted'}

def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],
        'mathtext.fontset':'stixsans','font.size':5.5,'axes.labelsize':5.5,'axes.titlesize':5.5,
        'axes.linewidth':.6,'xtick.direction':'in','ytick.direction':'in',
        'xtick.labelsize':5,'ytick.labelsize':5,'xtick.major.width':.6,'ytick.major.width':.6,
        'figure.dpi':150,'savefig.dpi':300,'svg.fonttype':'none','pdf.fonttype':42,
        'svg.hashsalt':'KvN-single-column-F4-v38'})
    fig=plt.figure(figsize=(88/25.4,170/25.4))
    def mm(x,y,w,h,**kwargs):return fig.add_axes([x/88,y/170,w/88,h/170],**kwargs)
    panel_positions={}
    def label(letter,x,y):
        assert letter not in panel_positions
        panel_positions[letter]=(x,y)
    data=shared_data('CFD_ROM', 'CFD_ROM', 'Re100', 'data')
    fields=np.load(data/'reconstructed_fields.npz')
    direct_index=list(fields['methods']).index('direct')
    extent=[-1,7,-2,2]
    boxes=[(1.756,-.8378,1.8,2.2),(3.7144,-1.2348,1.8,2.2)]
    for row,y in enumerate([146.0,126.0,106.0]):
        for column,x in enumerate([9.0,51.0]):
            ax=mm(x,y,35.0,17.5)
            values=(fields['cfd_fields'][row,1] if column==0 else
                    fields['fields'][direct_index,row,1])
            values=np.ma.array(values,mask=~fields['fluid_mask'])
            im=ax.imshow(values,origin='lower',extent=extent,interpolation='bilinear',
                cmap='RdBu_r',vmin=-.75,vmax=.75,aspect='equal')
            ax.add_patch(Circle((0,0),.5,facecolor='#b7b7b7',edgecolor='#606060',lw=.55))
            if row==0:
                for xx,yy,ww,hh in boxes:
                    ax.add_patch(Rectangle((xx,yy),ww,hh,fill=False,edgecolor=DARK,
                        lw=.6,linestyle=(0,(3,2)),alpha=.85))
            ax.set_xticks([0,3,6]);ax.set_yticks([-2,0,2]);ax.tick_params(length=2,pad=1.5)
            if row<2:ax.set_xticklabels([])
            else:ax.set_xlabel(r'$x/D$',labelpad=1)
            if column>0:ax.set_yticklabels([])
            else:ax.set_ylabel(r'$y/D$',labelpad=1)
            for spine in ax.spines.values():spine.set_color('#9eabb2');spine.set_linewidth(.5)
            if row==0:ax.set_title('FEM reference' if column==0 else 'Walsh–KvN',
                pad=3,fontsize=5.5,fontweight='bold')
        fig.text(2.5/88,(y+8.75)/170,rf'$t^*={fields["tstar"][row]:.2f}$',
            rotation=90,ha='center',va='center',fontsize=5.2)
    cax=mm(28,99.2,39,1.3)
    colorbar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.75,0,.75])
    colorbar.outline.set_visible(False);colorbar.ax.tick_params(length=2,pad=1,labelsize=5)
    fig.text(47.5/88,94.5/170,r'Transverse velocity $v/U_\infty$',
        ha='center',va='center',fontsize=5.2)
    label('a',.1,169.0)

    tr=read(data/'bc_displayed_training_trajectory.csv')
    en=read(data/'ensemble_trajectory.csv')
    ax=mm(9.0,37.0,32.0,22.5,projection='3d')
    rr=np.linspace(.62,1.01,12);tt=np.linspace(0,2*np.pi,50)
    rr,tt=np.meshgrid(rr,tt)
    ax.plot_surface(rr*np.cos(tt),rr*np.sin(tt),rr**2,color='#acbac6',
        alpha=.12,linewidth=0,shade=False,rstride=1,cstride=1)
    ax.plot(col(tr,'CFD_x'),col(tr,'CFD_y'),col(tr,'CFD_z'),color=DARK,lw=.65,label='FEM')
    ax.scatter(col(en,'direct_x'),col(en,'direct_y'),col(en,'direct_z'),
        color=BLUE,s=2.5,depthshade=False,label='Walsh–KvN')
    ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),zlim=(.35,1.05))
    ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.set_zticks([.4,1])
    ax.set_xlabel('');ax.set_ylabel('');ax.set_zlabel('')
    ax.text2D(.35,-.04,r'$x$',transform=ax.transAxes,ha='center',va='top',fontsize=5.5)
    ax.text2D(.99,.08,r'$y$',transform=ax.transAxes,ha='center',va='center',fontsize=5.5)
    ax.text2D(1.05,.55,r'$z$',transform=ax.transAxes,ha='center',va='center',fontsize=5.5)
    ax.tick_params(labelsize=5,pad=-4);ax.view_init(elev=23,azim=-58)
    ax.set_box_aspect((1,1,1.0));ax.grid(False)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
    ax.legend(frameon=False,fontsize=5,loc='upper left',
        ncol=2,handlelength=.65,handletextpad=.2,columnspacing=.3,borderaxespad=.1)
    label('c',.1,61.5)

    ax=mm(54.0,37.5,32.0,22.0)
    ax.set_box_aspect(1)
    ax.plot(col(tr,'CFD_r2'),col(tr,'CFD_z'),color=DARK,lw=.75,label='FEM')
    q=col(en,'direct_mean_r2');z=col(en,'direct_z')
    chosen=np.array([np.argmin(abs(q-t)) for t in np.linspace(q.min(),q.max(),5)])
    ax.scatter(q[chosen],z[chosen],s=9,facecolor='white',edgecolor=BLUE,lw=.8,
        label='Walsh–KvN',zorder=4)
    ax.plot([.5,1.02],[.5,1.02],color=RED,lw=.8,ls='--',zorder=1)
    ax.text(.57,.63,r'$z=r^2$',rotation=35,color=DARK,fontsize=5)
    ax.set(xlim=(.5,1.03),ylim=(.22,1.04),xlabel=r'$r^2$',ylabel=r'$z$')
    ax.set_xticks([.5,1]);ax.set_yticks([.4,.7,1])
    ax.tick_params(pad=2,length=2);ax.xaxis.labelpad=1;ax.yaxis.labelpad=1
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=5,loc='lower left',handlelength=.8,
        handletextpad=.2,labelspacing=.1,borderaxespad=.1)
    label('d',45.0,61.5)

    curves=read(PLOT/'final_panel_e/Figure4e_curves.csv')
    ax=mm(10.0,6.0,32.0,24.0)
    order=[('3-mode vs FEM',DARK,'-'),('Radial vs 3-mode',RED,'-'),('Circuit vs Radial',BLUE,'-')]
    e_marker_indices=[]
    for name,color,style in order:
        rows=[r for r in curves if r['series']==name]
        markers=list(range(len(rows))) if name == 'Circuit vs Radial' else None
        if markers is not None:e_marker_indices=markers
        ax.plot(col(rows,'time_t_star'),col(rows,'relative_field_error_pct'),color=color,lw=.75,
            marker='o' if name == 'Circuit vs Radial' else None,markevery=markers,ms=2,label=name)
    # This is a target for total RMS error, whereas the blue data are noiseless bias only.
    ax.axhline(1,color='#555555',lw=.6,ls=(0,(1.2,1.8)))
    ax.text(40.10,1.35,'1% RMS target',fontsize=5,color='#555555',ha='left',va='bottom')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,['One state: 3-mode vs FEM','One state: Radial vs 3-mode','Ensemble: Circuit vs Radial'],
        frameon=False,fontsize=5,loc='lower right',handlelength=.8,handletextpad=.2,
        borderaxespad=.1,labelspacing=0)
    ax.set(yscale='log',xlim=(39.75,50.07906260206943),ylim=(1e-4,20),
           xlabel=r'$t^*$',ylabel='Error (%)')
    ax.set_xticks([40,49.8]);ax.set_yticks([1e-4,1e-2,1])
    ax.get_xticklabels()[-1].set_ha('right')
    ax.tick_params(length=2,pad=1.5);ax.xaxis.labelpad=1;ax.yaxis.labelpad=1
    ax.spines[['top','right']].set_visible(False)
    label('e',.1,33.0)

    fem=read(PLOT/'force_output/fig4f_fem_forces.csv')
    force=read(PLOT/'force_output/fig4f_walsh_forces.csv')
    sampled_force=read(published_path('Data/Force_Sampling/force_815_shots.csv'))
    force_markers=np.arange(0,len(sampled_force),8)
    for k,(coordinate,y,limits,ticks) in enumerate([
        ('C_D',19,(1.11,1.38),[1.15,1.25,1.35]),('C_L',6,(-.42,.42),[-.3,0,.3])]):
        ax=mm(54,y,32,11)
        ax.axvspan(40,49.82906260206943,color='#dbeaf3',alpha=.9,zorder=0)
        ax.plot(col(fem,'time'),col(fem,coordinate),color=DARK,lw=.65,label='FEM')
        ax.plot(col(force,'time'),col(force,coordinate),color=BLUE,lw=.85,label='Exact mean')
        ax.plot(col(sampled_force,'time')[force_markers],
            col(sampled_force,coordinate+'_815_shot')[force_markers],
            linestyle='None',marker='o',ms=3.0,mfc='white',mec=BLUE,mew=.7,
            color=BLUE,label='815 shots',zorder=4)
        ax.set(xlim=(0,60),ylim=limits,ylabel=r'$C_\mathrm{'+coordinate[-1]+'}$')
        ax.set_yticks(ticks);ax.set_xticks([0,20,40,60]);ax.tick_params(length=2,pad=1.5)
        ax.yaxis.labelpad=1;ax.spines[['top','right']].set_visible(False)
        if k==0:ax.set_xticklabels([])
        else:ax.set_xlabel(r'$t^*$',labelpad=1)
        if k==0:
            force_handles,force_labels=ax.get_legend_handles_labels()
            ax.legend(force_handles,force_labels,frameon=False,fontsize=5,loc='upper left',
                ncol=1,handlelength=.9,handletextpad=.2,labelspacing=.1,borderaxespad=.1)
    label('f',45,33.0)

    stats=draw_degree(mm(10,67,76,24))
    label('b',.1,93.5)
    assert set(panel_positions)==set('abcdef')
    assert ''.join(sorted(panel_positions,key=lambda q:(-panel_positions[q][1],panel_positions[q][0])))=='abcdef'
    for letter in 'abcdef':
        x,y=panel_positions[letter];panel(fig,letter,x/88,y/170)
    out=OUTPUT;out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/'Figure_4.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/'Figure_4.svg',metadata={'Date':None})
    fig.savefig(out/'Figure_4.png',dpi=300)
    plt.close(fig)
    sources=[data/'reconstructed_fields.npz',data/'bc_displayed_training_trajectory.csv',
        data/'ensemble_trajectory.csv',PLOT/'final_panel_e/Figure4e_curves.csv',
        PLOT/'force_output/fig4f_fem_forces.csv',PLOT/'force_output/fig4f_walsh_forces.csv',
        published_path('Data/SINDy_Degree/cost_accuracy_candidates.csv'),
        published_path('Data/Force_Sampling/force_815_shots.csv')]
    stats.update({'panels':'a–f','width_mm':88,'height_mm':170,
        'a_grid_shape':[3,2],'a_rows':'time','a_columns':'method',
        'b_axes_height_mm':24,
        'panel_b_annotations':'Grid-size scaling O((log N_x)^K), fixed coordinate count and degree; N_x and K defined',
        'panel_b_filled_markers':'Pareto-nondominated candidates; all other candidates are open',
        'legends':{'c':'inside upper left','d':'inside lower left',
            'f':'inside upper-left drag axis'},
        'layout_rows':[['a'],['b'],['c','d'],['e','f: drag/lift vertically']],
        'force_axes_orientation':'vertical','force_axes_height_mm':11,
        'force_view':{'x_limits':[0,60],'drag_limits':[1.11,1.38],
            'lift_limits':[-.42,.42],'circuit_interval_shading':[40,49.82906260206943],
            'FEM_style':'solid','circuit_style':'solid'},
        'error_bar_display':{'shown':False,'legend_shown':False,'marker_size_pt':3.0,
            'standard_errors_retained_in_source_CSV':True},
        'panel_d_box_aspect':1.0,
        'panel_d_axes_size_mm':[22,22],
        'panel_mapping_from_previous':{'a':'a','b':'b','c':'c','d':'d','e':'e','f':'f','g':'removed; results retained in SI Note 1'},
        'panel_letter_positions_mm':panel_positions,
        'numerical_values':'Retained arrays/CSV only; no refitting or propagation',
        'panel_d_indices':chosen.tolist(),'panel_e_marker_indices':e_marker_indices,
        'panel_e_lines':'All 34 circuit values and markers displayed',
        'panel_e_legend':['One state: 3-mode vs FEM','One state: Radial vs 3-mode','Ensemble: Circuit vs Radial'],
        'panel_e_model':'Selected 12-coefficient quadratic model; classical curves replayed against retained FEM fields',
        'panel_e_blue':'Noiseless deterministic ensemble field bias only',
        'panel_e_dotted_line':'1% relative RMS target including sampling and implementation errors',
        'colorbar':'Centred below both field columns, with label below the ticks',
        'force_marker_indices':list(range(0,len(force),8)),
        'cost_axis':'Common 64-cell Cartesian interior-layer support bounds; not physical-resource estimates',
        'sources':{source_label(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    (out/'plot_audit.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n')
    print('Rendered Figure 4a–f from retained numerical arrays and CSVs')

if __name__=='__main__':main()
