#!/usr/bin/env python3
"""Build the 88 × 180 mm Fig. 3 with the matched generator and published LCHS reference."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,ConnectionPatch
from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator,NullFormatter,NullLocator
ROOT=Path(__file__).resolve().parents[1];MM=1/25.4
COLORS={'direct':'#275D88','nj_lchs':'#83699C','generator':'#A95745'}
MARKERS={'direct':'s','nj_lchs':'o','generator':'^'}
LABELS={'direct':'Walsh–KvN','nj_lchs':'LCHS–KvN','generator':'BE–KvN'}
ORDER=['direct','nj_lchs','generator'];BLUE='#E8EFF5';INK='#2C3135'
SELECTIONS = ('minimum_qubits', 'fastest_uncapped')
PANEL_B_QUBIT_WINDOW = (1e4, 1e6)
YEAR_SECONDS = 365.25 * 86400
TASK_TIME_WINDOWS = (
    (0, 24, (2, 20), 'h', 3600, 7.8, 'linear'),
    # End the day band below the BE–KvN minimum-qubit task time (10.6 yr).
    (1, 370, (100, 300), 'd', 86400, 7.35, 'linear'),
    # Log years separate the BE minimum (10.6 yr) and both LCHS designs (35.5 and 427.5 yr).
    (7, 650, (10, 100, 500), 'yr', YEAR_SECONDS, 11.95, 'log'),
)
SHOT_TIME_WINDOWS = (
    (0, 100, (10, 90), 's', 1, 6.2, 'linear'),
    (1, 105, (10, 90), 'h', 3600, 6.0, 'linear'),
    # Log days separate the BE minimum (4.8 d) and both LCHS designs (15.9 and 191.6 d).
    (2, 260, (10, 100), 'd', 86400, 9.2, 'log'),
)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def axis(fig,x,y,w,h):return fig.add_axes([x/88,y/180,w/88,h/180])
def style():
 for font_path in sorted((ROOT/'fonts').glob('*.ttf')):
  font_manager.fontManager.addfont(str(font_path))
 plt.rcParams.update({'font.family':'Liberation Sans','font.sans-serif':['DejaVu Sans','Arial','Liberation Sans'],
  'mathtext.fontset':'stix','font.size':6.2,'axes.labelsize':6.3,'xtick.labelsize':5.5,'ytick.labelsize':5.5,
  'axes.linewidth':.6,'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
  'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
  'svg.hashsalt':'figure5_single_column_20261105','figure.dpi':150,'savefig.dpi':300})
def trim(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(which='major',length=2,width=.6,pad=2);ax.tick_params(which='minor',length=1,width=.4,pad=2)
def callout(ax,text,point,label,ha='left',relpos=(.5,.5)):
 return ax.annotate(text,xy=point,xytext=label,ha=ha,va='center',fontsize=5.25,color=INK,
  bbox=dict(facecolor='white',edgecolor='none',pad=.15),
  arrowprops=dict(arrowstyle='-',lw=.5,color=INK,shrinkA=1.5,shrinkB=1.5,relpos=relpos),zorder=10)
def panel_a(fig,data,refs):
 ax=axis(fig,13,132,69,39);ax.patch.set_alpha(0);ax.set_zorder(1)
 ax.set(xscale='log',yscale='log',xlim=(3,1e8),ylim=(6e-5,.04))
 ax.set_xlabel('Physical qubits',labelpad=2);ax.xaxis.set_label_coords(.42,-.115);ax.set_ylabel('Physical error probability',labelpad=2.5)
 ax.set_xticks([1e1,1e2,1e4,1e6,1e8]);ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,5]));ax.xaxis.set_minor_formatter(NullFormatter())
 ax.yaxis.set_major_locator(FixedLocator([1e-4,1e-3,1e-2]));ax.yaxis.set_minor_locator(LogLocator(base=10,subs=[2,5]));ax.yaxis.set_minor_formatter(NullFormatter());trim(ax)
 ax.add_patch(Rectangle((1e4,1e-4),1e6-1e4,9e-4,facecolor=BLUE,edgecolor=COLORS['direct'],linewidth=.65,zorder=1))
 ax.text(1e5,3.2e-4,'EFTQC window\n(this study)',ha='center',va='center',fontsize=5.4,weight='bold',linespacing=1.05,zorder=5)
 ax.axhline(.01,color='#B4BEC5',ls=(0,(1.5,2.3)),lw=.55,zorder=0)
 ax.text(8e7,.012,'Model threshold',ha='right',fontsize=5.15,zorder=5)
 ax.axvline(1e6,color='#B4BEC5',ls=(0,(2.5,2.5)),lw=.55,zorder=0)
 ax.text(1.4e6,.027,'Study boundary',fontsize=5.15,zorder=5)
 labels={'zhang2026':('Zhang [9]',(35,.027)),'tiwari2025':('Tiwari [10]',(48,.0066)),'wang2026':('Wang [7]',(4,.00135))}
 for r in data.itertuples():
  er=np.array([[r.error_center-r.error_low],[r.error_high-r.error_center]])if r.display=='point_with_range'else None
  ax.errorbar(r.used_qubits,r.error_center,yerr=er,fmt='*',ms=4.6,color='black',mew=.45,elinewidth=.5,capsize=1.4,zorder=7)
  text,pos=labels[r.id];callout(ax,text,(r.used_qubits,r.error_center),pos)
 ax.plot(10,.0033,'*',ms=4.6,color='black',mew=.45,zorder=7);callout(ax,'Meng [8]',(10,.0033),(27,.00265))
 labels={'R1a':('Minimal QEC',(5,1.4e-4)),'R1b':('Minimal QEC',(65,.00185)),
  'R2':('STAR: partial QEC',(1.8e4,7.7e-5)),'R3':('RSA-2048',(1.6e6,7.6e-4))}
 for r in refs.itertuples():
  ax.plot(r.physical_qubits,r.physical_error_rate,marker='D',ms=3.7,mfc='white',mec='black',mew=.7,ls='none',zorder=7)
  text,pos=labels[r.source_id]
  reference = r.label.splitlines()[0].rsplit(' ', 1)[-1]
  callout(ax,f'{text} {reference}',(r.physical_qubits,r.physical_error_rate),pos,relpos=(0,.5)if r.source_id=='R2'else(.5,.5))
 for t in ax.get_xticklabels():t.set_bbox(dict(facecolor='white',edgecolor='none',pad=.1))
 return ax
def panel_b(fig,data):
 ax=axis(fig,13,87,31.5,30);ax.set_facecolor('white');ax.axvspan(1e4,1e6,color=BLUE,zorder=0);ax.set_zorder(1)
 ax.set(xscale='log',xlim=PANEL_B_QUBIT_WINDOW,ylim=(1e-4,1e-3));ax.set_xlabel('Physical qubits',labelpad=2);ax.set_ylabel('Physical error probability',labelpad=3)
 ax.xaxis.set_major_locator(FixedLocator([1e4,1e5,1e6,1e7]));ax.xaxis.set_minor_locator(NullLocator())
 ax.yaxis.set_major_locator(FixedLocator([1e-4,5e-4,1e-3]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1e-4:g}'))
 ax.text(0,1.018,r'$\times10^{-4}$',transform=ax.transAxes,fontsize=5.5);ax.axhline(5e-4,color='#A1BACB',lw=.65,ls=(0,(3,3)),zorder=1)
 count=0
 for model in ORDER:
  for filled, selection in zip((False, True), SELECTIONS):
   sub=data[(data.model==model)&(data.selection==selection)].sort_values('p_phys');assert len(sub)==8
   visible=sub.physical_qubits.between(*PANEL_B_QUBIT_WINDOW)
   marker_size=3.1 if model=='nj_lchs' else 3.6
   if model=='generator' and filled:marker_size=5.2
   # Draw the complete curves; the axis clips coordinates outside the chosen range.
   ax.plot(sub.physical_qubits,sub.p_phys,color=COLORS[model],lw=.85,ls='--' if model=='nj_lchs' else '-',marker=MARKERS[model],ms=marker_size,
    mfc=COLORS[model] if filled else 'white',mec='white' if filled else COLORS[model],mew=.55,zorder=(4 if filled else 3)+(.2 if model=='nj_lchs'else 0),clip_on=True);count+=int(visible.sum())
 trim(ax)
 return ax,count
def time_label(seconds, task):
    if task and seconds >= 4 * YEAR_SECONDS:
        return f'{seconds / YEAR_SECONDS:,.1f} yr'
    if seconds >= 48 * 3600:
        return f'{seconds / 86400:.1f} d'
    if task or seconds >= 3600:
        return f'{seconds / 3600:.2f} h'
    if seconds >= 60:
        return f'{seconds / 60:.1f} min'
    return f'{seconds:.2f} s' if seconds < 10 else f'{seconds:.1f} s'


def stacked_time_axes(fig, x, bottom, width, windows, numeric_ticks):
    gap = 1.45
    axes = []
    y = bottom
    for idx, (lo, hi, ticks, unit, scale, band_height, yscale) in enumerate(windows):
        ax = axis(fig, x, y, width, band_height)
        ax.patch.set_visible(False)  # Time bands share the white figure background.
        ax.set(xlim=(0, 1), ylim=(lo * scale, hi * scale), yscale=yscale, xticks=[])
        labels = [f'{t:g} {unit}' for t in ticks] if numeric_ticks else [''] * len(ticks)
        ax.set_yticks(np.asarray(ticks) * scale, labels=labels)
        if yscale == 'log':
            ax.yaxis.set_minor_locator(NullLocator())
        trim(ax)
        ax.spines['bottom'].set_visible(not numeric_ticks and idx == 0)
        axes.append(ax)
        if idx < len(windows) - 1:
            break_bottom = y + band_height
            for mark_y in (break_bottom + .31, break_bottom + gap - .31):
                fig.add_artist(Line2D([(x - .57) / 88, (x + .57) / 88],
                                      [(mark_y - .23) / 180, (mark_y + .23) / 180],
                                      transform=fig.transFigure, color=INK, lw=.6))
        y += band_height + gap
    return axes


def stacked_time_pair(axes, method, times, marker_x, marker_size, font_size, task):
    color = COLORS[method]
    target_axes = [next(ax for ax in axes if ax.get_ylim()[0] < t < ax.get_ylim()[1]) for t in times]
    positions = times
    if times[0] == times[1]:
        # Both callouts branch from one true ordinate when the times tie.
        ax = target_axes[0]
        lo, hi = ax.get_ylim()
        value = times[0]
        positions = value + np.array([.17, -.17]) * (hi - lo)
        branch_x = marker_x - .10
        ax.plot([0, branch_x], [value, value], color=color, lw=.65, alpha=.72)
        ax.plot([marker_x, branch_x, branch_x, marker_x],
                [positions[0], positions[0], positions[1], positions[1]],
                color=color, lw=.65, alpha=.72)
    elif target_axes[0] is target_axes[1]:
        target_axes[0].plot([marker_x, marker_x], times, color=color, lw=.8, alpha=.65)
    else:
        # Draw a single connector behind both markers, including the gap between bands.
        first, second = target_axes
        first.figure.add_artist(ConnectionPatch(
            (marker_x, times[0]), (marker_x, times[1]), first.transData, second.transData,
            axesA=first, axesB=second, shrinkA=0, shrinkB=0, color=color,
            lw=.8, alpha=.65, zorder=-1, clip_on=False))
    for filled, value, y, ax in zip((False, True), times, positions, target_axes):
        ax.plot(marker_x, y, marker=MARKERS[method], ms=marker_size, ls='none',
                mfc=color if filled else 'white', mec=color, mew=.65, zorder=4, clip_on=False)
        label_dy = 2.5 if not task and filled and method in ('direct','nj_lchs') else 0
        ax.annotate(time_label(value, task), (marker_x, y), xytext=(4.1, label_dy),
                    textcoords='offset points', ha='left', va='center', fontsize=font_size)


def panel_c(fig, data):
    task_axes = stacked_time_axes(fig, 56.55, 87, 13.15, TASK_TIME_WINDOWS, numeric_ticks=True)
    shot_axes = stacked_time_axes(fig, 71.7, 90.7, 10.6, SHOT_TIME_WINDOWS, numeric_ticks=False)
    fig.text(48.05 / 88, 102.0 / 180, 'Total task time', rotation=90,
             ha='center', va='center', fontsize=6.3)
    fig.text(69 / 88, 118.15 / 180, r'$p_{\mathrm{phys}}=5\times 10^{-4}$',
             ha='center', va='bottom', fontsize=5.5)
    fig.text(77.0 / 88, 88.45 / 180, 'Per accepted\nshot',
             ha='center', va='center', fontsize=5.1)
    count = 0
    for method in ORDER:
        rows = data[data.model.eq(method)].set_index('selection')
        assert set(rows.index) == set(SELECTIONS) and len(rows) == 2
        times = rows.loc[list(SELECTIONS)]
        stacked_time_pair(task_axes, method, times.complete_task_seconds.to_numpy(),
                          .23, 3.8, 5.2, task=True)
        stacked_time_pair(shot_axes, method, times.seconds_per_accepted_shot.to_numpy(),
                          .17, 3.35, 5.05, task=False)
        count += len(times)
    return task_axes[-1], count
