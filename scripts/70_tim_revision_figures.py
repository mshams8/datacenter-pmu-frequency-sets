from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'reproduced';O=D/'figures';O.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#0042A5','axes.labelcolor':'#0042A5','xtick.color':'#0042A5','ytick.color':'#0042A5'})
blue='#0072B2';teal='#009E73';orange='#D55E00';purple='#7C4D9B';gray='#6B7280'
def save(fig,name):fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.1,2.1));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
def box(x,y,w,h,text,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color+'10',edgecolor=color,linewidth=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.1,color='#0042A5')
def arrow(x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=gray,lw=1))
ax.text(0,2.85,'OFFLINE: FIXED MEASUREMENT CONFIGURATION',fontweight='bold',color=blue)
box(.05,1.9,2.2,.65,'Design events\nconfigured channels');box(2.8,1.9,2.1,.65,'Design model\ncentres, scales,\nfeatures and G');arrow(2.3,2.22,2.75,2.22)
box(5.55,1.9,2.05,.65,'Calibration events\nscores â†’ radius q',teal);box(8.15,1.9,1.65,.65,'Frozen response\nmodel',teal);arrow(4.95,2.22,5.5,2.22);arrow(7.65,2.22,8.1,2.22)
ax.text(0,1.52,'EVENT: PRECEDING DETECTOR SUPPORTS THE CATALOGUE HYPOTHESIS',fontweight='bold',color=blue)
box(.05,.45,1.7,.7,'Reported channels\nestimate fold');box(2.12,.45,1.6,.7,'Compatible\ncatalogue set');arrow(1.8,.8,2.08,.8)
box(4.1,.35,2.3,.9,'Unique fold: singleton\nG = 0: unresolved\nWeak channels:\ninsufficient support');arrow(3.77,.8,4.05,.8)
box(6.78,.45,1.25,.7,'Otherwise\nscore response');arrow(6.45,.8,6.73,.8)
box(8.4,.35,1.4,.9,'1: singleton\n>1: unresolved\n0: outside\nresponse',teal);arrow(8.08,.8,8.35,.8)
ax.text(.05,.02,'A wrong fold can exclude the truth. Response calibration does not detect every off-catalogue event.',fontsize=7.5,color=orange)
save(fig,'method')
d=pd.read_csv(D/'noise_decisions.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,2.25),layout='constrained')
for ax,kind,title in zip(axes[:2],['white','correlated'],['(a) 30 frames/s: white noise','(b) 30 frames/s: correlated noise']):
 g=d[(d.R==30)&(d.kind==kind)]
 for col,label,color,mark in [('fold_correct','Four-channel score',blue,'o'),('family_fold_correct','Angleâ€“frequency score',teal,'s')]:
  v=g.groupby('snr')[col].mean();ax.plot(v.index,100*v.values,marker=mark,color=color,label=label,ms=3)
 ax.set(title=title,xlabel='Oscillatory-component SNR (dB)',ylabel='Correct fold (%)',ylim=(40,103));ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=False)
for kind,color,mark in [('white',blue,'o'),('correlated',teal,'s')]:
 v=d[(d.R==60)&(d.kind==kind)].groupby('snr').correct.mean();axes[2].plot(v.index,100*v.values,marker=mark,color=color,label=kind.capitalize(),ms=3)
axes[2].set(title='(c) 60 frames/s: fixed calibration',xlabel='Oscillatory-component SNR (dB)',ylabel='Correct singleton (%)',ylim=(0,103));axes[2].grid(alpha=.2);axes[2].legend(frameon=False)
save(fig,'noise')
d=pd.read_csv(D/'waveform_summary.csv');h=pd.read_csv(D/'threshold_summary.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,2.3),layout='constrained')
for ax,kind,title in zip(axes[:2],['offset','drift'],['(a) Offset from catalogue','(b) Linear frequency drift']):
 for chain,proc,label,color,mark in [('SDFT-pre','reference','SDFT: reference',blue,'o'),('SDFT-pre','filtered','SDFT: filtered',teal,'s'),('TF-p2','filtered','TF: filtered',orange,'^')]:
  g=d[(d.chain==chain)&(d.processing==proc)&(d.calibration==proc)&(d.kind==kind)].sort_values('level');ax.plot(g.level,100*g.singleton,marker=mark,color=color,label=label,ms=3)
 ax.set(title=title,xlabel='Offset (Hz)' if kind=='offset' else 'Drift rate (Hz/s)',ylabel='Catalogue singleton issued (%)',ylim=(-3,103));ax.grid(alpha=.2)
axes[0].legend(fontsize=6.5,frameon=False,loc='center right')
v=h[h.cycle==2].pivot(index='floor',columns='tau',values='correct').sort_index()
im=axes[2].imshow(v.values*100,aspect='auto',origin='lower',vmin=75,vmax=100,cmap='Blues');axes[2].set_xticks(range(len(v.columns)),[str(x) for x in v.columns]);axes[2].set_yticks(range(len(v.index)),[f'{x:g}' for x in v.index]);axes[2].set(title='(c) Two-cycle filter, 60 frames/s',xlabel=r'Separation threshold $\tau$',ylabel=r'Scale floor $s_{\min}$')
fig.colorbar(im,ax=axes[2],shrink=.8,label='Correct singleton (%)')
save(fig,'stress')
