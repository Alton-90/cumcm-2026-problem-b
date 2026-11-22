"""Evidence-only plots; a dot represents one completed GUI practice case."""
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q3_practice_100_20'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
audit=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
assert audit['all_checks_pass'], 'Only checked public evidence is plotted'
rows=audit['rows']
selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8')) if (OUT/'selection.json').exists() else None
groups=[('G0','原冻结'),('G1','默认前瞻'),('G2','低门槛'),('G3','高门槛'),('G4','短移动')]
validation=[r for r in rows if r['stage']=='validation']
fig,axes=plt.subplots(1,2,figsize=(12.5,4.9),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
colors=['#7A8594','#367CAD','#DA8E3F','#3F9A8B','#AB6FAD']
for i,((ident,label),color) in enumerate(zip(groups,colors)):
    part=[r for r in rows if r['stage']=='tuning' and r['profile']==ident]
    values=[r['s_per_source'] for r in part]
    if not values: continue
    jitter=np.linspace(-.17,.17,len(values))
    axes[0].scatter(i+jitter,values,s=25,color=color,alpha=.6,edgecolors='none')
    mean=statistics.mean(values)
    axes[0].plot([i-.24,i+.24],[mean,mean],color=color,lw=3)
    axes[0].text(i,max(values)+9,f'{mean:.2f}\nn={len(values)}',ha='center',fontsize=9,color=color)
axes[0].set_xticks(range(5),[f'{x}\n{y}' for x,y in groups])
axes[0].set_title(f'调参：{sum(r["stage"]=="tuning" for r in rows)}/100 场',loc='left',fontweight='bold')
axes[0].set_ylabel('每场虚拟耗时 ÷ 真实源数（秒/源）')
axes[0].axhline(200,color='#BC4444',lw=1,ls='--',label='目标 200 秒/源')
axes[0].legend(loc='lower left',frameon=False,fontsize=9)
axes[0].grid(axis='y',alpha=.15)
if validation:
    values=[r['s_per_source'] for r in validation]
    cumulative=[statistics.mean(values[:i]) for i in range(1,len(values)+1)]
    axes[1].scatter(range(1,len(values)+1),values,s=24,color='#A4B8CB',label='单场')
    axes[1].plot(range(1,len(values)+1),cumulative,color='#215F91',lw=2,label='累计平均')
    axes[1].axhline(200,color='#BC4444',ls='--',lw=1)
    axes[1].annotate(f'平均 {cumulative[-1]:.2f}',(len(values),cumulative[-1]),xytext=(-8,14),textcoords='offset points',ha='right',color='#215F91')
    axes[1].set_title(f'固定 {selection["profile"]["id"]} 后的新案例：{len(values)}/20 场',loc='left',fontweight='bold')
    axes[1].set_xlim(.5,20.5)
    axes[1].set_xticks([1,5,10,15,20])
    axes[1].set_xlabel('验证场次（按发生顺序）')
    axes[1].legend(frameon=False,fontsize=9)
    axes[1].grid(axis='y',alpha=.15)
else:
    axes[1].axis('off')
    axes[1].text(.05,.7,'先完成 100 次调参\n再固定参数运行 20 个新案例',transform=axes[1].transAxes,fontsize=15,color='#215F91',linespacing=1.6)
fig.suptitle('第三问练习模拟器 · 每五次记录，保留全部案例',x=.01,ha='left',fontweight='bold',fontsize=15)
fig.savefig(OUT/'comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'comparison.pdf',bbox_inches='tight')
print(str(OUT/'comparison.png'))
