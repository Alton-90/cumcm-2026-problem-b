"""Generate paper tables and figures from saved case records, without simulation."""
from pathlib import Path
import argparse,hashlib,json
from statistics_helpers import summarize
ROOT=Path(__file__).resolve().parents[2]
BATCHES={
 'Q3':[('final_strict84','最早验证'),('final_strict84_80_20260912','第一批'),('final_strict84_100_20260913','第二批'),('final_strict84_continuous_20260913','持续演练')],
 'Q4':[('FREE_tasks_fifteen_20260913','前期演练'),('FREE_final_hundred_20260913','集中演练'),('FREE_continuous_20260913_1400','后续演练')]}

def table(caption,label,columns,heading,rows):
    return ('% Automatically generated from evidence/verified_results.json.\n'
      r'\begin{table}[H]\centering\small'+'\n'+rf'\caption{{{caption}}}\label{{{label}}}'+'\n'
      +rf'\begin{{tabular}}{{{columns}}}\toprule'+'\n'+heading+r'\\\midrule'+'\n'
      +'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')

def build_tables(e):
    s={q:summarize(e['practice'][q]) for q in BATCHES};out={}
    rows=[]
    for q in BATCHES:
        n=s[q]['n'];nf=len([r for r in e['formal'] if r['question']==q])
        sizes=[sum(r['batch']==b for r in e['practice'][q]) for b,_ in BATCHES[q]]
        rows.append([q+'（'+('strict84' if q=='Q3' else 'FREE')+'）','$'+'+'.join(map(str,sizes))+'$',n,nf,n+nf])
    out['run_counts.tex']=table('采用策略的模拟器运行次数及统计范围','tab:counts','llrrr','方法 & 演练批次构成 & 演练合计 & 正式测试 & 实际运行合计',rows)
    rows=[]
    for q in BATCHES:
        for b,name in BATCHES[q]:
            z=summarize([r for r in e['practice'][q] if r['batch']==b]);rows.append([q,name,z['n'],f"{z['cleared']}/{z['sources']}",f"{z['mean']:.2f}"])
    out['practice_batches.tex']=table('演练批次与场次等权均值（秒/源）','tab:batches','llrrr','题号 & 批次 & 场数 & 清除数/总源数 & 均值',rows)
    rows=[[q,z['n'],f"{z['mean']:.2f}",f"{z['sd']:.2f}",f"[{z['ci95'][0]:.2f}, {z['ci95'][1]:.2f}]",f"{z['p95']:.2f}"] for q,z in s.items()]
    out['practice_statistics.tex']=table(r'全部演练的统计结果；正式测试未纳入（秒/源）','tab:stats','lrrrrr',r'题号 & 场数 & 均值 & 标准差 & 均值95\%区间 & P95',rows)
    for q in BATCHES:
        rows=[[r['index'],r['case_id'],r['cleared'],f"{r['value']:.3f}",f"{r['runtime_s']:.3f}"] for r in e['formal'] if r['question']==q]
        out[q.lower()+'_formal.tex']=table('问题'+('三' if q=='Q3' else '四')+'正式测试结果',f'tab:{q.lower()}formal','rlrrr','场次 & 测试案例编码 & 清除数 & 平均定位清除时间 & 程序运行时间',rows)
    macros={}
    for q,prefix in [('Q3','QThree'),('Q4','QFour')]:
        z=s[q]
        for key,suffix in [('mean','Mean'),('sd','Sd'),('median','Median'),('min','Min'),('max','Max'),('pooled','Pooled')]:
            macros[prefix+suffix+'Four']=f"{z[key]:.4f}"
        macros[prefix+'PracticeCount']=str(z['n']);macros[prefix+'SourceCount']=str(z['sources']);macros[prefix+'MeanTwo']=f"{z['mean']:.2f}"
        macros[prefix+'RunCount']=str(z['n']+len([r for r in e['formal'] if r['question']==q]))
    b=summarize([r for r in e['practice']['Q4'] if r['batch']==BATCHES['Q4'][-1][0]])
    macros['QFourNewBatchMeanTwo']=f"{b['mean']:.2f}"
    out['auto_practice_numbers.tex']='% Generated from saved cases; formal cases excluded from practice summaries.\n'+'\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k,v in macros.items())+'\n'
    return out

def plot(e,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'savefig.dpi':220})
    colors={'Q3':'#296493','Q4':'#aa612f'};qnames={'Q3':'问题三：有限费用前瞻（strict84）','Q4':'问题四：冗余扫描删除（FREE）'}
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.25),layout='constrained')
    for ax,q in zip(axes,BATCHES):
        vals=[r['value'] for r in e['practice'][q]];z=summarize(e['practice'][q])
        ax.hist(vals,bins='fd',color=colors[q],alpha=.83,edgecolor='white',linewidth=.7)
        ax.axvline(z['mean'],color='#282828',ls='--',lw=1.3,label=f"均值 {z['mean']:.2f}")
        ax.set(title=qnames[q]+f"\n{z['n']} 场演练",xlabel='每源虚拟耗时（秒/源）',ylabel='场数')
        ax.legend(frameon=False,fontsize=9);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'figures/practice_current.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.25),layout='constrained')
    for ax,q in zip(axes,BATCHES):
        ns=list(range(10,17));zs=[summarize([r for r in e['practice'][q] if r['N']==n]) for n in ns]
        ys=[z['mean'] for z in zs];errs=[z['ci95'][1]-z['mean'] for z in zs]
        ax.errorbar(ns,ys,yerr=errs,fmt='o-',capsize=3,color=colors[q],lw=1.4,markersize=4)
        ax.set(title=qnames[q],xlabel='实际干扰源数量 N',ylabel='每源耗时均值（秒/源）',xticks=ns)
        low=min(y-er for y,er in zip(ys,errs));high=max(y+er for y,er in zip(ys,errs));span=high-low
        ax.set_ylim(low-.18*span,high+.18*span)
        for n,y,er,z in zip(ns,ys,errs,zs):ax.annotate(f"n={z['n']}",(n,y+er),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        ax.set_axisbelow(True);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'figures/practice_by_N.png');plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'paper_snapshot');ap.add_argument('--tables-only',action='store_true');args=ap.parse_args()
    e=json.loads((ROOT/'evidence/verified_results.json').read_text(encoding='utf-8'));args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    for name,content in build_tables(e).items():(args.output/name).write_text(content,encoding='utf-8')
    if not args.tables_only:plot(e,args.output)
    print(json.dumps({'status':'GENERATED','practice_cases':{q:len(e['practice'][q]) for q in BATCHES},'formal_cases':len(e['formal']),'tables_and_macros':len(build_tables(e)),'figures':0 if args.tables_only else 2},ensure_ascii=False))
if __name__=='__main__':main()
