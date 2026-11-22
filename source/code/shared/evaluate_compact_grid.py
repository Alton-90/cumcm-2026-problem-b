"""Independent paired evaluation; synthetic data only, no network."""
import json,math,random,statistics,hashlib
from pathlib import Path
from dataclasses import asdict
from offline_arena import Source,OfflineArena,make_cases
from simulator_client import SimulatorClient
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/compact_grid'
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cases():
 result=[]
 for q in (3,4):
  for c in make_cases(q):result.append(dict(question=q,split='development',**c))
  for i in range(24):
   seed=(61000 if q==3 else 62000)+i;rng=random.Random(seed);n=(10,13,16)[i%3];sources=[]
   for j,ch in enumerate(rng.sample(range(1,21),n)):
    a=rng.uniform(0,2*math.pi);rho=1800*math.sqrt(rng.random());x,y=rho*math.cos(a),rho*math.sin(a)
    if i//6==1:a=2*math.pi*j/n;x,y=1800*math.cos(a),1800*math.sin(a)
    if i//6==2:
     grid=[(x,y) for x in range(-1400,1401,700) for y in range(-1400,1401,700) if x*x+y*y<=1800**2];x,y=grid[j%len(grid)]
    if i//6==3 and j<3:x,y=[(0,0),(5,0),(20,0)][j]
    count=(0,1,n-1,n,n//2,n)[i%6] if q==4 else 0
    heading=(math.degrees(a) if i//6==1 else rng.uniform(0,360)) if j<count else None
    radius=1000 if i%3==0 else 1500 if i%3==1 else rng.uniform(1000,1500)
    sources.append(Source(ch,float(x),float(y),radius,heading))
   result.append(dict(question=q,case_id=f'new_q{q}_{seed}',split='holdout',seed=seed,error_mode=['fixed_hash','positive_edge','negative_edge'][i%3],sources=sources))
 return result
if __name__=='__main__':
 cc=cases(); data=[dict(**{k:v for k,v in c.items() if k!='sources'},sources=[asdict(s) for s in c['sources']]) for c in cc]
 if (OUT/'cases.json').exists():assert json.loads((OUT/'cases.json').read_text(encoding='utf-8'))==data
 else:save(OUT/'cases.json',data)
 import search_policy,search_policy_recovery,search_policy_compact
 inputs=['code/shared/search_policy.py','code/shared/search_policy_recovery.py','code/shared/search_policy_compact.py','code/shared/offline_arena.py','code/shared/evaluate_compact_grid.py','results/compact_grid/plan.md','results/compact_grid/cases.json']
 hashes=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in inputs];save(OUT/'inputs_before_run.json',hashes)
 rows=[]
 for c in cc:
  pair={}
  for method in ['old','compact']:
   arena=OfflineArena(c['sources'],c['seed'],c['error_mode']);client=SimulatorClient('synthetic',transport=arena,log_path=False);client.enter()
   run=search_policy_compact.run_search if method=='compact' else search_policy.run_search if c['question']==3 else search_policy_recovery.run_search
   state=run(client,question=c['question']);client.exit()
   count=len(arena.cleared);calc=arena.distance_m/5+5*arena.measure_count+arena.switch_count+3*arena.clear_count+2*count
   ok=state.get('completed') and count==len(c['sources']) and set(state['cleared_channels'])==arena.cleared and not(set(state['excluded_channels'])&set(arena.sources)) and abs(calc-arena.virtual_time_s)<1e-6
   pair[method]=dict(all_checks_pass=bool(ok),cleared=count,n=len(c['sources']),time_s=arena.virtual_time_s,mean_s=arena.virtual_time_s/count if count else None,movement_m=arena.distance_m,measures=arena.measure_count,clears=arena.clear_count,fallbacks=state['fallback_count'])
  rows.append(dict(question=c['question'],case_id=c['case_id'],split=c['split'],ratio=pair['compact']['time_s']/pair['old']['time_s'],**pair))
 summaries={}
 for q in (3,4):
  summaries[str(q)]={}
  for split in ['all','holdout']:
   rs=[x for x in rows if x['question']==q and (split=='all' or x['split']==split)]
   old=statistics.mean(x['old']['mean_s'] for x in rs);new=statistics.mean(x['compact']['mean_s'] for x in rs)
   summaries[str(q)][split]=dict(cases=len(rs),old_mean_s=old,compact_mean_s=new,reduction=1-new/old,max_ratio=max(x['ratio'] for x in rs))
 all_ok=all(x[m]['all_checks_pass'] for x in rows for m in ['old','compact'])
 accepted=all_ok and all(summaries[str(q)][sp]['reduction']>=.1 and summaries[str(q)][sp]['max_ratio']<=1.1 for q in (3,4) for sp in ['all','holdout'])
 save(OUT/'comparison.json',dict(scope='SYNTHETIC_ONLY',accepted=accepted,all_checks_pass=all_ok,summaries=summaries,rows=rows,inputs=hashes));print(json.dumps(dict(accepted=accepted,all_checks_pass=all_ok,summaries=summaries),ensure_ascii=False))
