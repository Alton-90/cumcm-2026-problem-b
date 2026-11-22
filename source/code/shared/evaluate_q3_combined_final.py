"""Frozen paired holdout for the Q3 combined candidate; no simulator access."""
from pathlib import Path
import math,random,json,hashlib,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/shared'))
from offline_arena import Source,OfflineArena
from simulator_client import SimulatorClient
import search_policy_joint_route_q3 as baseline
import search_policy_q3_combined as candidate
OUT=ROOT/'results/q3_combined/final_holdout'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'comparison.json').exists():raise RuntimeError('Preserve frozen result.')
 cases=[]
 for i in range(60):
  seed=104000+i;g=random.Random(seed);n=g.randint(10,16);sources=[]
  cx,cy=g.uniform(-700,700),g.uniform(-700,700)
  for j,ch in enumerate(g.sample(range(1,21),n)):
   a=g.uniform(0,2*math.pi);r=1800*math.sqrt(g.random())
   if i%5==2:r=g.uniform(1650,1800)
   x,y=r*math.cos(a),r*math.sin(a)
   if i%5==3:
    x,y=cx+g.gauss(0,320),cy+g.gauss(0,320)
    norm=math.hypot(x,y)
    if norm>1800:x,y=x*1799/norm,y*1799/norm
   if i%5==4 and j<3:x,y=g.uniform(-8,8),g.uniform(-8,8)
   radius=1000 if i%7==0 else 1500 if i%7==1 else g.uniform(1000,1500)
   sources.append(dict(channel=ch,x=x,y=y,radius=radius,heading=None))
  cases.append(dict(seed=seed,case_id=f'combined_q3_{seed}',question=3,sources=sources,error_mode=('fixed_hash','positive_edge','negative_edge')[i%3]))
 cp=OUT/'cases.json';cp.write_text(json.dumps(cases,indent=2),encoding='utf-8')
 names=['search_policy.py','search_policy_compact.py','search_policy_feasible_q3.py','search_policy_joint_route_q3.py','search_policy_q3_combined.py','route_multistart.py','offline_arena.py','simulator_client.py']
 files=[ROOT/'code/shared'/name for name in names]+[cp,Path(__file__)]
 inputs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
 (OUT/'inputs_before_run.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
 rows=[];start=time.monotonic()
 for case in cases:
  row=dict(case_id=case['case_id'],seed=case['seed'])
  for name,policy in [('joint_route',baseline),('combined',candidate)]:
   arena=OfflineArena([Source(**s) for s in case['sources']],case['seed'],case['error_mode'])
   cli=SimulatorClient('synthetic',transport=arena,log_path=False);cli.enter();s=policy.run_search(cli,question=3,method='main');cli.exit()
   n=len(case['sources']);groups=[set(s[k]) for k in ('cleared_channels','excluded_channels','unresolved_channels')]
   calc=arena.distance_m/5+5*arena.measure_count+arena.switch_count+3*arena.clear_count+2*len(arena.cleared)
   passed=len(arena.cleared)==n and s['completed'] and not groups[2] and set.union(*groups)==set(range(1,21)) and sum(map(len,groups))==20 and abs(calc-arena.virtual_time_s)<1e-6
   row[name]=dict(all_checks_pass=passed,cleared=len(arena.cleared),true_sources=n,time_s=arena.virtual_time_s,mean_s=arena.virtual_time_s/n,movement_s=arena.distance_m/5,measure_s=5*arena.measure_count,switch_s=arena.switch_count,clear_s=3*arena.clear_count+2*len(arena.cleared))
  rows.append(row)
 summaries={name:dict(cases=len(rows),mean_per_source_s=statistics.mean(row[name]['mean_s'] for row in rows),all_checks_pass=all(row[name]['all_checks_pass'] for row in rows)) for name in ('joint_route','combined')}
 allpass=all(v['all_checks_pass'] for v in summaries.values())
 improvement=1-summaries['combined']['mean_per_source_s']/summaries['joint_route']['mean_per_source_s']
 result=dict(scope='SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT',case_count=60,seed_range=[104000,104059],accepted=allpass and improvement>0,all_checks_pass=allpass,improvement=improvement,summaries=summaries,inputs=inputs,rows=rows,elapsed_s=time.monotonic()-start)
 (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','rows')}))
if __name__=='__main__':main()
