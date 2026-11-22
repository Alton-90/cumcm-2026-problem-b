"""Independent 60-seed Q3 holdout: joint route versus event-action v2."""
import hashlib,json,math,random,statistics
from dataclasses import asdict
from pathlib import Path
from offline_arena import OfflineArena,Source
import search_policy_joint_route_q3 as base
import search_policy_q3_event_action as cand
from evaluate_q3_measure_replan_development import run
R=Path(__file__).resolve().parents[2];O=R/'results'/'q3_event_action'/'final_holdout_v2';DEV=R/'results'/'q3_event_action'/'development_v2'/'comparison.json'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cases():
 out=[]
 for i,seed in enumerate(range(99000,99060)):
  rng=random.Random(seed);n=(10,13,16)[i%3];src=[]
  for j,ch in enumerate(rng.sample(range(1,21),n)):
   a=rng.uniform(0,2*math.pi);layout=i%6
   if layout==0:d=1800;a=2*math.pi*j/n
   elif layout==1:d=1000
   elif layout==2:d=1800*math.sqrt(rng.random())
   elif layout==3:d=(0,5,20,999.999,1000,1800)[j%6];a=math.radians(30+60*(j%6))
   elif layout==4:d=1800;a=math.radians(30+60*(j%6))
   else:d=1800*math.sqrt(rng.random())
   src.append(Source(ch,d*math.cos(a),d*math.sin(a),(1000,1250,1500)[i%3],None))
  out.append({'question':3,'case_id':f'event_action_q3_{seed}','split':'final_independent_holdout','seed':seed,'error_mode':('fixed_hash','positive_edge','negative_edge')[i%3],'sources':src})
 return out
def summ(rows,n):return {k:statistics.mean(x[n][k] for x in rows) for k in ('mean_s','movement_s','measure_s','switch_s','clears_s')}
def main():
 if json.loads(DEV.read_text(encoding='utf-8')).get('accepted') is not True:raise ValueError('development not accepted')
 O.mkdir(parents=True,exist_ok=True);cs=cases();payload=[{**{k:v for k,v in c.items() if k!='sources'},'sources':[asdict(x) for x in c['sources']]} for c in cs];cp=O/'cases.json'
 if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=payload:raise ValueError('case mismatch')
 if not cp.exists():save(cp,payload)
 files=['code/shared/search_policy.py','code/shared/search_policy_compact.py','code/shared/search_policy_feasible_q3.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q3_measure_replan.py','code/shared/search_policy_q3_event_action.py','code/shared/evaluate_q3_event_action_final.py','code/shared/offline_arena.py','results/q3_event_action/development_v2/comparison.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];ip=O/'inputs_before_run.json'
 if ip.exists() and json.loads(ip.read_text(encoding='utf-8'))!=inputs:raise ValueError('input mismatch')
 if not ip.exists():save(ip,inputs)
 rows=[]
 for c in cs:rows.append({'case_id':c['case_id'],'seed':c['seed'],'joint_route':run(base.run_search,c),'event_action':run(cand.run_search,c)})
 sums={n:summ(rows,n) for n in ('joint_route','event_action')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['event_action']['mean_s']/sums['joint_route']['mean_s'];out={'scope':'SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT','case_count':60,'seed_range':[99000,99059],'all_checks_pass':checks,'accepted':bool(checks and imp>0),'improvement_vs_joint_route':imp,'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
