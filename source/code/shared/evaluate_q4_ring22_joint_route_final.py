"""Independent 60-seed Q4 holdout: ring25 joint v3 versus ring22 joint v2."""
from __future__ import annotations
import hashlib,json,math,random,statistics
from dataclasses import asdict
from pathlib import Path
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
import search_policy_q4_ring25_joint_route as base
import search_policy_q4_ring22_joint_route as cand
R=Path(__file__).resolve().parents[2];O=R/'results'/'q4_ring22_joint_route'/'final_holdout_v2';DEV=R/'results'/'q4_ring22_joint_route'/'development_v2'/'comparison.json';POL={'ring25_joint':base.run_search,'ring22_joint':cand.run_search}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cases():
 out=[]
 for i,seed in enumerate(range(98000,98060)):
  rng=random.Random(seed);n=(10,13,16)[i%3];src=[]
  for j,ch in enumerate(rng.sample(range(1,21),n)):
   a=rng.uniform(0,2*math.pi);d=1800*math.sqrt(rng.random());x,y=d*math.cos(a),d*math.sin(a);layout=i%6
   if layout==1:a=2*math.pi*j/n;x,y=1800*math.cos(a),1800*math.sin(a)
   elif layout==2:
    grid=[(gx,gy) for gx in range(-1400,1401,700) for gy in range(-1400,1401,700) if gx*gx+gy*gy<=1800**2];x,y=grid[j%len(grid)]
   elif layout==3 and j<3:x,y=((0,0),(5,0),(20,0))[j]
   elif layout==4:a=math.radians(30+60*(j%6));x,y=1800*math.cos(a),1800*math.sin(a)
   dc=(0,1,n-1,n,n//2,n)[i%6];head=(math.degrees(a) if layout in (1,4) else rng.uniform(0,360)) if j<dc else None;rec=1000 if i%3==0 else 1500 if i%3==1 else rng.uniform(1000,1500);src.append(Source(ch,float(x),float(y),rec,head))
  out.append({'question':4,'case_id':f'ring22_joint_q4_{seed}','split':'final_independent_holdout','seed':seed,'error_mode':('fixed_hash','positive_edge','negative_edge')[i%3],'sources':src})
 return out
def run(policy,c):
 a=OfflineArena(c['sources'],c['seed'],c['error_mode']);cl=SimulatorClient('synthetic',transport=a,log_path=False);cl.enter();s=policy(cl,question=4,method='main');cl.exit();n=len(a.cleared);move=a.distance_m/5;meas=5*a.measure_count;sw=a.switch_count;clear=3*a.clear_count+2*n;groups=[set(s[x]) for x in ('cleared_channels','excluded_channels','unresolved_channels')];u=set().union(*groups);account=u==set(range(1,21)) and sum(map(len,groups))==len(u);false_excluded=bool(set(s['excluded_channels'])&set(a.sources));full=n==len(c['sources']) and set(s['cleared_channels'])==a.cleared and not false_excluded;return {'all_checks_pass':bool(s.get('completed') and full and account and abs(move+meas+sw+clear-a.virtual_time_s)<1e-6),'full_source_clear':full,'channels_accounted':account,'false_excluded_source':false_excluded,'time_s':a.virtual_time_s,'mean_s':a.virtual_time_s/n,'movement_s':move,'measure_s':meas,'switch_s':sw,'clears_s':clear,'fallback_attempts':s.get('fallback_clear_count',0),'maximum_count_stop_used':s.get('maximum_count_stop_used',False)}
def summary(rows,n):return {'cases':len(rows),'mean_per_source_s':statistics.mean(x[n]['mean_s'] for x in rows),'mean_movement_s':statistics.mean(x[n]['movement_s'] for x in rows),'mean_measure_s':statistics.mean(x[n]['measure_s'] for x in rows),'mean_switch_s':statistics.mean(x[n]['switch_s'] for x in rows),'mean_clears_s':statistics.mean(x[n]['clears_s'] for x in rows),'total_fallback_attempts':sum(x[n]['fallback_attempts'] for x in rows),'maximum_count_stops':sum(x[n]['maximum_count_stop_used'] for x in rows)}
def main():
 if json.loads(DEV.read_text(encoding='utf-8')).get('accepted') is not True:raise ValueError('development not accepted')
 O.mkdir(parents=True,exist_ok=True);cs=cases();payload=[{**{k:v for k,v in c.items() if k!='sources'},'sources':[asdict(x) for x in c['sources']]} for c in cs];cp=O/'cases.json'
 if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=payload:raise ValueError('case mismatch')
 if not cp.exists():save(cp,payload)
 files=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py','code/shared/search_policy_q4_reuse.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q4_ring25_joint_route.py','code/shared/search_policy_q4_ring22.py','code/shared/search_policy_q4_ring22_joint_route.py','results/optimization_200_400/ring22_continuous_certificate.json','code/shared/evaluate_q4_ring22_joint_route_final.py','results/q4_ring22_joint_route/development_v2/comparison.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];ip=O/'inputs_before_run.json'
 if ip.exists() and json.loads(ip.read_text(encoding='utf-8'))!=inputs:raise ValueError('input mismatch')
 if not ip.exists():save(ip,inputs)
 rows=[]
 for c in cs:
  x={'case_id':c['case_id'],'seed':c['seed']}
  for n,p in POL.items():x[n]=run(p,c)
  x['ring22_to_ring25_ratio']=x['ring22_joint']['time_s']/x['ring25_joint']['time_s'];rows.append(x)
 sums={n:summary(rows,n) for n in POL};checks=all(x[n]['all_checks_pass'] for x in rows for n in POL);imp=1-sums['ring22_joint']['mean_per_source_s']/sums['ring25_joint']['mean_per_source_s'];out={'scope':'SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT','artifact_type':'q4_ring22_joint_route_final_comparison','case_count':60,'seed_range':[98000,98059],'all_checks_pass':checks,'accepted':bool(checks and imp>0),'improvement_vs_ring25_joint':imp,'coverage_certificate':cand.coverage_certificate(),'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
