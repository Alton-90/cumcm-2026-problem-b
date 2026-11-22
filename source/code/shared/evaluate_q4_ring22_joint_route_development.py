"""36-case paired Q4 development screen: ring25 joint v3 versus ring22 joint."""
import hashlib,json,statistics
from pathlib import Path
import search_policy_q4_ring25_joint_route as base
import search_policy_q4_ring22_joint_route as cand
from evaluate_q4_ring25_joint_route_development import _run
R=Path(__file__).resolve().parents[2];O=R/'results'/'q4_ring22_joint_route'/'development_v2';SRC=R/'results'/'q4_reuse'/'development'/'cases.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def summary(rows,n):return {'cases':len(rows),'mean_per_source_s':statistics.mean(x[n]['mean_s'] for x in rows),'mean_movement_s':statistics.mean(x[n]['movement_s'] for x in rows),'mean_measure_s':statistics.mean(x[n]['measure_s'] for x in rows),'mean_switch_s':statistics.mean(x[n]['switch_s'] for x in rows),'mean_clears_s':statistics.mean(x[n]['clears_s'] for x in rows),'total_fallback_attempts':sum(x[n]['fallback_attempts'] for x in rows),'maximum_count_stops':sum(x[n].get('maximum_count_stop_used',False) for x in rows)}
def main():
 O.mkdir(parents=True,exist_ok=True);cs=json.loads(SRC.read_text(encoding='utf-8'));cp=O/'cases.json'
 if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=cs:raise ValueError('case mismatch')
 if not cp.exists():save(cp,cs)
 files=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py','code/shared/search_policy_q4_reuse.py','code/shared/search_policy_q4_ring22.py','code/shared/search_policy_q4_ring22_joint_route.py','code/shared/search_policy_q4_ring25_joint_route.py','code/shared/search_policy_joint_route_q3.py','results/optimization_200_400/ring22_continuous_certificate.json','code/shared/evaluate_q4_ring22_joint_route_development.py','results/q4_reuse/development/cases.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];ip=O/'inputs_before_run.json'
 if ip.exists() and json.loads(ip.read_text(encoding='utf-8'))!=inputs:raise ValueError('hash mismatch')
 if not ip.exists():save(ip,inputs)
 rows=[]
 for c in cs:
  x={'case_id':c['case_id'],'seed':c['seed'],'ring25_joint':_run(base.run_search,c),'ring22_joint':_run(cand.run_search,c)};x['ring22_to_ring25_ratio']=x['ring22_joint']['time_s']/x['ring25_joint']['time_s'];rows.append(x)
 sums={n:summary(rows,n) for n in ('ring25_joint','ring22_joint')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['ring22_joint']['mean_per_source_s']/sums['ring25_joint']['mean_per_source_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and imp>0),'improvement_vs_ring25_joint':imp,'coverage_certificate':cand.coverage_certificate(),'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
