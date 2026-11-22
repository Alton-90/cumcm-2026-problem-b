import hashlib,json,statistics
from pathlib import Path
import search_policy_q4_ring22_joint_route as base
import search_policy_q4_joint_feasible as cand
from evaluate_q4_ring25_joint_route_development import _run
R=Path(__file__).resolve().parents[2];O=R/'results'/'q4_joint_feasible'/'development';SRC=R/'results'/'q4_reuse'/'development'/'cases.json'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summ(rs,n):return {k:statistics.mean(x[n][k] for x in rs) for k in ('mean_s','movement_s','measure_s','switch_s','clears_s')}
def main():
 O.mkdir(parents=True,exist_ok=True);cs=json.loads(SRC.read_text(encoding='utf-8'));cp=O/'cases.json'
 if not cp.exists():save(cp,cs)
 files=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py','code/shared/search_policy_q4_reuse.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q4_ring22.py','code/shared/search_policy_q4_ring22_joint_route.py','code/shared/search_policy_q4_joint_feasible.py','results/optimization_200_400/ring22_continuous_certificate.json','code/shared/evaluate_q4_joint_feasible_development.py','results/q4_reuse/development/cases.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];save(O/'inputs_before_run.json',inputs);rows=[]
 for c in cs:rows.append({'case_id':c['case_id'],'seed':c['seed'],'ring22_joint':_run(base.run_search,c),'joint_feasible':_run(cand.run_search,c)})
 sums={n:summ(rows,n) for n in ('ring22_joint','joint_feasible')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['joint_feasible']['mean_s']/sums['ring22_joint']['mean_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and imp>=.03),'improvement_vs_ring22_joint':imp,'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
