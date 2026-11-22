import hashlib,json,statistics
from pathlib import Path
import search_policy_joint_route_q3 as base
import search_policy_q3_event_action as cand
from evaluate_q3_measure_replan_development import run
R=Path(__file__).resolve().parents[2];O=R/'results'/'q3_event_action'/'development_v2';SRC=R/'results'/'feasible_q3'/'cases.json'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summ(rows,n):return {k:statistics.mean(x[n][k] for x in rows) for k in ('mean_s','movement_s','measure_s','switch_s','clears_s')}
def main():
 O.mkdir(parents=True,exist_ok=True);cs=json.loads(SRC.read_text(encoding='utf-8'))[:36];cp=O/'cases.json'
 if not cp.exists():save(cp,cs)
 files=['code/shared/search_policy.py','code/shared/search_policy_compact.py','code/shared/search_policy_feasible_q3.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q3_measure_replan.py','code/shared/search_policy_q3_event_action.py','code/shared/evaluate_q3_event_action_development.py','code/shared/offline_arena.py','results/feasible_q3/cases.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];save(O/'inputs_before_run.json',inputs)
 rows=[]
 for c in cs:rows.append({'case_id':c['case_id'],'seed':c['seed'],'joint_route':run(base.run_search,c),'event_action':run(cand.run_search,c)})
 sums={n:summ(rows,n) for n in ('joint_route','event_action')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['event_action']['mean_s']/sums['joint_route']['mean_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and imp>0),'improvement_vs_joint_route':imp,'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
