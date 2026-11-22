"""36-case paired screen for Q3 current-pose measurement replanning."""
import hashlib,json,statistics
from pathlib import Path
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
import search_policy_joint_route_q3 as base
import search_policy_q3_measure_replan as candidate
R=Path(__file__).resolve().parents[2];O=R/'results'/'q3_measure_replan'/'development';SRC=R/'results'/'feasible_q3'/'cases.json'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run(policy,c):
 a=OfflineArena([Source(**x) for x in c['sources']],c['seed'],c['error_mode']);cl=SimulatorClient('synthetic',transport=a,log_path=False);cl.enter();s=policy(cl,question=3,method='main');cl.exit();n=len(a.cleared);m=a.distance_m/5;me=5*a.measure_count;sw=a.switch_count;ce=3*a.clear_count+2*n;gs=[set(s[x]) for x in ('cleared_channels','excluded_channels','unresolved_channels')];u=set().union(*gs);account=u==set(range(1,21)) and sum(map(len,gs))==len(u);full=n==len(c['sources']) and set(s['cleared_channels'])==a.cleared and not(set(s['excluded_channels'])&set(a.sources));return {'all_checks_pass':bool(s['completed'] and full and account and abs(m+me+sw+ce-a.virtual_time_s)<1e-6),'full_source_clear':full,'channels_accounted':account,'time_s':a.virtual_time_s,'mean_s':a.virtual_time_s/n,'movement_s':m,'measure_s':me,'switch_s':sw,'clears_s':ce,'fallback_attempts':s.get('fallback_clear_count',0)}
def summ(rows,n):return {k:statistics.mean(x[n][k] for x in rows) for k in ('mean_s','movement_s','measure_s','switch_s','clears_s')}
def main():
 O.mkdir(parents=True,exist_ok=True);cs=json.loads(SRC.read_text(encoding='utf-8'))[:36];cp=O/'cases.json'
 if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=cs:raise ValueError('case mismatch')
 if not cp.exists():save(cp,cs)
 files=['code/shared/search_policy.py','code/shared/search_policy_compact.py','code/shared/search_policy_feasible_q3.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q3_measure_replan.py','code/shared/evaluate_q3_measure_replan_development.py','code/shared/offline_arena.py','results/feasible_q3/cases.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':h(R/x)} for x in files];ip=O/'inputs_before_run.json'
 if ip.exists() and json.loads(ip.read_text(encoding='utf-8'))!=inputs:raise ValueError('hash mismatch')
 if not ip.exists():save(ip,inputs)
 rows=[]
 for c in cs:
  x={'case_id':c['case_id'],'seed':c['seed'],'joint_route':run(base.run_search,c),'measure_replan':run(candidate.run_search,c)};rows.append(x)
 sums={n:summ(rows,n) for n in ('joint_route','measure_replan')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['measure_replan']['mean_s']/sums['joint_route']['mean_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and imp>0),'improvement_vs_joint_route':imp,'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
