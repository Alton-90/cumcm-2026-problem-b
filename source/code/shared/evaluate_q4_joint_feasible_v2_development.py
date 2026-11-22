import hashlib,json,math,statistics
from pathlib import Path
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
import search_policy_q4_ring22_joint_route as base
import search_policy_q4_joint_feasible_v2 as cand
R=Path(__file__).resolve().parents[2];O=R/'results'/'q4_joint_feasible'/'development_v2';SRC=R/'results'/'q4_reuse'/'development'/'cases.json'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(policy,c,diagnose=False):
 src=[Source(**x) for x in c['sources']];a=OfflineArena(src,c['seed'],c['error_mode']);cl=SimulatorClient('x',transport=a,log_path=False);cl.enter();s=policy(cl,question=4,method='main');cl.exit();n=len(a.cleared);move=a.distance_m/5;meas=5*a.measure_count;sw=a.switch_count;clear=3*a.clear_count+2*n;groups=[set(s[x]) for x in ('cleared_channels','excluded_channels','unresolved_channels')];u=set().union(*groups);account=u==set(range(1,21)) and sum(map(len,groups))==len(u);false=bool(set(s['excluded_channels'])&set(a.sources));full=n==len(src) and set(s['cleared_channels'])==a.cleared and not false;survive=True;checked=0;empty=[]
 if diagnose:
  for source in src:
   hist=s.get('feasible_histories',{}).get(str(source.channel),[])
   if not any(x[1]=='direction' for x in hist):continue
   fc=cand.FeasibleClient(None);fc.history[source.channel]=hist;fc._initialize(source.channel);fc._rebuild(source.channel);q=fc.models[source.channel]['q'];checked+=1
   if not len(q):empty.append(source.channel);survive=False
   elif min(math.dist((float(x[0]),float(x[1])),(source.x,source.y)) for x in q)>15+1e-9:survive=False
 return {'all_checks_pass':bool(s.get('completed') and full and account and survive and abs(move+meas+sw+clear-a.virtual_time_s)<1e-6),'full_source_clear':full,'channels_accounted':account,'false_excluded_source':false,'truth_neighborhood_survival':survive,'diagnosed_true_models':checked,'empty_model_channels':empty,'time_s':a.virtual_time_s,'mean_s':a.virtual_time_s/n,'movement_s':move,'measure_s':meas,'switch_s':sw,'clears_s':clear,'fallback_attempts':s.get('fallback_clear_count',0),'feasible_fallbacks':s.get('feasible_fallback_count',0)}
def summ(rs,n):return {k:statistics.mean(x[n][k] for x in rs) for k in ('mean_s','movement_s','measure_s','switch_s','clears_s')}
def main():
 O.mkdir(parents=True,exist_ok=True);cs=json.loads(SRC.read_text(encoding='utf-8'));cp=O/'cases.json'
 if not cp.exists():save(cp,cs)
 files=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py','code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q4_ring22.py','code/shared/search_policy_q4_ring22_joint_route.py','code/shared/search_policy_q4_joint_feasible.py','code/shared/search_policy_q4_joint_feasible_v2.py','code/shared/evaluate_q4_joint_feasible_v2_development.py','results/optimization_200_400/ring22_continuous_certificate.json','results/q4_reuse/development/cases.json',str(cp.relative_to(R))];inputs=[{'path':x,'sha256':sha(R/x)} for x in files];save(O/'inputs_before_run.json',inputs);rows=[]
 for c in cs:rows.append({'case_id':c['case_id'],'seed':c['seed'],'ring22_joint':run(base.run_search,c),'joint_feasible_v2':run(cand.run_search,c,True)})
 sums={n:summ(rows,n) for n in ('ring22_joint','joint_feasible_v2')};checks=all(x[n]['all_checks_pass'] for x in rows for n in sums);imp=1-sums['joint_feasible_v2']['mean_s']/sums['ring22_joint']['mean_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and imp>=.03),'improvement_vs_ring22_joint':imp,'all_truth_neighborhood_survival':all(x['joint_feasible_v2']['truth_neighborhood_survival'] for x in rows),'summaries':sums,'rows':rows,'inputs':inputs};save(O/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':imp,'survival':out['all_truth_neighborhood_survival'],'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
