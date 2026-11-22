"""Frozen 36-case Q4 comparison: ring25 versus independently certified ring22."""
from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
import search_policy_q4_ring25 as ring25
import search_policy_q4_ring22 as ring22
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/'q4_ring22'/'development';SOURCE=ROOT/'results'/'q4_reuse'/'development'/'cases.json';POLICIES={'ring25':ring25.run_search,'ring22':ring22.run_search}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
 cases=json.loads(SOURCE.read_text(encoding='utf-8'));target=OUT/'cases.json'
 if len(cases)!=36:raise ValueError('expected frozen 36 cases')
 if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=cases:raise ValueError('case mismatch')
 if not target.exists():save(target,cases)
 files=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py','code/shared/search_policy_q4_reuse.py','code/shared/search_policy_q4_ring25.py','code/shared/search_policy_q4_ring22.py','results/optimization_200_400/ring22_continuous_certificate.json','code/shared/offline_arena.py','code/shared/evaluate_q4_ring22_development.py','results/q4_reuse/development/cases.json',str(target.relative_to(ROOT))]
 h=[{'path':x,'sha256':sha(ROOT/x)} for x in files];ih=OUT/'inputs_before_run.json'
 if ih.exists() and json.loads(ih.read_text(encoding='utf-8'))!=h:raise ValueError('input hash mismatch')
 if not ih.exists():save(ih,h)
 return cases,h
def run(policy,case):
 src=[Source(**x) for x in case['sources']];a=OfflineArena(src,case['seed'],case['error_mode']);c=SimulatorClient('synthetic',transport=a,log_path=False);c.enter();s=policy(c,question=4,method='main');c.exit();cleared=len(a.cleared);move=a.distance_m/5;measure=5*a.measure_count;switch=a.switch_count;clears=3*a.clear_count+2*cleared;groups=[set(s[x]) for x in ('cleared_channels','excluded_channels','unresolved_channels')];union=set().union(*groups);accounted=union==set(range(1,21)) and sum(map(len,groups))==len(union);full=cleared==len(src) and set(s['cleared_channels'])==a.cleared and not(set(s['excluded_channels'])&set(a.sources));return {'all_checks_pass':bool(s.get('completed') and full and accounted and abs(move+measure+switch+clears-a.virtual_time_s)<1e-6),'full_source_clear':full,'channels_accounted':accounted,'time_s':a.virtual_time_s,'mean_s':a.virtual_time_s/cleared if cleared else None,'movement_s':move,'measure_s':measure,'switch_s':switch,'clears_s':clears,'fallback_attempts':s.get('fallback_clear_count',0)}
def summary(rows,name):return {'cases':len(rows),'mean_per_source_s':statistics.mean(x[name]['mean_s'] for x in rows),'mean_movement_s':statistics.mean(x[name]['movement_s'] for x in rows),'mean_measure_s':statistics.mean(x[name]['measure_s'] for x in rows),'mean_switch_s':statistics.mean(x[name]['switch_s'] for x in rows),'mean_clears_s':statistics.mean(x[name]['clears_s'] for x in rows),'total_fallback_attempts':sum(x[name]['fallback_attempts'] for x in rows)}
def main():
 OUT.mkdir(parents=True,exist_ok=True);cases,h=freeze();rows=[]
 for case in cases:
  row={'case_id':case['case_id'],'seed':case['seed']}
  for n,p in POLICIES.items():row[n]=run(p,case)
  row['ring22_to_ring25_ratio']=row['ring22']['time_s']/row['ring25']['time_s'];rows.append(row)
 sums={n:summary(rows,n) for n in POLICIES};checks=all(r[n]['all_checks_pass'] for r in rows for n in POLICIES);improvement=1-sums['ring22']['mean_per_source_s']/sums['ring25']['mean_per_source_s'];out={'scope':'SYNTHETIC_DEVELOPMENT_ONLY','artifact_type':'q4_ring22_development_comparison','case_count':36,'all_checks_pass':checks,'accepted':bool(checks and improvement>0),'selection_rule':'advance only on full paired pass and strictly lower ring22 mean per source','improvement_vs_ring25':improvement,'coverage_certificate':ring22.coverage_certificate(),'summaries':sums,'rows':rows,'inputs':h};save(OUT/'comparison.json',out);print(json.dumps({'all_checks_pass':checks,'accepted':out['accepted'],'improvement':improvement,'summaries':sums},ensure_ascii=False))
if __name__=='__main__':main()
