"""Root review of the rejected Q3 parallax candidate; no online execution."""
from pathlib import Path
import hashlib,json,math
from evaluate_q3_robust_parallax_development import run
import search_policy_q3_combined as baseline
import search_policy_q3_robust_parallax as candidate
from verify_paper_q12_geometry import upper
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q3_robust_parallax/development'
data=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
errors=[]
for item in data['inputs']:
    if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']: errors.append(item['path'])
rows=data['rows']
selected=[]
for row in rows:
    saved=json.loads((OUT/'states'/f"{row['case_id']}__candidate.json").read_text(encoding='utf-8'))
    measures=[a for a in saved['actions'] if a['path']=='/measure']
    for event in saved['state']['robust_parallax_events']:
        if not event['used_offset']: continue
        q=event['selected_point'];p=event['before_polygon']
        j=upper(p,q,1.005,1.)
        jc=upper(p,event['original_center'],1.005,1.)
        i=event['measure_index']-1;a=measures[i]
        checks=[max(math.dist(q,v) for v in p)<=999.9+1e-8,
                math.dist(event['current_position'],q)<=math.dist(event['current_position'],event['original_center'])+50+1e-8,
                j<=.8*jc+1e-8,abs(j-event['best_offset_upper_bound_m'])<1e-6,
                a['channel']==event['channel'],math.dist(a['point'],q)<1e-8,
                a['response']['measure_result'] in ['near','direction'],0<=event['extra_used_before']<8]
        selected.append({'case_id':row['case_id'],'channel':event['channel'],'independent_upper_m':j,'checks_pass':all(checks)})
        if not all(checks):errors.append('event:'+row['case_id'])
best=min(rows,key=lambda r:r['candidate_to_baseline_ratio'])
worst=max(rows,key=lambda r:r['candidate_to_baseline_ratio'])
cases={c['case_id']:c for c in json.loads((OUT/'cases.json').read_text(encoding='utf-8'))}
reruns=[]
for row in [best,worst]:
    for name,policy in [('baseline',baseline.run_search),('candidate',candidate.run_search)]:
        metrics,_,_=run(policy,cases[row['case_id']],False)
        keys=['completed','full_source_clear','time_s','movement_s','measure_count','clear_attempt_count','public_action_fingerprint']
        ok=all(metrics[k]==row[name][k] for k in keys) and metrics['all_checks_pass']
        reruns.append({'case_id':row['case_id'],'method':name,'non_cpu_metrics_and_action_fingerprint_match':ok})
        if not ok:errors.append('rerun:'+row['case_id']+name)
result={'implementation_review_pass':not errors,'errors':errors,'independent_selected_events':selected,'extreme_case_reruns':reruns,
        'method_accepted':False,'reason':'predeclared development performance gates failed',
        'baseline_mean_s':data['summaries']['baseline']['mean_s_per_source'],
        'candidate_mean_s':data['summaries']['candidate']['mean_s_per_source'],
        'absolute_saving_s_per_source':data['absolute_saving_s_per_source'],
        'relative_improvement':data['relative_improvement'],'formal_and_practice_runs_started':False}
(OUT/'root_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='independent_selected_events'},ensure_ascii=False,indent=2))
print('independently verified adopted events',len(selected))
assert not errors

