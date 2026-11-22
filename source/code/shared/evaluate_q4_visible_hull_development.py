"""A single frozen paired screen of Q4 visible-hull recovery."""
from pathlib import Path
from collections import Counter
import json
import math
import statistics

import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_visible_hull as candidate
from evaluate_q4_domain_prior_development import run,save,sha,summarize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q4_visible_hull/development'
CASES=ROOT/'results/q4_reuse/development/cases.json'
RELATIVE_MIN=0.05
ABSOLUTE_MIN=10.0


def verify_routes(state,actions):
    errors=[]
    measures=[a for a in actions if a['path']=='/measure']
    for i,event in enumerate(state.get('visible_hull_events',[])):
        points=event['positive_points'];weights=event['weights'];q=event['point']
        if len(points)!=len(weights) or min(weights)<0 or abs(sum(weights)-1)>1e-10:
            errors.append(f'event {i}: invalid convex weights');continue
        expected=tuple(sum(w*p[k] for w,p in zip(weights,points)) for k in (0,1))
        if math.dist(expected,q)>1e-7:errors.append(f'event {i}: convex coordinate mismatch')
        idx=event['measure_index']-1
        if not 0<=idx<len(measures):errors.append(f'event {i}: invalid measurement index');continue
        action=measures[idx]
        if action['channel']!=event['channel'] or math.dist(action['point'],q)>1e-7:
            errors.append(f'event {i}: actual measurement differs')
        if action['response'].get('measure_result') not in ('direction','near'):
            errors.append(f'event {i}: guaranteed reception failed')
        history=[a['point'] for a in measures[:idx] if a['channel']==event['channel'] and a['response'].get('measure_result') in ('direction','near')]
        if any(not any(math.dist(p,a)<1e-7 for a in history) for p in points):
            errors.append(f'event {i}: anchor not a past public positive')
        if event['move_m']>event['old_move_m']+1e-7:errors.append(f'event {i}: move cap violated')
        if event['posterior_diameter_upper_m']>.8*event['prior_diameter_m']+1e-7:
            errors.append(f'event {i}: information threshold violated')
        if not 1<=event['extra_used']<8:errors.append(f'event {i}: local budget violated')
    return {'all_checks_pass':not errors,'errors':errors,'decision_count':len(state.get('visible_hull_events',[])),
            'shorter_planned_route_count':len(state.get('visible_hull_events',[]))}


def main():
    cases=json.loads(CASES.read_text(encoding='utf-8'))
    assert len(cases)==36
    rels=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py',
          'code/shared/search_policy_q4_reuse.py','code/shared/search_policy_q4_ring22.py',
          'code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q4_ring22_joint_route.py',
          'code/shared/search_policy_q4_visible_hull.py','code/shared/visible_hull_recovery.py',
          'code/shared/evaluate_q4_visible_hull_development.py','code/shared/evaluate_q4_domain_prior_development.py',
          'code/shared/evaluate_q4_shadow_recovery_development.py','code/shared/shadow_cone_geometry.py',
          'code/shared/offline_arena.py','code/shared/simulator_client.py',
          'results/optimization_200_400/ring22_continuous_certificate.json',
          'results/q4_reuse/development/cases.json','planning/q2_transfer_q4_visible_hull.md']
    inputs=[{'path':rel,'sha256':sha(ROOT/rel)} for rel in rels]
    manifest=OUT/'inputs_before_run.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=inputs:
        raise ValueError('Frozen inputs differ')
    save(manifest,inputs)
    save(OUT/'cases.json',cases)
    rows=[]
    for case in cases:
        row={'case_id':case['case_id'],'seed':case['seed'],'error_mode':case['error_mode']}
        for name,policy in [('baseline',baseline.run_search),('candidate',candidate.run_search)]:
            metrics,state,actions=run(policy,case,domain=False)
            if name=='candidate':
                route_checks=verify_routes(state,actions)
                metrics['route_checks']=route_checks
                metrics['all_checks_pass']=metrics['all_checks_pass'] and route_checks['all_checks_pass']
            row[name]=metrics
            save(OUT/'states'/f"{case['case_id']}__{name}.json",{'state':state,'actions':actions})
        row['candidate_to_baseline_ratio']=row['candidate']['time_s']/row['baseline']['time_s']
        rows.append(row)
        print(json.dumps({'case_id':case['case_id'],'baseline':row['baseline']['mean_s'],'candidate':row['candidate']['mean_s'],
                          'checks':row['baseline']['all_checks_pass'] and row['candidate']['all_checks_pass']}),flush=True)
    summaries={n:summarize(rows,n) for n in ('baseline','candidate')}
    saving=summaries['baseline']['mean_s_per_source']-summaries['candidate']['mean_s_per_source']
    relative=saving/summaries['baseline']['mean_s_per_source']
    checks=all(s['all_checks_pass'] for s in summaries.values())
    accepted=checks and saving>=ABSOLUTE_MIN and relative>=RELATIVE_MIN
    worst=max(rows,key=lambda r:r['candidate_to_baseline_ratio'])
    result={'scope':'PAIRED_DEVELOPMENT_ONLY_NOT_REAL_PRACTICE','all_hard_checks_pass':checks,'accepted':bool(accepted),
            'predeclared_gate':{'relative_min':RELATIVE_MIN,'absolute_min_s_per_source':ABSOLUTE_MIN},
            'absolute_saving_s_per_source':saving,'relative_improvement':relative,
            'worst_case':{'case_id':worst['case_id'],'ratio':worst['candidate_to_baseline_ratio']},
            'adopted_visible_hull_measurements':sum(r['candidate']['route_checks']['decision_count'] for r in rows),
            'route_decision_count':sum(r['candidate']['route_checks']['decision_count'] for r in rows),
            'summaries':summaries,'rows':rows,'inputs':inputs}
    save(OUT/'comparison.json',result)
    save(OUT/'decision.json',{'accepted':bool(accepted),'next_step':'independent_review_then_fresh_60' if accepted else 'stop_without_tuning_holdout_or_practice'})
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
