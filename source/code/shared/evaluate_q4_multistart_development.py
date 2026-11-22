"""A single frozen paired screen of Q4 multistart route ordering."""
from pathlib import Path
from collections import Counter
import json
import math
import statistics

import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_multistart as candidate
from evaluate_q4_domain_prior_development import run,save,sha,summarize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q4_multistart/development'
CASES=ROOT/'results/q4_reuse/development/cases.json'
RELATIVE_MIN=0.05
ABSOLUTE_MIN=10.0


def verify_routes(state,actions):
    errors=[];improved=0
    for i,event in enumerate(state.get('route_decisions',[])):
        node_keys=Counter((n[0],n[1],tuple(n[2])) for n in event['nodes'])
        position=event['position']
        lengths={}
        for name in ('anchor','proposal','selected'):
            route=event[f'{name}_route']
            if Counter((n[0],n[1],tuple(n[2])) for n in route)!=node_keys:
                errors.append(f'route {i}: {name} does not reconcile nodes')
            coords=[position]+[n[2] for n in route]
            length=sum(math.dist(a,b) for a,b in zip(coords,coords[1:]))
            lengths[name]=length
            if abs(length-event[f'{name}_length_m'])>1e-7:
                errors.append(f'route {i}: {name} length mismatch')
        if lengths['selected']>lengths['anchor']+1e-7:
            errors.append(f'route {i}: cost anchor violated')
        expected=event['proposal_route'] if lengths['proposal']<lengths['anchor']-1e-7 else event['anchor_route']
        if expected!=event['selected_route']:
            errors.append(f'route {i}: wrong anchor tie handling')
        improved+=lengths['selected']<lengths['anchor']-1e-7
        action_count=event['measure_count_before']+event['clear_count_before']
        if action_count>len(actions):
            errors.append(f'route {i}: invalid public action index')
        elif action_count and math.dist(actions[action_count-1]['point'],position)>1e-7:
            errors.append(f'route {i}: position differs from public history')
    return {'all_checks_pass':not errors,'errors':errors,'decision_count':len(state.get('route_decisions',[])),
            'shorter_planned_route_count':improved}


def main():
    cases=json.loads(CASES.read_text(encoding='utf-8'))
    assert len(cases)==36
    rels=['code/shared/search_policy.py','code/shared/search_policy_polygon.py','code/shared/search_policy_q4_batch.py',
          'code/shared/search_policy_q4_reuse.py','code/shared/search_policy_q4_ring22.py',
          'code/shared/search_policy_joint_route_q3.py','code/shared/search_policy_q4_ring22_joint_route.py',
          'code/shared/search_policy_q4_multistart.py','code/shared/route_multistart.py',
          'code/shared/evaluate_q4_multistart_development.py','code/shared/evaluate_q4_domain_prior_development.py',
          'code/shared/evaluate_q4_shadow_recovery_development.py','code/shared/shadow_cone_geometry.py',
          'code/shared/offline_arena.py','code/shared/simulator_client.py',
          'results/optimization_200_400/ring22_continuous_certificate.json',
          'results/q4_reuse/development/cases.json','planning/q4_multistart_experiment.md']
    inputs=[{'path:p':rel,'sha256':sha(ROOT/rel)} for rel in rels]
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
            'shorter_planned_route_count':sum(r['candidate']['route_checks']['shorter_planned_route_count'] for r in rows),
            'route_decision_count':sum(r['candidate']['route_checks']['decision_count'] for r in rows),
            'summaries':summaries,'rows':rows,'inputs':inputs}
    save(OUT/'comparison.json',result)
    save(OUT/'decision.json',{'accepted':bool(accepted),'next_step':'independent_review_then_fresh_60' if accepted else 'stop_without_tuning_holdout_or_practice'})
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
