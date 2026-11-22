"""Frozen paired development experiment for the user-supplied Q1 disk prior."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import time

from offline_arena import Source
from simulator_client import SimulatorClient
from evaluate_q4_shadow_recovery_development import TrackedArena
from search_policy import _bearing_triangle, _box_circle
from shadow_cone_geometry import contains_point, polygon_area
import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_domain_prior as candidate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q4_domain_prior/development'
CASES=ROOT/'results/q4_reuse/development/cases.json'
RELATIVE_MIN=0.05
ABSOLUTE_MIN=10.0
WORST_RATIO_MAX=1.10


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_tests():
    tests=[]
    # The entire disk lies within this square. Boundary points check the
    # implemented tangent orientation including points between tangent normals.
    square=[(-2000.,-2000.),(2000.,-2000.),(2000.,2000.),(-2000.,2000.)]
    clipped=candidate.clip_domain(square)
    maximum_radius=max(math.hypot(*p) for p in clipped)
    bound=(candidate.DOMAIN_RADIUS_M+candidate.DOMAIN_MARGIN_M)/math.cos(math.pi/candidate.DOMAIN_SIDES)
    boundary=[(1800*math.cos(2*math.pi*k/4096),1800*math.sin(2*math.pi*k/4096)) for k in range(4096)]
    tests.append({'name':'disk_boundary_retained','pass':all(contains_point(clipped,p,tolerance=1e-7) for p in boundary),
                  'point_count':len(boundary),'analytic_outer_vertex_radius_bound_m':bound,'observed_max_radius_m':maximum_radius})
    tests.append({'name':'outer_polygon_radial_bound','pass':maximum_radius<=bound+1e-7})
    small=[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]
    tests.append({'name':'interior_unchanged','pass':candidate.clip_domain(small)==small})
    # For each public bearing, construct the receiver from a boundary source.
    # This is a geometry test only, not a strategy decision interface.
    failed=[]
    for k in range(128):
        a=2*math.pi*k/128
        source=(1800*math.cos(a),1800*math.sin(a))
        for distance in (5.,750.,1500.):
            heading=a+0.79
            point=(source[0]-distance*math.cos(heading),source[1]-distance*math.sin(heading))
            for error in (-1.005,0.,1.005):
                direction=(math.degrees(heading)+error)%360
                poly=candidate.clip_domain(_bearing_triangle(point,direction))
                if not contains_point(poly,source,tolerance=1e-6):
                    failed.append([k,distance,error])
    tests.append({'name':'boundary_source_and_angle_endpoints','pass':not failed,'checks':1152,'failures':failed[:10]})
    triangle=[(0.,0.),(39.,0.),(19.5,39*math.sqrt(3)/2)]
    tests.append({'name':'diameter_under40_does_not_trigger_clear','pass':_box_circle(triangle)[1]>19.9,
                  'diameter_m':39.0,'minimum_enclosing_radius_m':39/math.sqrt(3)})
    return {'all_checks_pass':all(t['pass'] for t in tests),'tests':tests}


def run(policy,case,domain=False):
    sources=[Source(**s) for s in case['sources']]
    arena=TrackedArena(sources,case['seed'],case['error_mode'])
    client=SimulatorClient('offline',transport=arena,log_path=False)
    client.enter()
    start=time.perf_counter()
    state=policy(client,question=4,method='main')
    cpu=time.perf_counter()-start
    client.exit()
    groups=[set(state[k]) for k in ('cleared_channels','excluded_channels','unresolved_channels')]
    source_channels=set(arena.sources)
    errors=[]
    full_clear=arena.cleared==source_channels==groups[0]
    accounted=set.union(*groups)==set(range(1,21)) and sum(map(len,groups))==20
    false_excluded=bool(source_channels & groups[1])
    actions=arena.action_log
    measurements=[a for a in actions if a['path']=='/measure']
    clear_actions=[a for a in actions if a['path']=='/clear']
    position=(0.,0.);channel=1
    costs={'movement_s':0.,'measure_s':0.,'switch_s':0.,'successful_clear_s':0.,'failed_clear_s':0.}
    post_clear_measure_count=0;public_cleared=set()
    for a in actions:
        p=tuple(a['point'])
        costs['movement_s']+=math.dist(position,p)/5
        position=p
        if a['path']=='/measure':
            costs['measure_s']+=5
            costs['switch_s']+=int(a['channel']!=channel)
            channel=a['channel']
            post_clear_measure_count+=a['channel'] in public_cleared
        elif a['response'].get('clear_result')=='success':
            costs['successful_clear_s']+=5
            public_cleared.add(a['channel'])
        else:
            costs['failed_clear_s']+=3
    residual=sum(costs.values())-arena.virtual_time_s
    clock_ok=abs(residual)<1e-6 and abs(state['virtual_time_s']-arena.virtual_time_s)<1e-6
    counters_ok=state['measure_count']==len(measurements)==arena.measure_count and state['clear_count']==len(clear_actions)==arena.clear_count
    coverage_ok=state.get('coverage_points_total')==22 and state.get('coverage_geometry')==baseline.coverage_certificate()
    if domain:
        source_map={s.channel:s for s in sources}
        for idx,event in enumerate(state.get('domain_clips',[])):
            if event['empty_intersection_fallback']:
                errors.append(f'event {idx}: empty domain intersection')
            k=event['measure_index']-1
            if not 0<=k<len(measurements):
                errors.append(f'event {idx}: bad public measurement index');continue
            a=measurements[k]
            if a['channel']!=event['channel'] or math.dist(a['point'],event['point'])>1e-8 or a['response'].get('svd_deg')!=event['svd_deg']:
                errors.append(f'event {idx}: mismatch to public action history')
            s=source_map[event['channel']]
            truth=(s.x,s.y)
            if not all(contains_point(event[name],truth,tolerance=1e-5) for name in ('before_polygon','after_polygon')):
                errors.append(f'event {idx}: source omitted by polygon')
            if polygon_area(event['after_polygon'])>polygon_area(event['before_polygon'])+1e-5:
                errors.append(f'event {idx}: initial clip expanded polygon')
    valid_domain=all(math.hypot(s.x,s.y)<=1800+1e-7 for s in sources)
    passed=bool(state.get('completed') and not state.get('stopped') and full_clear and accounted and not false_excluded and clock_ok and counters_ok and coverage_ok and not errors and valid_domain and post_clear_measure_count==0)
    metrics={'all_checks_pass':passed,'completed':state.get('completed'),'full_source_clear':full_clear,
             'channels_accounted':accounted,'false_excluded_source':false_excluded,'clock_accounting_pass':clock_ok,
             'clock_residual_s':residual,'action_counter_pass':counters_ok,'coverage_certificate_pass':coverage_ok,
             'source_domain_valid':valid_domain,'domain_event_errors':errors,'domain_event_count':len(state.get('domain_clips',[])),
             'post_clear_measure_count':post_clear_measure_count,'source_count':len(sources),'time_s':arena.virtual_time_s,
             'mean_s':arena.virtual_time_s/len(sources),'policy_cpu_s':cpu,**costs,
             'measure_count':arena.measure_count,'clear_attempts':arena.clear_count,
             'fallback_count':state.get('fallback_clear_count',0),
             'public_action_fingerprint':hashlib.sha256(json.dumps(actions,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    return metrics,state,actions


def summarize(rows,name):
    metrics=[r[name] for r in rows]
    out={'case_count':len(metrics),'all_checks_pass':all(m['all_checks_pass'] for m in metrics),
         'mean_s_per_source':statistics.mean(m['mean_s'] for m in metrics),
         'max_policy_cpu_s':max(m['policy_cpu_s'] for m in metrics),
         'total_domain_events':sum(m['domain_event_count'] for m in metrics)}
    for k in ('movement_s','measure_s','switch_s','successful_clear_s','failed_clear_s'):
        out[f'mean_{k}_per_source']=statistics.mean(m[k]/m['source_count'] for m in metrics)
    return out


def main():
    cases=json.loads(CASES.read_text(encoding='utf-8'))
    assert len(cases)==36
    files=[CASES,Path(__file__),Path(candidate.__file__),Path(baseline.__file__),
           ROOT/'code/shared/search_policy.py',ROOT/'code/shared/search_policy_polygon.py',
           ROOT/'code/shared/search_policy_q4_batch.py',ROOT/'code/shared/search_policy_q4_reuse.py',
           ROOT/'code/shared/search_policy_q4_ring22.py',ROOT/'code/shared/search_policy_joint_route_q3.py',
           ROOT/'code/shared/evaluate_q4_shadow_recovery_development.py',ROOT/'code/shared/offline_arena.py',
           ROOT/'code/shared/simulator_client.py',ROOT/'code/shared/shadow_cone_geometry.py',
           ROOT/'results/optimization_200_400/ring22_continuous_certificate.json',
           ROOT/'planning/q1_reference_domain_prior_experiment.md']
    inputs=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files]
    manifest=OUT/'inputs_before_run.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=inputs:
        raise ValueError('Frozen input mismatch; use a new explicit experiment identity')
    save(manifest,inputs)
    tests=geometry_tests()
    save(OUT/'geometry_tests.json',tests)
    rows=[]
    for case in cases:
        row={'case_id':case['case_id'],'seed':case['seed'],'error_mode':case['error_mode']}
        for name,policy,domain in [('baseline',baseline.run_search,False),('candidate',candidate.run_search,True)]:
            metrics,state,actions=run(policy,case,domain)
            row[name]=metrics
            save(OUT/'states'/f"{case['case_id']}__{name}.json",{'state':state,'actions':actions})
        row['candidate_to_baseline_ratio']=row['candidate']['time_s']/row['baseline']['time_s']
        rows.append(row)
        print(json.dumps({'case_id':row['case_id'],'baseline':row['baseline']['mean_s'],'candidate':row['candidate']['mean_s'],
                          'checks':row['baseline']['all_checks_pass'] and row['candidate']['all_checks_pass']},ensure_ascii=False),flush=True)
    summaries={n:summarize(rows,n) for n in ('baseline','candidate')}
    saving=summaries['baseline']['mean_s_per_source']-summaries['candidate']['mean_s_per_source']
    relative=saving/summaries['baseline']['mean_s_per_source']
    worst=max(rows,key=lambda r:r['candidate_to_baseline_ratio'])
    checks=tests['all_checks_pass'] and all(s['all_checks_pass'] for s in summaries.values())
    accepted=checks and relative>=RELATIVE_MIN and saving>=ABSOLUTE_MIN and worst['candidate_to_baseline_ratio']<=WORST_RATIO_MAX
    result={'scope':'PAIRED_DEVELOPMENT_ONLY_NOT_REAL_PRACTICE','all_hard_checks_pass':checks,'accepted':bool(accepted),
            'predeclared_gate':{'relative_min':RELATIVE_MIN,'absolute_min_s_per_source':ABSOLUTE_MIN,'worst_ratio_max':WORST_RATIO_MAX},
            'absolute_saving_s_per_source':saving,'relative_improvement':relative,
            'worst_case':{'case_id':worst['case_id'],'ratio':worst['candidate_to_baseline_ratio']},
            'summaries':summaries,'rows':rows,'inputs':inputs,'geometry_tests':tests}
    save(OUT/'comparison.json',result)
    save(OUT/'decision.json',{'accepted':bool(accepted),'next_step':'independent_review_before_new_60' if accepted else 'stop_without_tuning_or_holdout_or_practice'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','inputs','geometry_tests')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
