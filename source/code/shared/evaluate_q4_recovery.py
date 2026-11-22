"""Paired v1/recovery evaluation on predeclared synthetic cases; no HTTP."""
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source, make_cases
from simulator_client import SimulatorClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/Q4/experiments/round2'


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def declared_cases():
    cases=[]
    for case in make_cases(4):
        cases.append(dict(split='development', **case))
    for index, seed in enumerate(range(41000, 41024)):
        rng=random.Random(seed)
        count=(10,13,16)[index % 3]
        channels=rng.sample(range(1,21),count)
        direction_count=(0,1,count-1,count,count//2,count)[index % 6]
        sources=[]
        for j, channel in enumerate(channels):
            a=rng.random()*2*math.pi
            rho=1800*math.sqrt(rng.random())
            x,y=rho*math.cos(a),rho*math.sin(a)
            family=index//6
            if family==1:
                a=2*math.pi*j/count
                x,y=1800*math.cos(a),1800*math.sin(a)
            elif family==2:
                grid=[(xx,yy) for xx in range(-1800,1801,600) for yy in range(-1800,1801,600)
                      if xx*xx+yy*yy<=1800**2]
                x,y=grid[j % len(grid)]
            elif family==3 and j<3:
                x,y=((0.0,0.0),(5.0,0.0),(20.0,0.0))[j]
            radius=1000.0 if index%3==0 else 1500.0 if index%3==1 else rng.uniform(1000,1500)
            heading=(math.degrees(a) if family==1 else rng.uniform(0,360)) if j<direction_count else None
            sources.append(Source(channel,float(x),float(y),radius,heading))
        cases.append(dict(case_id=f'holdout_q4_{seed}',seed=seed,split='holdout',sources=sources,
                          error_mode=('fixed_hash','positive_edge','negative_edge')[index%3]))
    return cases


def load_entry(name):
    path=ROOT/'code/Q4/python'/('q4_main.py' if name=='v1' else 'q4_recovery.py')
    spec=importlib.util.spec_from_file_location('evaluate_'+name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run,path


def evaluate():
    cases=declared_cases()
    data=[dict(**{k:v for k,v in c.items() if k!='sources'},sources=[asdict(s) for s in c['sources']]) for c in cases]
    casepath=OUT/'cases.json'
    # Freeze the prospective dataset before either method is evaluated.
    if casepath.exists():
        assert json.loads(casepath.read_text(encoding='utf-8'))==data, 'Declared dataset changed; use a new round.'
    else:
        save(casepath,data)
    all_rows={}
    for method in ('v1','recovery'):
        run,path=load_entry(method)
        rows=[]
        for case in cases:
            arena=OfflineArena(case['sources'],case['seed'],case['error_mode'])
            client=SimulatorClient('synthetic-team',transport=arena,log_path=False)
            started=time.perf_counter()
            error=None
            result={}
            try:
                client.enter()
                result=run(client)
            except Exception as exc:
                error=repr(exc)
            finally:
                if client.entered and not client.ended:
                    client.exit()
            elapsed=time.perf_counter()-started
            count=len(arena.cleared)
            calculated=arena.distance_m/5+5*arena.measure_count+arena.switch_count+3*arena.clear_count+2*count
            violations=[]
            if error or not result.get('completed'): violations.append('execution_not_completed')
            if count!=len(case['sources']): violations.append('sources_remain')
            if set(result.get('cleared_channels',[]))!=arena.cleared: violations.append('clear_evidence_mismatch')
            excluded=set(result.get('excluded_channels',[]))
            if excluded & set(arena.sources): violations.append('false_exclusion')
            if excluded | arena.cleared != set(range(1,21)): violations.append('channel_accounting')
            if abs(calculated-arena.virtual_time_s)>1e-6 or abs(client.virtual_time_s-arena.virtual_time_s)>1e-6:
                violations.append('time_accounting')
            if arena.virtual_time_s>360000 or elapsed>1200: violations.append('time_limit')
            if result.get('extra_bearing_count',0)>8*len(case['sources']): violations.append('extra_measure_budget')
            row=dict(case_id=case['case_id'],split=case['split'],source_count=len(case['sources']),cleared_count=count,
                     directional_count=sum(s.heading is not None for s in case['sources']),
                     virtual_time_s=arena.virtual_time_s,mean_time_per_source_s=arena.virtual_time_s/count if count else None,
                     measurements=arena.measure_count,clear_attempts=arena.clear_count,movement_m=arena.distance_m,
                     actual_execution_s=elapsed,policy=result,error=error,violations=violations)
            rows.append(row)
        save(OUT/'metrics'/f'{method}.json',dict(scope='synthetic_not_official',entry=ref(path),cases=rows))
        all_rows[method]=rows
        print(method,'cases',len(rows),'failures',sum(bool(x['violations']) for x in rows),flush=True)
    comparisons=[]
    for old,new in zip(all_rows['v1'],all_rows['recovery']):
        assert old['case_id']==new['case_id']
        comparisons.append(dict(case_id=old['case_id'],split=old['split'],
                                ratio=new['virtual_time_s']/old['virtual_time_s'],
                                v1_time_s=old['virtual_time_s'],recovery_time_s=new['virtual_time_s']))
    summaries={}
    for split in ('all','development','holdout'):
        selected={m:[x for x in rows if split=='all' or x['split']==split] for m,rows in all_rows.items()}
        old=statistics.mean(x['mean_time_per_source_s'] for x in selected['v1'])
        new=statistics.mean(x['mean_time_per_source_s'] for x in selected['recovery'])
        summaries[split]=dict(cases=len(selected['v1']),v1_mean_s=old,recovery_mean_s=new,reduction=1-new/old,
                             v1_clear_attempts=sum(x['clear_attempts'] for x in selected['v1']),
                             recovery_clear_attempts=sum(x['clear_attempts'] for x in selected['recovery']),
                             v1_fallbacks=sum(x['policy'].get('fallback_count',0) for x in selected['v1']),
                             recovery_fallbacks=sum(x['policy'].get('fallback_count',0) for x in selected['recovery']))
    constraints_ok=all(not x['violations'] for rows in all_rows.values() for x in rows)
    no_large_regression=all(x['ratio']<=1.10 for x in comparisons)
    accepted=constraints_ok and no_large_regression and summaries['all']['reduction']>0 and summaries['holdout']['reduction']>0
    result=dict(scope='SYNTHETIC_NOT_OFFICIAL',criteria_predeclared_in='planning/q4_recovery_round2_plan.md',
                accepted_by_predeclared_numeric_criteria=accepted,all_cases_constraint_pass=constraints_ok,
                max_case_time_ratio=max(x['ratio'] for x in comparisons),regressions_over_10_percent=[x for x in comparisons if x['ratio']>1.10],
                summaries=summaries,case_comparisons=comparisons,
                inputs=[ref(casepath),ref(ROOT/'planning/q4_recovery_round2_plan.md'),
                        ref(ROOT/'code/shared/search_policy.py'),ref(ROOT/'code/shared/search_policy_recovery.py'),
                        ref(ROOT/'code/shared/offline_arena.py'),ref(Path(__file__).resolve())],
                limitations=['Synthetic holdout error fields are not official error fields.',
                             'No claim of global optimality or official speedup.',
                             'A bounded recovery failure retains the original finite clearance sweep.'])
    save(OUT/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('accepted_by_predeclared_numeric_criteria','max_case_time_ratio','summaries')},ensure_ascii=False),flush=True)
    return accepted


if __name__=='__main__':
    raise SystemExit(0 if evaluate() else 1)
