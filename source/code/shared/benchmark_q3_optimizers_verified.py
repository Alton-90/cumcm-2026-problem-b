"""Bound-cache runner for the unchanged Q3 optimizer comparison.

Adds fail-closed cache identity checks without changing objectives, samplers,
case sets, budgets or existing evidence. Legacy baselines are recalculated;
legacy candidate caches must pass complete ordered-case and score checks before
being admitted with explicit migration provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import benchmark_q3_optimizers as b

OUT=b.OUT
PLAN=b.load(OUT/'plan.json')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def binding(cases,label,parameters=None):
    expected=b.load(OUT/(label+'_cases.json'))
    if cases!=expected:raise ValueError('Case contents or ordering differ from frozen '+label+' dataset')
    return {'plan_sha256':b.core.original.sha(OUT/'plan.json'),'case_file_sha256':b.core.original.sha(OUT/(label+'_cases.json')),
            'ordered_case_ids':[c['case_id'] for c in cases], 'dependency_manifest_sha256':digest(PLAN['inputs']),
            'parameters':None if parameters is None else list(map(float,parameters)),
            'versions':PLAN['versions']}


def validate_baseline(rows,cases):
    if len(rows)!=len(cases):raise ValueError('Baseline count differs from case count')
    for row,case in zip(rows,cases,strict=True):
        if row['source_count']!=len(case['sources']) or not row['all_checks_pass']:
            raise ValueError('Baseline source count or hard checks invalid')
        if abs(row['mean_s']-row['time_s']/len(case['sources']))>1e-9:
            raise ValueError('Baseline time denominator invalid')


def baseline(cases,label):
    expected=binding(cases,label)
    path=OUT/'bound_cache'/(label+'_baseline.json')
    if path.exists():
        cache=b.load(path)
        if cache['binding']!=expected or cache['data_sha256']!=digest(cache['data']):raise ValueError('Baseline cache binding changed')
        rows=cache['data']
    else:
        rows=[]
        for case in cases:
            metrics,_,_=b.run(b.core.baseline.run_search,case)
            rows.append(metrics)
        validate_baseline(rows,cases)
        b.save(path,{'binding':expected,'data':rows,'data_sha256':digest(rows),'provenance':'Recalculated all baseline cases under strict binding; no unbound baseline accepted.'})
    validate_baseline(rows,cases)
    return rows


def validate_row(row,parameters,cases,base):
    validate_baseline(base,cases)
    if list(map(float,row['parameters']))!=list(map(float,parameters)):raise ValueError('Cached parameters mismatch')
    if len(row['cases'])!=len(cases):raise ValueError('Cached candidate case count mismatch')
    errors=[]
    for recorded,case,reference in zip(row['cases'],cases,base,strict=True):
        if recorded['case_id']!=case['case_id']:raise ValueError('Candidate case identity/order mismatch')
        m=recorded['metrics']
        if m['source_count']!=len(case['sources']):raise ValueError('Candidate source count mismatch')
        if abs(m['mean_s']-m['time_s']/len(case['sources']))>1e-9:raise ValueError('Candidate denominator mismatch')
        if abs(recorded['delta_s_per_source']-(m['mean_s']-reference['mean_s']))>1e-8:raise ValueError('Cached paired delta mismatch')
        if abs(recorded['time_ratio']-m['time_s']/reference['time_s'])>1e-9:raise ValueError('Cached ratio mismatch')
        if recorded['hard_checks_pass']!=(m['all_checks_pass'] and not recorded['errors']):raise ValueError('Hard check aggregation mismatch')
        errors.append(recorded['delta_s_per_source'])
    if abs(row['paired_mean_delta_s_per_source']-statistics.mean(errors))>1e-8:raise ValueError('Cached objective mismatch')
    if row['all_checks_pass']!=all(r['hard_checks_pass'] for r in row['cases']):raise ValueError('Cached feasibility mismatch')
    signature=hashlib.sha256('|'.join(r['action_sha256'] for r in row['cases']).encode()).hexdigest()
    if row.get('behavior_signature')!=signature:raise ValueError('Behavior signature mismatch')


def evaluation(parameters,cases,base,label):
    expected=binding(cases,label,parameters)
    path=OUT/'bound_cache'/label/(b.key(parameters)+'.json')
    if path.exists():
        cache=b.load(path)
        if cache['binding']!=expected or cache['data_sha256']!=digest(cache['data']):raise ValueError('Candidate cache binding changed')
        row=cache['data'];cached=True
    else:
        legacy=OUT/('cache_'+label)/(b.key(parameters)+'.json')
        if legacy.exists():
            row=b.load(legacy);cached=True
            provenance={'kind':'Explicit migration from this unchanged frozen benchmark run','legacy_path':legacy.relative_to(b.ROOT).as_posix(),'legacy_sha256':b.core.original.sha(legacy)}
        else:
            row=b.core.evaluate(parameters,cases,base);cached=False
            row['behavior_signature']=hashlib.sha256('|'.join(r['action_sha256'] for r in row['cases']).encode()).hexdigest()
            row['created_at']=b.core.original.now()
            provenance={'kind':'New evaluation under bound-cache wrapper'}
            b.save(legacy,row)
        validate_row(row,parameters,cases,base)
        b.save(path,{'binding':expected,'data':row,'data_sha256':digest(row),'provenance':provenance})
    validate_row(row,parameters,cases,base)
    return row,cached


def check_existing():
    for label in ('training','validation'):
        cases=b.load(OUT/(label+'_cases.json'))
        paths=list((OUT/('cache_'+label)).glob('*.json'))
        if not paths:continue
        base=baseline(cases,label)
        for path in paths:
            row=b.load(path)
            if path.stem!=b.key(row['parameters']):raise ValueError('Legacy parameter cache key mismatch')
            evaluation(row['parameters'],cases,base,label)
    for path in (OUT/'studies').glob('*/result.json'):
        study=b.load(path);history=study['history']
        if len(history)!=48 or study['proposals']!=48:raise ValueError('Completed study budget mismatch')
        best=min(history,key=lambda h:(h['objective'],h['proposal']))
        if study['best']!=best:raise ValueError('Study best row mismatch')
        for row in history:
            cache=b.load(OUT/'bound_cache/training'/(b.key(row['parameters'])+'.json'))['data']
            if abs(row['objective']-b.objective(cache))>1e-8:raise ValueError('Study history cached score mismatch')
    # Negative controls: a truncated or reordered baseline/candidate must fail.
    cases=b.load(OUT/'training_cases.json');base=baseline(cases,'training')
    failure=False
    try:validate_baseline(base[:-1],cases)
    except ValueError:failure=True
    assert failure
    example=next((OUT/'cache_training').glob('*.json'))
    row=b.load(example)
    wrong=list(row['parameters']);wrong[0]=999.
    failure=False
    try:validate_row(row,wrong,cases,base)
    except ValueError:failure=True
    assert failure
    failure=False
    try:binding(list(reversed(cases)),'training')
    except ValueError:failure=True
    assert failure
    b.save(OUT/'cache_binding_audit.json',{'passed':True,'at':b.core.original.now(),
        'bound_training_caches':len(list((OUT/'bound_cache/training').glob('*.json'))),
        'negative_controls':['truncated baseline rejected','wrong parameter tuple rejected','reordered case set rejected'],
        'samplers_objectives_parameters_and_case_sets_unchanged':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run','confirm'])
    parser.add_argument('--method',choices=b.METHODS)
    parser.add_argument('--seed',type=int,choices=b.SEEDS)
    args=parser.parse_args()
    b.validate()
    addendum={'scope':'CACHE_BINDING_CORRECTION_ONLY','original_plan_sha256':b.core.original.sha(OUT/'plan.json'),
              'wrapper_sha256':b.core.original.sha(Path(__file__)),'no_objective_or_sampler_change':True,
              'tie_rule':'For equal feasible validation objective within 1e-9 s/source, prefer the smaller MOVE_SLACK, then smaller gain threshold, then offset scale closest to 1. This is a conservative tie convention, not a claim of unique statistical optimum.'}
    path=OUT/'cache_binding_addendum.json'
    if path.exists():
        if b.load(path)!=addendum:raise ValueError('Cache binding addendum changed')
    else:b.save(path,addendum)
    b.baseline=baseline;b.get_evaluation=evaluation
    check_existing()
    if args.action=='run':
        for method in ([args.method] if args.method else b.METHODS):
            for seed in ([args.seed] if args.seed else b.SEEDS):b.experiment(method,seed)
    elif args.action=='confirm':b.confirm()
    print('Bound-cache checks completed',flush=True)


if __name__=='__main__':main()
