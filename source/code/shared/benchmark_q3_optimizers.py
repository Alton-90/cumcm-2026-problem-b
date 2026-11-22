"""Equal-query-budget comparison of GP-EI, multivariate TPE, and DE.

Each optimizer receives identical initial points within each seed, identical
paired training cases and 48 total proposals. Cached duplicates still consume
proposal budget. Fresh confirmation cases are excluded from all optimizers.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time
import warnings

# Preserve the policy's already verified numerical runtime; add only missing
# optimizer packages from the project-local installation.
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/q3_optimizer_deps'))
import optuna
from scipy.optimize import differential_evolution
import scipy
import q3_parameter_optimizer as core
from evaluate_q3_robust_parallax_development import run

OUT=ROOT/'results/q3_optimizer_benchmark'
SEEDS=[20260911,20260912,20260913]
METHODS=['gp_ei','tpe','differential_evolution']
SCALE=np.array([25.,2.5,.25]);OFFSET=np.array([0.,0.,.5])


def save(path,data):core.save(path,data)
def load(path):return core.load(path)
def key(parameters):return hashlib.sha256(json.dumps(list(map(float,parameters))).encode()).hexdigest()[:20]
def vector(index):return (np.asarray(index,dtype=float)*SCALE+OFFSET).tolist()
def indices(parameters):return np.rint((np.asarray(parameters)-OFFSET)/SCALE).astype(int).tolist()


def balanced_cases(count,seed,label):
    # Each seven-case block contains all N=10..16. Geometry/range categories
    # are assigned by block, so boundary cases are not tied only to N=10.
    import math
    from dataclasses import asdict
    from offline_arena import Source
    cases=[]
    for i in range(count):
        rng=random.Random(seed+i); n=10+i%7; block=i//7
        sources=[]
        for ch in rng.sample(range(1,21),n):
            angle=rng.uniform(0,2*math.pi)
            distance=1800. if block%4==0 else 1800*math.sqrt(rng.random())
            radius=1000. if block%3==0 else 1500. if block%3==1 else rng.uniform(1000,1500)
            sources.append(asdict(Source(ch,distance*math.cos(angle),distance*math.sin(angle),radius,None)))
        cases.append({'question':3,'split':label,'case_id':f'{label}_{seed+i}','seed':seed+i,
                      'error_mode':['fixed_hash','positive_edge','negative_edge'][(block+i%7)%3],'sources':sources})
    return cases


def initialize():
    if (OUT/'plan.json').exists():raise RuntimeError('Existing benchmark plan will not be overwritten')
    train=balanced_cases(42,961100,'optimizer_benchmark_training')
    held=balanced_cases(84,971100,'optimizer_benchmark_selection_validation')
    save(OUT/'training_cases.json',train);save(OUT/'validation_cases.json',held)
    initial={}
    for seed in SEEDS:
        rng=random.Random(seed)
        points=[(100.,5.,1.)]
        points.extend(rng.sample([p for p in core.GRID if p!=points[0]],11))
        initial[str(seed)]=points
    dependencies=core.original.dependency_paths(['benchmark_q3_optimizers'])
    plan={'created_at':core.original.now(),'scope':'OFFLINE_OPTIMIZER_BENCHMARK_AND_PARAMETER_SELECTION',
          'methods':METHODS,'seeds':SEEDS,'proposal_budget_per_method_seed':48,'common_initial_points_per_seed':initial,
          'training_cases':42,'validation_cases':84,'validation_role':'Select among optimization winners; not a final independent performance estimate. GUI final 20 remain separate.',
          'duplicate_rule':'Every parameter proposal, including cached repeated parameters, consumes one of 48 slots. Unique configurations and physical synthetic executions are separately counted.',
          'same_seed_initial_design':True,'objective':'Mean of paired candidate-minus-baseline seconds/source over the same 42 training cases; any hard-check failure receives 1000000+failed-case-count.',
          'tpe':{'implementation':'Optuna TPESampler','multivariate':True,'n_startup_trials':12,'n_ei_candidates':64},
          'differential_evolution':{'implementation':'scipy.optimize.differential_evolution','strategy':'best1bin','population':12,'generations':3,'mutation':[.5,1.],'recombination':.7,'polish':False,'integrality':[True,True,True],'updating':'deferred'},
          'gp_ei':'q3_parameter_optimizer.gp_predict and expected_improvement; same Matérn model, marginal likelihood and EI rule as verified pilot',
          'versions':{'numpy':np.__version__,'optuna':optuna.__version__,'scipy':scipy.__version__,'python':sys.version},
          'inputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':core.original.sha(p)} for p in dependencies],
          'training_sha256':core.original.sha(OUT/'training_cases.json'),'validation_sha256':core.original.sha(OUT/'validation_cases.json'),
          'references':['https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html']}
    save(OUT/'plan.json',plan)
    print(json.dumps({'methods':METHODS,'seeds':SEEDS,'total_parameter_proposals':432,'train_cases':42,'validation_cases':84},ensure_ascii=False))


def validate():
    plan=load(OUT/'plan.json')
    for item in plan['inputs']:
        if core.original.sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Frozen benchmark input changed: '+item['path'])
    assert core.original.sha(OUT/'training_cases.json')==plan['training_sha256']
    assert core.original.sha(OUT/'validation_cases.json')==plan['validation_sha256']
    assert [np.__version__,optuna.__version__,scipy.__version__]==[plan['versions'][k] for k in ['numpy','optuna','scipy']]
    return plan


def baseline(cases,label):
    path=OUT/(label+'_baseline.json')
    if path.exists():return load(path)
    rows=[]
    for case in cases:
        metrics,_,_=run(core.baseline.run_search,case)
        assert metrics['all_checks_pass']
        rows.append(metrics)
    save(path,rows)
    return rows


def get_evaluation(parameters,cases,base,label):
    path=OUT/('cache_'+label)/(key(parameters)+'.json')
    if path.exists():return load(path),True
    row=core.evaluate(parameters,cases,base)
    row['behavior_signature']=hashlib.sha256('|'.join(r['action_sha256'] for r in row['cases']).encode()).hexdigest()
    row['created_at']=core.original.now()
    save(path,row)
    return row,False


def objective(row):
    if row['all_checks_pass']:return row['paired_mean_delta_s_per_source']
    return 1000000.+sum(not r['hard_checks_pass'] for r in row['cases'])


def experiment(method,seed):
    plan=validate()
    folder=OUT/'studies'/f'{method}_{seed}'
    if (folder/'result.json').exists():return load(folder/'result.json')
    cases=load(OUT/'training_cases.json');base=baseline(cases,'training')
    initial=plan['common_initial_points_per_seed'][str(seed)]
    history=[];rows=[]
    start=time.perf_counter()
    def evaluate(parameters):
        if len(history)>=48:raise RuntimeError('Optimizer exceeded equal proposal budget')
        parameters=list(map(float,parameters))
        row,cached=get_evaluation(parameters,cases,base,'training')
        value=objective(row)
        history.append({'proposal':len(history)+1,'parameters':parameters,'objective':value,'safe':row['all_checks_pass'],
                        'cache_hit':cached,'configuration_key':key(parameters),'behavior_signature':row['behavior_signature']})
        rows.append(row)
        save(folder/'progress.json',{'method':method,'seed':seed,'proposals':len(history),'best_objective':min(h['objective'] for h in history),'history':history})
        if len(history)%5==0 or len(history)==48:
            print(json.dumps({'method':method,'seed':seed,'proposals':len(history),'best_delta':min(h['objective'] for h in history)},ensure_ascii=False),flush=True)
        return value
    if method=='gp_ei':
        for parameters in initial:evaluate(parameters)
        while len(history)<48:
            parameters,_=core.suggest(rows)
            evaluate(parameters)
    elif method=='tpe':
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',optuna.exceptions.ExperimentalWarning)
            sampler=optuna.samplers.TPESampler(seed=seed,multivariate=True,n_startup_trials=12,n_ei_candidates=64)
        study=optuna.create_study(direction='minimize',sampler=sampler)
        distributions={name:optuna.distributions.IntDistribution(0,high) for name,high in zip(['slack','gain','offset'],[10,12,6])}
        for parameters in initial:
            value=evaluate(parameters)
            study.add_trial(optuna.trial.create_trial(params=dict(zip(distributions,indices(parameters))),distributions=distributions,value=value))
        while len(history)<48:
            trial=study.ask()
            idx=[trial.suggest_int(name,0,high) for name,high in zip(distributions,[10,12,6])]
            value=evaluate(vector(idx));study.tell(trial,value)
    elif method=='differential_evolution':
        result=differential_evolution(lambda x:evaluate(vector(np.rint(x).astype(int))),bounds=[(0,10),(0,12),(0,6)],
            strategy='best1bin',maxiter=3,popsize=4,tol=0.,atol=-1.,mutation=(.5,1.),recombination=.7,
            rng=np.random.default_rng(seed),polish=False,init=np.array([indices(p) for p in initial]),
            updating='deferred',workers=1,integrality=[True,True,True])
        assert result.nfev==48
    else:raise ValueError(method)
    assert len(history)==48
    best=min(history,key=lambda h:(h['objective'],h['proposal']))
    result={'method':method,'seed':seed,'proposals':48,'unique_parameters':len(set(h['configuration_key'] for h in history)),
            'distinct_behaviors_on_training':len(set(h['behavior_signature'] for h in history)),
            'best':best,'history':history,'wall_s':time.perf_counter()-start,'all_proposals_safe':all(h['safe'] for h in history)}
    save(folder/'result.json',result)
    return result


def confirm():
    validate()
    results=[load(OUT/'studies'/f'{method}_{seed}'/'result.json') for method in METHODS for seed in SEEDS]
    cases=load(OUT/'validation_cases.json');base=baseline(cases,'validation')
    checked=[]
    for result in results:
        row,cached=get_evaluation(result['best']['parameters'],cases,base,'validation')
        checked.append({'method':result['method'],'seed':result['seed'],'parameters':row['parameters'],
                        'training_delta':result['best']['objective'],'validation_delta':row['paired_mean_delta_s_per_source'],
                        'validation_mean':row['mean_s_per_source'],'safe':row['all_checks_pass'],'worst_case_ratio':row['worst_case_ratio'],
                        'validation_behavior_signature':row['behavior_signature']})
        print(json.dumps({'confirmation':checked[-1]},ensure_ascii=False),flush=True)
    aggregate={}
    for method in METHODS:
        part=[r for r in checked if r['method']==method]
        values=[r['validation_delta'] for r in part]
        aggregate[method]={'seeds':len(part),'all_safe':all(r['safe'] for r in part),'mean_validation_delta':statistics.mean(values),
                           'seed_sd':statistics.stdev(values),'worst_seed_delta':max(values)}
    unique={}
    for row in sorted(checked,key=lambda r:(not r['safe'],r['validation_delta'],r['worst_case_ratio'])):
        unique.setdefault(row['validation_behavior_signature'],row)
    save(OUT/'comparison.json',{'scope':'OFFLINE_OPTIMIZER_BENCHMARK_AND_PARAMETER_SELECTION','algorithms':aggregate,'winner_records':checked,
        'behavior_distinct_finalists':list(unique.values()),'training_baseline_mean':statistics.mean(x['mean_s'] for x in load(OUT/'training_baseline.json')),
        'validation_baseline_mean':statistics.mean(x['mean_s'] for x in base),'unique_training_configurations':len(list((OUT/'cache_training').glob('*.json'))),
        'actual_training_policy_runs':42*len(list((OUT/'cache_training').glob('*.json'))),'unique_validation_configurations':len(list((OUT/'cache_validation').glob('*.json'))),
        'gui_claims':False,'final_20_gui_validation_completed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','run','confirm'])
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='init':initialize()
    elif args.action=='confirm':confirm()
    else:
        for method in ([args.method] if args.method else METHODS):
            for seed in ([args.seed] if args.seed else SEEDS):experiment(method,seed)
