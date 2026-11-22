"""Sequential, paired-case Bayesian parameter optimization using only NumPy.

Synthetic evaluator owns hidden source state; the Q3 policy receives the normal
public-action client. New official-simulator cases are not used as paired cases.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time
import numpy as np

import q3_practice_campaign as original
from evaluate_q3_robust_parallax_development import run, contains_point
from offline_arena import Source
import q3_cost_rollout_v2 as helper
import search_policy_q3_cost_rollout_v2 as policy
import search_policy_q3_combined as baseline

ROOT=original.ROOT
OUT=ROOT/'results/q3_parameter_optimizer'
GRID=list(itertools.product(range(0,251,25), np.arange(0.,30.01,2.5).tolist(), [.5,.75,1.,1.25,1.5,1.75,2.]))
LOW=np.array([0.,0.,.5]);SPAN=np.array([250.,30.,1.5])


def save(path,obj): original.save(path,obj)
def load(path): return original.load(path)


def configure(values):
    slack,gain,scale=map(float,values)
    assert 0<=slack<=250 and 0<=gain<=30 and .5<=scale<=2
    helper.MOVE_SLACK=slack
    helper.MIN_PREDICTED_SAVING=gain
    helper.OFFSETS=tuple(scale*x for x in (.1,.25,.5))
    helper.BACKOFFS=(.25,.5)
    helper.MAX_HYPOTHESES=9
    helper.RECEPTION_LIMIT=999.9
    helper.BIASES=(-1.,0.,1.)
    return policy.run_search


def kernel(a,b,length):
    distance=np.linalg.norm((a[:,None,:]-b[None,:,:])/length,axis=2)
    z=math.sqrt(5)*distance
    return (1+z+z*z/3)*np.exp(-z)


def gp_predict(x,y,targets):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); targets=np.asarray(targets,dtype=float)
    center=float(y.mean());scale=max(float(y.std()),.1)
    z=(y-center)/scale
    fitted=[]
    for length in (.15,.3,.6,1.2):
        covariance=kernel(x,x,length)+np.eye(len(x))*1e-4
        chol=np.linalg.cholesky(covariance)
        alpha=np.linalg.solve(chol.T,np.linalg.solve(chol,z))
        nll=.5*float(z@alpha)+float(np.log(np.diag(chol)).sum())
        fitted.append((nll,length,chol,alpha))
    _,length,chol,alpha=min(fitted,key=lambda item:item[0])
    cross=kernel(x,targets,length)
    mean=center+scale*(cross.T@alpha)
    v=np.linalg.solve(chol,cross)
    sd=scale*np.sqrt(np.maximum(1.-np.sum(v*v,axis=0),1e-12))
    return mean,sd,length


def expected_improvement(mean,sd,best):
    advantage=best-mean-.01
    z=advantage/sd
    cdf=np.array([.5*(1+math.erf(float(v)/math.sqrt(2))) for v in z])
    density=np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    return np.maximum(advantage*cdf+sd*density,0.)


def suggest(rows):
    observed={tuple(r['parameters']) for r in rows}
    available=np.asarray([p for p in GRID if p not in observed],dtype=float)
    feasible=[r for r in rows if r['all_checks_pass']]
    if not feasible:
        raise RuntimeError('No safe initial point available to fit optimizer')
    x=(np.asarray([r['parameters'] for r in feasible])-LOW)/SPAN
    y=np.asarray([r['paired_mean_delta_s_per_source'] for r in feasible])
    mean,sd,length=gp_predict(x,y,(available-LOW)/SPAN)
    acquisition=expected_improvement(mean,sd,min(0.,float(y.min())))
    index=max(range(len(available)),key=lambda i:(acquisition[i],sd[i],-i))
    return available[index].tolist(),{'rule':'maximum_expected_improvement','predicted_delta_s_per_source':float(mean[index]),
            'predictive_sd':float(sd[index]),'expected_improvement':float(acquisition[index]),'matern_length_scale':length,
            'previous_observations':len(rows),'previous_safe_observations':len(feasible),'best_paired_delta_before':min(0.,float(y.min()))}


def make_cases(count,seed_base,label):
    cases=[]
    for i in range(count):
        rng=random.Random(seed_base+i)
        n=10+i%7
        sources=[]
        for ch in rng.sample(range(1,21),n):
            angle=rng.uniform(0,2*math.pi)
            distance=1800. if i%7==0 else 1800.*math.sqrt(rng.random())
            radius=1000. if i%5==0 else 1500. if i%5==1 else rng.uniform(1000.,1500.)
            sources.append(asdict(Source(ch,distance*math.cos(angle),distance*math.sin(angle),radius,None)))
        cases.append({'question':3,'split':label,'case_id':f'{label}_{seed_base+i}','seed':seed_base+i,
                      'error_mode':['fixed_hash','positive_edge','negative_edge'][i%3], 'sources':sources})
    return cases


def initialize():
    if (OUT/'plan.json').exists(): raise RuntimeError('Optimization plan already exists')
    original.validate()
    training=make_cases(21,931100,'bo_training')
    confirmation=make_cases(42,941100,'bo_finalist_confirmation')
    initial=[(100,5,1),(100,0,1),(100,15,1),(50,5,1),(0,0,.5),(250,30,2),(0,30,2),(250,0,.5),
             (125,15,.75),(50,25,1.5),(200,5,1.75),(175,22.5,1.25)]
    inputs=[{'path':p.relative_to(ROOT).as_posix(),'sha256':original.sha(p)} for p in original.dependency_paths(['q3_parameter_optimizer'])]
    plan={'created_at':original.now(),'scope':'PAIRED_SYNTHETIC_BAYESIAN_PARAMETER_SEARCH_NOT_GUI_RESULTS',
          'supersedes':'Remaining fixed five-profile rotation after user explicitly requested an optimization algorithm; retain first 30 GUI cases and the original plan.',
          'search_grid_size':len(GRID),'parameter_order':['MOVE_SLACK_m','MIN_PREDICTED_SAVING_s','OFFSET_SCALE'],
          'bounds':[[0,250],[0,30],[.5,2]], 'grid_steps':[25,2.5,.25],
          'fixed':{'reception_limit_m':999.9,'backoffs':[.25,.5],'hypotheses':9,'biases':[-1,0,1],'coverage_routing_and_clear_fallback':'unchanged'},
          'configuration_budget':48,'initial_points':initial,'training_cases':21,'finalist_confirmation_cases':42,
          'objective':'Minimize arithmetic mean paired (candidate T/N - frozen baseline T/N) on the same 21 cases. Every case must pass full-clear, accounting and geometry checks.',
          'surrogate':'Gaussian process, Matern 5/2 covariance, length-scale selected by marginal likelihood from [.15,.3,.6,1.2], standardized target, diagonal jitter 1e-4.',
          'acquisition':'Maximum expected improvement over min(0, best observed safe paired delta), 0.01 s exploration offset; never repeat a configuration.',
          'finalists':'Three safe configurations with lowest training objective; evaluate all on the same fresh 42 cases. Confirmation is part of selection, not a final performance claim.',
          'gui_budget':'The already executed 30 cases remain. Preserve a total of 100 tuning GUI cases, followed by 20 new cases with frozen winner; allocate remaining 70 prospectively after offline finalists are known.',
          'not_claimed':['global optimality outside bounded grid','synthetic scores equal GUI performance','final independent validation already passed'],
          'references':['https://papers.nips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html'], 'inputs':inputs}
    save(OUT/'training_cases.json',training);save(OUT/'confirmation_cases.json',confirmation)
    plan['training_cases_sha256']=original.sha(OUT/'training_cases.json')
    plan['confirmation_cases_sha256']=original.sha(OUT/'confirmation_cases.json')
    save(OUT/'plan.json',plan)
    print(json.dumps({'plan':str(OUT/'plan.json'),'grid_size':len(GRID)},ensure_ascii=False))


def validate():
    plan=load(OUT/'plan.json')
    for item in plan['inputs']:
        if original.sha(ROOT/item['path'])!=item['sha256']: raise RuntimeError('Optimizer input changed: '+item['path'])
    for label in ('training','confirmation'):
        assert original.sha(OUT/(label+'_cases.json'))==plan[label+'_cases_sha256']
    return plan


def evaluate(values,cases,base_rows):
    controller=configure(values)
    results=[]
    for case,base in zip(cases,base_rows):
        metrics,state,actions=run(controller,case)
        errors=[]
        source_map={s['channel']:(s['x'],s['y']) for s in case['sources']}
        for event in state.get('cost_rollout_events',[]):
            if not contains_point(event['before_polygon'],source_map[event['channel']],tolerance=1e-5): errors.append('source absent from prior polygon')
            if event['used_alternative']:
                if max(math.dist(event['selected_point'],v) for v in event['before_polygon'])>999.9+1e-7: errors.append('reception certificate')
                if event['response'].get('measure_result')=='no_signal': errors.append('reception lost')
                if math.dist(event['selected_point'],event['current_position']) > math.dist(event['original_center'],event['current_position'])+values[0]+1e-7: errors.append('movement limit')
                if event['original_score_s']-event['selected_score_s'] < values[1]-1e-7: errors.append('adoption gate')
        results.append({'case_id':case['case_id'],'metrics':metrics,'hard_checks_pass':metrics['all_checks_pass'] and not errors,
                        'errors':errors,'delta_s_per_source':metrics['mean_s']-base['mean_s'],
                        'time_ratio':metrics['time_s']/base['time_s'],
                        'selected_count':sum(e['used_alternative'] for e in state.get('cost_rollout_events',[])),
                        'action_sha256':hashlib.sha256(json.dumps(actions,sort_keys=True).encode()).hexdigest()})
    return {'parameters':list(values),'all_checks_pass':all(r['hard_checks_pass'] for r in results),
            'paired_mean_delta_s_per_source':statistics.mean(r['delta_s_per_source'] for r in results),
            'mean_s_per_source':statistics.mean(r['metrics']['mean_s'] for r in results),
            'worst_case_ratio':max(r['time_ratio'] for r in results),'cases':results}


def baselines(cases,label):
    path=OUT/(label+'_baseline.json')
    if path.exists(): return load(path)
    results=[]
    for case in cases:
        metrics,state,actions=run(baseline.run_search,case)
        assert metrics['all_checks_pass']
        results.append(metrics)
    save(path,results)
    return results


def optimize():
    plan=validate()
    cases=load(OUT/'training_cases.json')
    base=baselines(cases,'training')
    rows=[load(p) for p in sorted((OUT/'evaluations').glob('*.json'))]
    while len(rows)<plan['configuration_budget']:
        index=len(rows)
        if index<len(plan['initial_points']):
            values=plan['initial_points'][index];suggestion={'rule':'predeclared_initial_design','previous_observations':index}
        else: values,suggestion=suggest(rows)
        assert tuple(values) not in {tuple(r['parameters']) for r in rows}
        save(OUT/'suggestions'/f'{index+1:03}.json',{'parameters':values,**suggestion,'created_at':original.now()})
        start=time.perf_counter()
        row=evaluate(values,cases,base)
        row.update(iteration=index+1,suggestion=suggestion,evaluation_wall_s=time.perf_counter()-start)
        save(OUT/'evaluations'/f'{index+1:03}.json',row)
        rows.append(row)
        safe=[r for r in rows if r['all_checks_pass']]
        best=min(safe,key=lambda r:r['paired_mean_delta_s_per_source']) if safe else None
        save(OUT/'progress.json',{'evaluated_configurations':len(rows),'planned':plan['configuration_budget'],'best':None if best is None else {k:best[k] for k in ('parameters','paired_mean_delta_s_per_source','mean_s_per_source','worst_case_ratio')},'updated_at':original.now()})
        print(json.dumps({'iteration':index+1,'parameters':values,'safe':row['all_checks_pass'],'paired_delta':row['paired_mean_delta_s_per_source'],'best_delta':None if best is None else best['paired_mean_delta_s_per_source'],'rule':suggestion['rule']},ensure_ascii=False),flush=True)
    validate()
    finalists=sorted([r for r in rows if r['all_checks_pass']],key=lambda r:(r['paired_mean_delta_s_per_source'],r['worst_case_ratio']))[:3]
    save(OUT/'finalists_before_confirmation.json',{'selected_at':original.now(),'finalists':[{'parameters':r['parameters'],'training_iteration':r['iteration'],'training_delta':r['paired_mean_delta_s_per_source']} for r in finalists]})
    fresh=load(OUT/'confirmation_cases.json')
    base_fresh=baselines(fresh,'confirmation')
    checked=[]
    for i,candidate in enumerate(finalists,1):
        path=OUT/'confirmation'/f'candidate_{i}.json'
        row=load(path) if path.exists() else evaluate(candidate['parameters'],fresh,base_fresh)
        save(path,row);checked.append(row)
        print(json.dumps({'confirmed_candidate':i,'parameters':row['parameters'],'safe':row['all_checks_pass'],'paired_delta':row['paired_mean_delta_s_per_source'],'mean':row['mean_s_per_source']},ensure_ascii=False),flush=True)
    save(OUT/'result.json',{'scope':plan['scope'],'evaluated_configurations':len(rows),'all_training_cases_same_across_configs':True,'training_baseline_mean':statistics.mean(r['mean_s'] for r in base),
                           'confirmation_baseline_mean':statistics.mean(r['mean_s'] for r in base_fresh),'finalists':checked,'plan_sha256':original.sha(OUT/'plan.json'),
                           'next_step':'Use confirmation-safe finalists in a prospectively allocated remaining GUI budget. Do not substitute synthetic averages for 20-case GUI validation.'})


def selfcheck():
    # A known convex function checks minimization direction and numerical stability.
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[.5,.5,.5]],dtype=float)
    y=np.sum((x-.25)**2,axis=1)
    mean,sd,length=gp_predict(x,y,x)
    assert np.max(np.abs(mean-y))<.01 and np.all(sd>0)
    ei=expected_improvement(np.array([-2.,2.]),np.array([.1,.1]),0.)
    assert ei[0]>1.9 and ei[1]<1e-10
    x2=np.vstack([x,x[0]])
    gp_predict(x2,np.r_[y,y[0]],x)
    rows=[{'parameters':[100.,5.,1.],'all_checks_pass':True,'paired_mean_delta_s_per_source':-3.},
          {'parameters':[50.,5.,1.],'all_checks_pass':True,'paired_mean_delta_s_per_source':-1.}]
    first,details=suggest(rows);second,_=suggest(rows)
    assert first==second and first not in [r['parameters'] for r in rows] and tuple(first) in GRID
    save(OUT/'optimizer_selfcheck.json',{'passed':True,'checks':['GP interpolation','positive posterior variance','duplicate point Cholesky stability','EI chooses improvements for minimization','deterministic non-repeating bounded proposal'],'max_interpolation_error':float(np.max(np.abs(mean-y)))})
    print('Optimizer numerical selfchecks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','check','optimize'])
    args=parser.parse_args()
    if args.action=='init': initialize()
    elif args.action=='check': selfcheck()
    else: optimize()
