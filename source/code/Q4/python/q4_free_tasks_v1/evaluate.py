from pathlib import Path
import sys,json,time,statistics,math,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'code/Q4/python/q4_final_integration_20260912'),str(ROOT/'code/shared')]
import policy,bench
spec=importlib.util.spec_from_file_location('frozen_j06',ROOT/'code/Q4/python/q4_joint_parameters_v1/selected_entry.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
OUT=ROOT/'results/Q4/experiments/free_tasks_v1'
def run(case,fn):
 arena=bench.TrackedArena([bench.Source(**{k:s[k] for k in ('channel','x','y','radius','heading')}) for s in case['sources']],case['seed'],case['error_mode']);client=bench.SimulatorClient('offline-free-tasks',transport=arena,log_path=False);client.enter();start=time.perf_counter();state=fn(bench.PublicClient(client.measure,client.clear,lambda:client.virtual_time_s));cpu=time.perf_counter()-start;client.exit();metrics=bench.base_audit(case,state,arena.action_log)
 assert metrics['all_checks_pass'],state
 assert json.loads(json.dumps(bench.replay(fn,arena.action_log)))==json.loads(json.dumps(state))
 for event in state.get('free_tasks',{}).get('events',[]):
  proof=event['proof']
  if proof['kind']=='guaranteed_no_signal_range':
   source=next(s for s in case['sources'] if s['channel']==event['channel']);point=policy.ring22_points()[event['scan']]
   assert math.dist(point,(source['x'],source['y']))>1500
   assert proof['distance_lower_m']<=math.dist(point,(source['x'],source['y']))+1e-5
 return dict(state=state,actions=arena.action_log,metrics=metrics,cpu_s=cpu)
def stats(v):
 n=len(v);m=statistics.mean(v);sd=statistics.stdev(v);h={84:1.98895978,42:2.01954097}[n]*sd/math.sqrt(n)
 return dict(mean=m,sd=sd,mean_ci95=[m-h,m+h])
def stage(cases,name):
 rows=[]
 for i,case in enumerate(cases,1):
  a=run(case,baseline.run_search);b=run(case,policy.run_search)
  bench.save(OUT/name/'raw'/f"{case['case_id']}.json",dict(baseline=a,candidate=b))
  row=dict(case_id=case['case_id'],N=len(case['sources']),baseline=a['metrics']['costs']['total_s']/len(case['sources']),candidate=b['metrics']['costs']['total_s']/len(case['sources']),actions_identical=a['actions']==b['actions'],diagnostics={k:v for k,v in b['state']['free_tasks'].items() if k not in ('events','proofs')},events=len(b['state']['free_tasks']['events']),cpu_s=b['cpu_s'],baseline_costs=a['metrics']['costs'],candidate_costs=b['metrics']['costs']);rows.append(row)
  if i%7==0:print(name,i,flush=True)
 av=np.array([r['baseline'] for r in rows]);bv=np.array([r['candidate'] for r in rows]);a=stats(av.tolist());b=stats(bv.tolist());rng=np.random.default_rng(20260913);idx=rng.integers(0,len(rows),size=(30000,len(rows)));g=1-bv[idx].mean(axis=1)/av[idx].mean(axis=1)
 result=dict(cases=len(rows),sources=sum(r['N'] for r in rows),baseline=a,candidate=b,gain=1-b['mean']/a['mean'],gain_paired_bootstrap95=np.quantile(g,[.025,.975]).tolist(),bootstrap_seed=20260913,bootstrap_replicates=30000,all_core_and_replay_pass=True,all_actions_identical=all(r['actions_identical'] for r in rows),totals={k:sum(r['diagnostics'][k] for r in rows) for k in rows[0]['diagnostics']},max_cpu_s=max(r['cpu_s'] for r in rows),cost_delta_s_per_source={k:statistics.mean((r['candidate_costs'][k]-r['baseline_costs'][k])/r['N'] for r in rows) for k in rows[0]['baseline_costs']},rows=rows,mean_and_ci_improve=b['mean']<a['mean'] and b['mean_ci95'][1]<a['mean_ci95'][1])
 bench.save(OUT/name/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);return result

def main():
 assert not (OUT/'inputs.json').exists(),'No repeat'
 paths=[ROOT/'results/Q4/experiments/final_integration_20260912/independent42/cases.json',ROOT/'results/Q4/experiments/fallback_grid_v1/fresh42/cases.json'];validation=ROOT/'results/Q4/experiments/short_step_v1/fresh42/cases.json'
 cases=sum((bench.read(p) for p in paths),[]);assert len(cases)==len({bench.physical(c) for c in cases})==84
 bench.save(OUT/'inputs.json',dict(cases=84,validation_cases=42,previously_used=True,inputs=[bench.ref(p) for p in paths+[validation]+list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('plan.md')]],baseline='frozen J06',extra_measurements=0,tail_gate=False))
 train=stage(cases,'development')
 if train['mean_and_ci_improve']:
  tests=bench.read(validation);assert not ({bench.physical(c) for c in cases}&{bench.physical(c) for c in tests});stage(tests,'validation')
if __name__=='__main__':main()
