"""Frozen final Q3 validation: 84 fresh paired cases and 42 correlated-error stress rows."""
from __future__ import annotations
import argparse,ast,hashlib,json,math,os,random,statistics,sys,time,threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'code/Q3/python'),str(ROOT/'code/shared')]
import numpy as np
import q3_final_main as main_policy
import q3_final_baseline as baseline_policy
import search_policy_q3_cost_rollout_strict84_selected as strict84
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
from verify_low_source_v7 import independent_cover,close
from q3_shared_stop_v6 import certificate
OUT=ROOT/'results/Q3/experiments/final_integration_20260912'
LAYOUTS=('uniform','boundary','cluster','sector_edges','near_origin','two_clusters')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def local_sources(starts):
 todo=list(starts);found=set()
 while todo:
  p=todo.pop()
  if p in found:continue
  found.add(p)
  for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
   names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else []
   for name in names:
    for folder in (ROOT/'code/Q3/python',ROOT/'code/shared'):
     q=folder/(name.split('.')[0]+'.py')
     if q.is_file():todo.append(q);break
 return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(found)}
def make_cases(split):
 repeats=2;start=961000;result=[]
 for li,layout in enumerate(LAYOUTS):
  for ni in range(7):
   for rep in range(repeats):
    i=li*7*repeats+ni*repeats+rep;seed=start+i;g=random.Random(seed);n=10+ni
    error=('fixed_hash','positive_edge','negative_edge','spatial_smooth')[(li+ni+rep)%4]
    regime=('min1000','max1500','uniform')[(rep+ni)%3];rotation=g.uniform(0,2*math.pi);src=[]
    for j,ch in enumerate(g.sample(range(1,21),n)):
     a=g.uniform(0,2*math.pi);d=1800*math.sqrt(g.random());x,y=d*math.cos(a),d*math.sin(a)
     if layout=='boundary':d=g.uniform(1770,1800);x,y=d*math.cos(a),d*math.sin(a)
     elif layout=='sector_edges':
      a=(j%6+.5)*math.pi/3+(rotation if rep%2 else 0)+g.uniform(-.002,.002);d=g.choice((999.,1000.1,1799.5));x,y=d*math.cos(a),d*math.sin(a)
     elif layout in ('cluster','two_clusters'):
      center=1000. if layout=='cluster' else (1250. if j%2 else -1250.);d=(430. if layout=='cluster' else 250.)*math.sqrt(g.random());x=center*math.cos(rotation)+d*math.cos(a);y=center*math.sin(rotation)+d*math.sin(a)
     elif layout=='near_origin' and j<4:d=(0.,4.9,19.9,999.9)[j];x,y=d*math.cos(a),d*math.sin(a)
     radius=1000. if regime=='min1000' else 1500. if regime=='max1500' else g.uniform(1000,1500)
     src.append({'channel':ch,'x':float(x),'y':float(y),'radius':radius,'heading':None})
    result.append({'case_id':f'q3_final_{split}_{seed}','seed':seed,'N':n,'question':3,'layout':layout,'repeat':rep,'error_mode':error,'radius_regime':regime,'sources':src})
 return result

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 sources=local_sources([Path(main_policy.__file__),Path(baseline_policy.__file__)])
 lock={'version':main_policy.VERSION,'method_id':main_policy.METHOD_ID,'parameters':main_policy.PARAMETERS,'policy_sources':sources,'baseline_method':baseline_policy.METHOD_ID,'causal_boundary':'Only public measurement/clear feedback and virtual time; one serial policy per process.'}
 if main_policy.LOCK_PATH.exists():
  if load(main_policy.LOCK_PATH)!=lock:raise RuntimeError('Final code lock already exists and changed')
 else:save(main_policy.LOCK_PATH,lock)
 cases=make_cases('validation');stress=[]
 for c in cases:
  if c['repeat']==0:stress.append(dict(c,case_id=c['case_id'].replace('_validation_','_stress_'),error_mode='spatial_smooth_stress',physical_parent=c['case_id']))
 assert len(cases)==84 and len(stress)==42
 historical=set()
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('sources'),list) and x['sources']:historical.add(canon(x['sources']))
   else:
    for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for path in (ROOT/'results').rglob('*cases*.json'):
  if OUT in path.parents:continue
  try:walk(load(path))
  except (ValueError,OSError):continue
 if any(canon(c['sources']) in historical for c in cases):raise RuntimeError('Fresh final case collides with historical layout')
 dependencies=local_sources([Path(__file__),ROOT/'code/Q3/python/run_q3_final_practice.py'])
 dependencies[main_policy.LOCK_PATH.relative_to(ROOT).as_posix()]=sha(main_policy.LOCK_PATH)
 plan={'experiment':'final_integration_20260912','scope':'84 new physical layouts, 42 reused-layout stress cases; no official run in this command',
  'created_at':datetime.now(timezone.utc).isoformat(),'selected_method':main_policy.METHOD_ID,'baseline':baseline_policy.METHOD_ID,'parameters':main_policy.PARAMETERS,
  'code_lock_sha256':sha(main_policy.LOCK_PATH),'dependencies':dependencies,'case_hashes':{'validation':canon(cases),'stress':canon(stress)},'case_counts':{'validation':84,'stress':42},
  'selection':'strict84 selected from saved historical and low-source evidence; no combination trial justified; no tuning in final cases',
  'thresholds':{'mean_reduction_strictly_greater_than':.01,'worst_case_ratio_at_most':1.10,'p95_time_ratio_at_most':1.05,'worst_stratum_mean_ratio_at_most':1.05,'high_N_mean_ratio_at_most':1.02,'paired_bootstrap_lower_positive':True,'stress_mean_not_worse':True,'full_clear_coverage_clock_causal_boundary':True},
  'bootstrap':{'seed':55092026,'replicates':20000,'unit':'paired case T/N; selected single method only'},'workers':2,
  'cost_contract':'distance/5 + 5*measurements + switches + 3*clear_attempts + 2*successful_clears; clear never retunes measurement channel',
  'official_testing':'Separate practice-only entry after coordinator exclusive slot; formal scored runs excluded from current authorization.'}
 target=OUT/'protocol.json'
 if target.exists():
  previous=load(target)
  if previous['dependencies']!=dependencies or previous['case_hashes']!=plan['case_hashes']:raise RuntimeError('Frozen final protocol changed')
  return previous
 save(OUT/'validation_cases.json',cases);save(OUT/'stress_cases.json',stress);save(target,plan);return plan

def checked_plan():
 p=load(OUT/'protocol.json');main_policy.validate_code_lock()
 for name,digest in p['dependencies'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('Changed final dependency '+name)
 for split,digest in p['case_hashes'].items():
  if canon(load(OUT/f'{split}_cases.json'))!=digest:raise RuntimeError('Changed final cases')
 return p

class Arena(OfflineArena):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.actions=[]
 def _error(self,ch,x,y):
  if self.error_mode=='spatial_smooth':return math.sin((x+2*y)/400+ch)
  if self.error_mode=='spatial_smooth_stress':return math.sin((2*x-y)/350+ch)
  return super()._error(ch,x,y)
 def post(self,path,payload):
  duplicate=payload['request_id'] in self.cache;before=self.virtual_time_s;r=super().post(path,payload)
  if not duplicate and path in ('/measure','/clear'):self.actions.append(dict(path=path,point=[payload['position']['x'],payload['position']['y']],channel=payload['channel'],response=r.body,before_time_s=before,after_time_s=self.virtual_time_s))
  return r
class PublicClient:
 __slots__=('__client',)
 def __init__(self,c):self.__client=c
 @property
 def virtual_time_s(self):return self.__client.virtual_time_s
 def measure(self,x,y,ch):return self.__client.measure(x,y,ch)
 def clear(self,x,y,ch):return self.__client.clear(x,y,ch)

def audit_trace(row,case):
 errors=[];actions=row['actions'];state=row['state'];m=row['metrics'];sources={s['channel']:s for s in case['sources']}
 if canon(actions)!=row['actions_hash']:errors.append('actions_hash')
 position=(0.,0.);tuned=1;clock=distance=0.;measures=switches=clears=failed=0;cleared=set();last=0.;negative={ch:[] for ch in range(1,21)}
 a=math.radians(state.get('coverage_rotation_deg',0.));co,si=math.cos(a),math.sin(a)
 for i,action in enumerate(actions):
  x,y=action['point'];ch=action['channel'];reply=action['response']
  if not close(action['before_time_s'],clock):errors.append('before_clock');break
  step=math.dist(position,(x,y));distance+=step;clock+=step/5;position=(x,y)
  src=sources.get(ch) if ch not in cleared else None
  d=math.hypot(src['x']-x,src['y']-y) if src else math.inf
  if action['path']=='/clear':
   clears+=1;success=d<=20;clock+=5 if success else 3
   if success:cleared.add(ch);last=clock
   else:failed+=1
   if reply.get('clear_result')!=('success' if success else 'no_target_in_range'):errors.append('clear_response');break
  elif action['path']=='/measure':
   measures+=1;switches+=ch!=tuned;clock+=5+(ch!=tuned);tuned=ch
   result='no_signal' if src is None or d>src['radius'] else 'near' if d<=5 else 'direction'
   if reply.get('measure_result')!=result:errors.append('measure_response');break
   if result=='direction':
    mode=case['error_mode']
    if mode=='spatial_smooth_stress':error=math.sin((2*x-y)/350+ch)
    elif mode=='spatial_smooth':error=math.sin((x+2*y)/400+ch)
    elif mode in ('positive_edge','negative_edge'):error=1. if mode=='positive_edge' else -1.
    else:
     key=f"{case['seed']}:{ch}:{float(x).hex()}:{float(y).hex()}".encode();v=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big');error=2*v/(2**64-1)-1
    bearing=round((math.degrees(math.atan2(src['y']-y,src['x']-x))+error)%360,2)%360
    if not close(reply.get('svd_deg'),bearing):errors.append('bearing_response');break
   if result=='no_signal':
    pt=(x*co+y*si,-x*si+y*co)
    if pt not in negative[ch]:negative[ch].append(pt)
  else:errors.append('invalid_action');break
  if reply.get('accepted') is not True or not close(reply.get('virtual_time_s'),clock) or not close(action['after_time_s'],clock):errors.append('after_clock');break
 n=case['N'];excluded=set(state.get('excluded_channels',[]))
 if cleared!=set(sources) or cleared!=set(state.get('cleared_channels',[])) or cleared&excluded or cleared|excluded!=set(range(1,21)) or state.get('unresolved_channels'):errors.append('partition_or_full_clear')
 if not (state.get('completed') is True and state.get('stopped') is False):errors.append('termination')
 if len(cleared)!=16 and not independent_cover({ch:negative[ch] for ch in excluded}):errors.append('independent_analytic_cover')
 if len(cleared)!=16 and not all(certificate(region,tuple(negative[ch]))[0] for ch in excluded for region in range(7)):errors.append('continuous_coverage')
 expected={'cleared':len(cleared),'time_s':clock,'s_per_source':clock/n,'distance_m':distance,'measures':measures,'switches':switches,'failed_clears':failed,
  'move_s_per_source':distance/5/n,'measure_s_per_source':5*measures/n,'switch_s_per_source':switches/n,'success_clear_s_per_source':5*len(cleared)/n,'failed_clear_s_per_source':3*failed/n}
 if len(cleared)==n:expected['tail_after_last_clear_s']=clock-last
 if any(not close(m.get(k),v) for k,v in expected.items()):errors.append('reconstructed_metrics')
 if row['method']=='candidate' and state.get('low_source_parameters')!=row['parameters']:errors.append('executed_parameters')
 return errors

def execute(method,c,plan):
 if threading.current_thread() is not threading.main_thread():raise RuntimeError('Main-thread serial execution required')
 arena=Arena([Source(**x) for x in c['sources']],c['seed'],c['error_mode']);client=SimulatorClient('offline-q3-final',transport=arena,log_path=False);client.enter()
 before=tuple(getattr(strict84.helper,k) for k in strict84._NAMES);state={};error=None;t=time.perf_counter();cpu=time.process_time()
 try:state=(main_policy if method=='main' else baseline_policy).run_search(PublicClient(client),question=3,method='main')
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 wall=time.perf_counter()-t;cpu=time.process_time()-cpu
 if not client.ended:client.exit()
 n=c['N'];cleared=len(arena.cleared);last=max((a['after_time_s'] for a in arena.actions if a['path']=='/clear' and a['response'].get('clear_result')=='success'),default=0.)
 restored=before==tuple(getattr(strict84.helper,k) for k in strict84._NAMES)
 metrics={'N':n,'cleared':cleared,'time_s':arena.virtual_time_s,'s_per_source':arena.virtual_time_s/n,'distance_m':arena.distance_m,'measures':arena.measure_count,'switches':arena.switch_count,'failed_clears':arena.clear_count-cleared,
  'move_s_per_source':arena.distance_m/5/n,'measure_s_per_source':arena.measure_count*5/n,'switch_s_per_source':arena.switch_count/n,'success_clear_s_per_source':cleared*5/n,'failed_clear_s_per_source':(arena.clear_count-cleared)*3/n,
  'tail_after_last_clear_s':arena.virtual_time_s-last if cleared==n else None,'wall_s':wall,'cpu_s':cpu,'globals_restored':restored,'worker_pid':os.getpid(),'worker_main_thread':True,'error':error}
 row={'case_id':c['case_id'],'case_hash':canon(c),'method':method,'parameters':main_policy.PARAMETERS if method=='main' else {},'dependency_hash':canon(plan['dependencies']),'metrics':metrics,'state':state,'actions':arena.actions,'actions_hash':canon(arena.actions)}
 audit_case=c
 errors=audit_trace(row,audit_case)
 if error or not restored:errors.append('policy_exception_or_global_restore')
 if method=='main' and (state.get('final_parameters')!=main_policy.PARAMETERS or state.get('final_code_lock_sha256')!=sha(main_policy.LOCK_PATH)):errors.append('final_entry_binding')
 metrics['all_hard_checks']=not errors;row['audit_errors']=errors
 return row

def batch(method,split):
 plan=checked_plan();rows=[]
 for c in load(OUT/f'{split}_cases.json'):
  path=OUT/'rows'/split/(c['case_id']+'__'+method+'.json')
  if path.exists():
   row=load(path)
   if row['case_hash']!=canon(c) or row['dependency_hash']!=canon(plan['dependencies']) or row['method']!=method:raise RuntimeError('Changed final cache binding')
  else:row=execute(method,c,plan);save(path,row)
  rows.append(row)
 return rows

def summarize(rows,cases):
 metrics=[r['metrics'] for r in rows]
 return {'cases':len(rows),'sources':sum(m['N'] for m in metrics),'cleared':sum(m['cleared'] for m in metrics),'all_hard_checks':all(m['all_hard_checks'] for m in metrics),
  'mean_s_per_source':statistics.mean(m['s_per_source'] for m in metrics),'mean_distance_m':statistics.mean(m['distance_m'] for m in metrics),'mean_distance_m_per_source':statistics.mean(m['distance_m']/m['N'] for m in metrics),
  'p95':float(np.quantile([m['s_per_source'] for m in metrics],.95)),'total_policy_cpu_s':sum(m['cpu_s'] for m in metrics),
  'components':{k:statistics.mean(m[k] for m in metrics) for k in ('move_s_per_source','measure_s_per_source','switch_s_per_source','success_clear_s_per_source','failed_clear_s_per_source')},
  'by_N':{str(n):statistics.mean(m['s_per_source'] for m in metrics if m['N']==n) for n in range(10,17)},
  'strata':{field:{str(v):statistics.mean(r['metrics']['s_per_source'] for r,c in zip(rows,cases) if c[field]==v) for v in sorted({c[field] for c in cases})} for field in ('layout','N','radius_regime','error_mode')},
  'low_mean':statistics.mean(m['s_per_source'] for m in metrics if m['N']<=12),'high_mean':statistics.mean(m['s_per_source'] for m in metrics if m['N']>=13)}

def run():
 plan=checked_plan();t=time.monotonic();datasets={};rows_by={}
 try:
  with ProcessPoolExecutor(max_workers=2) as pool:
   for split in ('validation','stress'):
    futures={m:pool.submit(batch,m,split) for m in ('main','baseline')};rows_by[split]={m:f.result() for m,f in futures.items()}
    cases=load(OUT/f'{split}_cases.json');summaries={m:summarize(rows,cases) for m,rows in rows_by[split].items()};a=summaries['main'];b=summaries['baseline']
    ratios=[x['metrics']['time_s']/y['metrics']['time_s'] for x,y in zip(rows_by[split]['main'],rows_by[split]['baseline'])]
    delta=np.array([y['metrics']['s_per_source']-x['metrics']['s_per_source'] for x,y in zip(rows_by[split]['main'],rows_by[split]['baseline'])]);rng=np.random.default_rng(55092026);ix=rng.integers(0,len(delta),size=(20000,len(delta)));ci=np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist()
    datasets[split]={'summaries':summaries,'reduction':1-a['mean_s_per_source']/b['mean_s_per_source'],'distance_reduction':1-a['mean_distance_m_per_source']/b['mean_distance_m_per_source'],'worst_case_ratio':max(ratios),'p95_ratio':a['p95']/b['p95'],'high_N_ratio':a['high_mean']/b['high_mean'],'worst_stratum_ratio':max(a['strata'][field][v]/b['strata'][field][v] for field in a['strata'] for v in a['strata'][field]),'saving_95pct_interval':ci,'all_hard':a['all_hard_checks'] and b['all_hard_checks']}
    print(json.dumps({'stage':split,'reduction':datasets[split]['reduction'],'worst_ratio':max(ratios),'all_hard':datasets[split]['all_hard']}),flush=True)
 except Exception as exc:
  save(OUT/'execution_failure.json',{'error':f'{type(exc).__name__}: {exc}','stage':locals().get('split'),'completed_rows':len(list((OUT/'rows').glob('*/*.json'))),'protocol_sha256':sha(OUT/'protocol.json'),'dependencies':plan['dependencies']});raise
 a=datasets['validation'];s=datasets['stress'];checks={'full_clear_and_evidence':a['all_hard'] and s['all_hard'],'improvement_gt_1pct':a['reduction']>.01,'paired_interval_positive':a['saving_95pct_interval'][0]>0,'worst_case_at_most_1_10':max(a['worst_case_ratio'],s['worst_case_ratio'])<=1.10,'p95_ratio_at_most_1_05':a['p95_ratio']<=1.05,'stratum_ratio_at_most_1_05':a['worst_stratum_ratio']<=1.05,'high_N_ratio_at_most_1_02':a['high_N_ratio']<=1.02,'stress_no_mean_loss':s['reduction']>=0}
 result={'experiment':'final_integration_20260912','code_lock_sha256':sha(main_policy.LOCK_PATH),'protocol_sha256':sha(OUT/'protocol.json'),'datasets':datasets,'checks':checks,'accepted_under_user_rule':all(checks.values()),'elapsed_s':time.monotonic()-t,'policy_cases':252,'official_practice_runs':0,'formal_runs':0}
 for method in ('main','baseline'):save(OUT/f'metrics/{method}.json',{'method_id':main_policy.METHOD_ID if method=='main' else baseline_policy.METHOD_ID,'validation':datasets['validation']['summaries'][method],'stress':datasets['stress']['summaries'][method],'code_lock_sha256':sha(main_policy.LOCK_PATH)})
 save(OUT/'metrics/comparison.json',result);independent_audit();print(json.dumps({'final_accepted':result['accepted_under_user_rule'],'checks':checks}),flush=True)

def independent_audit():
 plan=checked_plan();checks=[];counts=0
 for split in ('validation','stress'):
  for c in load(OUT/f'{split}_cases.json'):
   for method in ('main','baseline'):
    path=OUT/'rows'/split/(c['case_id']+'__'+method+'.json');row=load(path);ac=c
    errors=audit_trace(row,ac)
    if method=='main' and (row['state'].get('final_parameters')!=main_policy.PARAMETERS or row['state'].get('final_code_lock_sha256')!=sha(main_policy.LOCK_PATH)):errors.append('final_entry_binding')
    if row['case_hash']!=canon(c) or row['dependency_hash']!=canon(plan['dependencies']) or not row['metrics']['all_hard_checks']:errors.append('input_binding_or_hard_check')
    counts+=len(row['actions']);checks.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'passed':not errors,'errors':errors})
 bound=dict(plan['dependencies']);bound.update({x['path']:x['sha256'] for x in checks});bound.update({(OUT/f'{s}_cases.json').relative_to(ROOT).as_posix():sha(OUT/f'{s}_cases.json') for s in ('validation','stress')});bound[(OUT/'protocol.json').relative_to(ROOT).as_posix()]=sha(OUT/'protocol.json')
 comparison=load(OUT/'metrics/comparison.json')
 for split in ('validation','stress'):
  cases=load(OUT/f'{split}_cases.json');all_rows={m:[load(OUT/'rows'/split/(c['case_id']+'__'+m+'.json')) for c in cases] for m in ('main','baseline')}
  means={m:statistics.mean(r['metrics']['s_per_source'] for r in rs) for m,rs in all_rows.items()}
  d=comparison['datasets'][split]
  if any(not close(means[m],d['summaries'][m]['mean_s_per_source']) for m in means):raise RuntimeError('Final mean replay mismatch')
  ratios=[x['metrics']['time_s']/y['metrics']['time_s'] for x,y in zip(all_rows['main'],all_rows['baseline'])]
  if not close(max(ratios),d['worst_case_ratio']) or not close(1-means['main']/means['baseline'],d['reduction']):raise RuntimeError('Final paired summary replay mismatch')
  delta=np.array([y['metrics']['s_per_source']-x['metrics']['s_per_source'] for x,y in zip(all_rows['main'],all_rows['baseline'])]);rng=np.random.default_rng(55092026);ci=np.quantile(delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(axis=1),[.025,.975])
  if any(not close(float(x),y) for x,y in zip(ci,d['saving_95pct_interval'])):raise RuntimeError('Final interval replay mismatch')
 a=comparison['datasets']['validation'];st=comparison['datasets']['stress']
 expected={'full_clear_and_evidence':a['all_hard'] and st['all_hard'],'improvement_gt_1pct':a['reduction']>.01,'paired_interval_positive':a['saving_95pct_interval'][0]>0,'worst_case_at_most_1_10':max(a['worst_case_ratio'],st['worst_case_ratio'])<=1.10,'p95_ratio_at_most_1_05':a['p95_ratio']<=1.05,'stratum_ratio_at_most_1_05':a['worst_stratum_ratio']<=1.05,'high_N_ratio_at_most_1_02':a['high_N_ratio']<=1.02,'stress_no_mean_loss':st['reduction']>=0}
 if expected!=comparison['checks'] or all(expected.values())!=comparison['accepted_under_user_rule']:raise RuntimeError('Final adoption gate replay mismatch')
 result={'all_pass':len(checks)==252 and all(x['passed'] for x in checks),'policy_cases':len(checks),'actions_replayed':counts,'comparison_sha256':sha(OUT/'metrics/comparison.json'),'bound_inputs':bound,'checks':checks,'scope':'Independent public-response and complete-cost replay plus closed-form seven-sector cover; no policy reruns.'}
 save(OUT/'metrics/independent_audit.json',result)
 if not result['all_pass']:raise RuntimeError('Final independent audit failed')

def no_network(event,args):
 if event in ('socket.connect','socket.bind','socket.sendto','socket.getaddrinfo'):raise RuntimeError('Final offline evaluator forbids network')
sys.addaudithook(no_network)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if args.prepare:print(json.dumps(prepare()['case_counts']))
 elif args.run:run()
 else:parser.print_help()
