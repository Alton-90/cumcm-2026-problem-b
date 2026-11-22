"""Large, frozen, multi-fidelity Q3 parameter study; all execution is offline."""
from __future__ import annotations
import argparse,ast,hashlib,importlib,json,math,os,random,statistics,sys,time,threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'code/Q3/python'),str(ROOT/'code/shared')]
import numpy as np
import q3_low_source_v7 as policy
import search_policy_q3_combined as combined
import search_policy_q3_cost_rollout_strict84_selected as strict
from offline_arena import OfflineArena,Source
from simulator_client import SimulatorClient
from q3_shared_stop_v6 import certificate
OUT=ROOT/'results/Q3/experiments/low_source_v7_20260912'
LAYOUTS=('uniform','boundary','cluster','sector_edges','near_origin','two_clusters')
WORKERS=4

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def configuration_id(p):return canon(policy.parameters(p))[:16]
def dependencies():
 todo=[Path(__file__),ROOT/'code/Q3/python/q3_low_source_v7.py',ROOT/'code/shared/search_policy_q3_cost_rollout_strict84_selected.py'];found=set()
 while todo:
  p=todo.pop()
  if p in found:continue
  found.add(p)
  for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
   names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for name in names:
    for folder in (ROOT/'code/Q3/python',ROOT/'code/shared'):
     q=folder/(name.split('.')[0]+'.py')
     if q.is_file():todo.append(q);break
 return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(found)}

def make_cases(split):
 repeats={'training':2,'selection':3,'holdout':4}[split];start={'training':951000,'selection':952000,'holdout':953000}[split];result=[]
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
    result.append({'case_id':f'low_v7_{split}_{seed}','seed':seed,'N':n,'question':3,'layout':layout,'repeat':rep,'error_mode':error,'radius_regime':regime,'sources':src})
 return result

def anchors():
 p=policy.DEFAULTS;out=[dict(p)]
 out += [dict(p,orientation_bins=x) for x in (4,8,12)]
 out += [dict(p,radius_cap_m=x) for x in (1240.,1400.,1560.)]
 out += [dict(p,scan_latency_weight=x,route_detour_cap_m=250.) for x in (.5,1.,2.,4.)]
 out += [dict(p,orientation_bins=8,radius_cap_m=1400.,scan_latency_weight=1.,route_detour_cap_m=250.)]
 return out

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);d=dependencies();pop={s:make_cases(s) for s in ('training','selection','holdout')}
 initial=[c['case_id'] for c in pop['training'] if c['repeat']==0 and LAYOUTS.index(c['layout']) in ((c['N']-10)%6,(c['N']-10+3)%6)]
 assert len(initial)==14 and len(set(initial))==14
 # Detect exact previously stored layouts; only evaluation reads source truth.
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
 hashes=[canon(c['sources']) for cs in pop.values() for c in cs]
 if len(hashes)!=len(set(hashes)) or any(h in historical for h in hashes):raise RuntimeError('Case collision across splits/history')
 plan={'schema_version':2,'artifact_type':'q3_low_source_v7_parameter_protocol','created_at':datetime.now(timezone.utc).isoformat(),
  'authorization':'用户要求针对低干扰源数新一轮优化，由助手设计大规模参数实验并寻找表现最好的一组参数。',
  'low_N':[10,11,12],'high_N':[13,14,15,16],'true_N_policy_input':False,
  'parameter_domains':policy.DOMAINS,'discrete_domain_size':math.prod(len(v) for v in policy.DOMAINS.values()),'protected_identity_parameters':policy.DEFAULTS,
  'dependencies':d,'case_hashes':{s:canon(cs) for s,cs in pop.items()},'case_counts':{s:len(cs) for s,cs in pop.items()},'initial_case_ids':initial,
  'search':{'unique_configurations':256,'sampler':'Optuna TPE','seed':20260912,'startup_trials':48,'batch_size':5,'workers':WORKERS,
    'max_tpe_proposals':1024,'after_limit':'Fill remaining unique configurations with preseeded uniform proposals',
    'isolation':'ProcessPoolExecutor only; each worker evaluates one policy at a time on its main thread; never concurrent policy threads; parameters restored after each call',
    'anchors':anchors(),'repeated_proposals':'Reuse only exact parameter/case/dependency cache; log every optimizer proposal',
    'successive_fidelity':'256 x 14 training; 32 best plus 8 diversity candidates x 84 training; five best plus protected identity x 126 selection; one locked configuration x 168 holdout',
    'maximum_unique_policy_case_budget':8064},
  'objective':'0.8*mean(low case candidate/base ratio)+0.2*mean(high ratio)+0.2*max(0,CVaR10 ratio-1.05)+5*max(0,high mean ratio-1.02)+5*max(0,worst ratio-1.15); any hard failure -> finite penalty >=1e6',
  'selection_rule':['full-clear/coverage/channel/time checks all pass','high-N equal-case mean T/N <= combined*1.02','worst case time ratio <=1.10','lowest low-N equal-case mean T/N','lower overall mean T/N','lexical configuration id'],
  'validation_gate':{'low_mean_reduction_vs_combined_at_least':.02,'low_not_worse_than_strict84':True,'low_paired_95pct_bootstrap_saving_lower_positive':True,
    'high_mean_ratio_vs_combined_at_most':1.02,'worst_case_ratio_at_most':1.10,'all_clear_and_continuous_coverage':True},
  'holdout_rule':'Do not inspect holdout scores or evaluate any additional configuration on holdout until one selection winner is saved; never retune from it.',
  'limitations':['Best among tested configurations, not a proof of the global best over all 160000 combinations.',
    'Synthetic model; no GUI/formal-test scores and no old 100+20 progress credit.','Full traces saved for selection/holdout and failures; training cache keeps metrics and deterministic action digests.']}
 target=OUT/'protocol.json'
 if target.exists():
  old=load(target)
  if old['dependencies']!=d or old['case_hashes']!=plan['case_hashes']:raise RuntimeError('Frozen protocol changed')
  return old
 for s,cs in pop.items():save(OUT/f'{s}_cases.json',cs)
 save(target,plan);return plan

def checked_plan():
 p=load(OUT/'protocol.json')
 if dependencies()!=p['dependencies']:raise RuntimeError('Source fingerprint changed after freeze')
 for split,h in p['case_hashes'].items():
  if canon(load(OUT/f'{split}_cases.json'))!=h:raise RuntimeError('Saved cases changed: '+split)
 return p

class Arena(OfflineArena):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.actions=[]
 def _error(self,ch,x,y):
  return math.sin((x+2*y)/400+ch) if self.error_mode=='spatial_smooth' else super()._error(ch,x,y)
 def post(self,path,payload):
  duplicate=payload['request_id'] in self.cache;before=self.virtual_time_s;out=super().post(path,payload)
  if not duplicate and path in ('/measure','/clear'):self.actions.append({'path':path,'point':[payload['position']['x'],payload['position']['y']],
    'channel':payload['channel'],'response':out.body,'before_time_s':before,'after_time_s':self.virtual_time_s})
  return out
class PublicClient:
 __slots__=('__client',)
 def __init__(self,c):self.__client=c
 @property
 def virtual_time_s(self):return self.__client.virtual_time_s
 def measure(self,x,y,ch):return self.__client.measure(x,y,ch)
 def clear(self,x,y,ch):return self.__client.clear(x,y,ch)

def run_one(method,p,c,plan,retain):
 if threading.current_thread() is not threading.main_thread():raise RuntimeError('Policy evaluation must run serially on worker main thread')
 rollout_keys=('MOVE_SLACK','MIN_PREDICTED_SAVING','OFFSETS','BACKOFFS','MAX_HYPOTHESES','RECEPTION_LIMIT','BIASES')
 before_globals=tuple(getattr(policy.rollout,k) for k in rollout_keys)
 arena=Arena([Source(**x) for x in c['sources']],c['seed'],c['error_mode']);client=SimulatorClient('offline-low-source-v7',transport=arena,log_path=False);client.enter()
 error=None;state={};t=time.perf_counter();cpu=time.process_time()
 try:state=combined.run_search(PublicClient(client)) if method=='combined' else strict.run_search(PublicClient(client)) if method=='strict84' else policy.run_search(PublicClient(client),parameters_override=p)
 except Exception as e:error=f'{type(e).__name__}: {e}'
 wall=time.perf_counter()-t;cpu=time.process_time()-cpu
 globals_restored=before_globals==tuple(getattr(policy.rollout,k) for k in rollout_keys)
 if not client.ended:client.exit()
 n=c['N'];cleared=len(arena.cleared);cc=set(state.get('cleared_channels',[]));excluded=set(state.get('excluded_channels',[]));unresolved=set(state.get('unresolved_channels',[]))
 angle=math.radians(state.get('coverage_rotation_deg',0.));ca,sa=math.cos(angle),math.sin(angle);negative={ch:[] for ch in range(1,21)}
 pos=(0.,0.);channel=1;clock=0.;clock_ok=True;last_clear=0.
 for a in arena.actions:
  clock+=math.dist(pos,a['point'])/5;pos=a['point'];ch=a['channel'];reply=a['response']
  if a['path']=='/measure':
   clock+=5+(channel!=ch);channel=ch
   if reply.get('measure_result')=='no_signal':
    x,y=a['point'];pt=(x*ca+y*sa,-x*sa+y*ca)
    if pt not in negative[ch]:negative[ch].append(pt)
  else:
   success=reply.get('clear_result')=='success';clock+=5 if success else 3
   if success:last_clear=a['after_time_s']
  clock_ok &= math.isclose(clock,a['after_time_s'],abs_tol=1e-6)
 cover=cleared==16 or all(certificate(region,tuple(negative[ch]))[0] for ch in excluded for region in range(7))
 partition=cc==arena.cleared and not (excluded&set(arena.sources)) and not unresolved and not (cc&excluded) and cc|excluded==set(range(1,21))
 ok=bool(cleared==n and partition and cover and clock_ok and state.get('completed') and state.get('stopped') is False and globals_restored and not error)
 m={'all_hard_checks':ok,'globals_restored':globals_restored,'worker_pid':os.getpid(),'worker_main_thread':True,'full_clear':cleared==n,'continuous_coverage':cover,'channel_partition':partition,'clock_check':bool(clock_ok),
  'N':n,'cleared':cleared,'time_s':arena.virtual_time_s,'s_per_source':arena.virtual_time_s/n,
  'move_s_per_source':arena.distance_m/5/n,'measure_s_per_source':arena.measure_count*5/n,'switch_s_per_source':arena.switch_count/n,
  'success_clear_s_per_source':cleared*5/n,'failed_clear_s_per_source':(arena.clear_count-cleared)*3/n,
  'distance_m':arena.distance_m,'measures':arena.measure_count,'switches':arena.switch_count,'failed_clears':arena.clear_count-cleared,
  'tail_after_last_clear_s':arena.virtual_time_s-last_clear if cleared==n else None,'wall_s':wall,'cpu_s':cpu,
  'rotation_deg':state.get('coverage_rotation_deg',0.),'route_changes':len(state.get('route_diagnostics',[])), 'error':error,'warnings':state.get('warnings',[])}
 row={'case_id':c['case_id'],'case_hash':canon(c),'method':method,'parameters':p,'dependencies_hash':canon(plan['dependencies']),
      'metrics':m,'actions_hash':canon(arena.actions)}
 if retain or not ok:row.update(actions=arena.actions,state=state)
 return row

def batch_job(job):
 method,p,split,ids,retain=job;plan=checked_plan();ident=method if method!='candidate' else configuration_id(p)
 cases={c['case_id']:c for c in load(OUT/f'{split}_cases.json')};rows=[];executed=reused=0
 for case_id in ids:
  c=cases[case_id];path=OUT/'cache'/ident/(case_id+'.json')
  if path.exists():
   row=load(path)
   if row['case_hash']!=canon(c) or row['dependencies_hash']!=canon(plan['dependencies']) or row['parameters']!=p or row['method']!=method:raise RuntimeError('Unbound cache reuse')
   if retain and 'actions' not in row:raise RuntimeError('Requested full evidence missing; do not silently rerun')
   reused+=1
  else:row=run_one(method,p,c,plan,retain);save(path,row);executed+=1
  rows.append(row['metrics'])
 return {'id':ident,'rows':rows,'executed':executed,'reused':reused}

def objective(rows,base):
 ratios=np.array([r['time_s']/b['time_s'] for r,b in zip(rows,base)]);low=[i for i,r in enumerate(rows) if r['N']<=12];high=[i for i,r in enumerate(rows) if r['N']>=13]
 if not all(r['all_hard_checks'] for r in rows):return 1e6+1000*sum(r['N']-r['cleared'] for r in rows)+float(ratios.mean())
 lo=float(ratios[low].mean());hi=float(ratios[high].mean());tail=float(np.sort(ratios)[-max(1,math.ceil(.1*len(ratios))):].mean())
 return .8*lo+.2*hi+.2*max(0,tail-1.05)+5*max(0,hi-1.02)+5*max(0,float(ratios.max())-1.15)

def evaluate_jobs(pool,jobs):
 futures=[pool.submit(batch_job,j) for j in jobs]
 return [f.result() for f in futures]

def baselines(pool,split):
 ids=[c['case_id'] for c in load(OUT/f'{split}_cases.json')]
 return {m:v['rows'] for m,v in zip(('combined','strict84'),evaluate_jobs(pool,[(m,None,split,ids,split!='training') for m in ('combined','strict84')]))}

def optuna_module():
 try:import optuna
 except ModuleNotFoundError:
  # NumPy is already imported from the primary environment before adding this runtime.
  for path in [ROOT/'.runtime/q3_optimizer_deps']:
   if (path/'optuna').is_dir():sys.path.append(str(path));break
  import optuna
 optuna.logging.set_verbosity(optuna.logging.WARNING);return optuna

def stage1(pool,plan,base):
 optuna=optuna_module();study=optuna.create_study(sampler=optuna.samplers.TPESampler(seed=20260912,n_startup_trials=48))
 for p in plan['search']['anchors']:study.enqueue_trial(p)
 history_file=OUT/'optimizer_history.json';history=load(history_file) if history_file.exists() else []
 def ask():
  t=study.ask();p={k:t.suggest_categorical(k,v) for k,v in policy.DOMAINS.items()};return t,p
 # Replay complete ask batches followed by ordered tells to preserve TPE RNG/state on resume.
 registry={};extra_rng=random.Random(7032026)
 for batch in history:
  asked=[]
  for item in batch['trials']:
   if item.get('uniform_enqueued') is not None:study.enqueue_trial(item['uniform_enqueued'])
   asked.append(ask())
  for (t,p),item in zip(asked,batch['trials']):
   if p!=item['parameters']:raise RuntimeError('Optimizer replay mismatch')
  for (t,p),item in zip(asked,batch['trials']):
   study.tell(t,item['objective']);registry[configuration_id(p)]={'parameters':p,'stage1_objective':item['objective']}
 cases=load(OUT/'training_cases.json');lookup={c['case_id']:i for i,c in enumerate(cases)};ids=plan['initial_case_ids'];b=[base['combined'][lookup[i]] for i in ids]
 proposals=sum(len(x['trials']) for x in history)
 if history and history[-1].get('extra_rng_state'):
  raw=history[-1]['extra_rng_state'];extra_rng.setstate((raw[0],tuple(raw[1]),raw[2]))
 while len(registry)<256:
  count=min(5,256-len(registry));asked=[];enqueues=[]
  for _ in range(count):
   enqueued=None
   if proposals>=1024:
    while True:
     p={k:extra_rng.choice(v) for k,v in policy.DOMAINS.items()}
     if configuration_id(p) not in registry:break
    study.enqueue_trial(p);enqueued=p
   enqueues.append(enqueued);asked.append(ask());proposals+=1
  new={configuration_id(p):p for _,p in asked if configuration_id(p) not in registry}
  vals=evaluate_jobs(pool,[('candidate',p,'training',ids,False) for p in new.values()])
  for p,v in zip(new.values(),vals):registry[v['id']]={'parameters':p,'stage1_objective':objective(v['rows'],b)}
  batch={'batch':len(history),'trials':[],'extra_rng_state':extra_rng.getstate()}
  for (t,p),enqueued in zip(asked,enqueues):
   score=registry[configuration_id(p)]['stage1_objective'];study.tell(t,score);batch['trials'].append({'trial':t.number,'parameters':p,'objective':score,'uniform_enqueued':enqueued})
  history.append(batch);save(history_file,history);save(OUT/'stage1_registry.json',registry)
  print(json.dumps({'stage':'TPE14','unique_configs':len(registry),'target':256,'proposals':proposals,'best_objective':min(v['stage1_objective'] for v in registry.values())}),flush=True)
 return registry

def diversity_distance(a,b):
 return sum(abs(policy.DOMAINS[k].index(a[k])-policy.DOMAINS[k].index(b[k]))/max(1,len(policy.DOMAINS[k])-1) for k in policy.DOMAINS)

def stage2(pool,reg,base):
 path=OUT/'stage2_selection.json'
 if path.exists():selected=load(path)['ids']
 else:
  order=sorted(reg,key=lambda k:(reg[k]['stage1_objective'],k));identity=configuration_id(policy.DEFAULTS)
  selected=[identity]+[k for k in order if k!=identity][:31]
  for _ in range(8):
   remaining=[k for k in order if k not in selected]
   k=max(remaining,key=lambda k:(min(diversity_distance(reg[k]['parameters'],reg[j]['parameters']) for j in selected),-reg[k]['stage1_objective'],k));selected.append(k)
  save(path,{'ids':selected,'rule':'identity+31 best, then 8 max-min parameter diversity outside the retained set'})
 ids=[c['case_id'] for c in load(OUT/'training_cases.json')];rows=[]
 for start in range(0,len(selected),5):
  group=selected[start:start+5];vals=evaluate_jobs(pool,[('candidate',reg[k]['parameters'],'training',ids,False) for k in group])
  for k,v in zip(group,vals):rows.append({'id':k,'parameters':reg[k]['parameters'],'training_objective':objective(v['rows'],base['combined']),'summary':summary(v['rows'],base['combined'])})
  save(OUT/'stage2_results.json',rows);print(json.dumps({'stage':'training84','completed':len(rows),'total':len(selected),'best_objective':min(x['training_objective'] for x in rows)}),flush=True)
 return rows

def summary(rows,base):
 ratios=[r['time_s']/b['time_s'] for r,b in zip(rows,base)];low=[i for i,r in enumerate(rows) if r['N']<=12];high=[i for i,r in enumerate(rows) if r['N']>=13]
 lowmean=statistics.mean(rows[i]['s_per_source'] for i in low);highmean=statistics.mean(rows[i]['s_per_source'] for i in high)
 bl=statistics.mean(base[i]['s_per_source'] for i in low);bh=statistics.mean(base[i]['s_per_source'] for i in high)
 hard=all(r['all_hard_checks'] for r in rows)
 return {'cases':len(rows),'all_hard_checks':hard,'cleared':sum(r['cleared'] for r in rows),'sources':sum(r['N'] for r in rows),
  'low_mean':lowmean,'high_mean':highmean,'mean':statistics.mean(r['s_per_source'] for r in rows),'low_reduction':1-lowmean/bl,'high_ratio':highmean/bh,
  'worst_case_ratio':max(ratios),'p95':float(np.quantile([r['s_per_source'] for r in rows],.95)),
  'selection_safe':hard and highmean<=bh*1.02 and max(ratios)<=1.10,
  'by_N':{str(n):{'mean':statistics.mean(r['s_per_source'] for r in rows if r['N']==n),'base_mean':statistics.mean(r['s_per_source'] for r in base if r['N']==n)} for n in range(10,17)},
  'low_components':{k:statistics.mean(rows[i][k] for i in low) if all(rows[i][k] is not None for i in low) else None for k in ('move_s_per_source','measure_s_per_source','switch_s_per_source','success_clear_s_per_source','failed_clear_s_per_source','tail_after_last_clear_s','cpu_s')} }

def select_final(pool,training):
 selected_file=OUT/'selected_parameters.json'
 if selected_file.exists():return load(selected_file)
 identity=configuration_id(policy.DEFAULTS);order=sorted(training,key=lambda x:(not x['summary']['selection_safe'],x['training_objective'],x['id']))
 finalists=[next(x for x in training if x['id']==identity)]+[x for x in order if x['id']!=identity][:5]
 save(OUT/'selection_candidates.json',[{'id':x['id'],'parameters':x['parameters']} for x in finalists])
 base=baselines(pool,'selection');ids=[c['case_id'] for c in load(OUT/'selection_cases.json')];ranked=[]
 vals=evaluate_jobs(pool,[('candidate',x['parameters'],'selection',ids,True) for x in finalists])
 for x,v in zip(finalists,vals):
  s=summary(v['rows'],base['combined']);ranked.append({'id':x['id'],'parameters':x['parameters'],'summary':s})
  save(OUT/'selection_results.json',ranked);print(json.dumps({'stage':'selection126','completed':len(ranked),'total':6,'id':x['id'],'low_mean':s['low_mean'],'safe':s['selection_safe']}),flush=True)
 feasible=[x for x in ranked if x['summary']['selection_safe']]
 best=min(feasible,key=lambda x:(x['summary']['low_mean'],x['summary']['mean'],x['id'])) if feasible else next(x for x in ranked if x['id']==identity)
 selected={'schema_version':2,'artifact_type':'q3_low_source_v7_locked_parameters','selected_before_holdout':True,'selected_at':datetime.now(timezone.utc).isoformat(),
  'configuration':best,'selection_feasible_count':len(feasible),'fallback_to_identity':not feasible,
  'source_hash':sha(ROOT/'code/Q3/python/q3_low_source_v7.py'),'protocol_sha256':sha(OUT/'protocol.json'),'selection_results_sha256':sha(OUT/'selection_results.json')}
 save(selected_file,selected);return selected

def holdout(pool,selected):
 if selected['protocol_sha256']!=sha(OUT/'protocol.json') or selected['source_hash']!=sha(ROOT/'code/Q3/python/q3_low_source_v7.py'):raise RuntimeError('Changed locked winner')
 base=baselines(pool,'holdout');ids=[c['case_id'] for c in load(OUT/'holdout_cases.json')];p=selected['configuration']['parameters']
 rows=evaluate_jobs(pool,[('candidate',p,'holdout',ids,True)])[0]['rows'];s=summary(rows,base['combined']);low=[i for i,r in enumerate(rows) if r['N']<=12]
 delta=np.array([base['combined'][i]['s_per_source']-rows[i]['s_per_source'] for i in low]);rng=np.random.default_rng(70260912);ix=rng.integers(0,len(low),size=(20000,len(low)));ci=np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist()
 strict_low=statistics.mean(base['strict84'][i]['s_per_source'] for i in low)
 checks={'all_hard':s['all_hard_checks'],'low_reduction_at_least_2pct':s['low_reduction']>=.02,'low_not_worse_than_strict84':s['low_mean']<=strict_low,
  'paired_low_95pct_lower_positive':ci[0]>0,'high_mean_ratio_at_most_1_02':s['high_ratio']<=1.02,'worst_case_ratio_at_most_1_10':s['worst_case_ratio']<=1.10}
 result={'schema_version':2,'artifact_type':'q3_low_source_v7_independent_holdout','scope':'NEW_SYNTHETIC_HELDOUT_CASES','parameters':p,'winner_id':configuration_id(p),
  'selected_parameters_sha256':sha(OUT/'selected_parameters.json'),'checks':checks,'accepted':all(checks.values()),'candidate':s,
  'combined':summary(base['combined'],base['combined']),'strict84':summary(base['strict84'],base['combined']),
  'low_paired_saving_95pct_interval':ci,'rows':[{'case_id':cid,'candidate':a,'combined':b,'strict84':c} for cid,a,b,c in zip(ids,rows,base['combined'],base['strict84'])]}
 save(OUT/'holdout_result.json',result);print(json.dumps({'stage':'holdout168','parameters':p,'accepted':result['accepted'],'checks':checks,'low_mean':s['low_mean'],'high_mean':s['high_mean'],'low_reduction':s['low_reduction']}),flush=True)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--search',action='store_true');parser.add_argument('--worker-count',type=int,choices=(4,),default=4);args=parser.parse_args()
 if args.prepare:
  p=prepare();print(json.dumps({'prepared':p['case_counts'],'parameter_domain':p['discrete_domain_size'],'budget':p['search']['maximum_unique_policy_case_budget']}));return
 if not args.search:parser.print_help();return
 plan=checked_plan();t=time.monotonic()
 with ProcessPoolExecutor(max_workers=WORKERS) as pool:
  base=baselines(pool,'training');reg=stage1(pool,plan,base);training=stage2(pool,reg,base);selected=select_final(pool,training);result=holdout(pool,selected)
 checked_plan()
 files=list((OUT/'cache').glob('*/*.json'));counts={}
 for path in files:
  d=load(path);key=('candidate' if d['method']=='candidate' else d['method']);counts[key]=counts.get(key,0)+1
 save(OUT/'run_summary.json',{'schema_version':2,'artifact_type':'q3_low_source_v7_search_run','completed':True,'elapsed_this_invocation_s':time.monotonic()-t,
   'unique_configurations':len(reg),'policy_case_counts':counts,'total_cached_policy_cases':len(files),'selected_parameters':selected,'holdout_accepted':result['accepted'],
   'protocol_sha256':sha(OUT/'protocol.json'),'holdout_result_sha256':sha(OUT/'holdout_result.json'),'official_simulator_runs':0})

# Child processes inherit the same explicit network prohibition; no subprocess UI or simulator.
def no_network(event,args):
 if event in ('socket.connect','socket.bind','socket.sendto','socket.getaddrinfo'):raise RuntimeError('Offline tuning forbids network')
sys.addaudithook(no_network)
if __name__=='__main__':main()
