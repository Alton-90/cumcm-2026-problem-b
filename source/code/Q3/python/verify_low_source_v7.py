"""Independent saved-evidence audit; executes no search policy and no network."""
from pathlib import Path
import hashlib,json,math,statistics,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'code/Q3/python'),str(ROOT/'code/shared')]
import tune_low_source_v7 as ev
from q3_shared_stop_v6 import certificate
OUT=ev.OUT

def close(a,b):return isinstance(a,(int,float)) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6)
def equivalent(a,b):
 if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):return close(a,b)
 if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
 return a==b

def independent_cover(negative):
 # Separate analytic sufficient proof: origin disk plus six annular sectors.
 # For r in [1000,1800], theta in [-pi/6,pi/6], maximum distance to
 # an on-axis anchor d is attained at r endpoints and theta endpoints.
 for ch,points in negative.items():
  if not any(math.hypot(x,y)<1e-10 for x,y in points):return False
  for k in range(6):
   ux,uy=math.cos(k*math.pi/3),math.sin(k*math.pi/3);witness=False
   for x,y in points:
    d=x*ux+y*uy;off=abs(x*uy-y*ux)
    if 1124-1e-7<=d<=1730+1e-7 and off<=1e-7:
     maximum=max(math.sqrt(r*r+d*d-math.sqrt(3)*r*d)+off for r in (1000.,1800.))
     if maximum<=1000-1e-5:witness=True;break
   if not witness:return False
 return True

def rebuild_summary(rows,base):
 import numpy as np
 low=[r for r in rows if r['N']<=12];high=[r for r in rows if r['N']>=13]
 lo=statistics.mean(r['s_per_source'] for r in low);hi=statistics.mean(r['s_per_source'] for r in high)
 bl=statistics.mean(r['s_per_source'] for r in base if r['N']<=12);bh=statistics.mean(r['s_per_source'] for r in base if r['N']>=13)
 ratios=[r['time_s']/b['time_s'] for r,b in zip(rows,base)];hard=all(r['all_hard_checks'] for r in rows)
 return {'cases':len(rows),'all_hard_checks':hard,'cleared':sum(r['cleared'] for r in rows),'sources':sum(r['N'] for r in rows),'low_mean':lo,'high_mean':hi,'mean':statistics.mean(r['s_per_source'] for r in rows),'low_reduction':1-lo/bl,'high_ratio':hi/bh,'worst_case_ratio':max(ratios),'p95':float(np.quantile([r['s_per_source'] for r in rows],.95)),'selection_safe':hard and hi<=bh*1.02 and max(ratios)<=1.10,
 'by_N':{str(n):{'mean':statistics.mean(r['s_per_source'] for r in rows if r['N']==n),'base_mean':statistics.mean(r['s_per_source'] for r in base if r['N']==n)} for n in range(10,17)},
 'low_components':{k:statistics.mean(r[k] for r in low) if all(r[k] is not None for r in low) else None for k in ('move_s_per_source','measure_s_per_source','switch_s_per_source','success_clear_s_per_source','failed_clear_s_per_source','tail_after_last_clear_s','cpu_s')}}

def audit_trace(row,case):
 errors=[];actions=row['actions'];state=row['state'];m=row['metrics'];sources={s['channel']:s for s in case['sources']}
 if ev.canon(actions)!=row['actions_hash']:errors.append('actions_hash')
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
    if mode=='spatial_smooth':error=math.sin((x+2*y)/400+ch)
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

def main():
 plan=ev.checked_plan();run=ev.load(OUT/'run_summary.json');selection=ev.load(OUT/'selected_parameters.json');hold=ev.load(OUT/'holdout_result.json')
 cases={c['case_id']:c for split in plan['case_counts'] for c in ev.load(OUT/f'{split}_cases.json')}
 registry=ev.load(OUT/'stage1_registry.json');promoted=ev.load(OUT/'stage2_selection.json')['ids'];finalists=ev.load(OUT/'selection_candidates.json')
 if len(registry)!=256 or len(promoted)!=40 or len(finalists)!=6:raise RuntimeError('Stage populations changed')
 if ev.sha(OUT/'selected_parameters.json')!=hold['selected_parameters_sha256'] or ev.sha(OUT/'selection_results.json')!=selection['selection_results_sha256']:raise RuntimeError('Selection binding mismatch')
 if selection['configuration']['parameters']!=hold['parameters'] or not selection['selected_before_holdout']:raise RuntimeError('Winner changed')
 files=sorted((OUT/'cache').glob('*/*.json'));checks=[];rows={};counts={};actions=full=0;cpu=wall=0.
 for index,path in enumerate(files):
  r=ev.load(path);c=cases[r['case_id']];errors=[];ident=path.parent.name;split=r['case_id'].split('_')[2]
  if r['case_hash']!=ev.canon(c) or r['dependencies_hash']!=ev.canon(plan['dependencies']):errors.append('input_binding')
  if (ev.configuration_id(r['parameters']) if r['method']=='candidate' else r['method'])!=ident:errors.append('configuration_binding')
  if r['method']=='candidate':
   if split=='holdout' and ident!=hold['winner_id']:errors.append('holdout_contamination')
   elif split=='selection' and ident not in {x['id'] for x in finalists}:errors.append('selection_population')
   elif split=='training' and ident not in registry:errors.append('training_population')
  m=r['metrics'];components=sum(m[k] for k in ('move_s_per_source','measure_s_per_source','switch_s_per_source','success_clear_s_per_source','failed_clear_s_per_source'))
  if not close(components,m['s_per_source']):errors.append('component_sum')
  if not m['all_hard_checks'] or not m['globals_restored'] or not m['worker_main_thread']:errors.append('saved_hard_failure')
  if 'actions' in r:errors+=audit_trace(r,c);actions+=len(r['actions']);full+=1
  elif split!='training':errors.append('missing_required_trace')
  cpu+=m['cpu_s'];wall+=m['wall_s'];key=f"{r['method']}:{split}";counts[key]=counts.get(key,0)+1;rows[(ident,c['case_id'])]=m
  checks.append({'file':path.relative_to(ROOT).as_posix(),'sha256':ev.sha(path),'passed':not errors,'errors':errors,'trace_replayed':'actions' in r})
  if (index+1)%1000==0:print(json.dumps({'audit_rows':index+1,'total':len(files),'traces':full}),flush=True)
 if len(files)!=8064 or len(files)!=run['total_cached_policy_cases']:raise RuntimeError('Incomplete or excess policy-case budget')
 # Independently replay the whole fidelity pipeline from bound training metrics.
 import numpy as np
 def score(metrics,base):
  ratios=np.array([x['time_s']/y['time_s'] for x,y in zip(metrics,base)])
  if not all(x['all_hard_checks'] for x in metrics):return 1e6+1000*sum(x['N']-x['cleared'] for x in metrics)+float(ratios.mean())
  lo=statistics.mean(float(v) for v,x in zip(ratios,metrics) if x['N']<=12);hi=statistics.mean(float(v) for v,x in zip(ratios,metrics) if x['N']>=13)
  tail=float(np.sort(ratios)[-max(1,math.ceil(.1*len(ratios))):].mean())
  return .8*lo+.2*hi+.2*max(0,tail-1.05)+5*max(0,hi-1.02)+5*max(0,float(ratios.max())-1.15)
 initial=plan['initial_case_ids'];ib=[rows[('combined',cid)] for cid in initial]
 for ident,item in registry.items():
  if ev.configuration_id(item['parameters'])!=ident or not close(score([rows[(ident,cid)] for cid in initial],ib),item['stage1_objective']):raise RuntimeError('Stage1 score replay')
 history=ev.load(OUT/'optimizer_history.json');history_ids=set()
 for batch in history:
  for item in batch['trials']:
   ident=ev.configuration_id(item['parameters']);history_ids.add(ident)
   if ident not in registry or not close(item['objective'],registry[ident]['stage1_objective']):raise RuntimeError('Optimizer history binding')
 if history_ids!=set(registry):raise RuntimeError('Optimizer history population')
 order=sorted(registry,key=lambda k:(registry[k]['stage1_objective'],k));identity=ev.configuration_id(ev.policy.DEFAULTS)
 expected_promoted=[identity]+[k for k in order if k!=identity][:31]
 def diversity(a,b):return sum(abs(plan['parameter_domains'][k].index(a[k])-plan['parameter_domains'][k].index(b[k]))/(len(plan['parameter_domains'][k])-1) for k in plan['parameter_domains'])
 for _ in range(8):
  remaining=[k for k in order if k not in expected_promoted]
  chosen=max(remaining,key=lambda k:(min(diversity(registry[k]['parameters'],registry[j]['parameters']) for j in expected_promoted),-registry[k]['stage1_objective'],k));expected_promoted.append(chosen)
 if promoted!=expected_promoted:raise RuntimeError('40-group promotion mismatch')
 training=ev.load(OUT/'stage2_results.json');training_ids=[c['case_id'] for c in ev.load(OUT/'training_cases.json')];tb=[rows[('combined',cid)] for cid in training_ids]
 if len(training)!=40 or {x['id'] for x in training}!=set(promoted):raise RuntimeError('Stage2 population')
 for item in training:
  metrics=[rows[(item['id'],cid)] for cid in training_ids]
  if item['parameters']!=registry[item['id']]['parameters'] or not equivalent(rebuild_summary(metrics,tb),item['summary']) or not close(score(metrics,tb),item['training_objective']):raise RuntimeError('Stage2 summary or score mismatch')
 order=sorted(training,key=lambda x:(not x['summary']['selection_safe'],x['training_objective'],x['id']))
 expected_finalists=[next(x for x in training if x['id']==identity)]+[x for x in order if x['id']!=identity][:5]
 if finalists!=[dict(id=x['id'],parameters=x['parameters']) for x in expected_finalists]:raise RuntimeError('Six-group finalist rule mismatch')
 # Reconstruct selection scores from independently replayed full traces, then rank.
 selection_ids=[c['case_id'] for c in ev.load(OUT/'selection_cases.json')]
 selection_scores=ev.load(OUT/'selection_results.json');rebuilt=[]
 if {x['id'] for x in selection_scores}!={x['id'] for x in finalists} or len(selection_scores)!=6:raise RuntimeError('Selection score population')
 for item in selection_scores:
  ident=item['id'];metrics=[rows[(ident,cid)] for cid in selection_ids];base=[rows[('combined',cid)] for cid in selection_ids]
  summary=rebuild_summary(metrics,base)
  if not equivalent(summary,item['summary']):raise RuntimeError('Selection summary replay mismatch: '+ident)
  if item['parameters']!=registry[ident]['parameters'] or ev.configuration_id(item['parameters'])!=ident:raise RuntimeError('Selection parameters mismatch')
  rebuilt.append(dict(id=ident,parameters=item['parameters'],summary=summary))
 eligible=[x for x in rebuilt if x['summary']['selection_safe']]
 expected=min(eligible,key=lambda x:(x['summary']['low_mean'],x['summary']['mean'],x['id'])) if eligible else next(x for x in rebuilt if x['id']==ev.configuration_id(ev.policy.DEFAULTS))
 if not equivalent(expected,selection['configuration']) or selection['selection_feasible_count']!=len(eligible) or selection['fallback_to_identity']!=bool(not eligible):raise RuntimeError('Predefined selection winner replay failed')
 for name,ident in [('candidate',hold['winner_id']),('combined','combined'),('strict84','strict84')]:
  ids=[c['case_id'] for c in ev.load(OUT/'holdout_cases.json')]
  rebuilt_hold=rebuild_summary([rows[(ident,cid)] for cid in ids],[rows[('combined',cid)] for cid in ids])
  if not equivalent(rebuilt_hold,hold[name]):raise RuntimeError('Full holdout summary mismatch')
 for row in hold['rows']:
  for name,ident in [('candidate',hold['winner_id']),('combined','combined'),('strict84','strict84')]:
   if row[name]!=rows[(ident,row['case_id'])]:raise RuntimeError('Holdout row binding')
 for name,ident in [('candidate',hold['winner_id']),('combined','combined'),('strict84','strict84')]:
  for group,predicate in [('low_mean',lambda n:n<=12),('high_mean',lambda n:n>=13),('mean',lambda n:True)]:
   value=statistics.mean(row[name]['s_per_source'] for row in hold['rows'] if predicate(row[name]['N']))
   if not close(value,hold[name][group]):raise RuntimeError('Holdout aggregate mismatch')
 low=[r for r in hold['rows'] if r['candidate']['N']<=12]
 delta=np.array([r['combined']['s_per_source']-r['candidate']['s_per_source'] for r in low]);rng=np.random.default_rng(70260912)
 ci=np.quantile(delta[rng.integers(0,len(low),size=(20000,len(low)))].mean(axis=1),[.025,.975]).tolist()
 if not equivalent(ci,hold['low_paired_saving_95pct_interval']):raise RuntimeError('Bootstrap replay mismatch')
 h=hold['candidate'];expected_checks={'all_hard':h['all_hard_checks'],'low_reduction_at_least_2pct':h['low_reduction']>=.02,'low_not_worse_than_strict84':h['low_mean']<=hold['strict84']['low_mean'],'paired_low_95pct_lower_positive':ci[0]>0,'high_mean_ratio_at_most_1_02':h['high_ratio']<=1.02,'worst_case_ratio_at_most_1_10':h['worst_case_ratio']<=1.10}
 if expected_checks!=hold['checks'] or all(expected_checks.values())!=hold['accepted']:raise RuntimeError('Acceptance rule replay mismatch')
 result={'schema_version':2,'artifact_type':'q3_low_source_v7_independent_readback','all_pass':all(x['passed'] for x in checks),'policy_cases':len(files),'full_traces_replayed':full,'actions_replayed':actions,'selection_rule_independently_recomputed':True,'full_fidelity_pipeline_independently_recomputed':True,'bootstrap_and_acceptance_recomputed':True,'independent_analytic_seven_sector_proof':True,
   'counts':counts,'total_policy_cpu_s':cpu,'total_policy_wall_s_summed':wall,'checks':checks,
   'inputs':{p.relative_to(ROOT).as_posix():ev.sha(p) for p in (Path(__file__),OUT/'protocol.json',OUT/'selected_parameters.json',OUT/'holdout_result.json',OUT/'run_summary.json',OUT/'selection_results.json',OUT/'selection_candidates.json',OUT/'stage1_registry.json',OUT/'stage2_results.json',OUT/'stage2_selection.json',OUT/'optimizer_history.json')},
   'limitations':['Training has compact metrics and deterministic action hashes, not independent full action replay.','Replay independently implements public response and cost equations; geometry checks both a separate closed-form seven-sector proof and the original subdivision certificate.','Synthetic heldout results are not official simulator scores.']}
 ev.save(OUT/'readback_audit.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('checks','inputs')}),flush=True)
 if not result['all_pass']:raise SystemExit(2)
if __name__=='__main__':main()
