"""Guarded practice-only launcher for accepted Q4 ring22 joint-route v2."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'/'shared'))
from practice_runner import RetryingClient,_call_with_pending_retries
from search_policy_q4_ring22_joint_route import VERSION,run_search
from simulator_client import SimulatorClient
FINAL=ROOT/'results'/'q4_ring22_joint_route'/'final_holdout_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
 c=json.loads((FINAL/'comparison.json').read_text(encoding='utf-8'));inputs=json.loads((FINAL/'inputs_before_run.json').read_text(encoding='utf-8'))
 if c.get('scope')!='SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT' or c.get('accepted') is not True or c.get('all_checks_pass') is not True or c.get('case_count')!=60 or c.get('inputs')!=inputs:raise ValueError('ring22 joint final evidence not accepted')
 for x in inputs:
  if set(x)!={'path','sha256'} or sha(ROOT/x['path'])!=x['sha256']:raise ValueError(f"frozen input mismatch: {x.get('path')}")
 return {'comparison_sha256':sha(FINAL/'comparison.json'),'validated_inputs':inputs,'summaries':c['summaries']}
def write(payload):
 d=ROOT/'results'/'Q4'/'practice'/'q4_ring22_joint_route'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);d.mkdir(parents=True,exist_ok=True);p=d/'summary.json';p.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return p
def main(argv=None,client_factory=SimulatorClient):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--practice-confirmed',action='store_true');p.add_argument('--question',type=int,choices=(4,));p.add_argument('--robot-id');p.add_argument('--port',type=int,default=2026);a=p.parse_args(argv)
 if not a.practice_confirmed or a.question is None or not a.robot_id:p.print_help();return 0
 try:evidence=validate()
 except Exception as e:print(json.dumps({'status':'refused_pre_client','practice_only':True,'error_fingerprint':hashlib.sha256(str(e).encode()).hexdigest()[:16]}),file=sys.stderr);return 2
 c=client_factory(a.robot_id,port=a.port);state=None;error=exit_error=None;entered=False;er=xr=None;t=time.monotonic()
 try:er=_call_with_pending_retries(c,'enter');entered=True;state=run_search(RetryingClient(c),question=4,method='main')
 except Exception as e:error=f'{type(e).__name__}: {e}'
 finally:
  if entered and not c.ended:
   try:xr=_call_with_pending_retries(c,'force_exit' if c.uncertain else 'exit')
   except Exception as e:exit_error=f'{type(e).__name__}: {e}'
 n=None if state is None else len(state.get('cleared_channels',[]));ok=bool(state and state.get('completed') and not error and not exit_error);interface=None
 if er and xr and all('real_timestamp_ms' in x for x in (er,xr)):interface=(xr['real_timestamp_ms']-er['real_timestamp_ms'])/1000
 payload={'artifact_type':'q4_ring22_joint_route_practice_summary','practice_only':True,'official_result':False,'question':4,'method':'ring22_joint_route','policy_version':VERSION,'policy_sha256':sha(ROOT/'code/shared/search_policy_q4_ring22_joint_route.py'),'status':'completed' if ok else 'failed_or_unresolved','local_elapsed_s':time.monotonic()-t,'interface_elapsed_s':interface,'virtual_time_s':c.virtual_time_s,'cleared_channel_count':n,'average_virtual_time_per_cleared_source_s':None if not n or c.virtual_time_s is None else c.virtual_time_s/n,'policy':state,'client_log_path':None if c.log_path is None else str(c.log_path),'validated_final_holdout':evidence,'error':error,'exit_error':exit_error};sp=write(payload);print(json.dumps({'status':payload['status'],'practice_only':True,'summary':str(sp),'log':payload['client_log_path']},ensure_ascii=False));return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
