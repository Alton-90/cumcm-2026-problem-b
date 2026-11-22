"""Remaining explicitly authorized Q4 formal sessions with the unchanged final FREE."""
from pathlib import Path
import sys,json,hashlib,argparse,importlib.util
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'code/shared'))
from run_formal import main as protocol
from simulator_client import SimulatorClient
spec=importlib.util.spec_from_file_location('q4_official_frozen_free',ROOT/'code/Q4/python/q4_free_tasks_v1/selected_entry.py')
selected=importlib.util.module_from_spec(spec);spec.loader.exec_module(selected)
OUT=ROOT/'results/Q4/formal/FREE_one_20260913'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate():
 data=selected.validate()
 choice=json.loads((ROOT/'planning/q4_current_selection.json').read_text(encoding='utf-8'))
 assert choice['selected_policy']=='q4_free_tasks_v1' and selected.VERSION=='q4_free_tasks_v1'
 for key in ('entry','parameters'):assert sha(ROOT/choice[key])==choice[key+'_sha256']
 return data

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--formal-confirmed',action='store_true');ap.add_argument('--check-only',action='store_true');ap.add_argument('--robot-id');ap.add_argument('--case-id');ap.add_argument('--index',type=int,choices=[2,3],required=True);args=ap.parse_args(argv)
 evidence=validate()
 if args.check_only:print(json.dumps({'frozen_FREE_valid':True,'policy':selected.VERSION,'simulator_calls':0}));return 0
 if not args.formal_confirmed or not args.robot_id or not args.case_id:ap.error('Explicit formal authorization, visible team and case id required')
 folder=OUT/f'run{args.index:02}';folder.mkdir(parents=True,exist_ok=True)
 attempt=folder/'enter_attempt.json'
 with attempt.open('x',encoding='utf-8') as f:json.dump({'case_id':args.case_id,'authorized_runs':3,'model':selected.VERSION,'no_automatic_restart':True},f)
 assert not (folder/'client.jsonl').exists()
 inputs=[Path(__file__),ROOT/'code/shared/run_formal.py',ROOT/'code/shared/practice_runner.py',ROOT/'code/shared/simulator_client.py',Path(selected.__file__),Path(selected.__file__).with_name('final_parameters.json')]
 hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
 def factory(robot_id,port=2026):return SimulatorClient(robot_id,port=port,log_path=folder/'client.jsonl')
 def writer(question,s):
  s.update(method=selected.VERSION,policy_version=selected.VERSION,case_id=args.case_id,code_hashes=hashes,validated_free_evidence=evidence,authorized_runs=3)
  p=folder/'summary.json';p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return p
 def checked():
  data=validate();return {'scope':'FROZEN_USER_SELECTED_FREE','parameters':{k:v for k,v in data.items() if k!='inputs'}},data['inputs']
 return protocol(['--formal-confirmed','--question','4','--robot-id',args.robot_id],client_factory=factory,q4_policy=selected.run_search,summary_writer=writer,q4_validator=checked)
if __name__=='__main__':raise SystemExit(main())
