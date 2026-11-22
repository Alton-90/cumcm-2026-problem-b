"""Explicit Q3 formal run with frozen policy and local request/response journal."""
import sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'code/Q3/python'),str(ROOT/'code/shared')]
import q3_final_main as policy
from run_q3_final_practice import validate_final_evidence
from run_formal import main as protocol
from simulator_client import SimulatorClient

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--formal-confirmed',action='store_true');ap.add_argument('--robot-id',required=True);ap.add_argument('--index',type=int,choices=[1,2,3],required=True);ap.add_argument('--case-id',required=True);args=ap.parse_args()
 if not args.formal_confirmed:raise RuntimeError('Formal authorization required')
 evidence=validate_final_evidence()
 folder=ROOT/'results/Q3/formal/final_strict84_three_20260913'/f'run{args.index:02}'
 if (folder/'client.jsonl').exists():raise RuntimeError('Already entered or attempted; no automatic restart')
 folder.mkdir(parents=True,exist_ok=True)
 def factory(robot_id,port=2026):return SimulatorClient(robot_id,port=port,log_path=folder/'client.jsonl')
 def writer(question,s):
  s.update(method=policy.METHOD_ID,policy_version=policy.VERSION,case_id=args.case_id,policy_sha256=hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest(),code_lock_sha256=hashlib.sha256(policy.LOCK_PATH.read_bytes()).hexdigest(),validated_evidence=evidence)
  p=folder/'summary.json';p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');return p
 return protocol(['--formal-confirmed','--question','3','--robot-id',args.robot_id],client_factory=factory,q3_policy=policy.run_search,summary_writer=writer)
if __name__=='__main__':raise SystemExit(main())
