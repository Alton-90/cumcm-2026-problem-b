"""Superseded ring22 reuse-A guard; deliberately not practice eligible."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'/'shared'))
from search_policy_q4_ring22 import VERSION,run_search
FINAL=ROOT/'results'/'q4_ring22'/'final_holdout'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
 c=json.loads((FINAL/'comparison.json').read_text(encoding='utf-8'));inputs=json.loads((FINAL/'inputs_before_run.json').read_text(encoding='utf-8'))
 if c.get('scope')!='SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT' or c.get('accepted') is not True or c.get('all_checks_pass') is not True or c.get('case_count')!=60 or c.get('inputs')!=inputs:raise ValueError('ring22 final evidence not accepted')
 for x in inputs:
  if set(x)!={'path','sha256'} or sha(ROOT/x['path'])!=x['sha256']:raise ValueError(f"frozen input mismatch: {x.get('path')}")
 return {'policy_version':VERSION,'comparison_sha256':sha(FINAL/'comparison.json'),'validated_inputs':inputs}
def main(argv=None):
 print(json.dumps({'status':'refused_superseded_baseline','practice_only':True,'reason':'ring22 reuse-A is superseded by the required ring22 joint-route comparison'}));return 2
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--practice-confirmed',action='store_true');p.add_argument('--question',type=int,choices=(4,));p.add_argument('--robot-id');a=p.parse_args(argv)
 if not a.practice_confirmed or a.question is None or not a.robot_id:p.print_help();return 0
 try:validate()
 except Exception as e:print(json.dumps({'status':'refused_pre_client','practice_only':True,'error_fingerprint':hashlib.sha256(str(e).encode()).hexdigest()[:16]}),file=sys.stderr);return 2
 print(json.dumps({'status':'guard_passed_no_simulator_started','practice_only':True,'policy_version':VERSION}));return 0
if __name__=='__main__':raise SystemExit(main())
