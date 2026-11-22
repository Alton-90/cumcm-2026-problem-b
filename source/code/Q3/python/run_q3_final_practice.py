"""Explicit practice-only Q3 launcher; final code and offline evidence are checked first."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import uuid

from q3_final_main import ROOT, VERSION, METHOD_ID, PARAMETERS, run_search, validate_code_lock, LOCK_PATH
from run_q3_combined_practice import main as run_practice

FINAL = ROOT / 'results/Q3/experiments/final_integration_20260912'


def validate_final_evidence():
    lock = validate_code_lock()
    result_path = FINAL / 'metrics/comparison.json'
    audit_path = FINAL / 'metrics/independent_audit.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not result['accepted_under_user_rule'] or not audit['all_pass']:
        raise ValueError('Final Q3 offline evidence did not pass')
    if result['code_lock_sha256'] != hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest():
        raise ValueError('Final Q3 offline evidence targets another code lock')
    if audit['comparison_sha256'] != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError('Changed final Q3 comparison')
    for relative, expected in audit['bound_inputs'].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed final Q3 audit input: ' + relative)
    return {'version': VERSION, 'parameters': PARAMETERS, 'code_lock': lock,
            'comparison_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'independent_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()}


def write_summary(summary):
    # The inherited launcher implements the tested request/retry/exit protocol.
    # Replace its descriptive combined labels with the policy actually executed.
    summary.update(artifact_type='q3_final_practice_summary', method=METHOD_ID,
                   policy_version=VERSION,
                   policy_sha256=hashlib.sha256(Path(__file__).with_name('q3_final_main.py').read_bytes()).hexdigest(),
                   code_lock_sha256=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest())
    folder = ROOT / 'results/Q3/practice/final_strict84' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def main(argv=None):
    return run_practice(argv, policy=run_search, validator=validate_final_evidence,
                        summary_writer=write_summary)


if __name__ == '__main__':
    raise SystemExit(main())
