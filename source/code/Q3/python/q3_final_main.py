"""Final Q3 strict84 entry; no network actions occur merely by importing it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/shared'))
import search_policy_q3_cost_rollout_strict84_selected as strict84

VERSION = 'q3_final_strict84_v1_20260912'
METHOD_ID = 'q3_final_strict84'
LOCK_PATH = Path(__file__).with_name('q3_final_lock.json')
PARAMETERS = {
    'move_slack_m': 175.0,
    'minimum_predicted_saving_s': 0.0,
    'offset_scale': 0.5,
    'offset_factors': [0.05, 0.125, 0.25],
    'backoffs': [0.25, 0.5],
    'maximum_hypotheses': 9,
    'guaranteed_reception_guard_m': 999.9,
    'bias_hypotheses_deg': [-1.0, 0.0, 1.0],
}


def validate_code_lock():
    lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    if lock['version'] != VERSION or lock['parameters'] != PARAMETERS:
        raise ValueError('Final Q3 version or parameter lock mismatch')
    if not lock['policy_sources']:
        raise ValueError('Empty final Q3 source lock')
    for relative, digest in lock['policy_sources'].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise ValueError('Invalid locked source path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed final Q3 dependency: ' + relative)
    return lock


def run_search(client, question=3, method='main'):
    if question != 3 or method != 'main':
        raise ValueError('Final Q3 main entry accepts question=3, method=main only')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run one policy serially on each process main thread')
    validate_code_lock()
    result = strict84.run_search(client, question=3, method='main')
    result.update(version=VERSION, final_method_id=METHOD_ID,
                  final_parameters=dict(PARAMETERS),
                  final_code_lock_sha256=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest())
    return result


if __name__ == '__main__':
    print('Use run_q3_final_practice.py for an explicitly confirmed practice session.')
