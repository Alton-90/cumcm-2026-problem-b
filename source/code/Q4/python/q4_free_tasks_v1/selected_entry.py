from pathlib import Path
import json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('q4_free_tasks_frozen_candidate',HERE/'candidate_entry.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
VERSION=candidate.VERSION

def validate():
 candidate.validate()
 data=json.loads((HERE/'final_parameters.json').read_text(encoding='utf-8'))
 for item in data['inputs']:
  assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
 return data

def run_search(client,question=4,method='main'):
 validate()
 return candidate.run_search(client,question=question,method=method)
