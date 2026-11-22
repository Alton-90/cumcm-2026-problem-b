from pathlib import Path
import sys,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'code/Q4/python'),str(ROOT/'code/shared')]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
load('pruning',HERE/'pruning.py')
load('localizer_free',HERE/'localizer_free.py')
policy=load('q4_free_tasks_selected_policy',HERE/'policy.py')
VERSION='q4_free_tasks_v1'
def validate():
 data=json.loads((HERE/'candidate_parameters.json').read_text(encoding='utf-8'))
 for item in data['inputs']:assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
 return data
def run_search(client,question=4,method='main'):
 validate()
 return policy.run_search(client,question=question,method=method)
