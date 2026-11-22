import sys,json,importlib,statistics,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'code/shared'))
from offline_arena import Source,OfflineArena
from simulator_client import SimulatorClient
modules=sys.argv[1:];cases=[c for c in json.loads((r/'results/compact_grid/cases.json').read_text(encoding='utf8')) if c['question']==3];rows=[]
for name in modules:
 policy=importlib.import_module(name)
 for c in cases:
  arena=OfflineArena([Source(**s) for s in c['sources']],c['seed'],c['error_mode']);client=SimulatorClient('synthetic',transport=arena,log_path=False);client.enter();s=policy.run_search(client);client.exit();n=len(c['sources']);calc=arena.distance_m/5+5*arena.measure_count+arena.switch_count+3*arena.clear_count+2*len(arena.cleared)
  rows.append(dict(module=name,case_id=c['case_id'],all_clear=len(arena.cleared)==n and s['completed'],clock=abs(calc-arena.virtual_time_s)<1e-6,mean=arena.virtual_time_s/n,move=arena.distance_m/5/n,measure=(5*arena.measure_count+arena.switch_count)/n,clears=arena.clear_count,fallbacks=s['fallback_count']))
out=r/'results/restructure';out.mkdir(exist_ok=True);(out/'q3_screen.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
for name in modules:
 rr=[a for a in rows if a['module']==name];print(name,{'all_clear':all(x['all_clear'] and x['clock'] for x in rr),'mean':statistics.mean(x['mean'] for x in rr),'move':statistics.mean(x['move'] for x in rr),'measure':statistics.mean(x['measure'] for x in rr)})
