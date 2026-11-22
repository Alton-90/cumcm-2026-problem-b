import sys,json,statistics,hashlib,random,math
from pathlib import Path
from dataclasses import asdict
r=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(r/'code/shared'))
from offline_arena import Source,OfflineArena
from simulator_client import SimulatorClient
import search_policy_compact as base,search_policy_hex_reuse as new
out=r/'results/hex_reuse_q3';out.mkdir(exist_ok=True)
cases=[c for c in json.loads((r/'results/compact_grid/cases.json').read_text(encoding='utf8')) if c['question']==3]
for c in cases:c['split']='development'
for i in range(24):
 rng=random.Random(76000+i);n=[10,13,16][i%3];sources=[]
 for ch in rng.sample(range(1,21),n):
  a=rng.uniform(0,2*math.pi);d=1800 if i%4==0 else 1800*math.sqrt(rng.random()); sources.append(asdict(Source(ch,d*math.cos(a),d*math.sin(a),[1000,1500,1250][i%3],None)))
 cases.append(dict(question=3,split='holdout',case_id=f'batch_{76000+i}',seed=76000+i,error_mode=['fixed_hash','positive_edge','negative_edge'][i%3],sources=sources))
(out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
inputs=[{'path':p,'sha256':hashlib.sha256((r/p).read_bytes()).hexdigest()} for p in ['code/shared/search_policy_hex_reuse.py','code/shared/search_policy_compact.py','code/shared/search_policy.py','code/shared/offline_arena.py','results/hex_reuse_q3/cases.json']]
(out/'inputs_before_run.json').write_text(json.dumps(inputs,indent=2),encoding='utf8')
rows=[]
for c in cases:
 pair={}
 for name,policy in [('compact',base),('batch',new)]:
  arena=OfflineArena([Source(**s) for s in c['sources']],c['seed'],c['error_mode']);client=SimulatorClient('synthetic',transport=arena,log_path=False);client.enter();state=policy.run_search(client);client.exit();n=len(c['sources']);ok=state['completed'] and len(arena.cleared)==n and not(set(state['excluded_channels'])&set(arena.sources));pair[name]=dict(ok=ok,mean_s=arena.virtual_time_s/n,time_s=arena.virtual_time_s,measures=arena.measure_count,clears=arena.clear_count,fallbacks=state['fallback_count'])
 rows.append(dict(case_id=c['case_id'],split=c['split'],ratio=pair['batch']['time_s']/pair['compact']['time_s'],**pair))
summaries={}
for split in ['all','holdout']:
 rr=[a for a in rows if split=='all' or a['split']==split];b=statistics.mean(x['compact']['mean_s'] for x in rr);v=statistics.mean(x['batch']['mean_s'] for x in rr);summaries[split]=dict(compact=b,batch=v,reduction=1-v/b,max_ratio=max(x['ratio'] for x in rr))
ok=all(row[m]['ok'] for row in rows for m in ['compact','batch']);result=dict(all_checks_pass=ok,accepted=ok and summaries['holdout']['reduction']>=.1 and summaries['all']['max_ratio']<=1.10,summaries=summaries,rows=rows,inputs=inputs)
(out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']}))
