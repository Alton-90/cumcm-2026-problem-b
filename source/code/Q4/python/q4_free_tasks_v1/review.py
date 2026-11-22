from pathlib import Path
import sys,json,math,types
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'code/Q4/python'),str(ROOT/'code/shared')]
import pruning as p
from search_policy_q4_ring22 import ring22_points
OUT=ROOT/'results/Q4/experiments/free_tasks_v1'
def main():
 checks={}
 assert p.outside_range((1500.,0.),[(0.,0.)]) is None
 assert p.outside_range((1500.0000005,0.),[(0.,0.)]) is None
 assert p.outside_range((1500.001,0.),[(0.,0.)]) is not None
 assert p.outside_range((0.,0.),[(-2000.,-1.),(2000.,1.)]) is None
 assert p.outside_range((9000.,0.),[]) is None
 checks['continuous_bbox_and_boundary']=True
 points=ring22_points()+[ring22_points()[8]];tasks=p.Tasks(points);remaining=list(range(23));c=types.SimpleNamespace(negatives={ch:[] for ch in range(1,21)},receipt_count=0)
 assert tasks.try_delete(8,1,remaining,c,{},set())
 assert not tasks.try_delete(22,1,remaining,c,{},set())
 checks['sequential_mutual_dependency']=True
 tasks=p.Tasks(ring22_points());c.negatives[2]=ring22_points()
 assert not tasks.try_delete(8,1,list(range(22)),c,{},set())
 checks['no_cross_channel_receipts']=True
 original=p.prove
 try:
  p.coverage.cache_clear();p.prove=lambda x:{'status':'UNKNOWN'}
  assert p.coverage(p.key(ring22_points()),8) is None
 finally:p.prove=original;p.coverage.cache_clear()
 checks['unknown_never_authorizes']=True
 def signature(a):return a['path'],a['channel'],tuple(a['point'])
 total=0;count=0
 for stage in ('development','validation'):
  for path in (OUT/stage/'raw').glob('*.json'):
   data=json.loads(path.read_text());a=data['baseline']['actions'];b=data['candidate']['actions'];i=0;skipped=0
   for action in b:
    while i<len(a) and signature(a[i])!=signature(action):
     assert a[i]['path']=='/measure' and a[i]['response']['measure_result']=='no_signal';skipped+=1;i+=1
    assert i<len(a)
    for field in ('measure_result','svd_deg','clear_result'):assert a[i]['response'].get(field)==action['response'].get(field)
    i+=1
   for action in a[i:]:assert action['path']=='/measure' and action['response']['measure_result']=='no_signal';skipped+=1
   assert skipped==data['candidate']['state']['free_tasks']['omitted_scan_measurements']
   assert data['candidate']['metrics']['costs']['movement_m']==data['baseline']['metrics']['costs']['movement_m']
   total+=skipped;count+=1
 assert count==126 and total==2735
 checks['126_cases_only_remove_original_no_signal']=True
 checks['126_cases_exact_movement_unchanged']=True
 result=dict(checks=checks,all_pass=all(checks.values()),case_count=count,verified_removed_measurements=total,reviewer='same visible controller; not independent agent',scope='targeted branch tests and saved-action comparison')
 (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
