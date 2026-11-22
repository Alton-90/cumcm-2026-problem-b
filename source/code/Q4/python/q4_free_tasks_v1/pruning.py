"""No additional measurements: proof-carrying per-channel obligations."""
import math,functools
from q4_joint_cover_engine_v1 import key,point_covered,witnesses,prove,proof_id

@functools.lru_cache(maxsize=4096)
def coverage(points,removed):
 if any(not point_covered(w,points) for w in witnesses(removed)):return None
 proof=prove(points)
 return proof if proof['status']=='COVERED' else None

def outside_range(point,polygon):
 if not polygon or any(not all(math.isfinite(v) for v in p) for p in polygon):return None
 x0=min(p[0] for p in polygon);x1=max(p[0] for p in polygon);y0=min(p[1] for p in polygon);y1=max(p[1] for p in polygon)
 lower=math.hypot(max(x0-point[0],0.,point[0]-x1),max(y0-point[1],0.,point[1]-y1))
 if lower<=1500.000001:return None
 return dict(kind='guaranteed_no_signal_range',bbox=[x0,y0,x1,y1],distance_lower_m=lower,polygon=polygon)

class Tasks:
 def __init__(self,points):
  self.points=points;self.pending={i:set(range(1,21)) for i in range(len(points))}
  self.stats=dict(coverage_checks=0,coverage_deletions=0,range_checks=0,range_deletions=0,removed_stops=0,omitted_scan_measurements=0,events=[],proofs={})
 def try_delete(self,i,ch,remaining,c,tracks,seen):
  if ch not in self.pending[i]:return True
  proof=None;future=None
  if ch in seen:
   self.stats['range_checks']+=1
   tr=tracks.get(ch)
   if tr:proof=outside_range(self.points[i],tr[2])
  else:
   self.stats['coverage_checks']+=1
   future=[self.points[j] for j in remaining if j!=i and ch in self.pending[j]]
   full=coverage(key(c.negatives[ch]+future),i)
   if full:
    ident=proof_id(full);self.stats['proofs'][ident]=full
    proof=dict(kind='continuous_cover',proof_id=ident,negative_points=key(c.negatives[ch]),future_points=future)
  if proof is None:return False
  self.pending[i].remove(ch)
  self.stats['range_deletions' if proof['kind']=='guaranteed_no_signal_range' else 'coverage_deletions']+=1
  self.stats['events'].append(dict(scan=i,channel=ch,receipt_count=c.receipt_count,proof=proof))
  return True
 def prune(self,remaining,c,tracks,seen,cleared):
  for i in list(remaining):
   for ch in sorted(self.pending[i]-cleared):self.try_delete(i,ch,remaining,c,tracks,seen)
   # Cleared channels need no scans in J06 either; do not count them as new deletions.
   if not (self.pending[i]-cleared):
    remaining.remove(i);self.stats['removed_stops']+=1
