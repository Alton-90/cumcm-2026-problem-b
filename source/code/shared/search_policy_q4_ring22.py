"""Q4 reuse-A localizer on the independently certified 22-point ring cover."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import search_policy_polygon as polygon_policy
from search_policy import _bearing_triangle,_box_circle,_resolve_clear
from search_policy_q4_batch import _TrackedClient,_localize,_state
from search_policy_q4_reuse import _supplement_from_current_pose

VERSION='q4_ring22_v1'
CERT=Path(__file__).resolve().parents[2]/'results'/'optimization_200_400'/'ring22_continuous_certificate.json'
CERT_SHA256=hashlib.sha256(CERT.read_bytes()).hexdigest()

def ring22_points():
 return [(0.,0.)]+[(990*math.cos(2*math.pi*k/7),990*math.sin(2*math.pi*k/7)) for k in range(7)]+[(1850*math.cos(math.pi*k/7),1850*math.sin(math.pi*k/7)) for k in range(14)]

def coverage_certificate():
 d=json.loads(CERT.read_text(encoding='utf-8'))
 required={'n_inner':7,'inner_radius_m':990,'outer_radius_m':1850,'receiver_radius_m':1000,'source_radius_m':1800}
 if d.get('certified') is not True or d.get('parameters')!=required or d.get('failed_boxes') or d.get('unprocessed_boxes')!=0:
  raise ValueError('ring22 continuous certificate is unavailable or incompatible')
 return {'certificate_path':'results/optimization_200_400/ring22_continuous_certificate.json','certificate_sha256':CERT_SHA256,'point_count':22,'accepted_cell_count':d['accepted_cell_count'],'min_receive_margin_m':d['min_receive_margin_m'],'min_hull_margin_m':d['min_hull_margin_m'],'continuous_sufficient':True}

def run_search(client,question=4,method='main'):
 if question!=4 or method!='main':raise ValueError('ring22 supports Q4 main only')
 points=ring22_points();state=_state(points);state.update(reuse_variant='A',reuse_measure_count=0,reuse_polygon_update_count=0,coverage_geometry=coverage_certificate())
 tracked=_TrackedClient(client);seen=set();remaining=list(points)
 try:
  while remaining:
   point=min(remaining,key=lambda p:(math.dist(tracked.position,p),p[1],p[0]));remaining.remove(point);pending={}
   for ch in range(1,21):
    if ch in state['cleared_channels']:continue
    response=tracked.measure(*point,ch);state['measure_count']+=1;state['coverage_measurements_by_channel'][ch]+=1;kind=response.get('measure_result')
    if kind=='no_signal':continue
    seen.add(ch)
    if kind=='near':_resolve_clear(tracked,point,ch,state,'near ring22 coverage')
    elif kind=='direction' and polygon_policy._valid_svd(response.get('svd_deg')):pending[ch]=(point,response['svd_deg'],_bearing_triangle(point,response['svd_deg']))
    else:state['warnings'].append(f'channel {ch}: invalid public response');state['unresolved_channels'].add(ch)
   state['batch_count']+=1
   while pending:
    ch,track=min(pending.items(),key=lambda x:math.inf if _box_circle(x[1][2]) is None else math.dist(tracked.position,_box_circle(x[1][2])[0]));del pending[ch]
    track,cleared=_supplement_from_current_pose(tracked,ch,track,state)
    if not cleared:_localize(tracked,track[0],track[1],ch,state,track[2])
  for ch in range(1,21):
   if ch not in state['cleared_channels']:
    (state['unresolved_channels'] if ch in seen else state['excluded_channels']).add(ch)
 except Exception as e:
  state['stopped']=True;state['warnings'].append(f'protocol/transport stop: {type(e).__name__}: {e}');state['unresolved_channels'].update(set(range(1,21))-state['cleared_channels']-state['excluded_channels'])
 for key in ('cleared_channels','excluded_channels','unresolved_channels'):state[key]=sorted(state[key])
 state.update(completed=not state['stopped'] and not state['unresolved_channels'] and len(state['cleared_channels'])+len(state['excluded_channels'])==20,question=4,method='main',version=VERSION,virtual_time_s=tracked.virtual_time_s)
 return state
