"""Bounded, causal joint service-stop coverage planner for Q4."""
from __future__ import annotations
import functools, hashlib, json, math
from pathlib import Path
from q4_cover_certificate_oracle import cell_certificate, hull
from q4_action_route_v1 import ring22_points, _route, _TrackedClient

ROOT = Path(__file__).resolve().parents[3]
CERT_PATH = ROOT/'results/Q4/experiments/under400_research/ring22_reproduced_certificate.json'
MAX_BOXES = 12000
MAX_DEPTH = 17

def key(points):
    return tuple(sorted(set(tuple(map(float,p)) for p in points)))

def point_covered(g, points):
    eligible=[p for p in points if math.dist(g,p)<=1000.0000001]
    h=hull(eligible)
    if len(h)<3:
        return False  # Screening rejection only, not a continuous impossibility claim.
    return all((b[0]-a[0])*(g[1]-a[1])-(b[1]-a[1])*(g[0]-a[0])>=-1e-5
               for a,b in zip(h,h[1:]+h[:1]))

@functools.lru_cache(maxsize=1)
def template():
    data=json.loads(CERT_PATH.read_text(encoding='utf-8-sig'))
    assert data['status']=='COVERED' and not data['uncertain_boxes'] and not data['unprocessed_boxes']
    return data

@functools.lru_cache(maxsize=22)
def witnesses(removed):
    points=ring22_points()
    other=points[:removed]+points[removed+1:]
    centers=[((c['box'][0]+c['box'][2])/2,(c['box'][1]+c['box'][3])/2)
             for c in template()['accepted_cells']]
    return tuple(p for p in centers if math.hypot(*p)<=1800 and not point_covered(p,other))

@functools.lru_cache(maxsize=64)
def prove_cached(points):
    rings=ring22_points()
    available={i for i,p in enumerate(rings) if p in points}
    reused=[]; stack=[]; accepted=[]; checked=0
    for i,c in enumerate(template()['accepted_cells']):
        if set(c['receiver_ids'])<=available:
            reused.append(i)
        else:
            b=c['box']; depth=round(math.log2(3600/(b[2]-b[0])))
            stack.append((*b,depth))
    while stack:
        if checked>=MAX_BOXES:
            return {'status':'UNKNOWN','reason':'box_budget','checked_boxes':checked}
        x0,y0,x1,y1,depth=stack.pop(); checked+=1
        cert=cell_certificate((x0,y0,x1,y1),points)
        if cert:
            accepted.append(cert); continue
        if depth>=MAX_DEPTH:
            return {'status':'UNKNOWN','reason':'depth_budget','checked_boxes':checked}
        mx,my=(x0+x1)/2,(y0+y1)/2
        stack.extend((xa,ya,xb,yb,depth+1) for xa,xb in ((x0,mx),(mx,x1))
                     for ya,yb in ((y0,my),(my,y1)))
    return {'status':'COVERED','points':points,
            'template_path':CERT_PATH.relative_to(ROOT).as_posix(),
            'template_sha256':hashlib.sha256(CERT_PATH.read_bytes()).hexdigest(),
            'reused_cell_ids':reused,'refined_cells':accepted,'checked_boxes':checked,
            'outside_cells_from_template':True}

def prove(points):
    return prove_cached(key(points))

def proof_id(proof):
    return hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class ReceiptClient(_TrackedClient):
    def __init__(self,client):
        super().__init__(client)
        self.observations={ch:[] for ch in range(1,21)}
        self.receipt_count=0
        self.service_mode=False
        self.last_service_index=0
        self.service_poses=[]
        self.detect_channel=1
    def arrived(self):
        self.receipt_count+=1
        if self.service_mode:
            self.last_service_index=self.receipt_count
            if all(math.dist(self.position,p['point'])>=50 for p in self.service_poses):
                self.service_poses.append({'point':self.position,'receipt_index':self.receipt_count})
                self.service_poses=self.service_poses[-64:]
    def measure(self,x,y,channel):
        result=super().measure(x,y,channel)
        self.detect_channel=channel
        self.arrived()
        self.observations[channel].append({'point':(x,y),'result':result.get('measure_result'),
                                           'receipt_index':self.receipt_count})
        return result
    def clear(self,x,y,channel):
        result=super().clear(x,y,channel)
        self.arrived()
        return result
    def negatives(self,ch):
        return key(o['point'] for o in self.observations[ch] if o['result']=='no_signal')

def route_length(position,nodes):
    route=_route(position,nodes)
    return sum(math.dist(a,b) for a,b in zip([position]+[n[2] for n in route],[n[2] for n in route]))

def choose_bundle(client,remaining,nodes,unknown,stats):
    if not unknown:
        return None
    rings=ring22_points()
    start=client.position
    base_distance=route_length(start,nodes)
    best=None
    for removed in remaining:
        prior=[p['point'] for p in client.service_poses if math.dist(start,p['point'])>=50]
        prior.sort(key=lambda p:(math.dist(p,rings[removed]),p))
        chosen=prior[:3]+([prior[-1]] if len(prior)>3 else [])
        bundles=[(start,)]+[(start,p) for p in dict.fromkeys(chosen)]
        future=[rings[i] for i in remaining if i!=removed]
        for bundle in bundles:
            stats['proposals']+=1
            geometries={key(list(client.negatives(ch))+future+list(bundle)) for ch in unknown}
            if any(any(not point_covered(w,g) for w in witnesses(removed)) for g in geometries):
                stats['finite_screen_reject']+=1; continue
            stats['finite_screen_pass']+=1
            proofs=[prove(g) for g in geometries]
            if any(p['status']!='COVERED' for p in proofs):
                stats['continuous_unknown']+=1; continue
            stats['continuous_pass']+=1
            paid=[p for p in bundle if any(p not in client.negatives(ch) for ch in unknown)]
            new_measures=sum(p not in client.negatives(ch) for p in bundle for ch in unknown)
            end=paid[-1] if paid else start
            visit_distance=sum(math.dist(a,b) for a,b in zip([start]+paid,paid))
            new_nodes=[n for n in nodes if not(n[0]=='scan' and n[1]==removed)]
            new_distance=visit_distance+route_length(end,new_nodes)
            move_saved=base_distance-new_distance
            action_overhead=6*new_measures-5*len(unknown)
            seconds_saved=move_saved/5-action_overhead
            if move_saved<50 or seconds_saved<=1:
                stats['cost_reject']+=1
                rejected={'removed_scan':removed,'bundle':bundle,'predicted_move_saved_m':move_saved,
                          'predicted_seconds_saved':seconds_saved,'auxiliary_measure_bound':new_measures}
                best_rejected=stats.setdefault('best_cost_rejections',[])
                best_rejected.append(rejected)
                best_rejected.sort(key=lambda x:x['predicted_seconds_saved'],reverse=True)
                del best_rejected[3:]
                continue
            proposal={'removed_scan':removed,'bundle':bundle,'paid_stops':paid,
                      'unknown_channels':sorted(unknown),'public_prefix_receipts':client.receipt_count,
                      'service_history':list(client.service_poses),'position':start,
                      'nodes':nodes,'base_route_m':base_distance,'new_route_m':new_distance,
                      'auxiliary_measure_bound':new_measures,'auxiliary_action_overhead_bound_s':action_overhead,
                      'predicted_move_saved_m':move_saved,'predicted_seconds_saved':seconds_saved}
            if best is None or (seconds_saved,move_saved)>(best['predicted_seconds_saved'],best['predicted_move_saved_m']):
                best=proposal
    return best
