"""Paid shared sensing stops and continuously certified residual sector coverage.

Policy inputs are public observations only. Geometry uses analytic maxima on
closed polar rectangles; unresolved subdivision never proves coverage.
"""
from __future__ import annotations
import math
from functools import lru_cache
import search_policy as old
import search_policy_q3_combined as base
from search_policy_feasible_q3 import localize
from search_policy_compact import _state

VERSION = "q3_shared_stop_v6"
PARAMETERS = dict(max_depth=12, max_nodes=2048, line_steps=8,
                  minimum_path_saving_m=2.0, shared_stops=True, residual_route=True)
LIMIT2 = (1000.0-1e-7)**2

def polar_max_distance2(point, bounds):
    """Exact extrema: convex in r; angle endpoints or opposite bearing."""
    r0,r1,a0,a1=bounds
    x,y=point
    angles=[a0,a1]
    opposite=math.atan2(y,x)+math.pi
    k=math.ceil((a0-opposite)/(2*math.pi))
    middle=opposite+k*2*math.pi
    if a0<=middle<=a1: angles.append(middle)
    return max((r*math.cos(a)-x)**2+(r*math.sin(a)-y)**2
               for r in (r0,r1) for a in angles)

def region_bounds(region):
    if region==0:return (0.,1000.,-math.pi,math.pi)
    angle=(region-1)*math.pi/3
    return (1000.,1800.,angle-math.pi/6,angle+math.pi/6)

@lru_cache(maxsize=32768)
def certificate(region, points):
    """Return proved/node count. points contains only no-signal or a proposed stop."""
    if not points:return False,0
    if region==0 and (0.,0.) in points:return True,1
    stack=[(region_bounds(region),0)];nodes=0
    while stack:
        bounds,depth=stack.pop();nodes+=1
        if nodes>PARAMETERS["max_nodes"]:return False,nodes
        if any(polar_max_distance2(p,bounds)<=LIMIT2 for p in points):continue
        r0,r1,a0,a1=bounds
        # A genuine uncovered witness permits immediate rejection, never acceptance.
        for r,a in ((r0,a0),(r0,a1),(r1,a0),(r1,a1),((r0+r1)/2,(a0+a1)/2)):
            q=(r*math.cos(a),r*math.sin(a))
            if all(math.dist(q,p)>1000.+1e-7 for p in points):return False,nodes
        if depth>=PARAMETERS["max_depth"]:return False,nodes
        if r1-r0>=r1*(a1-a0):
            mid=(r0+r1)/2
            children=((r0,mid,a0,a1),(mid,r1,a0,a1))
        else:
            mid=(a0+a1)/2
            children=((r0,r1,a0,mid),(r0,r1,mid,a1))
        stack.extend((c,depth+1) for c in children)
    return True,nodes

class SharedTracked(base.ValueTracked):
    def __init__(self,client):
        super().__init__(client)
        self.negatives={ch:[] for ch in range(1,21)}
        self.actions=[]
    def measure(self,x,y,ch):
        reply=super().measure(x,y,ch)
        self.actions.append(dict(path="/measure",point=[x,y],channel=ch,response=dict(reply)))
        if reply.get("measure_result")=="no_signal":
            p=(float(x),float(y))
            if p not in self.negatives[ch]:self.negatives[ch].append(p)
        return reply
    def clear(self,x,y,ch):
        reply=super().clear(x,y,ch)
        self.actions.append(dict(path="/clear",point=[x,y],channel=ch,response=dict(reply)))
        return reply
    def proof(self,ch,region,extra=None):
        points=tuple(self.negatives[ch])
        if extra is not None and tuple(extra) not in points:points+= (tuple(extra),)
        return certificate(region,points)[0]

def run_search(client,question=3,method="main"):
    if question!=3 or method!="main":raise ValueError("Q3 main only")
    client=SharedTracked(client)
    points=[(0.,0.)]+[(1124*math.cos(k*math.pi/3),1124*math.sin(k*math.pi/3)) for k in range(6)]
    state=_state(3,points)
    state.update(coverage_certificates=[],shared_stop_events=[],scan_shift_events=[])
    tracks={};seen=set();attempted=set();done={ch:set() for ch in range(1,21)}
    shared_checked=set()

    def unknown():return set(range(1,21))-seen-state["cleared_channels"]
    def refresh():
        for ch in sorted(unknown()):
            for region in range(7):
                if region not in done[ch] and client.proof(ch,region):
                    done[ch].add(region)
                    state["coverage_certificates"].append(dict(channel=ch,region=region,
                        action_count=len(client.actions),negative_points=[list(p) for p in client.negatives[ch]]))
    def absorb(point,ch,reply):
        result=reply.get("measure_result")
        if result=="no_signal":return
        if result=="near":
            seen.add(ch)
            if not old._resolve_clear(client,point,ch,state,"near shared scan"):
                state["warnings"].append(f"channel {ch}: near clear failed")
            return
        if result=="direction" and base._valid(reply.get("svd_deg")):
            seen.add(ch)
            if ch not in tracks:tracks[ch]=(point,reply,old._bearing_triangle(point,reply["svd_deg"]))
            else:
                p,a,poly=tracks[ch];new=old._add_bearing(poly,point,reply["svd_deg"])
                if not new:raise RuntimeError(f"channel {ch}: incompatible bearings")
                tracks[ch]=(p,a,new)
            return
        raise RuntimeError(f"channel {ch}: invalid public measurement")
    def scan(point,channels,reason):
        for ch in channels:
            if ch in state["cleared_channels"]:continue
            if ch in tracks:
                if not client.useful(ch,point):continue
                box=old._box_circle(tracks[ch][2])
                if box and box[1]<=old.LOCATE_CLEAR_RADIUS_M:continue
            reply=client.measure(*point,ch)
            state["measure_count"]+=1;state["coverage_measurements_by_channel"][ch]+=1
            absorb(point,ch,reply)
        refresh()
    def route_point(region,start,after,original,channels):
        if region==0 or not PARAMETERS["residual_route"]:return original
        def valid(p):return all(client.proof(ch,region,p) for ch in channels)
        def cost(p):return math.dist(start,p)+(0 if after is None else math.dist(p,after))
        best=original
        targets=[start]
        if after is not None:targets += [after,((start[0]+after[0])/2,(start[1]+after[1])/2)]
        # No convex-feasibility assumption: keep only independently certified trials.
        for target in targets:
            low,high=0.,1.
            if valid(target):trial=target
            else:
                trial=original
                for _ in range(PARAMETERS["line_steps"]):
                    fraction=(low+high)/2
                    p=tuple(original[k]+fraction*(target[k]-original[k]) for k in (0,1))
                    if valid(p):low=fraction;trial=p
                    else:high=fraction
            if cost(trial)<cost(best):best=trial
        if cost(original)-cost(best)<PARAMETERS["minimum_path_saving_m"]:return original
        state["scan_shift_events"].append(dict(region=region,original=list(original),selected=list(best),
            start=list(start),after=None if after is None else list(after),channels=channels,
            path_saving_proxy_m=cost(original)-cost(best),action_count=len(client.actions)))
        return best

    try:
        for iteration in range(500):
            if len(state["cleared_channels"])==16:break
            refresh()
            remaining=[i for i in range(7) if any(i not in done[ch] for ch in unknown())]
            pending=[ch for ch in tracks if ch not in state["cleared_channels"] and ch not in attempted]
            if not remaining and not pending:break
            here=tuple(client.position)
            # A paid observation can discharge a whole pending region. No speculative area threshold.
            if PARAMETERS["shared_stops"] and here not in shared_checked and math.hypot(*here)>1:
                shared_checked.add(here)
                eligible={}
                for ch in sorted(unknown()):
                    regions=[i for i in remaining if i not in done[ch] and client.proof(ch,i,here)]
                    if regions:eligible[ch]=regions
                if eligible:
                    state["shared_stop_events"].append(dict(point=list(here),channels=sorted(eligible),
                        prospective_regions=eligible,action_count=len(client.actions)))
                    scan(here,sorted(eligible),"shared")
                    continue
            nodes=[("scan",i,points[i]) for i in remaining]
            for ch in pending:
                box=client.estimate(ch) or old._box_circle(tracks[ch][2])
                if box:nodes.append(("source",ch,box[0]))
            if not nodes:break
            route=base._route(client.position,nodes);kind,index,destination=route[0]
            if kind=="scan":
                channels=[ch for ch in sorted(unknown()) if index not in done[ch]]
                after=route[1][2] if len(route)>1 else None
                destination=route_point(index,client.position,after,destination,channels)
                channels=sorted(set(channels)|{ch for ch in tracks if ch not in state["cleared_channels"]})
                scan(destination,channels,"anchor")
            else:
                ch=index;p,a,poly=tracks[ch];attempted.add(ch)
                if math.dist(client.position,p)>1 and client.useful(ch,client.position):
                    here=client.position;extra=client.measure(*here,ch)
                    state["measure_count"]+=1;state["extra_bearing_count"]+=1
                    if extra.get("measure_result")=="near" and old._resolve_clear(client,here,ch,state,"near current pose"):continue
                    if extra.get("measure_result")=="direction" and base._valid(extra.get("svd_deg")):
                        new=old._add_bearing(poly,here,extra["svd_deg"])
                        if new:poly=new
                    tracks[ch]=(p,a,poly);attempted.discard(ch);continue
                localize(client,p,a,ch,state,poly)
        else:raise RuntimeError("control iteration budget reached")
        refresh()
        excluded={ch for ch in unknown() if len(done[ch])==7}
        if len(state["cleared_channels"])==16:excluded=set(range(1,21))-state["cleared_channels"]
        state["excluded_channels"]=excluded
        state["unresolved_channels"]=set(range(1,21))-state["cleared_channels"]-excluded
    except Exception as error:
        state["stopped"]=True;state["warnings"].append(f"{type(error).__name__}: {error}")
        state["unresolved_channels"]=set(range(1,21))-state["cleared_channels"]-state["excluded_channels"]
    for key in ("cleared_channels","excluded_channels","unresolved_channels"):state[key]=sorted(state[key])
    state.update(completed=not state["stopped"] and not state["unresolved_channels"],
        question=3,method="main",version=VERSION,virtual_time_s=client.virtual_time_s,
        coverage_ledger={str(ch):sorted(v) for ch,v in done.items()},
        skipped_known_measures=client.skipped_known_measures,public_actions=client.actions,
        parameters=PARAMETERS.copy())
    return state

