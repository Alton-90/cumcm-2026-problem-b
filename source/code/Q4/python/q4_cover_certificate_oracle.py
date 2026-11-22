"""Continuous full-view coverage oracle for future Q4 scan synthesis.

This is geometry, not a robot policy: point lists are per-channel measurement
locations or proposed designs, never proof that an unmeasured channel was read.
A PASS proves a sufficient continuous certificate. UNKNOWN is not uncovered.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'code/shared'))
from q4_joint_compatibility import check_fixed_position

EPS = 1e-6

def hull(points):
    points=sorted(set(tuple(p) for p in points))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def cell_certificate(box, points, receive_radius=1000.0):
    x0,y0,x1,y1=box
    corners=[(x,y) for x in (x0,x1) for y in (y0,y1)]
    eligible=[i for i,p in enumerate(points)
              if max(math.dist(p,c) for c in corners)<=receive_radius-EPS]
    boundary=hull([points[i] for i in eligible])
    if len(boundary)<3:
        return None
    margin=min(((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/math.dist(a,b)
               for a,b in zip(boundary,boundary[1:]+boundary[:1]) for c in corners)
    if margin<EPS:
        return None
    return {'box':box,'receiver_ids':eligible,'hull':boundary,'hull_margin_m':margin,
            'range_margin_m':receive_radius-max(math.dist(points[i],c) for i in eligible for c in corners)}

def certify(points, *, max_depth=17, max_boxes=100000):
    points=[tuple(map(float,p)) for p in points]
    if not points or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('finite receiver coordinates required')
    stack=[(-1800.,-1800.,1800.,1800.,0)]
    accepted=[]; outside=[]; uncertain=[]; checked=0
    while stack and checked<max_boxes:
        x0,y0,x1,y1,depth=stack.pop(); checked+=1
        box=(x0,y0,x1,y1)
        if math.hypot(max(x0,0.,-x1),max(y0,0.,-y1))>1800+EPS:
            outside.append(box); continue
        certificate=cell_certificate(box,points)
        if certificate:
            accepted.append(certificate); continue
        if depth>=max_depth:
            uncertain.append(box); continue
        mx,my=(x0+x1)/2,(y0+y1)/2
        # A compatible interior point is an actual missed-source witness.
        if math.hypot(mx,my)<1800-EPS:
            result=check_fixed_position((mx,my),[{'point':p,'result':'no_signal'} for p in points],source_type='directional')
            if result.status=='compatible':
                return {'status':'UNCOVERED','certified':False,'point_count':len(points),
                        'witness':{'source':(mx,my),'radius_m':1000.,'heading_deg':result.orientation_witness_deg},
                        'checked_boxes':checked,'accepted_cells':accepted,'outside_cells':outside,
                        'unprocessed_boxes':len(stack),'uncertain_boxes':uncertain}
        stack.extend((xa,ya,xb,yb,depth+1) for xa,xb in ((x0,mx),(mx,x1)) for ya,yb in ((y0,my),(my,y1)))
    passed=not stack and not uncertain
    return {'status':'COVERED' if passed else 'UNKNOWN','certified':passed,
            'point_count':len(points),'checked_boxes':checked,'accepted_cells':accepted,
            'outside_cells':outside,'unprocessed_boxes':len(stack),'uncertain_boxes':uncertain}
