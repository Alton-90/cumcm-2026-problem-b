"""Offline Q1 reference geometry. Exact arithmetic AFTER angle conversion.

Halfplanes use a*x+b*y<=c. No artificial bounding box in classification.
O(m^3) vertex enumeration; diameter is O(h) calipers or O(h^2) baseline.
Trigonometric coefficients are binary64 approximations, not interval trig.
"""
from fractions import Fraction as F
from itertools import combinations
import math

def rational(v):
    if isinstance(v,float) and not math.isfinite(v):
        raise ValueError('non-finite coordinate')
    return F(v)

def halfplanes(rows):
    return [tuple(rational(v) for v in row) for row in rows]

def bearing_halfplanes(observations, half_angle=1.0):
    if not math.isfinite(half_angle) or not 0 < half_angle < 90:
        raise ValueError('half angle must be in (0,90) degrees')
    rows=[]
    for x,y,theta in observations:
        x,y=rational(x),rational(y)
        if not math.isfinite(theta): raise ValueError('non-finite bearing')
        lo,hi=[math.radians((theta+a)%360) for a in (-half_angle,half_angle)]
        for a,b in [(math.sin(lo),-math.cos(lo)),(-math.sin(hi),math.cos(hi))]:
            a,b=F(a),F(b)
            rows.append((a,b,a*x+b*y))
    return rows

def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def hull(points):
    ps=sorted(set(points))
    if len(ps)<3: return ps
    def chain(seq):
        out=[]
        for p in seq:
            while len(out)>1 and cross(out[-2],out[-1],p)<=0: out.pop()
            out.append(p)
        return out
    return chain(ps)[:-1]+chain(reversed(ps))[:-1]

def feasible(p, rows):
    return all(a*p[0]+b*p[1]<=c for a,b,c in rows)

def intersection(rows):
    rows=halfplanes(rows)
    candidates=[(F(0),F(0))]
    vertices=[]
    for a,b,c in rows:
        if a or b: candidates.append((a*c/(a*a+b*b),b*c/(a*a+b*b)))
    for (a,b,c),(d,e,f) in combinations(rows,2):
        det=a*e-b*d
        if det:
            p=((c*e-b*f)/det,(a*f-c*d)/det)
            if feasible(p,rows): vertices.append(p)
    witness=next((p for p in candidates+vertices if feasible(p,rows)),None)
    if witness is None: return {'status':'EMPTY','vertices':[],'witness':None}
    directions=[(F(1),F(0)),(F(-1),F(0)),(F(0),F(1)),(F(0),F(-1))]
    for a,b,c in rows:
        if a or b: directions.extend([(b,-a),(-b,a)])
    ray=next((d for d in directions if all(a*d[0]+b*d[1]<=0 for a,b,c in rows)),None)
    if ray is not None:
        return {'status':'UNBOUNDED','vertices':hull(vertices),'witness':witness,'recession_direction':ray}
    ps=hull(vertices)
    if not ps: raise ArithmeticError('bounded nonempty polyhedron without vertex')
    return {'status':{1:'POINT',2:'SEGMENT'}.get(len(ps),'POLYGON'),'vertices':ps,'witness':witness}

def distance2(p,q):
    return (p[0]-q[0])**2+(p[1]-q[1])**2

def diameter2(points, method='calipers'):
    """Diameter of a counterclockwise convex hull, without collinear vertices."""
    if len(points)<2: return F(0)
    if method=='exhaustive' or len(points)==2:
        return max(distance2(p,q) for p,q in combinations(points,2))
    if method!='calipers': raise ValueError('unknown diameter method')
    ps=list(points)
    if len(ps)<3: return diameter2(ps,'exhaustive')
    n=len(ps); j=1; best=F(0)
    for i in range(n):
        ni=(i+1)%n
        while cross(ps[i],ps[ni],ps[(j+1)%n])>cross(ps[i],ps[ni],ps[j]):
            j=(j+1)%n
        best=max(best,distance2(ps[i],ps[j]),distance2(ps[ni],ps[j]))
        if cross(ps[i],ps[ni],ps[(j+1)%n])==cross(ps[i],ps[ni],ps[j]):
            best=max(best,distance2(ps[i],ps[(j+1)%n]),distance2(ps[ni],ps[(j+1)%n]))
    return best

def sqrt_bounds(value):
    value=F(value)
    if value<0: raise ValueError('negative square')
    if not value: return 0.0,0.0
    # sqrt of numerator/denominator via their separate logs avoids ratio overflow.
    try: guess=math.sqrt(float(value))
    except OverflowError: raise ValueError('magnitude exceeds supported reporting range')
    if guess==0: raise ValueError('squared magnitude is below supported reporting range')
    if not math.isfinite(guess): raise ValueError('magnitude exceeds reporting range')
    lo=hi=guess
    while F(lo)**2>value: lo=math.nextafter(lo,0.0)
    while F(hi)**2<value: hi=math.nextafter(hi,math.inf)
    return lo,hi

def solve(rows,method='calipers',center=None):
    result=intersection(rows)
    if result['status']=='EMPTY': result['diameter_interval_m']=None
    elif result['status']=='UNBOUNDED': result['diameter_interval_m']=['infinity','infinity']
    else:
        d2=diameter2(result['vertices'],method)
        result['diameter_squared_exact']=str(d2)
        result['diameter_interval_m']=list(sqrt_bounds(d2))
        if center is not None:
            c=tuple(rational(v) for v in center)
            r2=max(distance2(p,c) for p in result['vertices'])
            result['specified_center_radius_interval_m']=list(sqrt_bounds(r2))
            result['clear_within_20m']=r2<=400
    return result

def clip(poly,row):
    if not poly: return []
    a,b,c=row; out=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        fp=a*p[0]+b*p[1]-c; fq=a*q[0]+b*q[1]-c
        if fp<=0: out.append(p)
        if (fp<=0)!=(fq<=0):
            t=fp/(fp-fq); out.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return hull(out)

def disk_interval(rows,radius=1800,center=(0,0),sides=64):
    if sides<8 or sides%4: raise ValueError('sides must be a multiple of 4, >=8')
    R=rational(radius); cx,cy=map(rational,center)
    if R<=0: raise ValueError('radius must be positive')
    rows=halfplanes(rows); inner=[]; support=[]
    # Rational support constants are certified by exact square comparison.
    for k in range(sides):
        a,b=F(math.cos(2*math.pi*k/sides)),F(math.sin(2*math.pi*k/sides))
        norm_upper=F(sqrt_bounds(a*a+b*b)[1])
        p=(cx+R*a/norm_upper,cy+R*b/norm_upper)
        assert distance2(p,(cx,cy))<=R*R
        inner.append(p)
        support.append((a,b,R*norm_upper+a*cx+b*cy))
    # This box is a proven enclosure of the disk, never used to classify P.
    outer=[(cx-R,cy-R),(cx+R,cy-R),(cx+R,cy+R),(cx-R,cy+R)]
    for row in support+rows: outer=clip(outer,row)
    for row in rows: inner=clip(inner,row)
    if not outer: return {'status':'EMPTY','interval_m':None,'sides':sides}
    lower=sqrt_bounds(diameter2(inner))[0] if inner else 0.0
    upper=sqrt_bounds(diameter2(outer))[1]
    return {'status':'ENCLOSED' if inner else 'OUTER_NONEMPTY_INNER_EMPTY','interval_m':[lower,upper],'sides':sides,'inner_vertices':inner,'outer_vertices':outer}

def serializable(value):
    if isinstance(value,F): return float(value)
    if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serializable(v) for v in value]
    return value

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows=bearing_halfplanes(data['observations'],data.get('half_angle_deg',1.0)) if 'observations' in data else halfplanes(data['halfplanes'])
    result=solve(rows,center=data.get('center'))
    if 'disk_radius_m' in data: result['disk_enhancement']=disk_interval(rows,data['disk_radius_m'])
    Path(args.output).write_text(json.dumps(serializable(result),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
