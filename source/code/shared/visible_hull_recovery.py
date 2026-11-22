"""Public-observation convexity certificate for Q4 recovery measurement points."""
import math
from search_policy import _box_circle,_clip_halfplane,EPSILON_DEG


def hull(points):
    points=sorted(set(tuple(p) for p in points))
    if len(points)<3:return points
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def diameter(poly):
    return max((math.dist(a,b) for i,a in enumerate(poly) for b in poly[i+1:]),default=0.)


def worst_posterior_upper(poly,point):
    worst=10.
    for degree in range(360):
        half=math.radians(EPSILON_DEG+.5)
        a=math.radians(degree)
        lower=(math.cos(a-half),math.sin(a-half))
        upper=(math.cos(a+half),math.sin(a+half))
        clipped=_clip_halfplane(_clip_halfplane(poly,point,lower,1.),point,upper,-1.)
        worst=max(worst,diameter(clipped))
    return worst


def proposal(positives,measured,poly,position,old_next):
    boundary=hull(positives)
    if len(boundary)<2:return None
    center=_box_circle(poly)[0]
    best=None
    # Recovery is called at a no-signal point. Projection onto boundary also
    # provides explicit nonnegative coefficients without a hull interior test.
    for a,b in zip(boundary,boundary[1:]+boundary[:1]):
        delta=(b[0]-a[0],b[1]-a[1]);squared=delta[0]**2+delta[1]**2
        if squared==0:continue
        t=max(0.,min(1.,((center[0]-a[0])*delta[0]+(center[1]-a[1])*delta[1])/squared))
        point=((1-t)*a[0]+t*b[0],(1-t)*a[1]+t*b[1])
        value=math.dist(center,point)
        if best is None or value<best[0]:best=(value,a,b,t)
    if best is None:return None
    _,a,b,t=best
    weights=[.0001/len(positives)]*len(positives)
    weights[positives.index(a)]+=.9999*(1-t)
    weights[positives.index(b)]+=.9999*t
    point=tuple(sum(w*p[axis] for w,p in zip(weights,positives)) for axis in (0,1))
    if min((math.dist(point,p) for p in measured),default=math.inf)<1.:return None
    if math.dist(position,point)>math.dist(position,old_next)+1e-7:return None
    prior=diameter(poly)
    posterior=worst_posterior_upper(poly,point)
    if posterior>.8*prior:return None
    return {'point':point,'positive_points':list(positives),'weights':weights,
            'before_polygon':poly,'prior_diameter_m':prior,'posterior_diameter_upper_m':posterior,
            'current_position':position,'old_binary_next':old_next,'move_m':math.dist(position,point),
            'old_move_m':math.dist(position,old_next)}
