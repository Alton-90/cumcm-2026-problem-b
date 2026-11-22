"""Independent geometry checks used by the integrated initial paper."""
from pathlib import Path
import json, math, itertools
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/paper_q12_geometry'
def clip(poly,a,b,c):
    result=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        fp=a*p[0]+b*p[1]-c; fq=a*q[0]+b*q[1]-c
        pin=fp<=1e-10; qin=fq<=1e-10
        if pin: result.append(p)
        if pin!=qin:
            t=fp/(fp-fq); result.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return result
def wedge(poly,s,theta,delta):
    lo=math.radians(theta-delta);hi=math.radians(theta+delta)
    poly=clip(poly,math.sin(lo),-math.cos(lo),math.sin(lo)*s[0]-math.cos(lo)*s[1])
    return clip(poly,-math.sin(hi),math.cos(hi),-math.sin(hi)*s[0]+math.cos(hi)*s[1])
def diameter(poly):
    return max((math.dist(p,q) for p,q in itertools.combinations(poly,2)),default=0.)
def outer_disk(poly,center,r,m):
    for i in range(m):
        a=math.cos(2*math.pi*i/m);b=math.sin(2*math.pi*i/m)
        poly=clip(poly,a,b,r+a*center[0]+b*center[1])
        if not poly: break
    return poly
def sector_outer(delta,m):
    z=1500*math.tan(math.radians(delta))
    return outer_disk([(0.,0.),(1500.,-z),(1500.,z)],(0.,0.),1500.,m)
def upper(poly,x,delta,step):
    maximum=10.
    count=round(360/step)
    for i in range(count):
        maximum=max(maximum,diameter(wedge(poly,x,i*step,delta+step/2)))
    return maximum
def farthest_sector(c,b,delta):
    d=math.radians(delta)
    return max(math.hypot(c,b),math.hypot(c-1500*math.cos(d),abs(b)+1500*math.sin(d)))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tri=[(0.,0.),(36.,0.),(18.,18*math.sqrt(3))]
    observations=[]
    p=[(-2000.,-2000.),(2000.,-2000.),(2000.,2000.),(-2000.,2000.)]
    for i,(a,z) in enumerate(zip(tri,tri[1:]+tri[:1])):
        e=((z[0]-a[0])/36,(z[1]-a[1])/36)
        s=(a[0]-1000*e[0],a[1]-1000*e[1]);theta=120*i+1
        observations.append({'point':s,'svd_deg':theta})
        p=wedge(p,s,theta,1.)
    center=(18.,6*math.sqrt(3))
    vertex_error=max(min(math.dist(a,z) for z in p) for a in tri)
    source_errors=[]
    for obs in observations:
        s=obs['point'];beta=math.degrees(math.atan2(center[1]-s[1],center[0]-s[0]))
        error=(obs['svd_deg']-beta+180)%360-180
        source_errors.append({'distance':math.dist(s,center),'bearing_error_deg':error})
    q1={'triangle':tri,'observations':observations,'intersection_vertices':p,'vertex_error_m':vertex_error,
        'diameter_m':diameter(p),'minimum_enclosing_radius_m':12*math.sqrt(3),
        'example_true_source':center,'example_radius_m':1200.,'source_checks':source_errors,
        'all_checks_pass':vertex_error<1e-7 and abs(diameter(p)-36)<1e-7 and all(abs(e['bearing_error_deg'])<=1 and e['distance']<=1200 for e in source_errors)}
    q2=[]
    for delta in [1.,1.005]:
        d=math.radians(delta);c=1500/(2*math.cos(d))
        bm=min(math.sqrt(1000**2-c*c),math.sqrt(1000**2-(c-1500*math.cos(d))**2)-1500*math.sin(d))
        b=bm-.01
        q2.append({'half_angle_deg':delta,'c_m':c,'b_max_m':bm,'safe_b_m':b,
                   'safe_exact_max_distance_m':farthest_sector(c,b,delta),
                   'reference_rounded_max_distance_m':farthest_sector(750.11,635.65,delta),
                   'axis_max_distance_m':c,'safe_movement_m':math.hypot(c,b)})
    d=q2[0];c=d['c_m'];b=d['safe_b_m'];convergence=[]
    for m,step in [(360,.5),(720,.25),(1440,.125)]:
        poly=sector_outer(1.,m)
        convergence.append({'circle_sides':m,'angle_bin_width_deg':step,'outer_vertices':len(poly),
                            'axis_upper_m':upper(poly,(c,0.),1.,step),
                            'left_upper_m':upper(poly,(c,b),1.,step),
                            'right_upper_m':upper(poly,(c,-b),1.,step),
                            'outer_vertex_max_distance_m':max(math.dist((c,b),v) for v in poly)})
    assert q1['all_checks_pass']
    assert all(row['safe_exact_max_distance_m']<1000 for row in q2)
    assert all(abs(row['left_upper_m']-row['right_upper_m'])<1e-6 for row in convergence)
    data={'scope':'constructed_geometry_examples_not_simulator_scores','q1':q1,'q2_constants':q2,
          'q2_bin_upper_bounds':convergence,'global_optimality_claimed':False}
    (OUT/'verification.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

