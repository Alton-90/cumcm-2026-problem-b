"""Offline decision interface. No movement, detector, GUI, or network calls.

The caller supplies the actual accumulated distance and refreshed public belief.
Without a fresh stopped observation this function cannot shrink that belief.
"""
import math
import select_second_detection_point as s

def fixed1000_baseline(first,bearing,budget,*,sides=72,step=25,bearing_bin=2):
    poly=s.initial_outer_polygon(first,bearing,circle_sides=sides)
    if len(poly)<3: return {'status':'INCONSISTENT_OBSERVATION','candidates':[]}
    anchor=s.complete_sector_anchor(first,bearing)
    local_anchor=s.local_coordinates(anchor,first,bearing)[0]
    points=[]
    # A genuinely different, restricted axis/normal candidate family.
    for a in [i*step for i in range(int(1500/step)+1)]+[local_anchor]:
        points.append(s.world_coordinates((a,0),first,bearing))
    for b in range(-1000,1001,step):
        points.append(s.world_coordinates((local_anchor,b),first,bearing))
    rows=[s.candidate_record(p,first,bearing,poly,bearing_bin,'conservative_1000') for p in points if math.dist(p,first)<=budget+1e-9 and s.maximum_vertex_distance(p,poly)<=999.9]
    return {'status':'FEASIBLE' if rows else 'NO_CERTIFIED_CANDIDATE','candidates':rows}

def choose_event(candidates,*,current,next_destination,total_budget,spent_distance,quality_cap,current_channel,target_channel,clear_radius=None):
    numbers=[*current,*next_destination,total_budget,spent_distance,quality_cap]
    if not all(math.isfinite(v) for v in numbers) or min(total_budget,spent_distance,quality_cap)<0:
        raise ValueError('finite nonnegative budget, spending and quality cap required')
    if spent_distance>total_budget: return {'action':'BUDGET_EXHAUSTED','candidates':[]}
    # Only a supplied continuous coverage radius, not a diameter, can trigger clear.
    if clear_radius is not None:
        if not math.isfinite(clear_radius) or clear_radius<0: raise ValueError('invalid coverage radius')
        if clear_radius<=19.9: return {'action':'CLEAR_AT_CURRENT_POINT','reason':'supplied continuous radius certificate','candidates':[]}
    rows=[]
    for record in candidates:
        p=record['world_m']; U=record['posterior_diameter_upper_m']
        if not all(math.isfinite(v) for v in [*p,U]) or U<0: continue
        cert=record.get('correlated_reception_certificate')
        safe=cert['feasible'] if cert else record.get('reception_margin_m',-math.inf)>=0.099999
        move=math.dist(current,p)
        if not safe or spent_distance+move>total_budget+1e-9: continue
        extra=max(0.,move+math.dist(p,next_destination)-math.dist(current,next_destination))
        rows.append({**record,'remaining_move_m':move,'cumulative_distance_at_event_m':spent_distance+move,'route_increment_m':extra,'immediate_measurement_increment_s':extra/5+5+int(current_channel!=target_channel)})
    eligible=[r for r in rows if r['posterior_diameter_upper_m']<=quality_cap]
    if not eligible:
        return {'action':'NO_CANDIDATE_MEETS_QUALITY_CAP','candidates':rows,'reason':'retain evidence; caller must revise budget/quality or use a certified fallback'}
    best=min(eligible,key=lambda r:(r['route_increment_m'],r['remaining_move_m'],r['posterior_diameter_upper_m'],tuple(r['world_m'])))
    return {'action':'MEASURE_NOW' if best['remaining_move_m']<=1e-9 else 'MOVE_THEN_STOP_AND_MEASURE','selected':best,'candidates':rows,'limits':'Local event choice only; no optimal stopping or future-time guarantee. Movement without a new stopped observation does not update belief.'}
