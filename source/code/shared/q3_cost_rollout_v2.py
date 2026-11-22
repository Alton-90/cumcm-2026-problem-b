"""Public-state scenario scoring; finite predictions are not performance bounds."""
from __future__ import annotations

import math
import numpy as np
import search_policy as old

OFFSETS = (0.1, 0.25, 0.5)
BACKOFFS = (0.25, 0.5)
MOVE_SLACK = 100.0
RECEPTION_LIMIT = 999.9
MIN_PREDICTED_SAVING = 5.0
MAX_HYPOTHESES = 9
BIASES = (-1.0, 0.0, 1.0)


def longest_axis(polygon):
    pairs = [(math.dist(a, b), a, b) for i, a in enumerate(polygon) for b in polygon[i+1:]]
    if not pairs:
        return np.array([1.0, 0.0])
    length, a, b = max(pairs, key=lambda item: item[0])
    return (np.asarray(b) - a) / length if length > 1e-8 else np.array([1.0, 0.0])


def compatible_hypotheses(cloud, polygon, history, axis):
    """Select representative grid hypotheses, never read arena/source objects."""
    if cloud is None or len(cloud) == 0:
        return []
    points = np.asarray(cloud, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    area = sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]
               - polygon[i][1]*polygon[(i+1)%len(polygon)][0] for i in range(len(polygon)))
    orientation = 1 if area >= 0 else -1
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        edge = np.asarray(b)-a
        delta = points-a
        keep &= orientation*(edge[0]*delta[:, 1]-edge[1]*delta[:, 0]) >= -1e-7
    minimum_range = np.full(len(points), 1000.0)
    negative_distances = []
    for event in history:
        delta = points-np.asarray(event['point'])
        distance = np.linalg.norm(delta, axis=1)
        if event['action'] == 'clear':
            if event['response'].get('clear_result') != 'success':
                keep &= distance > 20.0
            continue
        response = event['response']
        kind = response.get('measure_result')
        if kind == 'direction':
            angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            errors = np.abs((angles-float(response['svd_deg'])+180.0) % 360.0-180.0)
            keep &= (distance > 5.0) & (distance <= 1500.0) & (errors <= old.EPSILON_DEG+1e-9)
            minimum_range = np.maximum(minimum_range, distance)
        elif kind == 'no_signal':
            negative_distances.append(distance)
        elif kind == 'near':
            keep &= distance <= 5.0
    keep &= minimum_range <= 1500.0
    for distance in negative_distances:
        keep &= distance > minimum_range
    indices = np.flatnonzero(keep)
    if not len(indices):
        return []
    order = np.argsort(points[indices] @ axis, kind='stable')
    indices = indices[order]
    selected = np.unique(np.rint(np.linspace(0, len(indices)-1, min(MAX_HYPOTHESES, len(indices)))).astype(int))
    return [{'point': points[indices[i]].tolist(), 'range_m': float(minimum_range[indices[i]])}
            for i in selected]


def update_cloud(cloud, point, response):
    delta = cloud-np.asarray(point)
    distance = np.linalg.norm(delta, axis=1)
    kind = response['measure_result']
    if kind == 'no_signal':
        return cloud[distance >= 985.0]
    if kind == 'near':
        return cloud[distance <= 20.0]
    theta = math.radians(response['svd_deg'])
    axis = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    return cloud[(distance <= 1515.0) & (delta @ axis >= -15.0)
                 & (np.abs(delta @ normal) <= distance*math.sin(math.radians(1.005))+15.0)]


def predicted_completion(first_target, position, polygon, cloud, history, initial_point,
                         initial_svd, extra_used, hypothesis, bias, previous_center=None):
    """Execute the baseline local continuation against a constructed public hypothesis."""
    location = hypothesis['point']
    radius = hypothesis['range_m']
    position = tuple(position)
    cloud = np.array(cloud, dtype=float, copy=True)
    polygon = list(polygon)
    cost = 0.0
    cache = [(tuple(event['point']), event['response']) for event in history if event['action'] == 'measure']

    def move(point):
        nonlocal cost, position
        cost += math.dist(position, point)/5.0
        position = tuple(point)

    def clear(point):
        nonlocal cost
        move(point)
        success = math.dist(point, location) <= 20.0
        cost += 3.0+2.0*success
        return success

    def fallback():
        for point in old._clear_points(initial_point, initial_svd):
            if clear(point):
                return cost
        return math.inf

    def measure(point):
        nonlocal cost, cloud
        move(point)
        cost += 5.0
        for measured, response in cache:
            if math.dist(measured, point) <= 1e-8:
                cloud = update_cloud(cloud, point, response)
                return response
        distance = math.dist(point, location)
        if distance > radius:
            response = {'measure_result': 'no_signal'}
        elif distance <= 5.0:
            response = {'measure_result': 'near'}
        else:
            angle = math.degrees(math.atan2(location[1]-point[1], location[0]-point[0]))
            response = {'measure_result': 'direction', 'svd_deg': round((angle+bias) % 360.0, 2) % 360.0}
        cache.append((tuple(point), response))
        cloud = update_cloud(cloud, point, response)
        return response

    previous_target = previous_center
    for used in range(extra_used, old.MAX_EXTRA_BEARINGS+1):
        if used == extra_used:
            target = first_target
        else:
            box = old._box_circle(polygon)
            if box is None:
                return fallback()
            center, uncertainty = box
            if len(cloud):
                center = tuple(cloud.mean(axis=0))
                uncertainty = float(np.linalg.norm(cloud-center, axis=1).max()+15.0)
                if uncertainty <= 60.0 and clear(center):
                    return cost
            if uncertainty <= old.LOCATE_CLEAR_RADIUS_M:
                return cost if clear(center) else fallback()
            if used >= old.MAX_EXTRA_BEARINGS:
                return fallback()
            target = center
        if previous_target is not None and math.dist(target, previous_target) <= 1e-6:
            return fallback()
        previous_target = target
        response = measure(target)
        kind = response['measure_result']
        if kind == 'near':
            return cost if clear(target) else fallback()
        if kind == 'no_signal':
            return fallback()
        polygon = old._add_bearing(polygon, target, response['svd_deg'])
    raise AssertionError('unreachable rollout loop')


def choose_point(polygon, center, position, cloud, history, initial_point, initial_svd, extra_used, previous_center=None):
    center = tuple(center)
    axis = longest_axis(polygon)
    normal = np.array([-axis[1], axis[0]])
    uncertainty = old._box_circle(polygon)[1]
    candidates = [('baseline_center', center)]
    for factor in OFFSETS:
        offset = min(350.0, max(25.0, factor*uncertainty))
        for sign in (-1, 1):
            candidates.append((f'normal_{factor}_{sign}', tuple(np.asarray(center)+sign*offset*normal)))
    toward = np.asarray(position)-center
    original_move = float(np.linalg.norm(toward))
    if original_move > 1e-8:
        for factor in BACKOFFS:
            candidates.append((f'approach_{factor}', tuple(np.asarray(center)+toward/original_move*min(factor*uncertainty, original_move))))
    hypotheses = compatible_hypotheses(cloud, polygon, history, axis)
    records = []
    unique = []
    for label, point in candidates:
        if any(math.dist(point, known) <= 1e-6 for known in unique):
            continue
        unique.append(point)
        maximum_distance = max(math.dist(point, vertex) for vertex in polygon)
        distance = math.dist(position, point)
        fresh = all(math.dist(point, event['point']) > 1.0 for event in history if event['action'] == 'measure')
        valid = label == 'baseline_center' or (maximum_distance <= RECEPTION_LIMIT
                and distance <= original_move+MOVE_SLACK and fresh)
        record = {'label': label, 'point': list(point), 'max_vertex_distance_m': maximum_distance,
                  'movement_m': distance, 'fresh_point': fresh, 'eligible': valid, 'score_s': None}
        if valid and hypotheses:
            per_position = [max(predicted_completion(point, position, polygon, cloud, history,
                                initial_point, initial_svd, extra_used, item, bias, previous_center) for bias in BIASES)
                            for item in hypotheses]
            score = float(np.mean(per_position))
            if math.isfinite(score):
                record.update(score_s=score, scenario_maxima_s=per_position)
        records.append(record)
    original = records[0]
    scored = [record for record in records[1:] if record['score_s'] is not None]
    best = min(scored, key=lambda record: (record['score_s'], record['movement_m'])) if scored else None
    selected = bool(best is not None and original['score_s'] is not None
                    and original['score_s']-best['score_s'] >= MIN_PREDICTED_SAVING)
    chosen = best if selected else original
    return {'selected_point': chosen['point'], 'used_alternative': selected,
            'original_score_s': original['score_s'], 'selected_score_s': chosen['score_s'],
            'hypotheses': hypotheses, 'axis': axis.tolist(), 'polygon_radius_m': uncertainty,
            'candidates': records, 'no_compatible_grid_hypotheses': not bool(hypotheses)}
