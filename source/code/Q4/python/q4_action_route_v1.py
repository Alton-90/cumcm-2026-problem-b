"""Interruptible Q4 localization over the unchanged certified ring22 cover.

Only public receipts enter this module. A localizer yields after each measure,
so coverage and other source work can be selected before its next action.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / 'shared'
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient, _state
from search_policy_q4_ring22 import ring22_points, coverage_certificate
from search_policy_q4_ring22_joint_route import stop

VERSION = 'q4_action_route_v1'
DIRECT_RADIUS = 19.9
EXTRA_LIMIT = 8
COMMIT_RADIUS = 100.0
COMMIT_DISTANCE = 150.0

@dataclass
class Track:
    initial: tuple
    bearing: float
    polygon: list
    visible: tuple
    invisible: tuple | None = None
    last_measure: tuple | None = None
    tested_centers: list = field(default_factory=list)
    extra: int = 0
    started: bool = False
    blocked: bool = False
    near_point: tuple | None = None


def _next(track):
    bound = _box_circle(track.polygon)
    if track.near_point is not None:
        return 'clear', track.near_point
    if bound is None:
        return 'fallback', track.initial
    center, radius = bound
    if radius <= DIRECT_RADIUS:
        return 'clear', center
    if track.extra >= EXTRA_LIMIT:
        return 'fallback', center
    if track.invisible is not None:
        point = tuple((a+b)/2 for a, b in zip(track.visible, track.invisible))
        if min(math.dist(point, track.visible), math.dist(point, track.invisible)) <= 1e-6:
            return 'fallback', center
        return 'recover', point
    if any(math.dist(center, point) <= 1e-6 for point in track.tested_centers):
        return 'fallback', center
    return 'measure', center


def _observe(channel, track, point, response, state, *, local=False, recovery=False):
    kind = response.get('measure_result')
    if kind == 'no_signal':
        if local:
            track.invisible = point
            if not recovery:
                state['recovery_count'] += 1
        return
    if kind == 'near':
        track.near_point = point
        track.visible = point
        return
    if kind != 'direction' or not polygon_policy._valid_svd(response.get('svd_deg')):
        raise ValueError(f'invalid public measurement for channel {channel}')
    clipped = _add_bearing(track.polygon, point, response['svd_deg'])
    if not clipped:
        state['warnings'].append(f'channel {channel}: empty intersection retained')
    else:
        track.polygon = clipped
        state['polygon_track_update_count'] += 1
    track.visible = point
    if local and not recovery:
        track.invisible = None
    if recovery:
        state['recovery_positive_count'] += 1


def _fallback(client, channel, track, state):
    before = state['clear_count']
    success = polygon_policy._fallback_scan(client, track.initial, track.bearing,
                                           channel, state, track.polygon)
    state['fallback_events'].append({'channel': channel, 'extra_used': track.extra,
                                    'clears': state['clear_count']-before,
                                    'success': bool(success)})
    track.blocked = not success
    if not success:
        state['unresolved_channels'].add(channel)
    return success


def run_search(client, question=4, method='main', variant='A'):
    if question != 4 or method != 'main' or variant not in ('A', 'B'):
        raise ValueError('Q4 action-route supports main and variants A/B only')
    points = ring22_points()
    state = _state(points)
    state.update(coverage_geometry=coverage_certificate(),
                 version=VERSION+'_'+variant, variant=variant,
                 polygon_track_update_count=0, reuse_measure_count=0,
                 reuse_polygon_update_count=0, direct_clear_skip_measure_count=0,
                 maximum_count_stop_used=False, maximum_count_certificate=None,
                 action_route_decisions=[], local_events=[], fallback_events=[],
                 commitment_count=0, supplemental_duplicate_skip_count=0,
                 route_strategy='replan_after_each_public_local_measurement')
    c = _TrackedClient(client)
    tracks, seen = {}, set()
    remaining = list(range(len(points)))
    active = None
    try:
        while remaining or any(ch not in state['cleared_channels'] and not t.blocked
                               for ch, t in tracks.items()):
            if stop(state):
                break
            nodes = [('scan', i, points[i]) for i in remaining]
            tasks = {}
            for ch, track in sorted(tracks.items()):
                if ch in state['cleared_channels'] or track.blocked:
                    continue
                tasks[ch] = _next(track)
                nodes.append(('source', ch, tasks[ch][1]))
            if not nodes:
                break
            selected = _route(c.position, nodes)[0]
            committed = False
            if variant == 'B' and active in tasks:
                tr = tracks[active]
                bound = _box_circle(tr.polygon)
                if bound and bound[1] <= COMMIT_RADIUS and math.dist(c.position, tasks[active][1]) <= COMMIT_DISTANCE:
                    selected = ('source', active, tasks[active][1])
                    committed = True
                    state['commitment_count'] += 1
            kind, ident, dest = selected
            state['action_route_decisions'].append({
                'position': c.position, 'remaining_scans': list(remaining),
                'nodes': nodes, 'selected': selected, 'committed': committed,
                'measure_count': state['measure_count'], 'clear_count': state['clear_count'],
                'task': tasks.get(ident) if kind == 'source' else None})
            if kind == 'scan':
                remaining.remove(ident)
                active = None
                for ch in range(1, 21):
                    if stop(state):
                        break
                    if ch in state['cleared_channels']:
                        continue
                    track = tracks.get(ch)
                    if track and _next(track)[0] == 'clear':
                        continue
                    response = c.measure(*dest, ch)
                    state['measure_count'] += 1
                    state['coverage_measurements_by_channel'][ch] += 1
                    result = response.get('measure_result')
                    if result == 'no_signal':
                        if track:
                            track.last_measure = dest
                        continue
                    seen.add(ch)
                    if result == 'near':
                        _resolve_clear(c, dest, ch, state, 'near ring22 coverage')
                    elif result == 'direction' and polygon_policy._valid_svd(response.get('svd_deg')):
                        if track is None:
                            track = tracks[ch] = Track(dest, response['svd_deg'],
                                                     _bearing_triangle(dest, response['svd_deg']), dest)
                        else:
                            _observe(ch, track, dest, response, state)
                        track.last_measure = dest
                        track.blocked = False
                    else:
                        raise ValueError('invalid coverage receipt')
                state['batch_count'] += 1
                continue
            ch, track = ident, tracks[ident]
            active = ch
            action, target = tasks[ch]
            before_radius = _box_circle(track.polygon)
            event = {'channel': ch, 'position_before': c.position,
                     'polygon_before': list(track.polygon), 'visible_before': track.visible,
                     'invisible_before': track.invisible, 'extra_before': track.extra,
                     'action': action, 'target': target,
                     'radius_before': before_radius[1] if before_radius else None,
                     'measure_count_before': state['measure_count'],
                     'clear_count_before': state['clear_count']}
            state['local_events'].append(event)
            if action == 'clear':
                state['direct_clear_skip_measure_count'] += 1
                if not _resolve_clear(c, target, ch, state, 'public near or 19.9m action certificate'):
                    _fallback(c, ch, track, state)
                continue
            if action == 'fallback':
                _fallback(c, ch, track, state)
                continue
            if not track.started:
                track.started = True
                if math.dist(c.position, track.initial) >= 50.0:
                    if track.last_measure != c.position:
                        response = c.measure(*c.position, ch)
                        state['measure_count'] += 1
                        state['extra_bearing_count'] += 1
                        state['reuse_measure_count'] += 1
                        track.extra += 1
                        track.last_measure = c.position
                        event.update(action='supplement', target=c.position, response=response)
                        _observe(ch, track, c.position, response, state)
                        continue
                    state['supplemental_duplicate_skip_count'] += 1
            response = c.measure(*target, ch)
            track.extra += 1
            track.last_measure = target
            state['measure_count'] += 1
            state['extra_bearing_count'] += 1
            if action == 'recover':
                state['recovery_measure_count'] += 1
            else:
                track.tested_centers.append(target)
            event['response'] = response
            _observe(ch, track, target, response, state, local=True,
                     recovery=action == 'recover')
        if not state['maximum_count_stop_used']:
            for ch in range(1, 21):
                if ch not in state['cleared_channels']:
                    (state['unresolved_channels'] if ch in seen else state['excluded_channels']).add(ch)
    except Exception as error:
        state['stopped'] = True
        state['warnings'].append(f'protocol/transport stop: {type(error).__name__}: {error}')
        state['unresolved_channels'].update(set(range(1, 21))-state['cleared_channels']-state['excluded_channels'])
    state['remaining_scans'] = remaining
    state['source_extra_used'] = {ch: track.extra for ch, track in tracks.items()}
    for key in ('cleared_channels', 'excluded_channels', 'unresolved_channels'):
        state[key] = sorted(state[key])
    state.update(completed=not state['stopped'] and not state['unresolved_channels']
                 and len(state['cleared_channels'])+len(state['excluded_channels']) == 20,
                 question=4, method='main', virtual_time_s=c.virtual_time_s)
    return state
