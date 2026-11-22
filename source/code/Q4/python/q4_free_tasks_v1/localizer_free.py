"""Independent Q4 compact-49 candidate with batch-then-nearest localization.

At every compact coverage point, all currently uncleared channels are measured
before any directional localization.  Pending channels are then localized in
nearest-current-polygon-center order.  The policy only receives client
responses and does not import arena or source data.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy_compact import coverage_points
from search_policy import (LOCATE_CLEAR_RADIUS_M, MAX_EXTRA_BEARINGS, _add_bearing,
                           _bearing_triangle, _box_circle, _resolve_clear)


VERSION = "q4_batch49_v1"


class _TrackedClient:
    def __init__(self, client):
        self._client = client
        self.position = (0.0, 0.0)

    @property
    def virtual_time_s(self):
        return self._client.virtual_time_s

    def measure(self, x, y, channel):
        response = self._client.measure(x, y, channel)
        self.position = (x, y)
        return response

    def clear(self, x, y, channel):
        response = self._client.clear(x, y, channel)
        self.position = (x, y)
        return response


def _localize(client, initial_point, initial_svd, channel, state, feasible_polygon):
    """Recovery localizer copied from polygon policy, retaining its current polygon."""
    polygon = feasible_polygon
    last_visible = client.latest_visible.get(channel, initial_point) if client.info_enabled else initial_point
    previous_candidate = None
    extra_used = 0
    while True:
        if client.info_enabled and polygon: client.local_polygons[channel] = polygon
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        candidate, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if client.step_fraction < 1.0:
            center = candidate
            candidate = tuple(client.position[i]+client.step_fraction*(center[i]-client.position[i]) for i in (0,1))
            state.setdefault('short_step_events',[]).append({'fraction':client.step_fraction,'from':client.position,'center':center,'measurement':candidate})
        if client.early_enabled and early_clear(client, initial_point, initial_svd, channel, state, polygon, candidate):
            return True
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if previous_candidate and math.dist(candidate, previous_candidate) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        previous_candidate = candidate
        response = client.measure(candidate[0], candidate[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        extra_used += 1
        kind = response.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if kind == "direction":
            svd = response.get("svd_deg")
            if not polygon_policy._valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)

        state["recovery_count"] += 1
        invisible = candidate
        while True:
            if client.info_enabled and polygon: client.local_polygons[channel] = polygon
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            recovery_center, recovery_radius = box
            if recovery_radius <= LOCATE_CLEAR_RADIUS_M:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            midpoint = ((last_visible[0] + invisible[0]) / 2.0, (last_visible[1] + invisible[1]) / 2.0)
            if math.dist(midpoint, last_visible) <= 1e-6 or math.dist(midpoint, invisible) <= 1e-6:
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if client.early_enabled and early_clear(client, initial_point, initial_svd, channel, state, polygon, midpoint):
                return True
            response = client.measure(midpoint[0], midpoint[1], channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            state["recovery_measure_count"] += 1
            extra_used += 1
            kind = response.get("measure_result")
            if kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if kind == "direction":
                svd = response.get("svd_deg")
                if not polygon_policy._valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if kind == "no_signal":
                invisible = midpoint
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)


def early_clear(client, initial, angle, channel, state, polygon, next_measure):
 points=polygon_policy.polygon_fallback_points(polygon,initial,angle)
 if not points:return False
 alpha,beta,exit_weight=client.settings
 here=client.position;walk=0.;costs=[];strict_costs=[]
 for i,point in enumerate(points):
  walk+=math.dist(here,point)/5;here=point
  strict_costs.append(walk+3*i+5+math.dist(point,next_measure)/5)
  costs.append(walk+3*i+5+exit_weight*math.dist(point,next_measure)/5)
 worst=max(costs);mean=sum(costs)/len(costs)
 lower=math.dist(client.position,next_measure)/5+10
 # Equal-weight success-index proxy, not a calibrated source posterior.
 # Preserve original strict trigger and cap relaxed worst-case local allowance.
 strict=max(strict_costs)<lower-1e-6
 relaxed=client.relaxed_enabled and mean<alpha*lower-1e-6 and worst<=beta*lower
 if not (strict or relaxed):return False
 state.setdefault('early_clear_events',[]).append({'worst_local_s':worst,'continue_lower_s':lower,'points':len(points),'expected_proxy_s':mean,'relaxed_only':not strict})
 return polygon_policy._fallback_scan(client,initial,angle,channel,state,polygon)
