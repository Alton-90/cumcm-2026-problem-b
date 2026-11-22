"""Q4 ring22 joint route with conservative shadow-cone no-signal pruning."""
from __future__ import annotations

import math
from time import perf_counter

import search_policy_polygon as polygon_policy
from search_policy import _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _state
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate, ring22_points
from shadow_cone_geometry import polygon_area, shadow_outer_polygon

VERSION = "q4_shadow_recovery_v1"
DIRECT_RADIUS = 19.9
MAX_SOURCES = 16
MAX_EXTRA_BEARINGS = 8
GEOMETRY_EVENT_CPU_BUDGET_S = 0.05
GEOMETRY_CHANNEL_CPU_BUDGET_S = 0.20
POINT_TOLERANCE_M = 1e-6


class PublicHistoryClient:
    def __init__(self, client):
        self._client = client
        self.position = (0.0, 0.0)
        self.positives = {}
        self.negatives = {}
        self.measured = {}
        self.cleared = set()
        self.shadow_disabled = set()
        self.shadow_cpu_by_channel = {}

    @property
    def virtual_time_s(self):
        return self._client.virtual_time_s

    @staticmethod
    def _append_unique(mapping, channel, point):
        value = (float(point[0]), float(point[1]))
        values = mapping.setdefault(channel, [])
        if not any(math.dist(value, old) <= POINT_TOLERANCE_M for old in values):
            values.append(value)

    def measure(self, x, y, channel):
        response = self._client.measure(x, y, channel)
        self.position = (x, y)
        if response.get("accepted") is True and channel not in self.cleared:
            point = (x, y)
            self._append_unique(self.measured, channel, point)
            kind = response.get("measure_result")
            if kind == "near" or (kind == "direction" and polygon_policy._valid_svd(response.get("svd_deg"))):
                self._append_unique(self.positives, channel, point)
            elif kind == "no_signal":
                self._append_unique(self.negatives, channel, point)
        return response

    def clear(self, x, y, channel):
        response = self._client.clear(x, y, channel)
        self.position = (x, y)
        if response.get("accepted") is True and response.get("clear_result") == "success":
            self.cleared.add(channel)
        return response


def _stop(state):
    if len(state["cleared_channels"]) != MAX_SOURCES:
        return False
    state["excluded_channels"].update(set(range(1, 21)) - state["cleared_channels"])
    state["maximum_count_certificate"] = {
        "maximum_sources": MAX_SOURCES,
        "cleared_distinct_channels": MAX_SOURCES,
        "rule": "remaining channels excluded only after 16 successful distinct clears",
    }
    state["maximum_count_stop_used"] = True
    return True


def _box(track):
    return _box_circle(track[2])


def _direct(client, channel, track, state):
    box = _box(track)
    if box is None or box[1] > DIRECT_RADIUS:
        return False
    state["direct_clear_skip_measure_count"] += 1
    return _resolve_clear(client, box[0], channel, state, "19.9m public-polygon certificate")


def _same_point(point, points):
    return any(math.dist(point, old) <= POINT_TOLERANCE_M for old in points)


def _attempt_shadow(
    client, channel, polygon, negative, last_visible, invisible, extra_used, localization_id, state
):
    positives = client.positives.get(channel, [])
    old_next = ((last_visible[0] + invisible[0]) / 2, (last_visible[1] + invisible[1]) / 2)
    event = {
        "channel": channel,
        "localization_id": localization_id,
        "negative": [negative[0], negative[1]],
        "positive_count": len(positives),
        "positive_snapshot": [[point[0], point[1]] for point in positives],
        "positive_pair": None,
        "before_polygon": [[point[0], point[1]] for point in polygon],
        "after_polygon": [[point[0], point[1]] for point in polygon],
        "old_binary_next": [old_next[0], old_next[1]],
        "new_center": None,
        "new_center_previously_measured": None,
        "candidate_radius": None,
        "actual_next_kind": "old_binary",
        "extra_used": extra_used,
        "actual_exit_recovery": False,
        "status": "disabled" if channel in client.shadow_disabled else "not_attempted",
        "geometry_status": None,
    }
    state["shadow_events"].append(event)
    if channel in client.shadow_disabled:
        return None, event
    state["shadow_attempt_count"] += 1
    started = perf_counter()
    result = shadow_outer_polygon(polygon, positives, negative)
    elapsed = perf_counter() - started
    client.shadow_cpu_by_channel[channel] = client.shadow_cpu_by_channel.get(channel, 0.0) + elapsed
    state["shadow_geometry_cpu_s"] += elapsed
    event["geometry_cpu_s"] = elapsed
    event["status"] = result["status"]
    event["geometry_status"] = result["status"]
    event["positive_pair"] = (
        [[point[0], point[1]] for point in result["pair"]] if result.get("pair") else None
    )
    candidate = result.get("polygon", polygon)
    event["candidate_polygon"] = [[point[0], point[1]] for point in candidate]
    event["area_before"] = polygon_area(polygon)
    event["candidate_area"] = polygon_area(candidate)
    event["area_after"] = event["area_before"]
    if result["status"] == "negative_inside_positive_hull":
        state["shadow_conflict_count"] += 1
    if elapsed > GEOMETRY_EVENT_CPU_BUDGET_S or client.shadow_cpu_by_channel[channel] > GEOMETRY_CHANNEL_CPU_BUDGET_S:
        client.shadow_disabled.add(channel)
        state["shadow_cpu_fallback_count"] += 1
        event["status"] = "cpu_budget_exceeded"
        return None, event
    if result["status"] != "updated" or not polygon_policy._polygon_is_valid(candidate):
        state["shadow_no_progress_count"] += 1
        return None, event
    box = _box_circle(candidate)
    if box is None:
        state["shadow_no_progress_count"] += 1
        event["status"] = "invalid_updated_polygon"
        return None, event
    new_center = box[0]
    new_radius = box[1]
    event["new_center"] = [new_center[0], new_center[1]]
    event["candidate_radius"] = new_radius
    measured = client.measured.get(channel, [])
    event["new_center_previously_measured"] = _same_point(new_center, measured)
    if new_radius > DIRECT_RADIUS and (
        event["new_center_previously_measured"]
        or math.dist(new_center, negative) <= POINT_TOLERANCE_M
        or math.dist(new_center, old_next) <= POINT_TOLERANCE_M
    ):
        state["shadow_no_progress_count"] += 1
        event["status"] = "updated_but_next_action_unchanged"
        return None, event
    event["after_polygon"] = event["candidate_polygon"]
    event["area_after"] = event["candidate_area"]
    event["actual_next_kind"] = "direct_clear" if new_radius <= DIRECT_RADIUS else "new_measure"
    state["shadow_applied_count"] += 1
    return candidate, event


def _fallback(client, initial_point, initial_svd, channel, state, polygon):
    return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)


def _localize(client, initial_point, initial_svd, channel, state, feasible_polygon):
    state["shadow_localization_serial"] += 1
    localization_id = state["shadow_localization_serial"]
    polygon = feasible_polygon
    last_visible = initial_point
    previous_candidate = None
    extra_used = 0
    while True:
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)
        candidate, radius = box
        if radius <= DIRECT_RADIUS:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)
        if previous_candidate and math.dist(candidate, previous_candidate) <= POINT_TOLERANCE_M:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)
        previous_candidate = candidate
        response = client.measure(candidate[0], candidate[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        extra_used += 1
        kind = response.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)
        if kind == "direction":
            svd = response.get("svd_deg")
            if not polygon_policy._valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)

        state["recovery_count"] += 1
        invisible = candidate
        updated, event = _attempt_shadow(
            client,
            channel,
            polygon,
            candidate,
            last_visible,
            invisible,
            extra_used,
            localization_id,
            state,
        )
        if updated is not None:
            polygon = updated
            event["actual_exit_recovery"] = True
            state["shadow_exit_recovery_count"] += 1
            continue
        while True:
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            recovery_center, recovery_radius = box
            if recovery_radius <= DIRECT_RADIUS:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            midpoint = ((last_visible[0] + invisible[0]) / 2, (last_visible[1] + invisible[1]) / 2)
            if (
                math.dist(midpoint, last_visible) <= POINT_TOLERANCE_M
                or math.dist(midpoint, invisible) <= POINT_TOLERANCE_M
            ):
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            response = client.measure(midpoint[0], midpoint[1], channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            state["recovery_measure_count"] += 1
            extra_used += 1
            kind = response.get("measure_result")
            if kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return _fallback(client, initial_point, initial_svd, channel, state, polygon)
            if kind == "direction":
                svd = response.get("svd_deg")
                if not polygon_policy._valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return _fallback(client, initial_point, initial_svd, channel, state, polygon)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if kind == "no_signal":
                invisible = midpoint
                updated, event = _attempt_shadow(
                    client,
                    channel,
                    polygon,
                    midpoint,
                    last_visible,
                    invisible,
                    extra_used,
                    localization_id,
                    state,
                )
                if updated is not None:
                    polygon = updated
                    event["actual_exit_recovery"] = True
                    state["shadow_exit_recovery_count"] += 1
                    break
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {kind!r}")
            return _fallback(client, initial_point, initial_svd, channel, state, polygon)


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("Q4 shadow recovery only")
    points = ring22_points()
    state = _state(points)
    state.update(
        coverage_geometry=coverage_certificate(),
        route_strategy="receding_open_2opt_over_scans_and_public_polygon_centers",
        polygon_track_update_count=0,
        reuse_measure_count=0,
        reuse_polygon_update_count=0,
        direct_clear_skip_measure_count=0,
        maximum_count_stop_used=False,
        maximum_count_certificate=None,
        shadow_attempt_count=0,
        shadow_applied_count=0,
        shadow_exit_recovery_count=0,
        shadow_no_progress_count=0,
        shadow_conflict_count=0,
        shadow_cpu_fallback_count=0,
        shadow_geometry_cpu_s=0.0,
        shadow_localization_serial=0,
        shadow_events=[],
    )
    tracked = PublicHistoryClient(client)
    seen = set()
    tracks = {}
    attempted = set()
    remaining = list(range(len(points)))
    try:
        while remaining or any(ch not in state["cleared_channels"] and ch not in attempted for ch in tracks):
            if _stop(state):
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, track in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                box = _box(track)
                if box:
                    nodes.append(("source", channel, box[0]))
                elif math.dist(tracked.position, track[0]) >= 50:
                    nodes.append(("source", channel, tracked.position))
            if not nodes:
                break
            kind, identifier, destination = _route(tracked.position, nodes)[0]
            if kind == "scan":
                remaining.remove(identifier)
                for channel in range(1, 21):
                    if _stop(state):
                        break
                    if channel in state["cleared_channels"]:
                        continue
                    old_track = tracks.get(channel)
                    box = _box(old_track) if old_track else None
                    if box and box[1] <= DIRECT_RADIUS:
                        continue
                    response = tracked.measure(*destination, channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        _resolve_clear(tracked, destination, channel, state, "near ring22 joint coverage")
                    elif result == "direction" and polygon_policy._valid_svd(response.get("svd_deg")):
                        if old_track is None:
                            tracks[channel] = (
                                destination,
                                response["svd_deg"],
                                _bearing_triangle(destination, response["svd_deg"]),
                            )
                        else:
                            point, svd, polygon = old_track
                            merged = _add_bearing(polygon, destination, response["svd_deg"])
                            tracks[channel] = (point, svd, merged) if merged else old_track
                        attempted.discard(channel)
                        state["polygon_track_update_count"] += 1
                    else:
                        state["warnings"].append(f"channel {channel}: invalid public response")
                        state["unresolved_channels"].add(channel)
                state["batch_count"] += 1
            else:
                channel = identifier
                track = tracks[channel]
                attempted.add(channel)
                if _direct(tracked, channel, track, state):
                    continue
                track, cleared = _supplement_from_current_pose(tracked, channel, track, state)
                if cleared:
                    continue
                tracks[channel] = track
                if _direct(tracked, channel, track, state):
                    continue
                _localize(tracked, track[0], track[1], channel, state, track[2])
        if not state["maximum_count_stop_used"]:
            for channel in range(1, 21):
                if channel not in state["cleared_channels"]:
                    (state["unresolved_channels"] if channel in seen else state["excluded_channels"]).add(channel)
    except Exception as exc:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(exc).__name__}: {exc}")
        state["unresolved_channels"].update(
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
        )
    state["shadow_public_history"] = {
        str(channel): {
            "positive": [[point[0], point[1]] for point in tracked.positives.get(channel, [])],
            "negative": [[point[0], point[1]] for point in tracked.negatives.get(channel, [])],
            "measured": [[point[0], point[1]] for point in tracked.measured.get(channel, [])],
        }
        for channel in sorted(set(tracked.positives) | set(tracked.negatives) | set(tracked.measured))
    }
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state.update(
        completed=(
            not state["stopped"]
            and not state["unresolved_channels"]
            and len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20
        ),
        question=4,
        method="main",
        version=VERSION,
        virtual_time_s=tracked.virtual_time_s,
    )
    return state
