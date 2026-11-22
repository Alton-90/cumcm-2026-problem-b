"""Q3 adaptive annular coverage using public observations only.

The origin certifies the inner 1000 m disk. Full scans of every still-unseen
channel add conservative angular intervals for the 1000..1800 m annulus.
Observed-source estimates and localization retain the existing Q3 machinery.
"""
from __future__ import annotations

import math

import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked, localize
from search_policy_joint_route_q3 import _route

VERSION = "q3_adaptive_annular_cover_v1"
RECEPTION_M = 999.9
INNER_RADIUS_M = 1000.0
OUTER_RADIUS_M = 1800.0
SCAN_RADIUS_M = math.sqrt(OUTER_RADIUS_M**2 - RECEPTION_M**2)
MIN_NEW_ARC_DEG = 20.0
MAX_SOURCES = 16


def _valid(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value < 360
    )


def _half_width_deg(point):
    radius = math.hypot(*point)
    if radius <= 1e-12:
        return None
    values = []
    for source_radius in (INNER_RADIUS_M, OUTER_RADIUS_M):
        denominator = 2 * radius * source_radius
        cosine = (radius * radius + source_radius * source_radius - RECEPTION_M**2) / denominator
        if cosine < -1 or cosine > 1:
            return None
        values.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return min(values)


def _merge(intervals):
    ordered = sorted((max(0.0, a), min(360.0, b)) for a, b in intervals if b > a)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1e-10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _add_interval(intervals, center_deg, half_width_deg):
    center = center_deg % 360
    start, end = center - half_width_deg, center + half_width_deg
    pieces = list(intervals)
    if start < 0:
        pieces.extend(((0.0, end), (start + 360, 360.0)))
    elif end > 360:
        pieces.extend(((start, 360.0), (0.0, end - 360)))
    else:
        pieces.append((start, end))
    return _merge(pieces)


def _covered_degrees(intervals):
    return sum(end - start for start, end in _merge(intervals))


def _uncovered_arcs(intervals):
    merged = _merge(intervals)
    if not merged:
        return [(0.0, 360.0)]
    if _covered_degrees(merged) >= 360 - 1e-8:
        return []
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1) if merged[i + 1][0] > merged[i][1] + 1e-8]
    wrap_start, wrap_end = merged[-1][1], merged[0][0] + 360
    if wrap_end > wrap_start + 1e-8:
        gaps.append((wrap_start, wrap_end))
    return gaps


def _gain_degrees(intervals, point):
    half_width = _half_width_deg(point)
    if half_width is None:
        return 0.0
    angle = math.degrees(math.atan2(point[1], point[0]))
    return _covered_degrees(_add_interval(intervals, angle, half_width)) - _covered_degrees(intervals)


def _planned_scans(intervals):
    half_width = _half_width_deg((SCAN_RADIUS_M, 0.0))
    nodes = []
    for gap_index, (start, end) in enumerate(_uncovered_arcs(intervals)):
        length = end - start
        pieces = max(1, math.ceil(length / (2 * half_width)))
        piece_width = length / pieces
        for piece_index in range(pieces):
            if piece_width + 1e-8 < MIN_NEW_ARC_DEG:
                continue
            angle_deg = (start + (piece_index + 0.5) * piece_width) % 360
            angle = math.radians(angle_deg)
            point = (SCAN_RADIUS_M * math.cos(angle), SCAN_RADIUS_M * math.sin(angle))
            if _gain_degrees(intervals, point) + 1e-8 >= MIN_NEW_ARC_DEG:
                nodes.append(("scan", f"adaptive:{gap_index}:{piece_index}", point))
    return nodes


def _stop_at_maximum(state):
    if len(state["cleared_channels"]) != MAX_SOURCES:
        return False
    state["excluded_channels"].update(set(range(1, 21)) - state["cleared_channels"])
    state["maximum_count_certificate"] = {
        "maximum_sources": MAX_SOURCES,
        "cleared_distinct_channels": MAX_SOURCES,
    }
    state["maximum_count_stop_used"] = True
    return True


def run_search(client, question=3, method="main"):
    if question != 3 or method != "main":
        raise ValueError("Q3 main only")
    tracked = Tracked(client)
    fallback_points = [(0.0, 0.0)] + [
        (1150 * math.cos(k * math.pi / 3), 1150 * math.sin(k * math.pi / 3)) for k in range(6)
    ]
    state = _state(3, fallback_points)
    state.update(
        coverage_strategy="origin plus adaptive certified annular intervals; original seven-point fallback",
        coverage_reception_radius_m=RECEPTION_M,
        adaptive_scan_radius_m=SCAN_RADIUS_M,
        minimum_new_arc_deg=MIN_NEW_ARC_DEG,
        inner_disk_covered=False,
        annulus_intervals_deg=[],
        annulus_covered_degrees=0.0,
        annulus_coverage_complete=False,
        coverage_events=[],
        adaptive_scan_count=0,
        opportunistic_current_scan_count=0,
        fallback_coverage_used=False,
        fallback_coverage_scan_count=0,
        maximum_count_stop_used=False,
        maximum_count_certificate=None,
    )
    tracks = {}
    seen = set()
    attempted = set()
    scanned_points = set()
    intervals = []
    fallback_remaining = []

    def point_key(point):
        return round(point[0], 6), round(point[1], 6)

    def scan(point, scan_kind):
        nonlocal intervals
        unseen_before = sorted(set(range(1, 21)) - seen)
        newly_seen = []
        for channel in unseen_before:
            response = tracked.measure(*point, channel)
            state["measure_count"] += 1
            state["coverage_measurements_by_channel"][channel] += 1
            result = response.get("measure_result")
            if result == "no_signal":
                continue
            seen.add(channel)
            newly_seen.append(channel)
            if result == "near":
                old._resolve_clear(tracked, point, channel, state, "near adaptive coverage")
            elif result == "direction" and _valid(response.get("svd_deg")):
                if channel not in tracks:
                    tracks[channel] = (point, response, old._bearing_triangle(point, response["svd_deg"]))
                else:
                    first, first_response, polygon = tracks[channel]
                    merged = old._add_bearing(polygon, point, response["svd_deg"])
                    if merged:
                        tracks[channel] = (first, first_response, merged)
            else:
                state["warnings"].append(f"channel {channel}: invalid public response")
        scanned_points.add(point_key(point))
        event = {
            "kind": scan_kind,
            "point": [point[0], point[1]],
            "scanned_unseen_channels": unseen_before,
            "newly_seen_channels": newly_seen,
        }
        if math.hypot(*point) <= 1e-8:
            state["inner_disk_covered"] = True
        else:
            half_width = _half_width_deg(point)
            if half_width is not None:
                angle = math.degrees(math.atan2(point[1], point[0]))
                before = _covered_degrees(intervals)
                intervals = _add_interval(intervals, angle, half_width)
                event.update(annulus_center_deg=angle % 360, annulus_half_width_deg=half_width,
                             new_arc_deg=_covered_degrees(intervals) - before)
        state["coverage_events"].append(event)

    try:
        while True:
            if _stop_at_maximum(state):
                break
            coverage_complete = state["inner_disk_covered"] and not _uncovered_arcs(intervals)
            source_nodes = []
            for channel, (point, response, polygon) in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                estimate = tracked.estimate(channel) or old._box_circle(polygon)
                if estimate:
                    source_nodes.append(("source", channel, estimate[0]))
            scan_nodes = []
            if not state["inner_disk_covered"]:
                scan_nodes = [("scan", "origin", (0.0, 0.0))]
            elif not coverage_complete:
                if fallback_remaining:
                    scan_nodes = [("scan", f"fallback:{index}", fallback_points[index]) for index in fallback_remaining]
                else:
                    if point_key(tracked.position) not in scanned_points and _gain_degrees(intervals, tracked.position) + 1e-8 >= MIN_NEW_ARC_DEG:
                        scan_nodes.append(("scan", "current", tracked.position))
                    scan_nodes.extend(
                        node for node in _planned_scans(intervals) if point_key(node[2]) not in scanned_points
                    )
                    if not scan_nodes:
                        fallback_remaining = [
                            index for index in range(1, len(fallback_points))
                            if point_key(fallback_points[index]) not in scanned_points
                        ]
                        state["fallback_coverage_used"] = True
                        scan_nodes = [("scan", f"fallback:{index}", fallback_points[index]) for index in fallback_remaining]
            nodes = scan_nodes + source_nodes
            if not nodes:
                break
            kind, identifier, destination = _route(tracked.position, nodes)[0]
            if kind == "scan":
                if identifier == "current":
                    state["opportunistic_current_scan_count"] += 1
                    scan_kind = "current"
                elif str(identifier).startswith("fallback:"):
                    fallback_index = int(str(identifier).split(":", 1)[1])
                    fallback_remaining.remove(fallback_index)
                    state["fallback_coverage_scan_count"] += 1
                    scan_kind = "fallback"
                elif identifier == "origin":
                    scan_kind = "origin"
                else:
                    state["adaptive_scan_count"] += 1
                    scan_kind = "adaptive"
                scan(destination, scan_kind)
                continue
            channel = identifier
            first, response, polygon = tracks[channel]
            attempted.add(channel)
            if math.dist(tracked.position, first) > 1:
                here = tracked.position
                extra = tracked.measure(*here, channel)
                state["measure_count"] += 1
                state["extra_bearing_count"] += 1
                if extra.get("measure_result") == "near" and old._resolve_clear(
                    tracked, here, channel, state, "near current pose"
                ):
                    continue
                if extra.get("measure_result") == "direction" and _valid(extra.get("svd_deg")):
                    merged = old._add_bearing(polygon, here, extra["svd_deg"])
                    if merged:
                        polygon = merged
            localize(tracked, first, response, channel, state, polygon)
        if not state["maximum_count_stop_used"]:
            coverage_complete = state["inner_disk_covered"] and not _uncovered_arcs(intervals)
            if coverage_complete:
                state["excluded_channels"] = set(range(1, 21)) - seen
                state["unresolved_channels"] = seen - state["cleared_channels"]
            else:
                state["warnings"].append("continuous coverage certificate incomplete")
                state["unresolved_channels"] = set(range(1, 21)) - state["cleared_channels"]
    except Exception as exc:
        state["stopped"] = True
        state["warnings"].append(str(exc))
        state["unresolved_channels"].update(
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
        )
    state["annulus_intervals_deg"] = [[start, end] for start, end in _merge(intervals)]
    state["annulus_covered_degrees"] = _covered_degrees(intervals)
    state["annulus_coverage_complete"] = not _uncovered_arcs(intervals)
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state.update(
        completed=(
            not state["stopped"]
            and not state["unresolved_channels"]
            and len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20
        ),
        question=3,
        method="main",
        version=VERSION,
        virtual_time_s=tracked.virtual_time_s,
    )
    return state
