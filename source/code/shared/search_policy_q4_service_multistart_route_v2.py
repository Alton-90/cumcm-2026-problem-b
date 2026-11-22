"""Q4 v2: choose among frozen, v1-service, and multistart full routes.

The candidate changes only the public outer ordering and the v1 redundant
same-point first-measure suppression.  Ring22 coverage, localization budgets,
continuous clear certificate, and both fallbacks remain frozen dependencies.
"""
from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from route_multistart import route as _multistart_route
from search_policy import _add_bearing, _bearing_triangle, _resolve_clear
from search_policy_joint_route_q3 import _route as _anchor_route
from search_policy_q4_batch import _localize, _state
from search_policy_q4_reuse import REUSE_MIN_BASELINE_M
from search_policy_q4_ring22 import coverage_certificate, ring22_points
from search_policy_q4_service_aware_route import (
    ServiceTrackedClient,
    _box,
    _direct,
    _node_key,
    _proxy_cost,
    _scannable_channels,
    _service_profiles,
    _stop,
    _supplement_without_duplicate,
)


VERSION = "q4_service_multistart_route_v2"
MIN_PROXY_SAVING_S = 6.0
GEOMETRY_TOLERANCE_M = 1e-7
PROXY_TIE_TOLERANCE_S = 1e-7


def _route_length(position, route):
    points = [tuple(map(float, position))] + [tuple(map(float, node[2])) for node in route]
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _route_keys(route):
    return [_node_key(node) for node in route]


def _v1_service_route(position, nodes, profiles, scannable, last_measure_points, anchor, anchor_proxy):
    best_route = anchor
    best_proxy = anchor_proxy
    first_choices = []
    for first in sorted(nodes, key=_node_key):
        remaining = list(nodes)
        remaining.remove(first)
        route = [first] + _anchor_route(profiles[_node_key(first)]["exit"], remaining)
        proxy = _proxy_cost(position, route, profiles, scannable, last_measure_points)
        first_choices.append({"first": _node_key(first), **proxy})
        candidate_order = (proxy["total_s"], proxy["movement_s"], _node_key(first))
        best_order = (best_proxy["total_s"], best_proxy["movement_s"], _node_key(best_route[0]))
        if candidate_order < best_order:
            best_route, best_proxy = route, proxy
    return best_route, best_proxy, first_choices


def _plan(position, nodes, tracks, state, client):
    profiles = _service_profiles(nodes, tracks)
    scannable = _scannable_channels(state, tracks)
    last_measure_points = client.last_measure_point

    anchor = _anchor_route(position, nodes)
    anchor_proxy = _proxy_cost(position, anchor, profiles, scannable, last_measure_points)
    v1_route, v1_proxy, v1_first_choices = _v1_service_route(
        position, nodes, profiles, scannable, last_measure_points, anchor, anchor_proxy
    )
    multistart_route = _multistart_route(position, nodes)
    multistart_proxy = _proxy_cost(
        position, multistart_route, profiles, scannable, last_measure_points
    )

    anchor_length = _route_length(position, anchor)
    route_records = {}
    for name, route, proxy in (
        ("anchor", anchor, anchor_proxy),
        ("v1_service", v1_route, v1_proxy),
        ("multistart_service", multistart_route, multistart_proxy),
    ):
        geometry_length = _route_length(position, route)
        proxy_saving = anchor_proxy["total_s"] - proxy["total_s"]
        proxy_guard = proxy_saving >= MIN_PROXY_SAVING_S
        geometry_guard = geometry_length <= anchor_length + GEOMETRY_TOLERANCE_M
        route_records[name] = {
            "route": _route_keys(route),
            "proxy": proxy,
            "geometry_length_m": geometry_length,
            "proxy_saving_vs_anchor_s": proxy_saving,
            "proxy_guard_pass": proxy_guard,
            "geometry_guard_pass": geometry_guard,
            "eligible": name != "anchor" and proxy_guard and geometry_guard,
        }

    eligible_names = [
        name for name in ("v1_service", "multistart_service") if route_records[name]["eligible"]
    ]
    selected_name = "anchor"
    selection_reason = "no_nonanchor_route_passed_both_guards"
    tie_names = []
    if eligible_names:
        minimum_proxy = min(route_records[name]["proxy"]["total_s"] for name in eligible_names)
        tie_names = [
            name
            for name in eligible_names
            if abs(route_records[name]["proxy"]["total_s"] - minimum_proxy) <= PROXY_TIE_TOLERANCE_S
        ]
        if len(tie_names) == 1:
            selected_name = tie_names[0]
            selection_reason = "unique_lowest_eligible_public_service_proxy"
        else:
            selection_reason = "eligible_proxy_tie_falls_back_to_anchor"

    selected_routes = {
        "anchor": anchor,
        "v1_service": v1_route,
        "multistart_service": multistart_route,
    }
    selected = selected_routes[selected_name]
    state["joint_route_selection_counts"][selected_name] += 1
    state["joint_route_decisions"].append(
        {
            "position": tuple(map(float, position)),
            "nodes": [
                {"kind": node[0], "ident": int(node[1]), "destination": tuple(map(float, node[2]))}
                for node in nodes
            ],
            "profiles": [{"key": key, **profile} for key, profile in sorted(profiles.items())],
            "scannable_channels": sorted(scannable),
            "last_measure_points": {
                str(channel): tuple(map(float, point))
                for channel, point in sorted(last_measure_points.items())
            },
            "routes": route_records,
            "v1_first_choices": v1_first_choices,
            "minimum_proxy_saving_s": MIN_PROXY_SAVING_S,
            "geometry_tolerance_m": GEOMETRY_TOLERANCE_M,
            "proxy_tie_tolerance_s": PROXY_TIE_TOLERANCE_S,
            "eligible_names": eligible_names,
            "tie_names": tie_names,
            "selected_strategy": selected_name,
            "selected_route": _route_keys(selected),
            "selected_first": _node_key(selected[0]),
            "selection_reason": selection_reason,
            "measure_count_before": state["measure_count"],
            "clear_count_before": state["clear_count"],
        }
    )
    return selected


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("Q4 service-multistart v2 supports only question=4, method='main'")

    points = ring22_points()
    state = _state(points)
    state.update(
        coverage_geometry=coverage_certificate(),
        route_strategy="anchor_vs_v1_service_vs_multistart_service_with_dual_guard",
        service_proxy_version="entry_service_exit_v1_unchanged",
        service_proxy_min_saving_s=MIN_PROXY_SAVING_S,
        joint_route_decisions=[],
        joint_route_selection_counts={"anchor": 0, "v1_service": 0, "multistart_service": 0},
        duplicate_supplement_skip_count=0,
        duplicate_supplement_skips=[],
        compatibility_used_for_decision=False,
        polygon_track_update_count=0,
        reuse_measure_count=0,
        reuse_polygon_update_count=0,
        direct_clear_skip_measure_count=0,
        maximum_count_stop_used=False,
        maximum_count_certificate=None,
    )
    tracked = ServiceTrackedClient(client)
    seen = set()
    tracks = {}
    attempted = set()
    remaining = list(range(len(points)))

    try:
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if _stop(state):
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, track in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                bound = _box(track)
                if bound is not None:
                    nodes.append(("source", channel, bound[0]))
                elif math.dist(tracked.position, track[0]) >= REUSE_MIN_BASELINE_M:
                    nodes.append(("source", channel, tracked.position))
            if not nodes:
                break

            kind, ident, destination = _plan(tracked.position, nodes, tracks, state, tracked)[0]
            if kind == "scan":
                remaining.remove(ident)
                for channel in range(1, 21):
                    if _stop(state):
                        break
                    if channel in state["cleared_channels"]:
                        continue
                    old_track = tracks.get(channel)
                    bound = _box(old_track) if old_track else None
                    if bound is not None and bound[1] <= 19.9:
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
                            initial_point, initial_svd, polygon = old_track
                            clipped = _add_bearing(polygon, destination, response["svd_deg"])
                            tracks[channel] = (
                                (initial_point, initial_svd, clipped) if clipped else old_track
                            )
                        attempted.discard(channel)
                        state["polygon_track_update_count"] += 1
                    else:
                        state["warnings"].append(f"channel {channel}: invalid public response")
                        state["unresolved_channels"].add(channel)
                state["batch_count"] += 1
                continue

            channel = ident
            track = tracks[channel]
            attempted.add(channel)
            if _direct(tracked, channel, track, state):
                continue
            track, cleared = _supplement_without_duplicate(tracked, channel, track, state)
            if cleared:
                continue
            tracks[channel] = track
            if _direct(tracked, channel, track, state):
                continue
            _localize(tracked, track[0], track[1], channel, state, track[2])

        if not state["maximum_count_stop_used"]:
            for channel in range(1, 21):
                if channel not in state["cleared_channels"]:
                    target = state["unresolved_channels"] if channel in seen else state["excluded_channels"]
                    target.add(channel)
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(error).__name__}: {error}")
        state["unresolved_channels"].update(
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
        )

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


__all__ = ["VERSION", "run_search"]
