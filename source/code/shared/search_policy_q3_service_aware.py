"""Experimental Q3 service-aware rolling open route.

Coverage, source bookkeeping, local cost rollout, and finite fallback are kept
from the existing controllers.  Only the next outer task may be reordered by a
public-state service-cost macro, with the frozen first task as the default.
"""
from __future__ import annotations

import math
from time import perf_counter

from search_policy_compact import _state
import search_policy as old
import search_policy_q3_combined as combined
import search_policy_q3_cost_rollout_v2 as cost_policy
from q3_service_aware_rollout import choose_task


VERSION = "q3_service_aware_v1"
_route = combined._route
_valid = combined._valid


def run_search(client, question=3, method="main"):
    if question != 3 or method != "main":
        raise ValueError("Q3 main only")
    client = cost_policy.PublicTracked(client)
    points = [(0.0, 0.0)] + [
        (1124 * math.cos(index * math.pi / 3), 1124 * math.sin(index * math.pi / 3))
        for index in range(6)
    ]
    state = _state(3, points)
    tracks = {}
    seen = set()
    remaining = list(range(len(points)))
    attempted = set()
    state.update(
        service_route_events=[],
        service_route_override_count=0,
        service_route_scoring_cpu_s=0.0,
    )
    try:
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if len(state["cleared_channels"]) == 16:
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, (point, response, polygon) in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                box = client.estimate(channel) or old._box_circle(polygon)
                if box:
                    nodes.append(("source", channel, box[0]))
            if not nodes:
                break
            proxy_route = _route(client.position, nodes)
            scoring_started = perf_counter()
            selected, event = choose_task(
                client, proxy_route, tracks, state["cleared_channels"]
            )
            state["service_route_scoring_cpu_s"] += perf_counter() - scoring_started
            event["decision_index"] = len(state["service_route_events"])
            event["measure_count_before"] = state["measure_count"]
            event["action_count_before"] = state["measure_count"] + state["clear_count"]
            state["service_route_events"].append(event)
            if event["decision"]["override"]:
                state["service_route_override_count"] += 1
            kind, index, destination = selected
            if kind == "scan":
                remaining.remove(index)
                point = destination
                for channel in range(1, 21):
                    if channel in state["cleared_channels"]:
                        continue
                    if channel in tracks:
                        if not client.useful(channel, point):
                            continue
                        box = old._box_circle(tracks[channel][2])
                        if box and box[1] <= old.LOCATE_CLEAR_RADIUS_M:
                            continue
                    response = client.measure(*point, channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        old._resolve_clear(
                            client, point, channel, state, "near joint-route scan"
                        )
                    elif result == "direction" and _valid(response.get("svd_deg")):
                        if channel not in tracks:
                            tracks[channel] = (
                                point,
                                response,
                                old._bearing_triangle(point, response["svd_deg"]),
                            )
                        else:
                            initial_point, initial_response, polygon = tracks[channel]
                            clipped = old._add_bearing(
                                polygon, point, response["svd_deg"]
                            )
                            if clipped:
                                tracks[channel] = (
                                    initial_point,
                                    initial_response,
                                    clipped,
                                )
                    else:
                        state["warnings"].append(
                            f"channel {channel}: invalid public response"
                        )
            else:
                channel = index
                initial_point, initial_response, polygon = tracks[channel]
                attempted.add(channel)
                if (
                    math.dist(client.position, initial_point) > 1.0
                    and client.useful(channel, client.position)
                ):
                    here = client.position
                    extra = client.measure(*here, channel)
                    state["measure_count"] += 1
                    state["extra_bearing_count"] += 1
                    if extra.get("measure_result") == "near" and old._resolve_clear(
                        client, here, channel, state, "near current pose"
                    ):
                        continue
                    if extra.get("measure_result") == "direction" and _valid(
                        extra.get("svd_deg")
                    ):
                        clipped = old._add_bearing(
                            polygon, here, extra["svd_deg"]
                        )
                        if clipped:
                            polygon = clipped
                    tracks[channel] = (initial_point, initial_response, polygon)
                    attempted.discard(channel)
                    continue
                cost_policy.localize(
                    client,
                    initial_point,
                    initial_response,
                    channel,
                    state,
                    polygon,
                )
        state["excluded_channels"] = set(range(1, 21)) - seen
        state["unresolved_channels"] = seen - state["cleared_channels"]
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(str(error))
        state["unresolved_channels"] = (
            set(range(1, 21))
            - state["cleared_channels"]
            - state["excluded_channels"]
        )
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state["skipped_known_measures"] = client.skipped_known_measures
    state.update(
        completed=(
            not state["stopped"]
            and not state["unresolved_channels"]
            and len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20
        ),
        question=3,
        method="main",
        version=VERSION,
        virtual_time_s=client.virtual_time_s,
    )
    return state
