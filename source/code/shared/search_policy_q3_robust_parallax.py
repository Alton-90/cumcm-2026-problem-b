"""Q3 combined v1 with robust local normal-offset measurement selection only."""
from __future__ import annotations

import math
from time import perf_counter

import search_policy as old
from search_policy_compact import _state
import search_policy_q3_combined as combined
from q3_robust_parallax_geometry import choose_measurement_point


VERSION = "q3_robust_parallax_v1"


def localize(client, initial_point, initial_response, channel, state, polygon):
    initial_svd = initial_response.get("svd_deg")
    if not combined._valid(initial_svd):
        state["warnings"].append(f"channel {channel}: direction response lacks numeric svd_deg")
        return False
    previous_target = None
    for extra_used in range(old.MAX_EXTRA_BEARINGS + 1):
        box = old._box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return old._fallback_scan(client, initial_point, initial_svd, channel, state)
        center, radius = box
        estimate = client.estimate(channel)
        if estimate is not None:
            center, radius = estimate
            if radius <= 60:
                if old._resolve_clear(client, center, channel, state, "candidate check"):
                    return True
        if radius <= old.LOCATE_CLEAR_RADIUS_M:
            if old._resolve_clear(client, center, channel, state, "bearing localization"):
                return True
            return old._fallback_scan(client, initial_point, initial_svd, channel, state)
        if extra_used >= old.MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return old._fallback_scan(client, initial_point, initial_svd, channel, state)

        started = perf_counter()
        proposal = choose_measurement_point(polygon, center, client.position, initial_svd)
        state["robust_parallax_scoring_cpu_s"] += perf_counter() - started
        target = proposal["selected_point"]
        if previous_target and math.dist(target, previous_target) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization target")
            return old._fallback_scan(client, initial_point, initial_svd, channel, state)
        previous_target = target
        event = {
            "channel": channel,
            "extra_used_before": extra_used,
            "initial_svd_deg": initial_svd,
            "current_position": [client.position[0], client.position[1]],
            "before_polygon": [[point[0], point[1]] for point in polygon],
            "original_center": [center[0], center[1]],
            "selected_point": [target[0], target[1]],
            **{key: value for key, value in proposal.items() if key != "selected_point"},
            "measure_result": None,
            "svd_deg": None,
            "measure_index": None,
        }
        state["robust_parallax_events"].append(event)
        if proposal["used_offset"]:
            state["robust_parallax_selected_count"] += 1
        else:
            state["robust_parallax_original_center_count"] += 1
        follow_up = client.measure(target[0], target[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        event["measure_index"] = state["measure_count"]
        event["measure_result"] = follow_up.get("measure_result")
        event["svd_deg"] = follow_up.get("svd_deg")
        kind = follow_up.get("measure_result")
        if kind == "near":
            if old._resolve_clear(client, target, channel, state, "near follow-up"):
                return True
            return old._fallback_scan(client, initial_point, initial_svd, channel, state)
        if kind == "direction":
            next_svd = follow_up.get("svd_deg")
            if not combined._valid(next_svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks svd_deg")
                return old._fallback_scan(client, initial_point, initial_svd, channel, state)
            polygon = old._add_bearing(polygon, target, next_svd)
            continue
        if proposal["used_offset"] and kind == "no_signal":
            state["robust_parallax_guarantee_failure_count"] += 1
            event["guarantee_failure"] = True
        state["warnings"].append(
            f"channel {channel}: follow-up no_signal/unknown; retaining fallback"
        )
        return old._fallback_scan(client, initial_point, initial_svd, channel, state)
    raise AssertionError("unreachable extra-bearing loop")


def run_search(client, question=3, method="main"):
    if question != 3 or method != "main":
        raise ValueError("Q3 robust parallax supports only question=3, method='main'")
    client = combined.ValueTracked(client)
    points = [(0.0, 0.0)] + [
        (1124 * math.cos(index * math.pi / 3), 1124 * math.sin(index * math.pi / 3))
        for index in range(6)
    ]
    state = _state(3, points)
    state.update(
        robust_parallax_events=[],
        robust_parallax_selected_count=0,
        robust_parallax_original_center_count=0,
        robust_parallax_guarantee_failure_count=0,
        robust_parallax_scoring_cpu_s=0.0,
        coverage_anchor_points=[[point[0], point[1]] for point in points],
    )
    tracks = {}
    seen = set()
    remaining = list(range(len(points)))
    attempted = set()
    try:
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if len(state["cleared_channels"]) == 16:
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, (_, _, polygon) in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                box = client.estimate(channel) or old._box_circle(polygon)
                if box:
                    nodes.append(("source", channel, box[0]))
            if not nodes:
                break
            kind, index, destination = combined._route(client.position, nodes)[0]
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
                        old._resolve_clear(client, point, channel, state, "near joint-route scan")
                    elif result == "direction" and combined._valid(response.get("svd_deg")):
                        if channel not in tracks:
                            tracks[channel] = (
                                point,
                                response,
                                old._bearing_triangle(point, response["svd_deg"]),
                            )
                        else:
                            first, initial, polygon = tracks[channel]
                            updated = old._add_bearing(polygon, point, response["svd_deg"])
                            if updated:
                                tracks[channel] = (first, initial, updated)
                    else:
                        state["warnings"].append(f"channel {channel}: invalid public response")
            else:
                channel = index
                point, initial, polygon = tracks[channel]
                attempted.add(channel)
                if math.dist(client.position, point) > 1 and client.useful(
                    channel, client.position
                ):
                    here = client.position
                    extra = client.measure(*here, channel)
                    state["measure_count"] += 1
                    state["extra_bearing_count"] += 1
                    if extra.get("measure_result") == "near" and old._resolve_clear(
                        client, here, channel, state, "near current pose"
                    ):
                        continue
                    if extra.get("measure_result") == "direction" and combined._valid(
                        extra.get("svd_deg")
                    ):
                        updated = old._add_bearing(polygon, here, extra["svd_deg"])
                        if updated:
                            polygon = updated
                    tracks[channel] = (point, initial, polygon)
                    attempted.discard(channel)
                    continue
                localize(client, point, initial, channel, state, polygon)
        state["excluded_channels"] = set(range(1, 21)) - seen
        state["unresolved_channels"] = seen - state["cleared_channels"]
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"{type(error).__name__}: {error}")
        state["unresolved_channels"] = (
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
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
