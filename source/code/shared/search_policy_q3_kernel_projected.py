"""Q3 combined route with one guaranteed-reception-kernel measurement per replan."""
from __future__ import annotations

import math
import numpy as np

import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import localize
import search_policy_q3_combined as combined
from q3_reception_kernel import (
    FEASIBILITY_TOLERANCE_M,
    KERNEL_RADIUS_M,
    cloud_hash,
    convex_hull,
    maximum_distance,
    projected_kernel_target,
)


VERSION = "q3_kernel_projected_v1"
MINIMUM_BASELINE_M = 50.0
MAX_REPLAN_BEARINGS = old.MAX_EXTRA_BEARINGS
CLOUD_ENVELOPE_M = 15.0


class KernelTracked(combined.ValueTracked):
    def __init__(self, client):
        super().__init__(client)
        self.observations = []
        self.action_role = "unclassified"
        self.default_action_role = "unclassified"

    def measure(self, x, y, channel):
        response = super().measure(x, y, channel)
        cloud = self.clouds.get(channel)
        role = (
            self.action_role
            if self.action_role != "unclassified"
            else self.default_action_role
        )
        self.observations.append(
            {
                "observation_index": len(self.observations),
                "channel": channel,
                "point": [float(x), float(y)],
                "action_role": role,
                "measure_result": response.get("measure_result"),
                "svd_deg": response.get("svd_deg"),
                "cloud_count_after": 0 if cloud is None else int(len(cloud)),
                "cloud_hash_after": None if cloud is None else cloud_hash(cloud),
            }
        )
        self.action_role = "unclassified"
        return response


def _projection_proposal(client, channel, start, successor):
    estimate = client.estimate(channel)
    cloud = client.clouds.get(channel)
    if estimate is None or cloud is None or len(cloud) == 0:
        return None, {
            "status": "mean_fallback",
            "fallback_reason": "empty_cloud",
            "used_kernel": False,
            "mean_target": None if estimate is None else list(estimate[0]),
        }
    mean = tuple(map(float, estimate[0]))
    hull = convex_hull(cloud)
    base = {
        "cloud_hash_before": cloud_hash(cloud),
        "cloud_count_before": int(len(cloud)),
        "convex_hull_vertices": hull.tolist(),
        "start": [float(start[0]), float(start[1])],
        "route_successor": [float(successor[0]), float(successor[1])],
        "mean_target": [mean[0], mean[1]],
        "kernel_radius_m": KERNEL_RADIUS_M,
        "cloud_envelope_m": CLOUD_ENVELOPE_M,
        "minimum_baseline_m": MINIMUM_BASELINE_M,
    }
    result = projected_kernel_target(cloud, start, successor)
    if result.get("status") != "ok":
        return mean, {
            **base,
            **{key: value for key, value in result.items() if key != "hull"},
            "status": "mean_fallback",
            "fallback_reason": result.get("status"),
            "used_kernel": False,
            "useful": None,
            "baseline_m": None,
        }
    projected = result["point"]
    useful = client.useful(channel, projected)
    last = client.last_positions.get(channel)
    baseline_m = None if last is None else math.dist(projected, last)
    fallback_reason = None
    if not useful:
        fallback_reason = "value_tracked_not_useful"
    elif baseline_m is not None and baseline_m < MINIMUM_BASELINE_M:
        fallback_reason = "baseline_below_50m"
    if fallback_reason is not None:
        return mean, {
            **base,
            **{key: value for key, value in result.items() if key not in ("hull", "point")},
            "projected_point": [projected[0], projected[1]],
            "status": "mean_fallback",
            "fallback_reason": fallback_reason,
            "used_kernel": False,
            "useful": useful,
            "baseline_m": baseline_m,
        }
    max_cloud_distance = maximum_distance(projected, cloud)
    return projected, {
        **base,
        **{key: value for key, value in result.items() if key not in ("hull", "point")},
        "projected_point": [projected[0], projected[1]],
        "status": "kernel_projection",
        "fallback_reason": None,
        "used_kernel": True,
        "useful": useful,
        "baseline_m": baseline_m,
        "max_cloud_distance_m": max_cloud_distance,
        "kernel_residual_m": max_cloud_distance - KERNEL_RADIUS_M,
        "stationary_projection": math.dist(projected, start) <= 1e-9,
    }


def _planned_route(client, remaining, points, tracks, attempted, cleared):
    nodes = [("scan", index, points[index]) for index in remaining]
    means = {}
    for channel, (_, _, polygon) in tracks.items():
        if channel in attempted or channel in cleared:
            continue
        estimate = client.estimate(channel) or old._box_circle(polygon)
        if estimate:
            means[channel] = estimate
            nodes.append(("source", channel, estimate[0]))
    preliminary = combined._route(client.position, nodes)
    proposals = {}
    replacements = {}
    for index, (kind, channel, _) in enumerate(preliminary):
        if kind != "source" or means[channel][1] <= 60:
            continue
        successor = preliminary[index + 1][2] if index + 1 < len(preliminary) else client.position
        target, proposal = _projection_proposal(client, channel, client.position, successor)
        if target is not None:
            replacements[channel] = target
            proposals[channel] = proposal
    projected_nodes = [
        (kind, identifier, replacements.get(identifier, point) if kind == "source" else point)
        for kind, identifier, point in preliminary
    ]
    return combined._route(client.position, projected_nodes), proposals


def _original_localize(client, point, initial, channel, state, polygon):
    client.default_action_role = "original_localizer_fallback"
    try:
        return localize(client, point, initial, channel, state, polygon)
    finally:
        client.default_action_role = "unclassified"


def run_search(client, question=3, method="main"):
    if question != 3 or method != "main":
        raise ValueError("Q3 kernel-projected policy supports only question=3, method='main'")
    client = KernelTracked(client)
    points = [(0.0, 0.0)] + [
        (1124 * math.cos(index * math.pi / 3), 1124 * math.sin(index * math.pi / 3))
        for index in range(6)
    ]
    state = _state(3, points)
    state.update(
        route_strategy="combined_radial_route_with_kernel_projected_single_action_replan",
        coverage_anchor_points=[[point[0], point[1]] for point in points],
        coverage_scan_points_visited=[],
        kernel_radius_m=KERNEL_RADIUS_M,
        cloud_envelope_m=CLOUD_ENVELOPE_M,
        minimum_baseline_m=MINIMUM_BASELINE_M,
        maximum_replan_bearings=MAX_REPLAN_BEARINGS,
        kernel_projection_events=[],
        kernel_projection_count=0,
        stationary_projection_count=0,
        mean_fallback_count=0,
        kernel_no_signal_count=0,
        original_localizer_fallback_count=0,
        per_channel_replan_measure_count={str(channel): 0 for channel in range(1, 21)},
    )
    tracks = {}
    seen = set()
    remaining = list(range(len(points)))
    attempted = set()
    replan_used = {channel: 0 for channel in range(1, 21)}
    try:
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if len(state["cleared_channels"]) == 16:
                break
            route, proposals = _planned_route(
                client, remaining, points, tracks, attempted, state["cleared_channels"]
            )
            if not route:
                break
            kind, identifier, destination = route[0]
            if kind == "scan":
                remaining.remove(identifier)
                state["coverage_scan_points_visited"].append([destination[0], destination[1]])
                for channel in range(1, 21):
                    if channel in state["cleared_channels"]:
                        continue
                    if channel in tracks:
                        if not client.useful(channel, destination):
                            continue
                        box = old._box_circle(tracks[channel][2])
                        if box and box[1] <= old.LOCATE_CLEAR_RADIUS_M:
                            continue
                    client.action_role = "coverage_scan"
                    response = client.measure(*destination, channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        old._resolve_clear(client, destination, channel, state, "near kernel-route scan")
                    elif result == "direction" and combined._valid(response.get("svd_deg")):
                        if channel not in tracks:
                            tracks[channel] = (
                                destination,
                                response,
                                old._bearing_triangle(destination, response["svd_deg"]),
                            )
                        else:
                            point, initial, polygon = tracks[channel]
                            updated = old._add_bearing(polygon, destination, response["svd_deg"])
                            if updated:
                                tracks[channel] = (point, initial, updated)
                        attempted.discard(channel)
                    else:
                        state["warnings"].append(f"channel {channel}: invalid public response")
                continue

            channel = identifier
            point, initial, polygon = tracks[channel]
            estimate = client.estimate(channel)
            if estimate is not None and estimate[1] <= 60:
                attempted.add(channel)
                _original_localize(client, point, initial, channel, state, polygon)
                continue
            if replan_used[channel] >= MAX_REPLAN_BEARINGS:
                attempted.add(channel)
                state["original_localizer_fallback_count"] += 1
                _original_localize(client, point, initial, channel, state, polygon)
                continue

            proposal = proposals.get(channel)
            if proposal is None:
                target = (client.estimate(channel) or old._box_circle(polygon))[0]
                proposal = {
                    "status": "mean_fallback",
                    "fallback_reason": "projection_plan_unavailable",
                    "used_kernel": False,
                    "mean_target": [target[0], target[1]],
                    "useful": None,
                    "baseline_m": None,
                }
            else:
                target = destination
            replan_used[channel] += 1
            state["per_channel_replan_measure_count"][str(channel)] += 1
            if proposal["used_kernel"]:
                state["kernel_projection_count"] += 1
                if proposal.get("stationary_projection"):
                    state["stationary_projection_count"] += 1
                client.action_role = "kernel_projection"
            else:
                state["mean_fallback_count"] += 1
                client.action_role = "mean_anchor_replan"
            response = client.measure(target[0], target[1], channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            event = {
                **proposal,
                "channel": channel,
                "target": [float(target[0]), float(target[1])],
                "replan_measure_number": replan_used[channel],
                "observation_index": len(client.observations) - 1,
                "measure_result": response.get("measure_result"),
                "svd_deg": response.get("svd_deg"),
            }
            state["kernel_projection_events"].append(event)
            result = response.get("measure_result")
            if result == "near":
                if old._resolve_clear(client, target, channel, state, "near projected replan"):
                    continue
                attempted.add(channel)
                state["original_localizer_fallback_count"] += 1
                _original_localize(client, point, initial, channel, state, polygon)
                continue
            if result == "direction" and combined._valid(response.get("svd_deg")):
                updated = old._add_bearing(polygon, target, response["svd_deg"])
                if updated:
                    tracks[channel] = (point, initial, updated)
                attempted.discard(channel)
                continue

            if proposal["used_kernel"] and result == "no_signal":
                state["kernel_no_signal_count"] += 1
                event["model_inconsistency"] = "guaranteed_reception_kernel_returned_no_signal"
            attempted.add(channel)
            state["original_localizer_fallback_count"] += 1
            _original_localize(client, point, initial, channel, state, polygon)

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
    state["public_observations"] = client.observations
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
