"""Pure public-state scorer for the predeclared Q3 edge opportunity probe.

The finite public-hypothesis scenarios are a deterministic screening model, not
a performance or localization guarantee.  Reception and collinearity checks are
certificates over the saved public feasible set and selected route edge.
"""
from __future__ import annotations

import math
import numpy as np

import search_policy as old
import search_policy_q3_combined as combined
from q3_cost_rollout_v2 import (
    BIASES,
    compatible_hypotheses,
    longest_axis,
    update_cloud,
)


VERSION = "q3_edge_opportunity_scorer_v1"
WAYPOINT_FRACTIONS = (0.25, 0.50, 0.75)
MIN_EDGE_LENGTH_M = 200.0
VISIBILITY_LIMIT_M = 999.9
MIN_LAST_MEASURE_SEPARATION_M = 1.0
COLLINEARITY_TOLERANCE_M = 1e-7
MIN_WORST_CASE_RADIUS_SHRINK_M = 20.0
SAME_CHANNEL_EQUIVALENT_M = 25.0
SWITCH_CHANNEL_EQUIVALENT_M = 30.0
REORDER_BOUND_EPSILON_M = 1e-6


def cloud_radius(cloud):
    points = np.asarray(cloud, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return None
    center = points.mean(axis=0)
    return float(np.linalg.norm(points-center, axis=1).max()+15.0)


def _predicted_response(hypothesis, waypoint, bias):
    location = hypothesis["point"]
    distance = math.dist(location, waypoint)
    if distance <= 5.0:
        return {"measure_result": "near"}
    angle = math.degrees(
        math.atan2(location[1]-waypoint[1], location[0]-waypoint[0])
    )
    return {
        "measure_result": "direction",
        "svd_deg": round((angle+bias) % 360.0, 2) % 360.0,
    }


def choose_edge_opportunity(client, channel, start, end, polygon, history):
    """Return a JSON-safe, deterministic decision from public state only."""
    start, end = tuple(start), tuple(end)
    cloud = client.clouds.get(channel)
    edge_length = math.dist(start, end)
    active_channel = client.active_channel
    equivalent = (
        SAME_CHANNEL_EQUIVALENT_M
        if active_channel == channel
        else SWITCH_CHANNEL_EQUIVALENT_M
    )
    decision = {
        "scorer_version": VERSION,
        "channel": int(channel),
        "start": list(start),
        "original_endpoint": list(end),
        "edge_length_m": float(edge_length),
        "active_channel_before": active_channel,
        "measure_equivalent_m": equivalent,
        "candidate_records": [],
        "selected": False,
        "selected_fraction": None,
        "selected_point": None,
        "reason": None,
    }
    if edge_length < MIN_EDGE_LENGTH_M:
        decision["reason"] = "edge_shorter_than_predeclared_minimum"
        return decision
    if cloud is None or len(cloud) == 0 or not polygon:
        decision["reason"] = "missing_public_feasible_set"
        return decision
    current_radius = cloud_radius(cloud)
    if current_radius is None or current_radius <= 60.0:
        decision["reason"] = "already_clearable_or_invalid_public_cloud"
        return decision
    hypotheses = compatible_hypotheses(
        cloud,
        list(polygon),
        list(history),
        longest_axis(list(polygon)),
    )
    decision["public_cloud_count"] = int(len(cloud))
    decision["current_public_cloud_radius_m"] = current_radius
    decision["hypotheses"] = hypotheses
    if not hypotheses:
        decision["reason"] = "no_compatible_public_grid_hypotheses"
        return decision

    feasible_points = [tuple(map(float, point)) for point in np.asarray(cloud)]
    feasible_points.extend(tuple(map(float, point)) for point in polygon)
    last_point = client.last_positions.get(channel)
    eligible = []
    for fraction in WAYPOINT_FRACTIONS:
        waypoint = (
            start[0]+fraction*(end[0]-start[0]),
            start[1]+fraction*(end[1]-start[1]),
        )
        collinearity_residual = (
            math.dist(start, waypoint)+math.dist(waypoint, end)-edge_length
        )
        maximum_feasible_distance = max(
            math.dist(waypoint, point) for point in feasible_points
        )
        last_separation = (
            math.inf if last_point is None else math.dist(waypoint, last_point)
        )
        replaced_m = math.dist(waypoint, end)
        net_m = replaced_m-equivalent
        scenario_radii = []
        scenario_valid = True
        for hypothesis in hypotheses:
            for bias in BIASES:
                response = _predicted_response(hypothesis, waypoint, bias)
                updated = update_cloud(np.asarray(cloud), waypoint, response)
                radius = cloud_radius(updated)
                if radius is None or not math.isfinite(radius):
                    scenario_valid = False
                    break
                scenario_radii.append(radius)
            if not scenario_valid:
                break
        worst_after = max(scenario_radii) if scenario_radii else None
        worst_shrink = (
            None if worst_after is None else current_radius-worst_after
        )
        checks = {
            "collinearity_zero_increment": abs(collinearity_residual)
            <= COLLINEARITY_TOLERANCE_M,
            "guaranteed_reception": maximum_feasible_distance
            <= VISIBILITY_LIMIT_M,
            "fresh_same_channel_point": last_separation
            > MIN_LAST_MEASURE_SEPARATION_M,
            "strict_positive_break_even": net_m > 0.0,
            "all_prediction_scenarios_valid": scenario_valid
            and bool(scenario_radii),
            "material_worst_case_public_shrink": worst_shrink is not None
            and worst_shrink >= MIN_WORST_CASE_RADIUS_SHRINK_M,
        }
        record = {
            "fraction": fraction,
            "point": list(waypoint),
            "collinearity_residual_m": collinearity_residual,
            "maximum_public_feasible_distance_m": maximum_feasible_distance,
            "last_same_channel_measure_separation_m": (
                None if math.isinf(last_separation) else last_separation
            ),
            "estimated_replaced_dedicated_movement_m": replaced_m,
            "measure_equivalent_m": equivalent,
            "predicted_net_m": net_m,
            "predicted_scenario_radius_m": scenario_radii,
            "predicted_worst_radius_after_m": worst_after,
            "predicted_worst_radius_shrink_m": worst_shrink,
            "checks": checks,
            "eligible": all(checks.values()),
        }
        decision["candidate_records"].append(record)
        if record["eligible"]:
            eligible.append(record)
    if not eligible:
        decision["reason"] = "no_waypoint_passed_all_predeclared_checks"
        return decision
    chosen = min(
        eligible,
        key=lambda record: (
            -record["predicted_net_m"],
            record["predicted_worst_radius_after_m"],
            record["fraction"],
        ),
    )
    decision.update(
        selected=True,
        selected_fraction=chosen["fraction"],
        selected_point=chosen["point"],
        reason="strict_positive_public_edge_opportunity",
    )
    return decision


def route_length(start, route):
    position = tuple(start)
    total = 0.0
    for _kind, _identifier, point in route:
        total += math.dist(position, point)
        position = tuple(point)
    return total


def closed_fallback_loop_length(proxy, initial_point, initial_svd):
    points = old._clear_points(initial_point, initial_svd)
    if not points:
        return 0.0
    return (
        math.dist(proxy, points[0])
        + sum(math.dist(first, second) for first, second in zip(points, points[1:]))
        + math.dist(points[-1], proxy)
    )


def conservative_reorder_decision(position, nodes, current_source, tracks):
    """Compare two executable public proxy tours with common closed fallbacks."""
    current = next(
        (node for node in nodes if node[0] == "source" and node[1] == current_source),
        None,
    )
    output = {
        "reorder": False,
        "reason": None,
        "current_source": int(current_source),
        "free_first": None,
        "forced_first": None,
    }
    if current is None:
        output["reason"] = "current_source_missing_from_public_nodes"
        return output
    others = [node for node in nodes if node is not current]
    forced_route = [current]+combined._route(current[2], others)
    free_route = combined._route(position, list(nodes))
    common_fallback = 0.0
    source_proxy = {
        node[1]: node[2] for node in nodes if node[0] == "source"
    }
    for channel, proxy in source_proxy.items():
        initial_point, initial_response, _polygon = tracks[channel]
        svd = initial_response.get("svd_deg")
        if not combined._valid(svd):
            output["reason"] = "invalid_initial_public_bearing"
            return output
        common_fallback += closed_fallback_loop_length(
            proxy, initial_point, svd
        )
    forced_route_m = route_length(position, forced_route)
    free_route_m = route_length(position, free_route)
    forced_bound = forced_route_m+common_fallback
    free_bound = free_route_m+common_fallback
    output.update(
        forced_first=[forced_route[0][0], forced_route[0][1]],
        free_first=[free_route[0][0], free_route[0][1]],
        forced_route=[
            {"kind": kind, "identifier": identifier, "point": list(point)}
            for kind, identifier, point in forced_route
        ],
        free_route=[
            {"kind": kind, "identifier": identifier, "point": list(point)}
            for kind, identifier, point in free_route
        ],
        forced_public_route_m=forced_route_m,
        free_public_route_m=free_route_m,
        common_closed_fallback_upper_m=common_fallback,
        forced_upper_bound_m=forced_bound,
        free_upper_bound_m=free_bound,
        strict_bound_decrease_m=forced_bound-free_bound,
    )
    different_first = output["free_first"] != output["forced_first"]
    strict_decrease = free_bound < forced_bound-REORDER_BOUND_EPSILON_M
    output["different_first_task"] = different_first
    output["strict_upper_bound_decrease"] = strict_decrease
    output["reorder"] = bool(different_first and strict_decrease)
    output["reason"] = (
        "different_first_and_strict_public_upper_bound_decrease"
        if output["reorder"]
        else "cache_only_no_strict_reorder_certificate"
    )
    return output


def self_test():
    start, end = (0.0, 0.0), (400.0, 0.0)
    records = []
    for fraction in WAYPOINT_FRACTIONS:
        point = (fraction*400.0, 0.0)
        residual = math.dist(start, point)+math.dist(point, end)-math.dist(start, end)
        records.append({"fraction": fraction, "residual_m": residual})
    checks = {
        "three_fixed_fractions": len(records) == 3,
        "straight_segment_triangle_equality": all(
            abs(record["residual_m"]) <= COLLINEARITY_TOLERANCE_M
            for record in records
        ),
        "fixed_break_even_constants": (
            SAME_CHANNEL_EQUIVALENT_M == 25.0
            and SWITCH_CHANNEL_EQUIVALENT_M == 30.0
        ),
        "visibility_below_reception_lower_bound": VISIBILITY_LIMIT_M < 1000.0,
    }
    return {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "records": records,
    }
