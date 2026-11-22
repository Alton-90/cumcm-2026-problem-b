"""Q3 edge-opportunity experiment; unchanged combined policy is the fallback."""
from __future__ import annotations

import math

import search_policy as old
from search_policy_compact import _state
import search_policy_q3_combined as combined
from search_policy_q3_combined import ValueTracked
from search_policy_feasible_q3 import localize
from q3_edge_opportunity_v1 import (
    choose_edge_opportunity,
    conservative_reorder_decision,
)


VERSION = "q3_edge_opportunity_v1"
_route = combined._route
_valid = combined._valid


class PublicTracked(ValueTracked):
    """Retain only normal public action/response history for the scorer."""

    def __init__(self, client):
        super().__init__(client)
        self.public_history = {}

    @property
    def active_channel(self):
        return self.c.channel

    def measure(self, x, y, channel):
        response = super().measure(x, y, channel)
        self.public_history.setdefault(channel, []).append(
            {
                "action": "measure",
                "point": [float(x), float(y)],
                "response": dict(response),
            }
        )
        return response

    def clear(self, x, y, channel):
        response = super().clear(x, y, channel)
        self.public_history.setdefault(channel, []).append(
            {
                "action": "clear",
                "point": [float(x), float(y)],
                "response": dict(response),
            }
        )
        return response


def _public_nodes(client, points, remaining, tracks, cleared, attempted, include_channel=None):
    nodes = [("scan", index, points[index]) for index in remaining]
    for channel, (_point, _response, polygon) in tracks.items():
        if channel in cleared:
            continue
        if channel in attempted and channel != include_channel:
            continue
        box = client.estimate(channel) or old._box_circle(polygon)
        if box:
            nodes.append(("source", channel, box[0]))
    return nodes


def run_search(client, question=3, method="main", *, allow_reorder=True):
    if question != 3 or method != "main":
        raise ValueError("Q3 main only")
    client = PublicTracked(client)
    points = [(0.0, 0.0)] + [
        (1124*math.cos(index*math.pi/3), 1124*math.sin(index*math.pi/3))
        for index in range(6)
    ]
    state = _state(3, points)
    tracks = {}
    seen = set()
    remaining = list(range(len(points)))
    attempted = set()
    probed_edges = set()
    state.update(
        edge_opportunity_events=[],
        edge_opportunity_inserted_count=0,
        edge_opportunity_hit_count=0,
        edge_opportunity_reorder_count=0,
        edge_opportunity_guarantee_failure_count=0,
        edge_opportunity_allow_reorder=bool(allow_reorder),
    )
    try:
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if len(state["cleared_channels"]) == 16:
                break
            nodes = _public_nodes(
                client,
                points,
                remaining,
                tracks,
                state["cleared_channels"],
                attempted,
            )
            if not nodes:
                break
            kind, index, destination = _route(client.position, nodes)[0]
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
                            first_point, first_response, polygon = tracks[channel]
                            clipped = old._add_bearing(
                                polygon, point, response["svd_deg"]
                            )
                            if clipped:
                                tracks[channel] = (
                                    first_point,
                                    first_response,
                                    clipped,
                                )
                    else:
                        state["warnings"].append(
                            f"channel {channel}: invalid public response"
                        )
                continue

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
                    clipped = old._add_bearing(polygon, here, extra["svd_deg"])
                    if clipped:
                        polygon = clipped
                tracks[channel] = (initial_point, initial_response, polygon)
                attempted.discard(channel)
                continue

            target = client.estimate(channel) or old._box_circle(polygon)
            endpoint = destination if target is None else target[0]
            edge_key = (
                channel,
                round(client.position[0], 9),
                round(client.position[1], 9),
                round(endpoint[0], 9),
                round(endpoint[1], 9),
            )
            decision = choose_edge_opportunity(
                client,
                channel,
                client.position,
                endpoint,
                polygon,
                client.public_history.get(channel, []),
            )
            if decision["selected"] and edge_key in probed_edges:
                decision.update(
                    selected=False,
                    selected_fraction=None,
                    selected_point=None,
                    reason="same_geometric_edge_already_probed",
                )
            event = {
                "event_index": len(state["edge_opportunity_events"]),
                "action_count_before": state["measure_count"]+state["clear_count"],
                "decision": decision,
                "inserted": False,
                "measure_result": None,
                "reorder_decision": None,
                "reorder_applied": False,
                "edge_key": list(edge_key),
                "baseline_next_real_action_certificate": {
                    "certified": bool(decision["selected"]),
                    "path": "/measure" if decision["selected"] else None,
                    "channel": channel,
                    "point": list(endpoint) if decision["selected"] else None,
                    "estimate_radius_m": None if target is None else target[1],
                    "current_pose_reuse_branch_already_failed": True,
                    "localize_extra_used_at_entry": 0,
                    "reason": (
                        "public estimate radius above clear threshold; localize next measures its center"
                        if decision["selected"]
                        else "no inserted action; no counterfactual next-action claim required"
                    ),
                },
            }
            state["edge_opportunity_events"].append(event)
            if not decision["selected"]:
                localize(
                    client,
                    initial_point,
                    initial_response,
                    channel,
                    state,
                    polygon,
                )
                continue

            waypoint = tuple(decision["selected_point"])
            probed_edges.add(edge_key)
            response = client.measure(*waypoint, channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            state["edge_opportunity_inserted_count"] += 1
            event["inserted"] = True
            event["measure_result"] = response.get("measure_result")
            event["response_svd_deg"] = response.get("svd_deg")
            if response.get("measure_result") != "no_signal":
                state["edge_opportunity_hit_count"] += 1
            else:
                state["edge_opportunity_guarantee_failure_count"] += 1
                event["guaranteed_reception_failure"] = True
                old._fallback_scan(
                    client,
                    initial_point,
                    initial_response["svd_deg"],
                    channel,
                    state,
                )
                continue
            if response.get("measure_result") == "near":
                if old._resolve_clear(
                    client, waypoint, channel, state, "near edge opportunity"
                ):
                    continue
                old._fallback_scan(
                    client,
                    initial_point,
                    initial_response["svd_deg"],
                    channel,
                    state,
                )
                continue
            if response.get("measure_result") != "direction" or not _valid(
                response.get("svd_deg")
            ):
                state["warnings"].append(
                    f"channel {channel}: invalid edge-opportunity response"
                )
                old._fallback_scan(
                    client,
                    initial_point,
                    initial_response["svd_deg"],
                    channel,
                    state,
                )
                continue

            clipped = old._add_bearing(polygon, waypoint, response["svd_deg"])
            if clipped:
                polygon = clipped
            tracks[channel] = (initial_point, initial_response, polygon)
            post_nodes = _public_nodes(
                client,
                points,
                remaining,
                tracks,
                state["cleared_channels"],
                attempted,
                include_channel=channel,
            )
            reorder = conservative_reorder_decision(
                client.position, post_nodes, channel, tracks
            )
            event["reorder_decision"] = reorder
            if allow_reorder and reorder["reorder"]:
                attempted.discard(channel)
                state["edge_opportunity_reorder_count"] += 1
                event["reorder_applied"] = True
                continue
            localize(
                client,
                initial_point,
                initial_response,
                channel,
                state,
                polygon,
            )

        state["excluded_channels"] = set(range(1, 21))-seen
        state["unresolved_channels"] = seen-state["cleared_channels"]
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
            and len(state["cleared_channels"])+len(state["excluded_channels"]) == 20
        ),
        question=3,
        method="main",
        version=VERSION+("_full" if allow_reorder else "_cache_only"),
        virtual_time_s=client.virtual_time_s,
    )
    return state


def run_search_cache_only(client, question=3, method="main"):
    return run_search(client, question=question, method=method, allow_reorder=False)
