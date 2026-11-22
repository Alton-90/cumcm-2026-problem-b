"""Compact offline candidate: altered coverage only, with existing localization intact.

The caller owns /enter and /exit.  This module receives only the public
SimulatorClient surface and never accesses an arena or source collection.
"""

from __future__ import annotations

import search_policy as q3_policy
import search_policy_recovery as q4_policy


VERSION = "compact_v1"
CHANNELS = range(1, 21)


def coverage_points(question: int) -> list[tuple[float, float]]:
    """Return an origin-first, duplicate-free row-snake coverage route."""
    if question == 3:
        values = (-1800, -600, 600, 1800)
    elif question == 4:
        values = (-2100, -1400, -700, 0, 700, 1400, 2100)
    else:
        raise ValueError("question must be 3 or 4")
    grid: list[tuple[float, float]] = []
    for row, y in enumerate(values):
        xs = values if row % 2 == 0 else tuple(reversed(values))
        grid.extend((float(x), float(y)) for x in xs)
    origin = (0.0, 0.0)
    return [origin] + [point for point in grid if point != origin]


def _state(question: int, points: list[tuple[float, float]]) -> dict:
    state = {
        "cleared_channels": set(), "excluded_channels": set(), "unresolved_channels": set(),
        "measure_count": 0, "clear_count": 0, "extra_bearing_count": 0,
        "fallback_count": 0, "fallback_clear_count": 0, "warnings": [], "stopped": False,
        "coverage_points_total": len(points),
        "coverage_measurements_by_channel": {channel: 0 for channel in CHANNELS},
    }
    if question == 4:
        state.update({"recovery_count": 0, "recovery_measure_count": 0, "recovery_positive_count": 0})
    return state


def run_search(client, question: int = 3, method: str = "main") -> dict:
    """Run the compact candidate with unchanged question-specific localizers."""
    if question not in (3, 4):
        raise ValueError("question must be 3 or 4")
    if method != "main":
        raise ValueError("compact candidate supports only method='main'")

    points = coverage_points(question)
    state = _state(question, points)
    seen: set[int] = set()
    try:
        for scan_point in points:
            for channel in CHANNELS:
                if channel in state["cleared_channels"]:
                    continue
                response = client.measure(scan_point[0], scan_point[1], channel)
                state["measure_count"] += 1
                state["coverage_measurements_by_channel"][channel] += 1
                kind = response.get("measure_result")
                if kind == "no_signal":
                    continue
                seen.add(channel)
                if kind == "near":
                    q3_policy._resolve_clear(client, scan_point, channel, state, "near coverage point")
                elif kind == "direction":
                    if question == 3:
                        q3_policy._handle_direction(client, scan_point, response, channel, state, method="main")
                    else:
                        q4_policy._handle_direction(client, scan_point, response, channel, state)
                else:
                    state["warnings"].append(f"channel {channel}: unknown measure result {kind!r}")
                    state["unresolved_channels"].add(channel)
        for channel in CHANNELS:
            if channel in state["cleared_channels"]:
                continue
            if channel in seen:
                state["unresolved_channels"].add(channel)
            else:
                state["excluded_channels"].add(channel)
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(error).__name__}: {error}")
        state["unresolved_channels"].update(set(CHANNELS) - state["cleared_channels"] - state["excluded_channels"])

    state["cleared_channels"] = sorted(state["cleared_channels"])
    state["excluded_channels"] = sorted(state["excluded_channels"])
    state["unresolved_channels"] = sorted(state["unresolved_channels"])
    state["completed"] = (not state["stopped"] and not state["unresolved_channels"] and
                          len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20)
    state["question"], state["method"] = question, method
    state["version"] = VERSION
    state["virtual_time_s"] = getattr(client, "virtual_time_s", None)
    return state
