"""Deterministic convex geometry for the Q3 guaranteed-reception kernel."""
from __future__ import annotations

import hashlib
import math
import numpy as np


KERNEL_RADIUS_M = 985.0
FEASIBILITY_TOLERANCE_M = 1e-8


def cloud_hash(cloud):
    data = np.ascontiguousarray(np.asarray(cloud, dtype="<f8"))
    return hashlib.sha256(data.tobytes()).hexdigest()


def convex_hull(points):
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    def half(sequence):
        output = []
        for point in sequence:
            while len(output) >= 2 and cross(output[-2], output[-1], point) <= 0:
                output.pop()
            output.append(point)
        return output

    lower = half(unique)
    upper = half(reversed(unique))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def maximum_distance(point, centers):
    if len(centers) == 0:
        return math.inf
    return float(np.linalg.norm(np.asarray(centers) - np.asarray(point), axis=1).max())


def _project_to_discs(point, centers, radius, max_cycles=600):
    """Dykstra projection onto an intersection of equal-radius closed discs."""
    centers = np.asarray(centers, dtype=float)
    current = np.asarray(point, dtype=float).copy()
    corrections = np.zeros_like(centers)
    if len(centers) == 0:
        return None, math.inf, 0
    for cycle in range(1, max_cycles + 1):
        cycle_start = current.copy()
        for index, center in enumerate(centers):
            shifted = current + corrections[index]
            delta = shifted - center
            distance = float(np.linalg.norm(delta))
            projected = center + delta * (radius / distance) if distance > radius else shifted
            corrections[index] = shifted - projected
            current = projected
        residual = maximum_distance(current, centers) - radius
        if residual <= FEASIBILITY_TOLERANCE_M / 4 and float(
            np.linalg.norm(current - cycle_start)
        ) <= FEASIBILITY_TOLERANCE_M / 4:
            return current, residual, cycle
    residual = maximum_distance(current, centers) - radius
    return current, residual, max_cycles


def _segment_kernel_point(start, end, centers, radius):
    """Return the first point on start--end lying in every disc, if one exists."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    coefficient = float(direction @ direction)
    if coefficient == 0:
        return start.copy() if maximum_distance(start, centers) <= radius else None
    low, high = 0.0, 1.0
    for center in centers:
        offset = start - center
        linear = 2 * float(offset @ direction)
        constant = float(offset @ offset) - radius * radius
        discriminant = linear * linear - 4 * coefficient * constant
        if discriminant < 0:
            if constant > 0:
                return None
            continue
        root = math.sqrt(max(0.0, discriminant))
        low = max(low, (-linear - root) / (2 * coefficient))
        high = min(high, (-linear + root) / (2 * coefficient))
        if low > high + 1e-14:
            return None
    point = start + min(1.0, max(0.0, low)) * direction
    return point if maximum_distance(point, centers) <= radius + FEASIBILITY_TOLERANCE_M else None


def insertion_cost(point, start, successor):
    return (
        math.dist(point, start)
        + math.dist(point, successor)
        - math.dist(start, successor)
    )


def _objective_gradient(point, start, successor):
    gradient = np.zeros(2)
    for focus in (start, successor):
        delta = point - focus
        distance = float(np.linalg.norm(delta))
        if distance > 1e-10:
            gradient += delta / distance
    return gradient


def projected_kernel_target(cloud, start, successor, radius=KERNEL_RADIUS_M):
    """Minimize route insertion cost over the public candidate-cloud reception kernel."""
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1:] != (2,) or len(cloud) == 0:
        return {"status": "empty_cloud", "point": None}
    hull = convex_hull(cloud)
    start = np.asarray(start, dtype=float)
    successor = np.asarray(successor, dtype=float)

    mean = cloud.mean(axis=0)
    anchor, anchor_residual, feasibility_cycles = _project_to_discs(mean, hull, radius)
    if anchor is None or anchor_residual > FEASIBILITY_TOLERANCE_M:
        return {
            "status": "empty_or_unresolved_kernel",
            "point": None,
            "hull": hull.tolist(),
            "feasibility_residual_m": anchor_residual,
            "feasibility_cycles": feasibility_cycles,
        }

    segment_point = _segment_kernel_point(start, successor, hull, radius)
    candidates = [("feasible_anchor", anchor)]
    if maximum_distance(mean, hull) <= radius + FEASIBILITY_TOLERANCE_M:
        candidates.append(("cloud_mean", mean))
    if segment_point is not None:
        candidates.append(("route_segment", segment_point))
    for label, trial in (
        ("projected_start", start),
        ("projected_successor", successor),
        ("projected_midpoint", (start + successor) / 2),
    ):
        projected, residual, _ = _project_to_discs(trial, hull, radius)
        if projected is not None and residual <= FEASIBILITY_TOLERANCE_M:
            candidates.append((label, projected))

    label, point = min(
        candidates,
        key=lambda item: (insertion_cost(item[1], start, successor), item[0]),
    )
    objective = insertion_cost(point, start, successor)
    optimization_iterations = 0
    if label != "route_segment":
        for iteration in range(80):
            gradient = _objective_gradient(point, start, successor)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= 1e-10:
                break
            direction = gradient / gradient_norm
            step = radius
            improved = False
            for _ in range(28):
                trial, residual, _ = _project_to_discs(point - step * direction, hull, radius)
                if trial is not None and residual <= FEASIBILITY_TOLERANCE_M:
                    trial_objective = insertion_cost(trial, start, successor)
                    if trial_objective < objective - 1e-9:
                        point, objective = trial, trial_objective
                        improved = True
                        break
                step /= 2
            optimization_iterations = iteration + 1
            if not improved or step <= 1e-7:
                break

    max_hull_distance = maximum_distance(point, hull)
    max_cloud_distance = maximum_distance(point, cloud)
    residual = max_cloud_distance - radius
    if residual > FEASIBILITY_TOLERANCE_M:
        return {
            "status": "solver_residual_exceeded",
            "point": None,
            "hull": hull.tolist(),
            "max_hull_distance_m": max_hull_distance,
            "max_cloud_distance_m": max_cloud_distance,
            "kernel_residual_m": residual,
            "feasibility_cycles": feasibility_cycles,
            "optimization_iterations": optimization_iterations,
        }
    return {
        "status": "ok",
        "point": (float(point[0]), float(point[1])),
        "hull": hull.tolist(),
        "max_hull_distance_m": max_hull_distance,
        "max_cloud_distance_m": max_cloud_distance,
        "kernel_residual_m": residual,
        "insertion_cost_m": objective,
        "feasibility_cycles": feasibility_cycles,
        "optimization_iterations": optimization_iterations,
        "optimizer_anchor": label,
    }
