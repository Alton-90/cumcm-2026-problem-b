"""Deterministic fixed-position Q4 compatibility from public receipts only.

This module eliminates the unknown reception radius and intersects continuous
heading arcs.  It is deliberately not a search policy or a continuous source-
area exclusion certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal, Mapping, Sequence


TAU = 2.0 * math.pi
DEFAULT_REGION_RADIUS_M = 1800.0
DEFAULT_BEARING_TOLERANCE_DEG = 1.005
MIN_RANGE_M = 1000.0
MAX_RANGE_M = 1500.0
NEAR_RADIUS_M = 5.0
CLEAR_RADIUS_M = 20.0

Status = Literal["compatible", "incompatible", "uncertain"]
SourceType = Literal["directional", "omnidirectional", "either"]


@dataclass(frozen=True)
class PublicObservation:
    """One normal public measurement response for a single channel."""

    point: tuple[float, float]
    result: Literal["direction", "near", "no_signal"]
    bearing_deg: float | None = None


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    left_closed: bool
    right_closed: bool


@dataclass(frozen=True)
class CompatibilityResult:
    status: Status
    keep_candidate: bool
    source_type: SourceType
    nominal_compatible: bool
    directional_nominal_compatible: bool
    omnidirectional_nominal_compatible: bool
    range_star_m: float
    nominal_range_m: float | None
    orientation_witness_deg: float | None
    orientation_intervals_deg: tuple[dict, ...]
    forced_negative_indices: tuple[int, ...]
    definitely_forced_negative_indices: tuple[int, ...]
    possibly_forced_negative_indices: tuple[int, ...]
    blocking_reasons: tuple[str, ...]
    uncertainty_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _point(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    return (_finite_number(value[0], f"{name}[0]"),
            _finite_number(value[1], f"{name}[1]"))


def _observation(value: PublicObservation | Mapping, index: int) -> PublicObservation:
    if isinstance(value, PublicObservation):
        result = value
    elif isinstance(value, Mapping):
        kind = value.get("result", value.get("measure_result"))
        bearing = value.get("bearing_deg", value.get("svd_deg"))
        result = PublicObservation(_point(value.get("point"), f"observations[{index}].point"),
                                   kind, bearing)
    else:
        raise ValueError(f"observations[{index}] must be PublicObservation or mapping")
    if result.result not in ("direction", "near", "no_signal"):
        raise ValueError(f"observations[{index}].result is not a public measurement result")
    point = _point(result.point, f"observations[{index}].point")
    if result.result == "direction":
        bearing = _finite_number(result.bearing_deg, f"observations[{index}].bearing_deg")
        if not 0.0 <= bearing < 360.0:
            raise ValueError(f"observations[{index}].bearing_deg must be in [0, 360)")
    else:
        bearing = None
    return PublicObservation(point, result.result, bearing)


def circular_difference_deg(a: float, b: float) -> float:
    """Smallest unsigned angle difference, including the 0/360 seam."""

    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _contains(segment: _Segment, value: float) -> bool:
    if value < segment.start or value > segment.end:
        return False
    if value == segment.start and not segment.left_closed:
        return False
    if value == segment.end and not segment.right_closed:
        return False
    return True


def _intersection(first: _Segment, second: _Segment) -> _Segment | None:
    start = max(first.start, second.start)
    end = min(first.end, second.end)
    if start > end:
        return None
    left_closed = _contains(first, start) and _contains(second, start)
    right_closed = _contains(first, end) and _contains(second, end)
    if start == end and not (left_closed and right_closed):
        return None
    return _Segment(start, end, left_closed, right_closed)


def _merge(segments: Iterable[_Segment]) -> list[_Segment]:
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    merged: list[_Segment] = []
    for item in ordered:
        if not merged:
            merged.append(item)
            continue
        old = merged[-1]
        touches = item.start == old.end and (old.right_closed or item.left_closed)
        if item.start < old.end or touches:
            if item.end > old.end:
                merged[-1] = _Segment(old.start, item.end, old.left_closed,
                                      item.right_closed)
            elif item.end == old.end:
                merged[-1] = _Segment(old.start, old.end, old.left_closed,
                                      old.right_closed or item.right_closed)
        else:
            merged.append(item)
    return merged


def _arc(center: float, threshold: float, closed: bool) -> list[_Segment]:
    """Angles satisfying cos(theta-center) >= threshold."""

    if threshold > 1.0:
        return []
    if threshold <= -1.0:
        return [_Segment(0.0, TAU, True, True)]
    width = math.acos(max(-1.0, min(1.0, threshold)))
    center %= TAU
    low, high = center - width, center + width
    if low < 0.0:
        return [_Segment(0.0, high, True, closed),
                _Segment(low + TAU, TAU, closed, True)]
    if high > TAU:
        return [_Segment(0.0, high - TAU, True, closed),
                _Segment(low, TAU, closed, True)]
    return [_Segment(low, high, closed, closed)]


def _intersect_arcs(arcs: Iterable[list[_Segment]]) -> list[_Segment]:
    feasible = [_Segment(0.0, TAU, True, True)]
    for parts in arcs:
        feasible = _merge(hit for old in feasible for new in parts
                          if (hit := _intersection(old, new)) is not None)
        if not feasible:
            break
    return feasible


def _heading_segments(positive: Iterable[tuple[float, float]],
                      negative: Iterable[tuple[float, float]], *,
                      threshold: float, nominal_strict_negative: bool) -> list[_Segment]:
    arcs: list[list[_Segment]] = []
    for vector in positive:
        # A zero positive vector is on the closed visible boundary for every h.
        if vector == (0.0, 0.0):
            continue
        arcs.append(_arc(math.atan2(vector[1], vector[0]), threshold, True))
    for vector in negative:
        # At the source itself, h dot 0 < 0 is impossible for every h.
        if vector == (0.0, 0.0):
            return []
        center = math.atan2(-vector[1], -vector[0])
        arcs.append(_arc(center, threshold, not nominal_strict_negative))
    return _intersect_arcs(arcs)


def _witness(segments: Sequence[_Segment]) -> float | None:
    if not segments:
        return None
    segment = max(segments, key=lambda item: item.end - item.start)
    if segment.start < segment.end:
        return ((segment.start + segment.end) / 2.0) % TAU
    return segment.start % TAU


def _segments_as_degrees(segments: Sequence[_Segment]) -> tuple[dict, ...]:
    return tuple({"start_deg": math.degrees(item.start),
                  "end_deg": math.degrees(item.end),
                  "left_closed": item.left_closed,
                  "right_closed": item.right_closed,
                  "degenerate": item.start == item.end}
                 for item in segments)


def check_fixed_position(
        candidate: Sequence[float],
        observations: Iterable[PublicObservation | Mapping],
        failed_clear_points: Iterable[Sequence[float]] = (), *,
        source_type: SourceType = "directional",
        region_center: Sequence[float] = (0.0, 0.0),
        region_radius_m: float = DEFAULT_REGION_RADIUS_M,
        bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE_DEG,
        distance_tolerance_m: float = 1e-7,
        angle_tolerance_deg: float = 1e-10,
        cosine_tolerance: float = 1e-12,
) -> CompatibilityResult:
    """Check one candidate without truth, simulator access, or heading sampling.

    ``status == 'incompatible'`` is the only result safe for candidate deletion.
    A nominally valid boundary-only solution is returned as ``uncertain`` and
    retained, while ``nominal_compatible`` records the exact float-model answer.
    Use ``source_type='either'`` when Q4's public history has not established
    whether the source is omnidirectional or directional.
    """

    if source_type not in ("directional", "omnidirectional", "either"):
        raise ValueError("source_type must be directional, omnidirectional, or either")
    candidate = _point(candidate, "candidate")
    region_center = _point(region_center, "region_center")
    region_radius_m = _finite_number(region_radius_m, "region_radius_m")
    bearing_tolerance_deg = _finite_number(bearing_tolerance_deg,
                                           "bearing_tolerance_deg")
    distance_tolerance_m = _finite_number(distance_tolerance_m,
                                           "distance_tolerance_m")
    angle_tolerance_deg = _finite_number(angle_tolerance_deg,
                                         "angle_tolerance_deg")
    cosine_tolerance = _finite_number(cosine_tolerance, "cosine_tolerance")
    if region_radius_m < 0.0 or bearing_tolerance_deg < 0.0:
        raise ValueError("region and bearing tolerances must be nonnegative")
    if distance_tolerance_m <= 0.0 or angle_tolerance_deg <= 0.0:
        raise ValueError("numerical distance and angle tolerances must be positive")
    if not 0.0 < cosine_tolerance < 1.0:
        raise ValueError("cosine_tolerance must be in (0, 1)")

    observations = tuple(_observation(value, index)
                         for index, value in enumerate(observations))
    failed_clear_points = tuple(_point(value, f"failed_clear_points[{index}]")
                                for index, value in enumerate(failed_clear_points))

    exact_reasons: list[str] = []
    uncertainty_reasons: list[str] = []
    robust_scalar = True
    relaxed_scalar = True
    nominal_scalar = True

    def closed_upper(name: str, value: float, bound: float, tolerance: float) -> None:
        nonlocal robust_scalar, relaxed_scalar, nominal_scalar
        nominal = value <= bound
        robust = value <= bound - tolerance
        relaxed = value <= bound + tolerance
        nominal_scalar = nominal_scalar and nominal
        robust_scalar = robust_scalar and robust
        relaxed_scalar = relaxed_scalar and relaxed
        if not nominal:
            exact_reasons.append(f"{name}: {value:.12g} > {bound:.12g}")
        if not robust and relaxed:
            uncertainty_reasons.append(f"{name} lies within its numerical boundary band")

    def strict_lower(name: str, value: float, bound: float, tolerance: float) -> None:
        nonlocal robust_scalar, relaxed_scalar, nominal_scalar
        nominal = value > bound
        robust = value > bound + tolerance
        relaxed = value > bound - tolerance
        nominal_scalar = nominal_scalar and nominal
        robust_scalar = robust_scalar and robust
        relaxed_scalar = relaxed_scalar and relaxed
        if not nominal:
            exact_reasons.append(f"{name}: {value:.12g} <= {bound:.12g}")
        if not robust and relaxed:
            uncertainty_reasons.append(f"{name} lies within its numerical boundary band")

    region_distance = math.dist(candidate, region_center)
    closed_upper("candidate source-region distance", region_distance,
                 region_radius_m, distance_tolerance_m)

    positive_vectors: list[tuple[float, float]] = []
    negatives: list[tuple[int, tuple[float, float], float]] = []
    positive_distances: list[float] = []
    for index, observation in enumerate(observations):
        vector = (observation.point[0] - candidate[0],
                  observation.point[1] - candidate[1])
        distance = math.hypot(*vector)
        if observation.result == "direction":
            strict_lower(f"direction[{index}] distance", distance, NEAR_RADIUS_M,
                         distance_tolerance_m)
            true_bearing = math.degrees(math.atan2(candidate[1] - observation.point[1],
                                                   candidate[0] - observation.point[0])) % 360.0
            error = circular_difference_deg(true_bearing, observation.bearing_deg)
            closed_upper(f"direction[{index}] circular bearing error", error,
                         bearing_tolerance_deg, angle_tolerance_deg)
            positive_vectors.append(vector)
            positive_distances.append(distance)
        elif observation.result == "near":
            closed_upper(f"near[{index}] distance", distance, NEAR_RADIUS_M,
                         distance_tolerance_m)
            positive_vectors.append(vector)
            positive_distances.append(distance)
        else:
            negatives.append((index, vector, distance))

    for index, point in enumerate(failed_clear_points):
        strict_lower(f"failed_clear[{index}] distance", math.dist(candidate, point),
                     CLEAR_RADIUS_M, distance_tolerance_m)

    range_star = max((MIN_RANGE_M, *positive_distances))
    closed_upper("R*", range_star, MAX_RANGE_M, distance_tolerance_m)

    force_band = 2.0 * distance_tolerance_m
    forced = tuple(index for index, _, distance in negatives if distance <= range_star)
    definitely_forced = tuple(index for index, _, distance in negatives
                              if distance <= range_star - force_band)
    possibly_forced = tuple(index for index, _, distance in negatives
                            if distance <= range_star + force_band)
    nominal_negative = [vector for index, vector, _ in negatives if index in forced]
    definite_negative = [vector for index, vector, _ in negatives
                         if index in definitely_forced]
    possible_negative = [vector for index, vector, _ in negatives
                         if index in possibly_forced]
    if set(definitely_forced) != set(possibly_forced):
        uncertainty_reasons.append("at least one no_signal distance is near the closed R* boundary")

    nominal_segments = _heading_segments(positive_vectors, nominal_negative,
                                         threshold=0.0,
                                         nominal_strict_negative=True)
    robust_segments = _heading_segments(positive_vectors, possible_negative,
                                        threshold=cosine_tolerance,
                                        nominal_strict_negative=False)
    relaxed_segments = _heading_segments(positive_vectors, definite_negative,
                                         threshold=-cosine_tolerance,
                                         nominal_strict_negative=False)
    if bool(nominal_segments) and not bool(robust_segments):
        uncertainty_reasons.append("directional feasibility exists only on or near a heading boundary")
    elif not nominal_segments and relaxed_segments:
        uncertainty_reasons.append("heading infeasibility is not separated from the numerical boundary")

    directional_nominal = nominal_scalar and bool(nominal_segments)
    directional_robust = robust_scalar and bool(robust_segments)
    directional_relaxed = relaxed_scalar and bool(relaxed_segments)
    omnidirectional_nominal = nominal_scalar and not forced
    omnidirectional_robust = robust_scalar and not possibly_forced
    omnidirectional_relaxed = relaxed_scalar and not definitely_forced

    if source_type == "directional":
        nominal, robust, relaxed = (directional_nominal, directional_robust,
                                    directional_relaxed)
    elif source_type == "omnidirectional":
        nominal, robust, relaxed = (omnidirectional_nominal, omnidirectional_robust,
                                    omnidirectional_relaxed)
    else:
        nominal = directional_nominal or omnidirectional_nominal
        robust = directional_robust or omnidirectional_robust
        relaxed = directional_relaxed or omnidirectional_relaxed

    if source_type == "directional" and not nominal_segments:
        exact_reasons.append("continuous directional-heading intersection is empty")
    elif source_type == "omnidirectional" and forced:
        exact_reasons.append("omnidirectional source cannot explain no_signal at distance <= R*")
    elif source_type == "either" and not nominal_segments and forced:
        exact_reasons.append("neither source type can explain all forced no_signal receipts")

    status: Status = "compatible" if robust else "incompatible" if not relaxed else "uncertain"
    witness = _witness(nominal_segments) if directional_nominal else None
    return CompatibilityResult(
        status=status,
        keep_candidate=status != "incompatible",
        source_type=source_type,
        nominal_compatible=nominal,
        directional_nominal_compatible=directional_nominal,
        omnidirectional_nominal_compatible=omnidirectional_nominal,
        range_star_m=range_star,
        nominal_range_m=range_star if nominal else None,
        orientation_witness_deg=None if witness is None else math.degrees(witness) % 360.0,
        orientation_intervals_deg=_segments_as_degrees(nominal_segments),
        forced_negative_indices=forced,
        definitely_forced_negative_indices=definitely_forced,
        possibly_forced_negative_indices=possibly_forced,
        blocking_reasons=tuple(dict.fromkeys(exact_reasons)),
        uncertainty_reasons=tuple(dict.fromkeys(uncertainty_reasons)),
    )


__all__ = ["CompatibilityResult", "PublicObservation", "check_fixed_position",
           "circular_difference_deg"]
