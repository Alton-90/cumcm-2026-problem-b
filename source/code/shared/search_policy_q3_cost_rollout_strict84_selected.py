"""Thin wrapper for the strictly selected Q3 cost-rollout configuration.

Provenance: ``results/q3_optimizer_benchmark/comparison_strict_v2.json`` chose
``tpe:20260912`` with ``[MOVE_SLACK=175, MIN_PREDICTED_SAVING=0,
OFFSET_SCALE=0.5]`` on the 84-case selection-validation set.  This module is a
versioned convenience wrapper, not the frozen main strategy and not independent
final-validation evidence.  It performs no evaluation and is not imported by
the source-residual v2 experiment.
"""
from __future__ import annotations

from threading import RLock

import q3_cost_rollout_v2 as helper
import search_policy_q3_cost_rollout_v2 as base_policy


VERSION = "q3_cost_rollout_strict84_selected_tpe_20260912_v1"
SELECTION_RECORD_ID = "tpe:20260912"
SELECTED_PARAMETERS = (175.0, 0.0, 0.5)
SELECTION_EVIDENCE = "results/q3_optimizer_benchmark/comparison_strict_v2.json"
STATUS = "selection_validation_only_not_frozen_main"
_LOCK = RLock()
_NAMES = (
    "MOVE_SLACK",
    "MIN_PREDICTED_SAVING",
    "OFFSETS",
    "BACKOFFS",
    "MAX_HYPOTHESES",
    "RECEPTION_LIMIT",
    "BIASES",
)
_SELECTED_VALUES = (
    175.0,
    0.0,
    (0.05, 0.125, 0.25),
    (0.25, 0.5),
    9,
    999.9,
    (-1.0, 0.0, 1.0),
)


def configuration():
    return {
        "selection_record_id": SELECTION_RECORD_ID,
        "move_slack_m": SELECTED_PARAMETERS[0],
        "minimum_predicted_saving_s": SELECTED_PARAMETERS[1],
        "offset_scale": SELECTED_PARAMETERS[2],
        "offset_factors": list(_SELECTED_VALUES[2]),
        "status": STATUS,
        "evidence": SELECTION_EVIDENCE,
    }


def run_search(client, question=3, method="main"):
    """Run serially with the selected globals, then restore ambient defaults."""
    with _LOCK:
        previous = tuple(getattr(helper, name) for name in _NAMES)
        try:
            for name, value in zip(_NAMES, _SELECTED_VALUES):
                setattr(helper, name, value)
            state = base_policy.run_search(client, question=question, method=method)
        finally:
            for name, value in zip(_NAMES, previous):
                setattr(helper, name, value)
    base_version = state.get("version")
    state.update(
        version=VERSION,
        wrapped_base_version=base_version,
        strict84_selection_configuration=configuration(),
    )
    return state
