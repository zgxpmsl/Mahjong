"""Evaluation helpers for comparing generated logs with ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..data.log_schema import Event, RoundLog


@dataclass
class EventComparison:
    predicted: Event
    reference: Event
    action_match: bool
    tile_match: bool
    actor_match: bool


@dataclass
class EvaluationReport:
    total_events: int
    matched_events: int
    action_accuracy: float
    tile_accuracy: float
    actor_accuracy: float


def compare_logs(predicted: RoundLog, reference: RoundLog) -> EvaluationReport:
    pairs = zip(predicted.events, reference.events)
    comparisons = [
        EventComparison(
            predicted=pred,
            reference=ref,
            action_match=pred.action == ref.action,
            tile_match=set(pred.tiles) == set(ref.tiles),
            actor_match=pred.actor == ref.actor,
        )
        for pred, ref in pairs
    ]
    total = len(reference.events)
    matched = sum(1 for item in comparisons if item.action_match and item.tile_match and item.actor_match)
    action_accuracy = sum(1 for item in comparisons if item.action_match) / max(total, 1)
    tile_accuracy = sum(1 for item in comparisons if item.tile_match) / max(total, 1)
    actor_accuracy = sum(1 for item in comparisons if item.actor_match) / max(total, 1)
    return EvaluationReport(
        total_events=total,
        matched_events=matched,
        action_accuracy=action_accuracy,
        tile_accuracy=tile_accuracy,
        actor_accuracy=actor_accuracy,
    )
