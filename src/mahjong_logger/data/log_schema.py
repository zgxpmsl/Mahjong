"""Data structures that represent the weakly supervised event logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional

ActionType = Literal[
    "draw",
    "discard",
    "pong",
    "kong_open",
    "kong_concealed",
    "chi",
    "hu",
    "flower",
    "pass",
    "unknown",
]


@dataclass
class Event:
    """Single log event for a Mahjong action."""

    timestamp: float
    actor: str
    action: ActionType
    tiles: List[str] = field(default_factory=list)
    notes: str = ""
    source_event: Optional[int] = None
    confidence: float = 1.0


@dataclass
class RoundLog:
    """Collection of ordered events for a Mahjong round."""

    round_id: str
    dealer: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    events: List[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "dealer": self.dealer,
            "created_at": self.created_at.isoformat(),
            "events": [
                {
                    "timestamp": event.timestamp,
                    "actor": event.actor,
                    "action": event.action,
                    "tiles": event.tiles,
                    "notes": event.notes,
                    "source_event": event.source_event,
                    "confidence": event.confidence,
                }
                for event in self.events
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RoundLog":
        log = cls(
            round_id=payload["round_id"],
            dealer=payload["dealer"],
            created_at=datetime.fromisoformat(payload.get("created_at")),
        )
        for idx, event in enumerate(payload.get("events", [])):
            log.events.append(
                Event(
                    timestamp=event.get("timestamp", float(idx)),
                    actor=event.get("actor", "unknown"),
                    action=event.get("action", "unknown"),
                    tiles=list(event.get("tiles", [])),
                    notes=event.get("notes", ""),
                    source_event=event.get("source_event"),
                    confidence=event.get("confidence", 1.0),
                )
            )
        return log


def load_round_log(path: Path) -> RoundLog:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return RoundLog.from_dict(data)


def save_round_log(log: RoundLog, path: Path) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(log.to_dict(), handle, ensure_ascii=False, indent=2)


def merge_events(log: RoundLog, events: Iterable[Event]) -> RoundLog:
    merged = RoundLog(round_id=log.round_id, dealer=log.dealer, created_at=log.created_at)
    merged.events.extend(log.events)
    merged.events.extend(events)
    merged.events.sort(key=lambda event: event.timestamp)
    return merged
