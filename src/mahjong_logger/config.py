"""Project-wide configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


DEFAULT_TILE_CLASSES: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red",
    "flower", "season", "tile_back", "tile_unknown",
]


@dataclass
class ProjectPaths:
    """Collection of filesystem paths used by the application."""

    root: Path
    data_root: Path = field(init=False)
    model_root: Path = field(init=False)
    export_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = self.root / "data"
        self.model_root = self.root / "models"
        self.export_root = self.root / "exports"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)


def load_project_paths(base: Path | None = None) -> ProjectPaths:
    """Return :class:`ProjectPaths` anchored at ``base`` or the CWD."""

    base_path = base or Path.cwd()
    return ProjectPaths(root=base_path)
