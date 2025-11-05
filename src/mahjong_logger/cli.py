"""Command line helpers to launch the GUI or batch tasks."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_project_paths
from .gui.app import launch_gui
from .models.yolo_manager import ModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mahjong logger control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    gui_parser = sub.add_parser("gui", help="Launch the desktop annotation and training GUI")
    gui_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gui":
        paths = load_project_paths(args.root)
        registry = ModelRegistry(paths)
        launch_gui(paths, registry)


if __name__ == "__main__":  # pragma: no cover
    main()
