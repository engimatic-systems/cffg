"""Replay the compact cff-nbo9 recurrence smoothing diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from cffg.experiments.recurrence_smoothing import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    write_report(args.output, repo)


if __name__ == "__main__":
    main()
