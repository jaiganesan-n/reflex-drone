"""Seed splits (SPEC §10). Level ``stress`` always runs on STRESS seeds."""
from __future__ import annotations

import secrets

TRAIN = range(0, 100_000)
VAL = range(100_000, 100_050)
TEST = range(200_000, 200_100)
STRESS = range(300_000, 300_100)

SPLITS = {"train": TRAIN, "val": VAL, "test": TEST, "stress": STRESS}


def seeds_for(split: str, level: str) -> list[int]:
    """Seeds for a split and level. ``stress`` always gets STRESS seeds, whatever the split."""
    if level == "stress":
        return list(STRESS)
    try:
        return list(SPLITS[split])
    except KeyError:
        raise ValueError(f"unknown split {split!r}; choose one of {tuple(SPLITS)}") from None


def split_of(seed: int) -> str | None:
    for name, rng in SPLITS.items():
        if seed in rng:
            return name
    return None


def play_seed() -> int:
    """Fresh random seed for an app run (SPEC §10)."""
    return secrets.randbelow(2**31)
