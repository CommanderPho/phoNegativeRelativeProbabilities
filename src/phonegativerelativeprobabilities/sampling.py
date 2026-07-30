from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import numpy as np

"""Bag generation and sampling helpers for white/orange ball experiments.

from phonegativerelativeprobabilities import generate_bag, sample_from_bag, observation_string_to_bag, bag_to_observation_string

"""

_OBS_CHAR_TO_COLOR: dict[str, str] = {
    "o": "orange",
    "w": "white",
    "orange": "orange",
    "white": "white",
}
_COLOR_TO_OBS_CHAR: dict[str, str] = {
    "orange": "o",
    "white": "w",
}


def generate_bag(n_total: int = 25, ball_color_counts_dict: dict | None = None, ball_color_prob_dict: dict | None = None) -> list[str]:
    """Generate a new bag distribution from counts or probabilities."""
    assert (ball_color_prob_dict is not None) or (ball_color_counts_dict is not None)
    if ball_color_prob_dict is not None:
        colors = list(ball_color_prob_dict.keys())
        probabilities = list(ball_color_prob_dict.values())
        bag_of_balls = [
            str(x) for x in np.random.choice(colors, size=n_total, p=probabilities)
        ]
    else:
        colors = list(ball_color_counts_dict.keys())
        counts = list(ball_color_counts_dict.values())
        bag_of_balls = random.choices(colors, weights=counts, k=n_total)

    return bag_of_balls


def sample_from_bag(bag_of_balls: list[str], n_sequential_samples: int = 25, ball_observation_error: float = 0.01) -> list[str]:
    """Draw samples from the bag without replacement, with optional observation error."""
    bag = bag_of_balls.copy()
    samples = []
    n_samples = min(n_sequential_samples, len(bag))

    for _ in range(n_samples):
        if not bag:
            break
        idx = random.randint(0, len(bag) - 1)
        real_color = bag.pop(idx)
        # Flip color with probability = ball_observation_error
        if random.random() < ball_observation_error:
            observed_color = "white" if (real_color == "orange") else "orange"
        else:
            observed_color = real_color
        samples.append(observed_color)
    ## END for _ in range(n_samples)....

    return samples


def repeat_sample_from_bag(
    bag_of_balls: list[str],
    n_repeats: int = 1000,
    n_sequential_samples: int = 25,
    ball_observation_error: float = 0.01,
) -> tuple[list[list[str]], list[dict[str, int]]]:
    """Independently sample from the same bag ``n_repeats`` times without replacement.

    Uses vectorized numpy permutations so each repeat is an independent draw of
    ``min(n_sequential_samples, len(bag))`` balls, then optional observation flips.

    Returns:
        samples: list of length ``n_repeats``, each an observed color sequence
        color_counts: per-repeat tallies ``{'orange': int, 'white': int}``
    """
    if n_repeats < 1:
        return [], []

    bag_arr = np.asarray(bag_of_balls, dtype=object)
    n_bag = bag_arr.size
    if n_bag == 0:
        empty_counts = [{"orange": 0, "white": 0} for _ in range(n_repeats)]
        return [[] for _ in range(n_repeats)], empty_counts

    n_samples = min(n_sequential_samples, n_bag)
    # Random permutation per repeat via argsort of uniform noise; take first n_samples
    perm_idx = np.argsort(np.random.random((n_repeats, n_bag)), axis=1)[:, :n_samples]
    sampled = bag_arr[perm_idx]

    if ball_observation_error > 0.0:
        flip = np.random.random((n_repeats, n_samples)) < ball_observation_error
        flipped = np.where(sampled == "orange", "white", "orange")
        sampled = np.where(flip, flipped, sampled)

    n_orange = (sampled == "orange").sum(axis=1)
    n_white = (sampled == "white").sum(axis=1)
    color_counts = [
        {"orange": int(o), "white": int(w)} for o, w in zip(n_orange, n_white)
    ]

    return sampled.tolist(), color_counts


def observation_string_to_bag(obs_seq: str) -> list[str]:
    """Convert a string of white/orange observations into a full bag.

    Accepts compact chars ('o'/'w') or full color names separated by
    non-letters (e.g. 'orange,white,orange' or 'o w o').

    Usage:

        obs_seq_string_example_original_paper: str = 'owowwowwwwwwwwwwwwwwwowww'
        example_bag_from_paper = observation_string_to_bag(obs_seq_string_example_original_paper)
    """
    tokens = [
        t
        for t in "".join(ch if ch.isalpha() else " " for ch in obs_seq.lower()).split()
        if t
    ]
    # Compact form with no separators: treat each character as an observation
    if len(tokens) == 1 and all(ch in _OBS_CHAR_TO_COLOR for ch in tokens[0]):
        tokens = list(tokens[0])
    try:
        return [_OBS_CHAR_TO_COLOR[t] for t in tokens]
    except KeyError as exc:
        raise ValueError(
            f"Unknown observation token {exc.args[0]!r}; expected o/w or orange/white"
        ) from exc


def bag_to_observation_string(bag: list[str]) -> str:
    """Convert a bag of white/orange colors into a compact observation string.

    Inverse of ``observation_string_to_bag`` for the compact 'o'/'w' encoding.
    """
    try:
        return "".join(_COLOR_TO_OBS_CHAR[color.lower()] for color in bag)
    except KeyError as exc:
        raise ValueError(
            f"Unknown bag color {exc.args[0]!r}; expected orange/white"
        ) from exc
