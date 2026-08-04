from __future__ import annotations

import itertools
import random
from fractions import Fraction
from typing import Optional

import pytest

from trainguard.clairvoyant import (
    build_frontier,
    build_joint_frontier,
    clairvoyant_report,
)


@pytest.mark.parametrize(
    ("hot_rated", "low_rated", "run", "hot_budget", "low_budget", "expected"),
    (
        (
            [(0.0, 1.0), (1.0, 1.0)],
            [(1.0, 1.0), (0.0, 1.0)],
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            [(0.0, 1.0), (1.0, 1.0), (1.0, 0.25)],
            [(1.0, 1.0), (0.0, 1.0), (1.0, 0.25)],
            0.25,
            0.25,
            0.25,
            0.5,
        ),
    ),
)
def test_joint_bound_is_one_attainable_two_budget_optimum(
    hot_rated,
    low_rated,
    run,
    hot_budget,
    low_budget,
    expected,
):
    hot = build_frontier(hot_rated)
    low = build_frontier(low_rated)
    joint = build_joint_frontier(hot_rated, low_rated)

    report = clairvoyant_report(
        hot,
        low,
        joint,
        run_seconds=run,
        hot_cost=hot_budget,
        low_battery_cost=low_budget,
        hot_ref_c=35.0,
    )

    # Taking min(hot-only, low-only) would incorrectly return 1.0 or 1.25.
    assert hot.value_at(hot_budget) == 1.0 + hot_budget
    assert low.value_at(low_budget) == 1.0 + low_budget
    assert report["bound_run_seconds"] == expected
    assert report["gap_seconds"] == expected - run


def _exact_joint_lp(
    items: list[tuple[Fraction, int, Fraction]],
    hot_budget: Fraction,
    low_budget: Fraction,
) -> Fraction:
    """Enumerate vertices of the two-constraint continuous LP."""

    best = Fraction(0)

    def feasible_objective(values: list[Fraction]) -> Optional[Fraction]:
        if any(value < 0 or value > items[index][2] for index, value in enumerate(values)):
            return None
        hot = sum(items[index][0] * value for index, value in enumerate(values))
        low = sum(items[index][1] * value for index, value in enumerate(values))
        if hot > hot_budget or low > low_budget:
            return None
        return sum(values)

    indices = tuple(range(len(items)))
    for free_count in range(3):
        for free in itertools.combinations(indices, free_count):
            fixed = tuple(index for index in indices if index not in free)
            for upper in itertools.product((False, True), repeat=len(fixed)):
                values = [Fraction(0) for _ in items]
                for index, use_upper in zip(fixed, upper):
                    if use_upper:
                        values[index] = items[index][2]

                candidates: list[list[Fraction]] = []
                if free_count == 0:
                    candidates.append(values)
                elif free_count == 1:
                    index = free[0]
                    used_hot = sum(items[item][0] * values[item] for item in fixed)
                    used_low = sum(items[item][1] * values[item] for item in fixed)
                    for coefficient, remaining in (
                        (items[index][0], hot_budget - used_hot),
                        (Fraction(items[index][1]), low_budget - used_low),
                    ):
                        if coefficient:
                            candidate = list(values)
                            candidate[index] = remaining / coefficient
                            candidates.append(candidate)
                else:
                    first, second = free
                    used_hot = sum(items[item][0] * values[item] for item in fixed)
                    used_low = sum(items[item][1] * values[item] for item in fixed)
                    hot_remaining = hot_budget - used_hot
                    low_remaining = low_budget - used_low
                    determinant = (
                        items[first][0] * items[second][1] - items[second][0] * items[first][1]
                    )
                    if determinant:
                        candidate = list(values)
                        candidate[first] = (
                            hot_remaining * items[second][1] - items[second][0] * low_remaining
                        ) / determinant
                        candidate[second] = (
                            items[first][0] * low_remaining - hot_remaining * items[first][1]
                        ) / determinant
                        candidates.append(candidate)

                for candidate in candidates:
                    objective = feasible_objective(candidate)
                    if objective is not None:
                        best = max(best, objective)
    return best


def test_joint_frontier_matches_an_independent_exact_rational_lp():
    rng = random.Random(0x4A4F494E54)
    rates = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3)]
    durations = [Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)]

    for _ in range(80):
        items = [
            (rng.choice(rates), rng.randrange(2), rng.choice(durations))
            for _ in range(rng.randint(1, 6))
        ]
        total_hot = sum(rate * duration for rate, _, duration in items)
        total_low = sum(low * duration for _, low, duration in items)
        hot_budget = Fraction(rng.randrange(int(total_hot * 4) + 1), 4)
        low_budget = Fraction(rng.randrange(int(total_low * 4) + 1), 4)
        hot_rated = [(float(rate), float(duration)) for rate, _, duration in items]
        low_rated = [(float(low), float(duration)) for _, low, duration in items]

        expected = _exact_joint_lp(items, hot_budget, low_budget)
        actual = build_joint_frontier(hot_rated, low_rated).value_at(
            float(hot_budget),
            float(low_budget),
        )
        assert actual == pytest.approx(float(expected), abs=1e-12)


def test_report_clamps_float_noise_and_handles_a_zero_bound():
    frontier = build_frontier([(0.0, 100.0)])
    joint = build_joint_frontier([(0.0, 100.0)], [(0.0, 100.0)])
    report = clairvoyant_report(
        frontier,
        frontier,
        joint,
        run_seconds=100.00000000000003,
        hot_cost=0.0,
        low_battery_cost=0.0,
        hot_ref_c=35.0,
    )
    assert report["efficiency"] == 1.0
    assert report["gap_seconds"] == 0.0

    empty = build_frontier([])
    empty_report = clairvoyant_report(
        empty,
        empty,
        build_joint_frontier([], []),
        run_seconds=0.0,
        hot_cost=0.0,
        low_battery_cost=0.0,
        hot_ref_c=35.0,
    )
    assert empty_report["bound_run_seconds"] == 0.0
    assert empty_report["efficiency"] is None
    assert empty_report["hot_only_hindsight_threshold_c"] is None
