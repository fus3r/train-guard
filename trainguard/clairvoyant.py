"""Exact offline work bounds for replayed policies on one recorded trace.

Each replay selects intervals during which work is permitted. Relaxing those
selections to fractions and giving the selector the full trace produces a
continuous linear program: maximize permitted time under the candidate's hot
and low-battery exposure budgets. Low-battery cost is binary per interval, so
the joint optimum reduces to sorted hot-cost prefix frontiers for low and
non-low intervals.

The result is a replay diagnostic, not a physical prediction. Recorded
temperature and charge came from the original run and remain exogenous.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Optional, Sequence

if __package__:
    from .model import Observation, PowerSource
else:  # Keep direct CLI execution from a checkout working.
    from model import Observation, PowerSource


@dataclass(frozen=True)
class CostFrontier:
    """Fractional-knapsack frontier for one exposure axis."""

    free_seconds: float
    rates: tuple[float, ...]
    cost_prefix: tuple[float, ...]
    value_prefix: tuple[float, ...]

    @property
    def total_seconds(self) -> float:
        return self.free_seconds + (self.value_prefix[-1] if self.value_prefix else 0.0)

    @property
    def total_cost(self) -> float:
        return self.cost_prefix[-1] if self.cost_prefix else 0.0

    def value_at(self, budget: float) -> float:
        """Maximum permitted seconds within one non-negative budget."""

        if not self.rates or budget <= 0.0:
            return self.free_seconds
        index = bisect_left(self.cost_prefix, budget)
        if index == len(self.rates):
            return self.free_seconds + self.value_prefix[-1]
        previous_cost = self.cost_prefix[index - 1] if index else 0.0
        previous_value = self.value_prefix[index - 1] if index else 0.0
        return (
            self.free_seconds
            + previous_value
            + (budget - previous_cost) / self.rates[index]
        )

    def marginal_rate_at(self, budget: float) -> Optional[float]:
        """Rate bought by the budget's final fraction, if it still binds."""

        if not self.rates:
            return None
        if budget <= 0.0:
            return 0.0
        index = bisect_left(self.cost_prefix, budget)
        if index == len(self.rates):
            return None
        return self.rates[index]

    def cost_for_value(self, seconds: float) -> float:
        """Minimum exposure required to buy ``seconds`` from this group."""

        priced_seconds = seconds - self.free_seconds
        if priced_seconds <= 0.0 or not self.rates:
            return 0.0
        index = bisect_left(self.value_prefix, priced_seconds)
        if index == len(self.rates):
            return self.total_cost
        previous_cost = self.cost_prefix[index - 1] if index else 0.0
        previous_value = self.value_prefix[index - 1] if index else 0.0
        return previous_cost + (priced_seconds - previous_value) * self.rates[index]


@dataclass(frozen=True)
class JointCostFrontier:
    """Exact fractional frontier under hot and low-battery budgets."""

    hot: CostFrontier
    non_low: CostFrontier
    low: CostFrontier
    hot_low_prefix: tuple[float, ...]
    hot_item_is_low: tuple[bool, ...]
    free_low_seconds: float

    def minimum_low_at_hot_optimum(self, hot_budget: float) -> float:
        """Least low-battery time among hot-only optimal schedules."""

        if not self.hot.rates or hot_budget >= self.hot.total_cost:
            return self.low.total_seconds
        if hot_budget <= 0.0:
            return self.free_low_seconds

        index = bisect_left(self.hot.cost_prefix, hot_budget)
        previous_cost = self.hot.cost_prefix[index - 1] if index else 0.0
        previous_low = self.hot_low_prefix[index - 1] if index else 0.0
        marginal_seconds = (hot_budget - previous_cost) / self.hot.rates[index]
        if self.hot_item_is_low[index]:
            previous_low += marginal_seconds
        return self.free_low_seconds + previous_low

    def value_at(self, hot_budget: float, low_battery_budget: float) -> float:
        """Maximum work satisfying both budgets at the same time."""

        hot_budget = max(0.0, hot_budget)
        low_battery_budget = max(0.0, low_battery_budget)
        hot_only = self.hot.value_at(hot_budget)
        if low_battery_budget >= self.minimum_low_at_hot_optimum(hot_budget):
            return hot_only

        low_cost = self.low.cost_for_value(low_battery_budget)
        remaining_hot = max(0.0, hot_budget - low_cost)
        return low_battery_budget + self.non_low.value_at(remaining_hot)


def build_frontier(
    rated_intervals: Sequence[tuple[float, float]],
) -> CostFrontier:
    """Build one frontier from ``(cost per second, duration)`` pairs."""

    free_seconds = 0.0
    priced: list[tuple[float, float]] = []
    for rate, duration in rated_intervals:
        if duration <= 0.0:
            continue
        if rate <= 0.0:
            free_seconds += duration
        else:
            priced.append((rate, duration))
    priced.sort(key=lambda item: item[0])

    rates: list[float] = []
    cost_prefix: list[float] = []
    value_prefix: list[float] = []
    cost = 0.0
    value = 0.0
    for rate, duration in priced:
        cost += rate * duration
        value += duration
        rates.append(rate)
        cost_prefix.append(cost)
        value_prefix.append(value)
    return CostFrontier(
        free_seconds=free_seconds,
        rates=tuple(rates),
        cost_prefix=tuple(cost_prefix),
        value_prefix=tuple(value_prefix),
    )


def build_joint_frontier(
    hot_rated: Sequence[tuple[float, float]],
    low_battery_rated: Sequence[tuple[float, float]],
) -> JointCostFrontier:
    """Build the joint frontier from aligned one-axis interval rates."""

    if len(hot_rated) != len(low_battery_rated):
        raise ValueError("joint frontier inputs must contain the same intervals")

    items: list[tuple[float, bool, float]] = []
    free_low_seconds = 0.0
    non_low: list[tuple[float, float]] = []
    low: list[tuple[float, float]] = []
    for (hot_rate, duration), (low_rate, low_duration) in zip(
        hot_rated, low_battery_rated
    ):
        if duration != low_duration:
            raise ValueError("joint frontier interval durations must align")
        if duration <= 0.0:
            continue
        is_low = low_rate > 0.0
        items.append((hot_rate, is_low, duration))
        (low if is_low else non_low).append((hot_rate, duration))
        if is_low and hot_rate <= 0.0:
            free_low_seconds += duration

    # For a tied hot rate, choose non-low time first. This constructs the
    # minimum-low member of the hot-only optimum set.
    items.sort(key=lambda item: (item[0], item[1]))
    hot = build_frontier([(rate, duration) for rate, _, duration in items])
    hot_low_prefix: list[float] = []
    hot_item_is_low: list[bool] = []
    cumulative_low = 0.0
    for rate, is_low, duration in items:
        if rate <= 0.0:
            continue
        if is_low:
            cumulative_low += duration
        hot_low_prefix.append(cumulative_low)
        hot_item_is_low.append(is_low)

    return JointCostFrontier(
        hot=hot,
        non_low=build_frontier(non_low),
        low=build_frontier(low),
        hot_low_prefix=tuple(hot_low_prefix),
        hot_item_is_low=tuple(hot_item_is_low),
        free_low_seconds=free_low_seconds,
    )


def hot_rated_intervals(
    observations: Sequence[Observation],
    durations: Sequence[float],
    hot_ref_c: float,
) -> list[tuple[float, float]]:
    """Rate each interval by degree-seconds per permitted second."""

    rated: list[tuple[float, float]] = []
    for observation, duration in zip(observations, durations):
        rate = 0.0
        if observation.temperature_c is not None:
            rate = max(0.0, observation.temperature_c - hot_ref_c)
        rated.append((rate, duration))
    return rated


def low_battery_rated_intervals(
    observations: Sequence[Observation],
    durations: Sequence[float],
    low_battery_ref_pct: float,
) -> list[tuple[float, float]]:
    """Rate each interval by low-battery seconds per permitted second."""

    rated: list[tuple[float, float]] = []
    for observation, duration in zip(observations, durations):
        low = (
            observation.source is PowerSource.BATTERY
            and observation.percent is not None
            and observation.percent <= low_battery_ref_pct
        )
        rated.append((1.0 if low else 0.0, duration))
    return rated


def clairvoyant_report(
    hot: CostFrontier,
    low_battery: CostFrontier,
    joint: JointCostFrontier,
    *,
    run_seconds: float,
    hot_cost: float,
    low_battery_cost: float,
    hot_ref_c: float,
) -> dict[str, Any]:
    """Bound one candidate at its own realized exposure budgets."""

    hot_bound = hot.value_at(hot_cost)
    low_bound = low_battery.value_at(low_battery_cost)
    bound = joint.value_at(hot_cost, low_battery_cost)
    marginal = hot.marginal_rate_at(hot_cost)

    # Real-arithmetic dominance is independently tested with exact rationals.
    # Clamp only the possible floating aggregation noise in the public report.
    gap = max(0.0, bound - run_seconds)
    efficiency: Optional[float] = None
    if bound > 0.0:
        efficiency = min(1.0, run_seconds / bound)
    return {
        "bound_run_seconds": bound,
        "hot_bound_run_seconds": hot_bound,
        "low_battery_bound_run_seconds": low_bound,
        "efficiency": efficiency,
        "gap_seconds": gap,
        "hot_only_hindsight_threshold_c": (
            None if marginal is None else hot_ref_c + marginal
        ),
    }
