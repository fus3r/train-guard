from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from trainguard import cli
from trainguard.config import ConfigError, PolicyConfig
from trainguard.native import KernelRow
from trainguard.robustness import SensitivityBounds, SensitivityResult, analyze_sensitivity
from trainguard.simulation import load_trace
from trainguard.sweep import (
    SweepError,
    _interval_front_flags,
    _pareto_flags,
    _python_rows,
    expand_grid,
    parse_grid,
    run_sweep,
)

EXAMPLE_TRACE = Path(__file__).parents[1] / "examples" / "power-trace.jsonl"


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ("[]", "non-empty object"),
        ("{}", "non-empty object"),
        ('{"nope":[1]}', "unknown policy field"),
        ('{"temp_pause_c":[]}', "non-empty list"),
        ('{"temp_pause_c":[[42]]}', "must be scalars"),
        ("{", "invalid JSON"),
    ),
)
def test_grid_parser_rejects_ambiguous_inputs(text, message):
    with pytest.raises(SweepError, match=message):
        parse_grid(text)


def test_grid_expansion_validates_and_deduplicates_candidates():
    grid = parse_grid('{"temp_pause_c":[44,30,44],"temp_resume_c":[36,36]}')
    candidates, rejected = expand_grid(grid, PolicyConfig())

    assert [overrides["temp_pause_c"] for overrides, _ in candidates] == [44]
    assert len(rejected) == 1
    assert "temp_resume_c" in rejected[0]["error"]


def test_sweep_matches_hand_calculation_and_joint_bound():
    report = run_sweep(
        PolicyConfig(),
        parse_grid('{"run_on_battery":[true]}'),
        load_trace(EXAMPLE_TRACE),
        engine="python",
    )

    baseline = report["baseline"]
    assert baseline["metrics"]["action_seconds"] == {
        "full": 600.0,
        "gentle": 300.0,
        "stop": 900.0,
    }
    assert baseline["metrics"]["hot_run_degc_seconds"] == 600.0
    assert report["trace_facts"]["hot_degc_seconds"] == 3900.0
    assert baseline["clairvoyant"] == {
        "bound_run_seconds": 1200.0,
        "hot_bound_run_seconds": 1200.0,
        "low_battery_bound_run_seconds": 1800.0,
        "efficiency": 0.75,
        "gap_seconds": 300.0,
        "hot_only_hindsight_threshold_c": 36.0,
    }
    candidate = report["candidates"][0]
    assert candidate["delta_vs_baseline"]["run_seconds"] == 300.0
    assert candidate["clairvoyant"]["efficiency"] == 1.0
    assert candidate["pareto_optimal"] is True


def test_pareto_pass_matches_hand_checked_and_pairwise_oracles():
    def row(run: float, hot: float, low: float) -> KernelRow:
        return KernelRow(run, 0.0, 0.0, run, hot, low, 0, 0, "0" * 16, None)

    def pairwise_oracle(rows: list[KernelRow]) -> list[bool]:
        flags = []
        for candidate in rows:
            dominated = any(
                other is not candidate
                and other.run_seconds >= candidate.run_seconds
                and other.hot_degc_seconds <= candidate.hot_degc_seconds
                and other.low_battery_run_seconds <= candidate.low_battery_run_seconds
                and (
                    other.run_seconds > candidate.run_seconds
                    or other.hot_degc_seconds < candidate.hot_degc_seconds
                    or other.low_battery_run_seconds < candidate.low_battery_run_seconds
                )
                for other in rows
            )
            flags.append(not dominated)
        return flags

    mixed = [
        row(10.0, 5.0, 5.0),
        row(9.0, 6.0, 6.0),
        row(10.0, 5.0, 5.0),
        row(10.0, 4.0, 7.0),
        row(11.0, 8.0, 2.0),
        row(8.0, 4.0, 8.0),
        row(10.0, 6.0, 4.0),
    ]
    assert _pareto_flags(mixed) == [True, False, True, True, True, False, True]

    rng = random.Random(20260810)
    randomized = [
        row(float(rng.randrange(8)), float(rng.randrange(8)), float(rng.randrange(8)))
        for _ in range(500)
    ]
    randomized.extend([row(9.0, 1.0, 3.0), row(9.0, 1.0, 3.0), row(8.0, 0.0, 4.0)])
    assert _pareto_flags(randomized) == pairwise_oracle(randomized)


def test_interval_front_matches_hand_checked_and_pairwise_oracles():
    def row(
        run_min: float,
        run_max: float,
        hot_min: float,
        hot_max: float,
        low_min: float,
        low_max: float,
    ) -> SensitivityResult:
        return SensitivityResult(
            run_min,
            run_max,
            hot_min,
            hot_max,
            low_min,
            low_max,
            0.0,
            0.0,
            0,
            0,
        )

    def pairwise_oracle(rows: list[SensitivityResult]) -> list[bool]:
        return [
            not any(
                other is not candidate
                and other.minimum_run_seconds >= candidate.maximum_run_seconds
                and other.maximum_hot_run_degc_seconds <= candidate.minimum_hot_run_degc_seconds
                and other.maximum_low_battery_run_seconds
                <= candidate.minimum_low_battery_run_seconds
                and (
                    other.minimum_run_seconds > candidate.maximum_run_seconds
                    or other.maximum_hot_run_degc_seconds < candidate.minimum_hot_run_degc_seconds
                    or other.maximum_low_battery_run_seconds
                    < candidate.minimum_low_battery_run_seconds
                )
                for other in rows
            )
            for candidate in rows
        ]

    mixed = [
        row(10, 12, 1, 2, 1, 2),  # dominates only boxes it clears worst-to-best
        row(8, 9, 3, 4, 3, 4),  # dominated on every axis
        row(13, 14, 4, 5, 1, 2),  # run/exposure trade-off
        row(9, 11, 1.5, 2.5, 1.5, 2.5),  # overlap stays unresolved
        row(10, 10, 2, 2, 2, 2),  # equality alone is not strict dominance
        row(10, 10, 2, 2, 2, 2),  # duplicate stays co-optimal
        row(9, 9, 2, 2, 2, 2),  # strict run-time loss is dominated
    ]
    assert _interval_front_flags(mixed) == [True, False, True, True, True, True, False]

    rng = random.Random(0x1A7E2A1)
    randomized = []
    for _ in range(500):
        lower = [float(rng.randrange(20)) for _ in range(3)]
        upper = [value + float(rng.randrange(4)) for value in lower]
        randomized.append(row(lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]))
    randomized.extend(mixed)
    assert _interval_front_flags(randomized) == pairwise_oracle(randomized)


def test_direct_policy_construction_enforces_sweep_invariants():
    with pytest.raises(ConfigError, match="temp_resume_c"):
        PolicyConfig(temp_resume_c=44, temp_pause_c=42)
    with pytest.raises(ConfigError, match="must be a number"):
        PolicyConfig(temp_pause_c="hot")  # type: ignore[arg-type]

    assert PolicyConfig(temp_pause_c=42).to_dict() == PolicyConfig(temp_pause_c=42.0).to_dict()
    assert str(PolicyConfig(battery_floor_pct=-0.0).battery_floor_pct) == "0.0"


def test_sweep_cli_is_offline_and_states_its_limitations(tmp_path, monkeypatch, capsys):
    state_home = tmp_path / "must-not-be-created"
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            {
                "temp_pause_c": [40, 44],
                "run_on_battery": [True, False],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAIN_GUARD_HOME", str(state_home))
    monkeypatch.delenv("TRAIN_GUARD_KERNEL", raising=False)
    monkeypatch.setattr("trainguard.sweep.find_kernel", lambda: None)

    assert (
        cli.main(
            [
                "sweep",
                str(EXAMPLE_TRACE),
                "--grid",
                str(grid),
                "--top",
                "2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "candidates: 4 evaluated" in output
    assert "joint clairvoyant bound: 1200s" in output
    assert "efficiency 75%" in output
    assert "hot-only cutoff 36C" in output
    assert "not battery-life predictions" in output
    assert not state_home.exists()

    assert cli.main(["sweep", str(EXAMPLE_TRACE), "--grid", str(grid), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["engine"] == "python"
    assert payload["candidates_evaluated"] == 4
    assert payload["baseline"]["clairvoyant"]["efficiency"] == 0.75
    assert "uncertainty" not in payload
    assert all("interval_nondominated" not in row for row in payload["candidates"])

    bounds = [
        "--temperature-uncertainty-c",
        "0.5",
        "--charge-uncertainty-pct",
        "1",
    ]
    assert cli.main(["sweep", str(EXAMPLE_TRACE), "--grid", str(grid), *bounds]) == 0
    bounded_output = capsys.readouterr().out
    assert "conservative interval-front enclosure" in bounded_output
    assert "not an exact robust Pareto set" in bounded_output

    assert cli.main(["sweep", str(EXAMPLE_TRACE), "--grid", str(grid), *bounds, "--json"]) == 0
    bounded = json.loads(capsys.readouterr().out)
    assert bounded["schema_version"] == 2
    assert bounded["engine"] == "python"
    assert bounded["uncertainty"]["engine"] == "python"
    assert bounded["uncertainty"]["frontier_method"] == "marginal_interval_separation"
    assert bounded["interval_front_size"] == sum(
        candidate["interval_nondominated"] for candidate in bounded["candidates"]
    )
    assert bounded["baseline"]["sensitivity"]["run_seconds"] == {
        "minimum": 600.0,
        "maximum": 1500.0,
    }
    assert all(
        "action_change_margin" not in candidate["sensitivity"]
        for candidate in bounded["candidates"]
    )
    assert not state_home.exists()


def test_sweep_rejects_unavailable_or_invalid_engines(monkeypatch):
    observations = load_trace(EXAMPLE_TRACE)
    grid = parse_grid('{"run_on_battery":[true]}')
    with pytest.raises(SweepError, match="engine must be"):
        run_sweep(PolicyConfig(), grid, observations, engine="rust")
    monkeypatch.delenv("TRAIN_GUARD_KERNEL", raising=False)
    monkeypatch.setattr("trainguard.sweep.find_kernel", lambda: None)
    with pytest.raises(SweepError, match="engine=native requires"):
        run_sweep(PolicyConfig(), grid, observations, engine="native")

    def nominal_kernel(_kernel, policies, kernel_observations, timestamps, **references):
        durations = [
            (timestamps[index + 1] - timestamps[index]).total_seconds()
            for index in range(len(timestamps) - 1)
        ] + [0.0]
        return _python_rows(
            policies,
            kernel_observations,
            durations,
            hot_ref_c=references["hot_ref_c"],
            low_battery_ref_pct=references["low_battery_ref_pct"],
        )

    monkeypatch.setattr("trainguard.sweep.find_kernel", lambda: Path("nominal-kernel"))
    monkeypatch.setattr("trainguard.sweep.run_kernel", nominal_kernel)

    def sensitivity_kernel(
        _kernel, policies, kernel_observations, timestamps, bounds, **references
    ):
        durations = [
            (timestamps[index + 1] - timestamps[index]).total_seconds()
            for index in range(len(timestamps) - 1)
        ] + [0.0]
        return [
            analyze_sensitivity(
                policy,
                kernel_observations,
                durations,
                bounds,
                hot_ref_c=references["hot_ref_c"],
                low_battery_ref_pct=references["low_battery_ref_pct"],
            )
            for policy in policies
        ]

    monkeypatch.setattr("trainguard.sweep.run_sensitivity_kernel", sensitivity_kernel)
    bounded = run_sweep(
        PolicyConfig(),
        grid,
        observations,
        engine="native",
        sensitivity=SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
    )
    assert bounded["engine"] == "native"
    assert bounded["kernel_verified_against_reference"] is True
    assert bounded["uncertainty"]["engine"] == "native"
    assert bounded["uncertainty"]["kernel_verified_against_reference"] is True
