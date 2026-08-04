from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainguard import cli
from trainguard.config import ConfigError, PolicyConfig
from trainguard.simulation import load_trace
from trainguard.sweep import SweepError, expand_grid, parse_grid, run_sweep

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


def test_sweep_rejects_unavailable_or_invalid_engines():
    observations = load_trace(EXAMPLE_TRACE)
    grid = parse_grid('{"run_on_battery":[true]}')
    with pytest.raises(SweepError, match="engine must be"):
        run_sweep(PolicyConfig(), grid, observations, engine="rust")
    with pytest.raises(SweepError, match="not available"):
        run_sweep(PolicyConfig(), grid, observations, engine="native")
