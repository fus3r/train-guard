from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainguard import cli
from trainguard.config import PolicyConfig
from trainguard.model import Observation, PowerSource
from trainguard.simulation import (
    TraceError,
    compare_policies,
    load_trace,
    simulate_policy,
)

EXAMPLE_TRACE = Path(__file__).parents[1] / "examples" / "power-trace.jsonl"


def _observation(timestamp: str) -> Observation:
    return Observation(
        source=PowerSource.AC,
        percent=80.0,
        temperature_c=30.0,
        charging=False,
        observed_at=timestamp,
    )


def test_replay_and_comparison_match_the_hand_checked_trace():
    observations = load_trace(EXAMPLE_TRACE)
    report = simulate_policy(PolicyConfig(), observations)

    assert report["samples"] == 7
    assert report["elapsed_seconds"] == 1800.0
    assert report["action_seconds"] == {
        "full": 600.0,
        "gentle": 300.0,
        "stop": 900.0,
    }
    assert report["action_percent"]["stop"] == 50.0
    assert report["action_transitions"] == 5
    assert len(report["transitions"]) == report["decision_transitions"] + 1
    assert len(report["policy_sha256"]) == 64
    assert len(report["observations_sha256"]) == 64

    comparison = compare_policies(
        PolicyConfig(),
        PolicyConfig(run_on_battery=True),
        observations,
    )
    assert comparison["delta"]["action_seconds"] == {
        "full": 0.0,
        "gentle": 300.0,
        "stop": -300.0,
    }
    assert comparison["delta"]["action_disagreement_seconds"] == 300.0
    assert (
        comparison["baseline"]["observations_sha256"]
        == (comparison["candidate"]["observations_sha256"])
    )


def test_terminal_journal_observation_closes_the_last_interval(tmp_path):
    trace = tmp_path / "completed.events.jsonl"
    first = _observation("2026-07-26T09:00:00Z").to_dict()
    terminal = _observation("2026-07-26T10:00:00Z").to_dict()
    trace.write_text(
        "\n".join(
            (
                json.dumps({"event": "decision", "observation": first}),
                json.dumps(
                    {
                        "event": "stopped",
                        "observation": terminal,
                        "trace_terminal": True,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = simulate_policy(PolicyConfig(), load_trace(trace))
    assert report["elapsed_seconds"] == 3600.0
    assert report["action_seconds"]["full"] == 3600.0


@pytest.mark.parametrize(
    "timestamp",
    (
        "2026-07-26T09:00:00.5Z",
        "2026-07-26T09:00:00.12Z",
        "2026-07-26t09:00:00.123456789z",
        "2026-07-26T09:00:00.1234+02:00",
    ),
)
def test_rfc3339_fraction_lengths_are_accepted(timestamp):
    assert simulate_policy(PolicyConfig(), [_observation(timestamp)])["samples"] == 1


@pytest.mark.parametrize(
    ("rows", "message", "line"),
    (
        (
            [
                {
                    "source": "ac",
                    "temp_c": 45.0,
                    "observed_at": "2026-07-26T09:00:00Z",
                }
            ],
            "unknown observation key",
            1,
        ),
        (
            [
                {
                    "source": "ac",
                    "temperature_c": float("nan"),
                    "observed_at": "2026-07-26T09:00:00Z",
                }
            ],
            "temperature_c must be between",
            1,
        ),
        (
            [
                {
                    "source": "ac",
                    "observed_at": "2026-07-26T09:01:00Z",
                },
                {
                    "source": "ac",
                    "observed_at": "2026-07-26T09:00:00Z",
                },
            ],
            "ordered",
            2,
        ),
        (
            [{"source": "ac", "observed_at": "2026-07-26T09:00:00"}],
            "UTC offset",
            1,
        ),
    ),
)
def test_trace_errors_identify_the_file_and_line(tmp_path, rows, message, line):
    trace = tmp_path / "bad.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(TraceError, match=message) as caught:
        load_trace(trace)
    assert f"{trace}:{line}" in str(caught.value)


def test_only_a_torn_final_line_is_tolerated(tmp_path):
    complete = json.dumps({"observation": _observation("2026-07-26T09:00:00Z").to_dict()})
    live = tmp_path / "live.jsonl"
    live.write_text(complete + "\n" + '{"observation":{"sou', encoding="utf-8")
    assert len(load_trace(live)) == 1

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"bad\n' + complete + "\n", encoding="utf-8")
    with pytest.raises(TraceError, match="invalid JSON"):
        load_trace(corrupt)


def test_simulate_cli_is_offline_and_emits_versioned_json(tmp_path, monkeypatch, capsys):
    state_home = tmp_path / "must-not-be-created"
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"run_on_battery": true}', encoding="utf-8")
    monkeypatch.setenv("TRAIN_GUARD_HOME", str(state_home))

    assert (
        cli.main(
            [
                "simulate",
                str(EXAMPLE_TRACE),
                "--compare-config",
                str(candidate),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["delta"]["action_seconds"]["stop"] == -300.0
    assert payload["candidate_config_source"] == str(candidate)
    assert not state_home.exists()


def test_value_equal_inputs_have_equal_fingerprints():
    timestamp = "2026-07-26T09:00:00Z"
    with_negative_zero = Observation(
        PowerSource.AC,
        -0.0,
        30.0,
        False,
        timestamp,
    )
    with_zero = Observation(PowerSource.AC, 0.0, 30.0, False, timestamp)

    assert (
        simulate_policy(PolicyConfig(), [with_negative_zero])["observations_sha256"]
        == simulate_policy(PolicyConfig(), [with_zero])["observations_sha256"]
    )
    assert (
        simulate_policy(PolicyConfig(temp_pause_c=42), [_observation(timestamp)])["policy_sha256"]
        == simulate_policy(PolicyConfig(temp_pause_c=42.0), [_observation(timestamp)])[
            "policy_sha256"
        ]
    )
