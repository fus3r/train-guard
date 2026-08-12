from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trainguard import cli
from trainguard.config import PolicyConfig
from trainguard.model import Action, DecisionReason, Observation, PowerSource
from trainguard.native import (
    KernelError,
    KernelRow,
    encode_request,
    encode_sensitivity_request,
    epoch_microseconds,
    fnv1a_decisions,
    resolve_thread_count,
    run_kernel,
    run_sensitivity_kernel,
)
from trainguard.policy import PolicyEngine
from trainguard.robustness import SensitivityBounds, SensitivityResult, analyze_sensitivity
from trainguard.simulation import load_trace, ordered_timestamps
from trainguard.sweep import SweepError, _python_rows, parse_grid, run_sweep

EXAMPLE_TRACE = Path(__file__).parents[1] / "examples" / "power-trace.jsonl"
KERNEL_SOURCE = Path(__file__).parents[1] / "native" / "replay_kernel.cpp"


@pytest.fixture(scope="module")
def kernel_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the kernel once per test run, or skip when no compiler exists."""

    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler available")
    binary_name = "train-guard-kernel.exe" if os.name == "nt" else "train-guard-kernel"
    binary = tmp_path_factory.mktemp("kernel") / binary_name
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-pthread",
            str(KERNEL_SOURCE),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.skip(f"kernel build failed: {build.stderr.strip()[:200]}")
    return binary


def test_epoch_microseconds_is_exact():
    # Expected value cross-checked with calendar.timegm, which counts
    # whole seconds without any float conversion.
    instant = datetime(2026, 7, 26, 9, 0, 0, 123456, tzinfo=timezone.utc)
    assert epoch_microseconds(instant) == 1785056400123456
    assert epoch_microseconds(datetime(1970, 1, 1, tzinfo=timezone.utc)) == 0


def test_fnv1a_matches_reference_vector():
    # FNV-1a over bytes 0x00 0x08 (full/ac_policy in the public enum order).
    assert fnv1a_decisions([(Action.FULL, DecisionReason.AC_POLICY)]) == (
        format(((0xCBF29CE484222325 ^ 0) * 0x100000001B3 ^ 8) * 0x100000001B3 % 2**64, "016x")
    )


def test_request_encoding_round_trips_optional_fields():
    observations = load_trace(EXAMPLE_TRACE)
    timestamps = ordered_timestamps(observations)
    request = encode_request(
        [PolicyConfig()],
        observations,
        timestamps,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        emit_actions=True,
    )
    lines = request.splitlines()
    assert lines[0] == "TGK 1 1 7 1 {} {}".format((35.0).hex(), (20.0).hex())
    assert lines[1].startswith("P 0 ")
    assert lines[2].split()[1] == str(epoch_microseconds(timestamps[0]))
    assert hashlib.sha256(request.encode()).hexdigest() == (
        "fc058f476ab41b0b974558f7589f8b741b8d4cdb38d84ca896a2c9368b807c71"
    )

    sensitivity_request = encode_sensitivity_request(
        [PolicyConfig()],
        observations,
        timestamps,
        SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
    )
    sensitivity_lines = sensitivity_request.splitlines()
    assert sensitivity_lines[0] == "TGS 1 1 7 {} {} {} {}".format(
        (35.0).hex(), (20.0).hex(), (0.5).hex(), (1.0).hex()
    )
    assert sensitivity_lines[1:] == lines[1:]


def test_kernel_agrees_bit_for_bit_with_python_reference(kernel_binary: Path):
    observations = [
        *load_trace(EXAMPLE_TRACE),
        Observation(
            source=PowerSource.NO_BATTERY,
            percent=None,
            temperature_c=30.0,
            charging=None,
            observed_at="2026-07-26T09:35:00Z",
        ),
    ]
    timestamps = ordered_timestamps(observations)
    durations = [0.0] * len(observations)
    for index in range(len(observations) - 1):
        durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()
    policies = [
        PolicyConfig(),
        PolicyConfig(run_on_battery=True, battery_floor_pct=15.0),
        PolicyConfig(run_on_battery=True, battery_floor_pct=65.0),
        PolicyConfig(temp_pause_c=40.0, temp_gentle_c=36.0),
        PolicyConfig(ac_band="gentle", temp_resume_c=30.0),
    ]
    reasons = set()
    for policy in policies:
        policy_engine = PolicyEngine()
        reasons.update(policy_engine.decide(policy, item).reason for item in observations)
    assert reasons == set(DecisionReason)

    kernel_rows = run_kernel(
        kernel_binary,
        policies,
        observations,
        timestamps,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        emit_actions=True,
    )
    python_rows = _python_rows(
        policies,
        observations,
        durations,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        emit_actions=True,
    )

    assert kernel_rows == python_rows

    bounds = SensitivityBounds(temperature_c=0.5, charge_pct=1.0)
    kernel_sensitivity = run_sensitivity_kernel(
        kernel_binary,
        policies,
        observations,
        timestamps,
        bounds,
        hot_ref_c=35.0,
        low_battery_ref_pct=60.0,
        threads=3,
    )
    python_sensitivity = [
        analyze_sensitivity(
            policy,
            observations,
            durations,
            bounds,
            hot_ref_c=35.0,
            low_battery_ref_pct=60.0,
        )
        for policy in policies
    ]
    assert kernel_sensitivity == python_sensitivity


def test_sweep_auto_uses_only_a_verified_kernel(kernel_binary: Path, monkeypatch):
    monkeypatch.setenv("TRAIN_GUARD_KERNEL", str(kernel_binary))
    report = run_sweep(
        PolicyConfig(),
        parse_grid('{"run_on_battery": [true]}'),
        load_trace(EXAMPLE_TRACE),
        sensitivity=SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
    )

    assert report["engine"] == "native"
    assert report["kernel_verified_against_reference"] is True
    assert report["baseline"]["metrics"]["action_seconds"] == {
        "full": 600.0,
        "gentle": 300.0,
        "stop": 900.0,
    }
    assert report["uncertainty"]["engine"] == "native"
    assert report["uncertainty"]["kernel_verified_against_reference"] is True
    assert report["baseline"]["sensitivity"]["run_seconds"] == {
        "minimum": 600.0,
        "maximum": 1500.0,
    }


def test_sweep_refuses_a_kernel_baseline_mismatch(monkeypatch):
    bad_row = KernelRow(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "0000000000000000", None)
    monkeypatch.setattr("trainguard.sweep.find_kernel", lambda: Path("fake-kernel"))
    monkeypatch.setattr("trainguard.sweep.run_kernel", lambda *args, **kwargs: [bad_row, bad_row])

    with pytest.raises(SweepError, match="disagrees with the Python reference"):
        run_sweep(
            PolicyConfig(),
            parse_grid('{"run_on_battery": [true]}'),
            load_trace(EXAMPLE_TRACE),
            engine="native",
        )

    def matching_nominal(_kernel, policies, observations, timestamps, **kwargs):
        durations = [0.0] * len(observations)
        for index in range(len(observations) - 1):
            durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()
        return _python_rows(
            policies,
            observations,
            durations,
            hot_ref_c=kwargs["hot_ref_c"],
            low_battery_ref_pct=kwargs["low_battery_ref_pct"],
        )

    bad_sensitivity = SensitivityResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    monkeypatch.setattr("trainguard.sweep.run_kernel", matching_nominal)
    monkeypatch.setattr(
        "trainguard.sweep.run_sensitivity_kernel",
        lambda *args, **kwargs: [bad_sensitivity, bad_sensitivity],
    )
    with pytest.raises(SweepError, match="Python sensitivity reference"):
        run_sweep(
            PolicyConfig(),
            parse_grid('{"run_on_battery": [true]}'),
            load_trace(EXAMPLE_TRACE),
            engine="native",
            sensitivity=SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
        )


def test_kernel_protocol_errors_fail_closed_through_the_cli(tmp_path, monkeypatch, capsys):
    observations = load_trace(EXAMPLE_TRACE)
    timestamps = ordered_timestamps(observations)
    truncated = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="TGK 1 OK\nEND\n", stderr=""
    )
    monkeypatch.setattr("trainguard.native.subprocess.run", lambda *args, **kwargs: truncated)
    with pytest.raises(KernelError, match="truncated"):
        run_kernel(
            tmp_path / "kernel",
            [PolicyConfig()],
            observations,
            timestamps,
            hot_ref_c=35.0,
            low_battery_ref_pct=20.0,
        )

    bad_sensitivity_responses = (
        ("TGK 1 OK\nEND\n", "unexpected sensitivity handshake"),
        ("TGS 1 OK\nEND\n", "truncated sensitivity response"),
        (
            "TGS 1 OK\nS 0x1p+1 0x1p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 1 0\nEND\n",
            "invalid sensitivity bounds",
        ),
        (
            "TGS 1 OK\nS 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 6 0\nEND\n",
            "invalid sensitivity bounds",
        ),
    )
    for stdout, message in bad_sensitivity_responses:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "trainguard.native.subprocess.run",
            lambda *args, _completed=completed, **kwargs: _completed,
        )
        with pytest.raises(KernelError, match=message):
            run_sensitivity_kernel(
                tmp_path / "kernel",
                [PolicyConfig()],
                observations,
                timestamps,
                SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
                hot_ref_c=35.0,
                low_battery_ref_pct=20.0,
            )

    grid = tmp_path / "grid.json"
    grid.write_text('{"run_on_battery": [true]}', encoding="utf-8")
    monkeypatch.setenv("TRAIN_GUARD_KERNEL", str(tmp_path / "missing-kernel"))
    assert cli.main(["sweep", str(EXAMPLE_TRACE), "--grid", str(grid), "--engine", "native"]) == 2
    assert "TRAIN_GUARD_KERNEL does not exist" in capsys.readouterr().err


def test_resolve_thread_count_contract():
    # The count can never exceed the policy count, emit_actions forces
    # the sequential path, and a nonsensical request fails loudly
    # instead of silently running single-threaded.
    assert resolve_thread_count(1) == 1
    assert resolve_thread_count(500) >= 1
    assert resolve_thread_count(3, requested=8) == 3
    assert resolve_thread_count(500, requested=8) == 8
    assert resolve_thread_count(500, emit_actions=True, requested=8) == 1
    with pytest.raises(KernelError, match="at least 1"):
        resolve_thread_count(4, requested=0)
    with pytest.raises(KernelError, match="at most 4096"):
        resolve_thread_count(5000, requested=4097)


def test_kernel_output_is_byte_identical_across_thread_counts(kernel_binary: Path):
    # Threads split whole policies and rows are buffered back into input
    # order, so any thread count must reproduce the single-threaded
    # bytes exactly.
    observations = load_trace(EXAMPLE_TRACE)
    timestamps = ordered_timestamps(observations)
    policies = [
        PolicyConfig(),
        PolicyConfig(run_on_battery=True, battery_floor_pct=15.0),
        PolicyConfig(temp_pause_c=40.0, temp_gentle_c=36.0),
        PolicyConfig(ac_band="gentle", temp_resume_c=30.0),
        PolicyConfig(temp_pause_c=44.0),
    ]
    request = encode_request(
        policies,
        observations,
        timestamps,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        emit_actions=False,
    )
    sensitivity_request = encode_sensitivity_request(
        policies,
        observations,
        timestamps,
        SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
    )
    for protocol_request in (request, sensitivity_request):
        outputs = []
        for threads in ("1", "3"):
            result = subprocess.run(
                [str(kernel_binary), threads],
                input=protocol_request,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1]

    rows = run_kernel(
        kernel_binary,
        policies,
        observations,
        timestamps,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        threads=3,
    )
    durations = [0.0] * len(observations)
    for index in range(len(observations) - 1):
        durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()
    reference = _python_rows(
        policies, observations, durations, hot_ref_c=35.0, low_battery_ref_pct=20.0
    )
    for kernel_row, python_row in zip(rows, reference):
        for name in (
            "full_seconds",
            "gentle_seconds",
            "stop_seconds",
            "run_seconds",
            "hot_degc_seconds",
            "low_battery_run_seconds",
            "action_transitions",
            "decision_transitions",
            "checksum",
        ):
            assert getattr(kernel_row, name) == getattr(python_row, name)


def test_kernel_rejects_a_bad_thread_argument(kernel_binary: Path):
    for argument in ("0", "banana"):
        result = subprocess.run(
            [str(kernel_binary), argument],
            input="",
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "threads" in result.stderr


def test_kernel_rejects_malformed_requests(kernel_binary: Path):
    unordered = (
        "TGK 1 1 2 0 0x0p+0 0x0p+0\n"
        "P 0 0x1p+4 1 0 0x1p+5 0x1p+6 0x1p+4 0x1p+6 0x1p+5\n"
        "O 2000000 0 - - 0\n"
        "O 1000000 0 - - 0\n"
    )
    result = subprocess.run(
        [str(kernel_binary)],
        input=unordered,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "ordered" in result.stderr

    wrong_version = subprocess.run(
        [str(kernel_binary)],
        input="TGK 99 1 1 0 0x0p+0 0x0p+0\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_version.returncode == 3

    malformed = (
        ("BAD 1 1 1 0 0x0p+0 0x0p+0\n", "TGK or TGS"),
        ("TGS 1 0 1 0x0p+0 0x0p+0 0x0p+0 0x0p+0\n", "at least 1"),
        ("TGS 1 1 1 0x0p+0 0x0p+0 -0x1p+0 0x0p+0\n", "half-widths"),
        ("TGS 1 1 1 0x0p+0 0x0p+0 0x1.2dp+8 0x0p+0\n", "supported domain"),
    )
    for request, message in malformed:
        result = subprocess.run(
            [str(kernel_binary)],
            input=request,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert message in result.stderr
