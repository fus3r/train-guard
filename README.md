# train-guard

![train-guard — Power-aware supervision for one long-running job.](https://raw.githubusercontent.com/fus3r/train-guard/c8f03a2da6a896d6b3bdd6f5a3a146a5796cf1cf/docs/assets/train-guard-banner.webp)

Power-aware supervision for one long-running job.

[![CI](https://github.com/fus3r/train-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/fus3r/train-guard/actions/workflows/ci.yml)

`train-guard` watches a laptop's power source, charge level and available
battery temperature, then applies `full`, `gentle` or `stop` to one named
process tree. Replay the same policy against a recorded trace before allowing
it to control a live job.

Train Guard is a workload policy, not a hardware safety controller. It does
not set a charge limit or predict battery life, temperature, energy use or
throughput.

## Install

Python 3.9 or later is required.

```bash
pipx install train-guard
train-guard doctor
```

`doctor` reports which sensors the machine exposes. Missing battery
temperature is common on some systems, especially Windows, and disables the
temperature rules for that observation.

## First job

Start with a disposable command:

```bash
train-guard config --init
train-guard run --name quickstart -- \
  python3 -c "import time; print('quickstart running', flush=True); time.sleep(120)"
train-guard status
train-guard events quickstart --limit 10
train-guard stop quickstart
```

`stop` releases changes owned by Train Guard and detaches. Add `--kill` only
when the worker process tree should also end.

## Continue

- [Documentation](https://train-guard.readthedocs.io/en/latest/)
- [Getting started](https://train-guard.readthedocs.io/en/latest/getting-started/)
- [Architecture and process lifecycle](https://train-guard.readthedocs.io/en/latest/architecture/)
- [Failure recovery](https://train-guard.readthedocs.io/en/latest/failure-model/)
- [Offline replay and bounded sensitivity](https://train-guard.readthedocs.io/en/latest/policy-replay/)
- [Policy sweep and limitations](https://train-guard.readthedocs.io/en/latest/policy-sweep/)

Development setup and validation commands are in
[CONTRIBUTING.md](https://github.com/fus3r/train-guard/blob/main/CONTRIBUTING.md).
Report vulnerabilities through the
[private security channel](https://github.com/fus3r/train-guard/security/policy),
not a public issue.

## Central limits

- Temperature availability depends on hardware and drivers.
- `gentle` is a scheduling hint, not a power cap.
- Match attachment can include unrelated commands containing the same text;
  prefer a PID when available.
- An application still needs durable checkpoints for reboot recovery.
- Replay and sweep hold the recorded trace fixed. Their outputs are exposure
  accounting, not causal or physical predictions.

Train Guard runs on macOS, Linux and Windows. Platform-specific actions and
limitations are documented in the
[platform guide](https://train-guard.readthedocs.io/en/latest/platforms/).

MIT licensed.
