<p align="center">
  <a href="https://train-guard.readthedocs.io/en/latest/">
    <img
      src="https://raw.githubusercontent.com/fus3r/train-guard/c8f03a2da6a896d6b3bdd6f5a3a146a5796cf1cf/docs/assets/train-guard-banner.webp"
      alt="train-guard — Power-aware supervision for one long-running job."
      width="1086"
    >
  </a>
</p>

<p align="center">
  <a href="https://train-guard.readthedocs.io/en/latest/">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-read%20the%20guide-2f81f7?style=flat-square&amp;logo=readthedocs&amp;logoColor=white">
  </a>
  <a href="https://pypi.org/project/train-guard/">
    <img alt="PyPI version" src="https://img.shields.io/pypi/v/train-guard?style=flat-square&amp;color=2f81f7">
  </a>
  <a href="https://pypi.org/project/train-guard/">
    <img alt="Python 3.9 or later" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  </a>
  <a href="https://github.com/fus3r/train-guard/actions/workflows/ci.yml">
    <img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/fus3r/train-guard/ci.yml?branch=main&amp;style=flat-square&amp;label=CI">
  </a>
  <a href="https://github.com/fus3r/train-guard/blob/main/LICENSE">
    <img alt="MIT license" src="https://img.shields.io/github/license/fus3r/train-guard?style=flat-square">
  </a>
</p>

<p align="center">
  <a href="https://train-guard.readthedocs.io/en/latest/getting-started/"><strong>Getting started</strong></a>
  ·
  <a href="https://train-guard.readthedocs.io/en/latest/cli/">CLI reference</a>
  ·
  <a href="https://train-guard.readthedocs.io/en/latest/platforms/">Platforms</a>
  ·
  <a href="https://github.com/fus3r/train-guard/blob/main/CHANGELOG.md">Changelog</a>
</p>

`train-guard` watches a laptop's power source, charge level and available
battery temperature, then applies `full`, `gentle` or `stop` to one named
process tree. Replay the same policy against a recorded trace before allowing
it to control a live job.

**Scope:** Train Guard is a workload policy, not a hardware safety controller.
It does not set a charge limit or predict battery life, temperature, energy use
or throughput.

## Why train-guard

- **One-job scope.** Supervise a named process tree without changing the
  machine's system-wide power policy.
- **Three explicit actions.** Run normally, apply a platform scheduling hint,
  or suspend the job until the policy allows it to resume.
- **Replay before control.** Evaluate a policy against JSON Lines observations
  without reading live sensors or changing process state.
- **Recovery-aware ownership.** Track PID and creation time, record only the
  process changes Train Guard owns, and retain incomplete cleanup state for an
  explicit recovery attempt.
- **Cross-platform adapters.** Use the same policy on macOS, Linux and Windows,
  with platform behavior and sensor limits documented separately.

## Install

Python 3.9 or later is required. Install the command in its own environment,
then inspect the sensors and integrations available on the machine:

```bash
pipx install train-guard
train-guard doctor
```

Missing battery temperature is common on some systems, especially Windows. It
disables temperature rules for that observation, but is not by itself an
installation failure.

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

`train-guard stop quickstart` releases changes owned by Train Guard and
detaches; the worker keeps running. Add `--kill` only when the worker process
tree should also end.

| Policy action | Effect on the named process tree |
|---|---|
| `full` | Release Train Guard-owned suspension and scheduling changes |
| `gentle` | Apply the documented platform-specific scheduling hint |
| `stop` | Suspend the tree until a later policy decision resumes it |

The policy action `stop` is different from the CLI command
`train-guard stop NAME`, which requests cleanup and detachment.

## Replay before live use

Clone the repository examples, then replay the default policy without touching
live sensors, processes or the Train Guard state directory:

```bash
git clone --depth 1 https://github.com/fus3r/train-guard.git
cd train-guard
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json
```

The included synthetic trace assigns 600 seconds to `full`, 300 seconds to
`gentle` and 900 seconds to `stop`. Those values are policy outputs on a fixed
trace, not measured changes in battery life or temperature.

## Documentation

| Goal | Guide |
|---|---|
| Complete a disposable launch, inspection and detach cycle | [Getting started](https://train-guard.readthedocs.io/en/latest/getting-started/) |
| Understand process identity, ownership and lifecycle | [Architecture](https://train-guard.readthedocs.io/en/latest/architecture/) |
| Configure thresholds and live reload | [Configuration](https://train-guard.readthedocs.io/en/latest/configuration/) |
| Inspect every command and option | [CLI reference](https://train-guard.readthedocs.io/en/latest/cli/) |
| Replay a policy or inspect bounded sensitivity | [Replay](https://train-guard.readthedocs.io/en/latest/policy-replay/) · [Sensitivity](https://train-guard.readthedocs.io/en/latest/replay-sensitivity/) |
| Evaluate a policy grid and its limitations | [Policy sweep](https://train-guard.readthedocs.io/en/latest/policy-sweep/) |
| Recover after interrupted cleanup | [Failure recovery](https://train-guard.readthedocs.io/en/latest/failure-model/) |

## Limits to read before relying on it

- Temperature availability depends on hardware and drivers.
- `gentle` is a scheduling hint, not a power cap.
- Match attachment can include unrelated commands containing the same text;
  prefer a PID when available.
- An application still needs durable checkpoints for reboot recovery.
- Replay and sweep hold the recorded trace fixed. Their outputs are exposure
  accounting, not causal or physical predictions.

See the [platform guide](https://train-guard.readthedocs.io/en/latest/platforms/)
for the exact macOS, Linux and Windows actions and limitations.

## Contributing and security

Development setup and validation commands are in
[CONTRIBUTING.md](https://github.com/fus3r/train-guard/blob/main/CONTRIBUTING.md).
Report vulnerabilities through the
[private security channel](https://github.com/fus3r/train-guard/security/policy),
not a public issue.

Train Guard is available under the
[MIT license](https://github.com/fus3r/train-guard/blob/main/LICENSE).
