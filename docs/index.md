# Power-aware supervision for one long-running job

`train-guard` watches the laptop's power source, charge level and available
battery temperature. It applies one of three actions to a named process tree:
`full`, `gentle` or `stop`.

Use it when a local training, indexing or build job should react to unplugging
or heat without changing unrelated processes. The same policy can be replayed
against a JSON Lines trace before it controls a live job.

## Start here

Install the command with `pipx`, inspect what the machine exposes, then try a
disposable job:

```bash
pipx install train-guard
train-guard doctor
```

The [getting started guide](getting-started.md) walks through one complete
launch, inspection and detach cycle. Continue with the
[architecture and lifecycle](architecture.md) guide before supervising a long
job that matters.

## Two operating paths

Live supervision acts on one named process tree. It records PID and process
creation time, tracks the changes it owns and retains recovery state when it
cannot safely clean up.

Offline analysis has no process or state-directory side effects. `simulate`
replays one policy or compares two policies. `sweep` evaluates a policy grid
against the same recorded observations.

## Scope

`train-guard` is a workload policy. It is not a hardware safety controller,
charge limiter or thermal model. Firmware and the operating system remain
responsible for electrical and thermal protection.

Replay and sweep re-weight a fixed recording. They do not predict temperature,
energy use, battery life or throughput under another policy. Missing battery
temperature is common on some machines, especially Windows, and disables the
temperature rules for that observation.

Read [platform behavior and limitations](platforms.md) before relying on a
specific sensor or scheduling action.
