# Getting started

This guide uses a disposable Python process. Do not begin with an important
training run.

## Install

Train Guard requires Python 3.9 or later. Install the command in its own
environment with [pipx](https://pipx.pypa.io/stable/how-to/install-pipx.html):

```bash
pipx install train-guard
train-guard --version
```

Run the installation check before starting a job:

```bash
train-guard doctor
```

`doctor` checks configuration, state access, sensors and stale jobs. A note
that battery temperature is unavailable is not an installation failure. It
means temperature rules cannot run on that observation; power source and
charge rules can still apply when those values are present.

## Launch a disposable job

Create the default policy, then start a short Python process under a stable
name:

```bash
train-guard config --init
train-guard run --name quickstart -- \
  python3 -c "import time; print('quickstart running', flush=True); time.sleep(120)"
```

The worker and supervisor detach from the terminal. Worker output is written
to `~/.train-guard/logs/quickstart.log`.

Inspect the current decision and recent transitions:

```bash
train-guard status
train-guard events quickstart --limit 10
```

Detach while leaving the worker alive:

```bash
train-guard stop quickstart
```

Use `train-guard stop quickstart --kill` only when the worker process tree
should also end.

## Expected transcript

This transcript is from `train-guard 0.4.0` installed with pipx outside a
checkout, on the macOS release test machine. User paths and PIDs are redacted;
sensor values are whatever the machine reports at that moment.

```text
$ train-guard --version
train-guard 0.4.0

$ train-guard doctor
train-guard 0.4.0  Python 3.13.7  macOS-15.7.7-arm64-arm-64bit-Mach-O
  ok   config: {'poll': 20.0, 'run_on_battery': False, 'battery_floor_pct': 30.0, 'battery_band': 'gentle', 'ac_band': 'full', 'temp_gentle_c': 38.0, 'temp_pause_c': 42.0, 'temp_resume_c': 36.0, 'charge_cool_until_pct': 80.0, 'temp_charge_gentle_c': 35.0}
  ok   state_directory: ~/.train-guard
  ok   state_files: {'errors': []}
  ok   sensors: available
  ok   supervisors: {'stale': []}
  ok   login_agent: {'installed': True}

$ train-guard run --name quickstart -- \
    python3 -c "import time; print('quickstart running', flush=True); time.sleep(120)"
[train-guard] 'quickstart': job pid=NNNNN  guard pid=NNNNN  out=~/.train-guard/logs/quickstart.log
[train-guard] policy: AC=full  battery=pause   |   train-guard status

$ train-guard status
power / battery
  source: AC   charge: 100%   charging: no   pack temp: 30.4°C
  Cycle Count: 278
  Condition: Normal
  Maximum Capacity: 96 %

policy (config.json)
  AC=full  battery=pause
  thermal: gentle >= 38°C  pause >= 42°C  resume <= 36°C   charge cool: <80% and >= 35°C

active guards
  quickstart [run]  guard=running (pid NNNNN)  state=full  pids=[NNNNN]
    last decision: full (ac_policy)

$ train-guard stop quickstart
[train-guard] stop requested for 'quickstart'; the supervisor acts after the next policy poll. (login restart removed)

$ train-guard events quickstart --limit 10
2026-08-12T18:15:01.913Z  started                   START mode=run jobpid=NNNNN
2026-08-12T18:15:01.934Z  decision                  -> full (ac_policy; power=ac batt=100% temp=30.41C charging=no)
2026-08-12T18:15:21.959Z  stopped                   STOP: released owned process changes, detaching
```

The `stopped` event confirms the supervisor released the changes it owned and
detached; the worker process itself keeps running because `--kill` was not
used. `login_agent: {'installed': True}` reflects this test machine; a first
installation reports it as not installed until `install-agent` is run.

## Replay before live use

Clone the repository examples or prepare a JSON Lines trace using the
[documented schema](policy-replay.md). Replay does not read sensors or control
processes:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json
```

The included example assigns 600 seconds to `full`, 300 seconds to `gentle`
and 900 seconds to `stop`. These values are policy outputs on a synthetic
trace, not measured changes in battery life or temperature.

## Use your own job

Once the disposable cycle is understood, launch a checkpointed command:

```bash
train-guard run --name experiment -- \
  python3 train.py --epochs 200
```

Train Guard cannot restore Python, CUDA or model state after reboot. The
application still needs durable checkpoints. Read the
[failure recovery guide](failure-model.md) before enabling login restart.
