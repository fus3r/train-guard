# train-guard

[![CI](https://github.com/fus3r/train-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/fus3r/train-guard/actions/workflows/ci.yml)

`train-guard` supervises one long-running job on a laptop. It can pause the
process tree when AC power is removed, lower its scheduling priority when the
battery is warm, pause at a configured temperature, and continue the same live
processes after power returns.

The operating system remains responsible for CPU and SoC protection. This tool
only applies a job-level battery and power policy.

```bash
train-guard run --name train -- python train.py --epochs 200
train-guard attach --match "python train.py"
train-guard status
train-guard stop train
```

This repository is a public release snapshot. The package is MIT licensed.

## What pause and gentle mode mean

Pause and resume use host process APIs through `psutil`. On POSIX systems this
is equivalent to `SIGSTOP` and `SIGCONT`. The process stays in memory, and
multiprocessing children are included. `train-guard` records the PID and
creation time only after it successfully suspends a process. A process that was
already stopped remains externally owned and is never resumed by the guard. If
another actor resumes a suspension that the guard does own while the policy
still says `stop`, the guard reapplies that suspension on the next poll.

Gentle mode is a best-effort scheduling request:

- macOS applies `taskpolicy -b` and later clears that hint with `taskpolicy
  -B`. The kernel still chooses the cores, so this does not pin a job to
  efficiency cores. `taskpolicy` cannot read and reconstruct an external
  background hint that existed before `train-guard`.
- Linux captures the current CPU affinity and I/O class before changing them,
  then restores those captured values.
- Windows captures the current priority class before switching to
  `IDLE_PRIORITY_CLASS`, then restores that exact class.

Owned suspensions and scheduling captures follow process identity across
controller replacement. They are also released when a process leaves the
target set, including when a match-based target becomes empty.

Charging tools such as AlDente or `batt` manage charge limits. System tools such
as TLP and auto-cpufreq manage broader machine policy. `train-guard` only
controls the named job.

## Install

The package is not on PyPI. Install it from a checkout:

```bash
git clone https://github.com/fus3r/train-guard.git
cd train-guard
python -m pip install -e .
```

You can also run `python trainguard/cli.py status` without installing the
command. A standalone macOS shell implementation is under [`macos/`](macos/).

## Platform behavior

| Platform | Pause and resume | Gentle mode | Battery temperature | Login restart |
|---|---|---|---|---|
| macOS | process suspend and resume | background scheduling hint | `ioreg`, no sudo | launchd starts a new process |
| Linux | process suspend and resume | reduced CPU affinity, idle I/O | psutil sensors or sysfs | systemd user service starts a new process |
| Windows | process suspend and resume | idle priority class | usually unavailable | scheduled task starts a new process |

CI runs the policy, CLI, persistence, sensor and child-process tests on macOS,
Linux and Windows with Python 3.9 and 3.13. Battery temperature still depends
on the hardware and driver. When no pack sensor is available, thermal rules are
skipped while power-source and charge rules remain active.

## Usage

```bash
# Start a command. Its output is written to ~/.train-guard/logs/<name>.log.
train-guard run --name bigtrain -- python train.py --epochs 200

# Adopt an existing process.
train-guard attach --match "python train.py" --name bigtrain
train-guard attach --pid 12345

train-guard list
train-guard status
train-guard events bigtrain --limit 10
train-guard list --json
train-guard status --json
train-guard doctor --json
train-guard stop bigtrain
train-guard stop bigtrain --kill
# After status reports a dead supervisor, release its recorded process changes.
train-guard recover bigtrain
train-guard config --init
train-guard config --check

# Replay and compare policies without starting a worker or reading sensors.
train-guard simulate examples/power-trace.jsonl --json
train-guard simulate examples/power-trace.jsonl \
  --compare-config examples/battery-enabled-policy.json

# Evaluate a Cartesian grid of policy values on the same trace.
train-guard sweep examples/power-trace.jsonl \
  --grid examples/policy-grid.json --engine python
```

### Sleep is not reboot

Sleep preserves RAM. A suspended job can continue after the machine wakes.

A reboot destroys the process and its in-memory state. With
`--restart-on-login`, `train-guard` saves the command line and starts a new
process at the next login:

```bash
train-guard run --restart-on-login --name bigtrain -- \
  python train.py --resume-from-checkpoint latest
train-guard install-agent
```

The application must load its own checkpoint if it needs to continue prior
work. An attached job can wait for a matching process or define a start command:

```bash
train-guard attach --restart-on-login --name dataprep \
  --match "build_index.py" --start "bash run_prep.sh"
```

`--persist` remains as a compatibility alias. It restarts or reattaches after
login; it cannot restore RAM.

## Policy

The supervisor rereads `~/.train-guard/config.json` while it runs. An invalid
edit is reported in the guard log and runtime state, while the supervisor keeps
using its last valid policy.

| Key | Default | Meaning |
|---|---|---|
| `run_on_battery` | `false` | pause when unplugged |
| `battery_floor_pct` | `30` | pause at or below this charge when battery running is enabled |
| `ac_band` | `full` | scheduling band on AC while cool |
| `battery_band` | `gentle` | scheduling band on battery |
| `temp_charge_gentle_c` | `35` | gentle mode while warm and charging below the cutoff |
| `temp_gentle_c` | `38` | gentle mode on AC at or above this pack temperature |
| `temp_pause_c` | `42` | pause at or above this pack temperature |
| `temp_resume_c` | `36` | leave a thermal pause at or below this pack temperature |
| `charge_cool_until_pct` | `80` | charge cutoff for the warm-charging rule |

The default pack thresholds are 35, 38 and 42 degrees Celsius, with resume at
36. They are conservative choices for this tool, not manufacturer safety
limits. Apple's published 10 to 35 degree range is an
[ambient operating range](https://support.apple.com/en-us/102336), not a battery
pack range.

Version 0.1 keys `temp_ecore_c` and `temp_charge_ecore_c` still load for
compatibility. New configurations use `gentle` because no supported API
guarantees a particular core class.

## Diagnostics and offline evaluation

`train-guard config --check` validates the complete policy without starting a
job. `train-guard doctor [--json]` checks the configuration, state-directory
access, sensors, supervisor liveness and every metadata, runtime and login
restart record. Corrupt and orphaned recovery state is reported but never
repaired or deleted automatically. `status --json` follows the same rule and
returns a non-zero status while still emitting a versioned, machine-readable
report. These JSON reports use `schema_version: 1`.

`simulate` accepts raw observation JSON Lines or structured event records that
contain an `observation`. It never creates the state directory, samples a
sensor or controls a process. Each observation is treated as holding until the
next timestamp; the last observation therefore has zero duration unless a
handled terminal event from the live journal closes the interval. The first
row in `transitions` is the initial decision, so a report contains
`decision_transitions + 1` transition rows.

The included trace covers normal AC work, warm charging, thermal hysteresis and
one battery interval. With the default policy its hand-checked 1,800 seconds
split into 600 seconds `full`, 300 seconds `gentle` and 900 seconds `stop`.
Comparison mode evaluates the baseline and candidate on the same immutable
observations and reports action-second and percentage-point deltas,
disagreement duration and transition deltas. Canonical SHA-256 fingerprints
identify the policy and observation values used by a report.

`sweep` validates and deduplicates the Cartesian product in a JSON grid such as
`{"temp_pause_c": [40, 42, 44], "run_on_battery": [true, false]}`. It reports
permitted work, hot degree-seconds while work is permitted, low-battery work,
trace-level facts and the Pareto front. Invalid combinations are counted and
shown rather than silently coerced.

Each candidate also carries an exact fractional clairvoyant bound at its own
hot and low-battery exposure budgets. This is one joint two-budget schedule;
the separate hot-only and low-battery-only optima remain diagnostics and are
not combined with `min`. The reported temperature cutoff is explicitly the
hot-only hindsight cutoff, not a threshold for the joint schedule. Tests compare
the joint optimizer with an independent exact-rational enumeration of compact
linear-program vertices.

Replay and sweep re-weight a trace recorded under the original actions. They do
not model how another policy would have changed temperature or charge, and are
not battery-life or throughput predictions. `auto` currently uses the Python
policy engine. The `native` engine selector is reserved for the optional replay
kernel and fails explicitly while that kernel is absent.

## Supervisor loop

Every `poll` seconds, the supervisor takes one battery snapshot, reads the pack
temperature when the platform exposes it, then evaluates one ordered list of
rules and keeps the first match:

1. an active thermal cooldown;
2. the pause threshold `temp_pause_c`;
3. the battery rules, in the order `run_on_battery`, `battery_floor_pct`,
   `battery_band`;
4. the warm-charging rule, below `charge_cool_until_pct` and at or above
   `temp_charge_gentle_c`;
5. the warm-AC rule at or above `temp_gentle_c`;
6. `ac_band`, reported as `no_battery` on a host that exposes no battery.

The match selects `full`, `gentle` or `stop` for the process tree and names the
rule that produced it. The short guard log and the structured JSON Lines event
journal record a decision only when its action, reason or observation changes.
When a managed supervisor exit follows a stable interval, the terminal event
repeats the last observation with the exit time so that interval has an explicit
end without logging every sensor poll.

Thermal pause has hysteresis: with the defaults, a job paused at 42 degrees
stays paused until the pack reaches 36.

### What one observation contains

Power source, charge and charging state come from a single battery snapshot, so
a decision never mixes a plugged-in reading with a charge read after the cable
was pulled. Each observation is stamped in UTC to the millisecond, because
`poll` may be set below one second.

`charging` is a portable inference, plugged in and below 100 percent. It is not
a charge-current measurement, because the cross-platform battery API does not
expose one.

A measurement this host does not provide stays absent instead of being replaced
by a plausible default, and the guard log records why it is missing. An absent
measurement skips only the rules that need it: a host with no battery at all is
treated as mains-powered and climbs the same thermal ladder as AC. The one
exception is an active thermal cooldown, which a reading that disappears cannot
end.

On Linux, `power_supply/*/temp` is read as tenths of a degree, the unit that
interface defines, so a cool pack reporting `45` is 4.5 °C. A temperature that
arrives through psutil carries no guaranteed scale, so only there is a value
too large for a battery pack rescaled.

Closing the lid needs no special path because the supervisor and worker sleep
with the laptop. Login persistence creates a new process after reboot.

## Local state

State lives under `~/.train-guard` by default. Set `TRAIN_GUARD_HOME` to move
it; relative values are resolved before a detached supervisor starts, so the
parent and child keep using the same directory.

Job names are limited to 1–64 letters, digits, dots, underscores or hyphens.
Path components and Windows device names such as `con`, `nul` and `com1` are
rejected before a worker starts.

Metadata, supervisor identity and current runtime state use separate,
schema-versioned JSON files under `run/`. Writes are flushed to a temporary file
and atomically replace the prior value; non-finite JSON numbers are rejected.
An advisory `<name>.lock` also prevents two commands from creating the same job
at once. The lock file remains in place for reuse.

Each job also has `logs/<name>.guard.log` for short human-readable messages and
`logs/<name>.events.jsonl` for independent structured events. The supervisor is
the single writer while it is alive. `train-guard events <name>` ignores a
corrupt line and applies `--limit` to valid events rather than raw lines.

Process identity is the pair of PID and creation time, not the PID alone. Every
tree lookup checks both values before controlling a recorded process. Legacy
PID-only metadata is upgraded only when the current process predates that
metadata; otherwise the state is preserved and the migration is refused.
Match-based attachment also excludes the exact CLI process that launched its
supervisor, even when the pattern appears in that CLI's own command line.
Runtime state records only suspensions and reversible scheduling changes that
the controller actually applied. A failed restoration keeps its identity and
captured values available for a later retry rather than discarding recovery
evidence.

Malformed state is reported by `status` rather than guessed or deleted. A
runtime record left without its metadata is likewise preserved and blocks a
new job from silently adopting the same name. `run` and `attach` report success
only after the supervisor writes a readiness record containing its own
PID-and-creation-time identity. A failed start terminates the children created
by that attempt and rolls back only that attempt's state.

`SIGINT`, `SIGTERM`, a soft `stop`, root-process exit and handled supervisor
errors all release the suspensions and scheduling changes owned by the guard.
If a dead supervisor could not release every change, `train-guard recover
<name>` retries from the validated runtime identities. A permission-denied
identity remains recorded as `*_incomplete`, so recovery can be retried without
targeting a reused PID.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest
bash -n macos/train-guard.sh
```

The GitHub Actions matrix runs on `ubuntu-latest`, `macos-latest` and
`windows-latest`.

## License

MIT, 2026.
