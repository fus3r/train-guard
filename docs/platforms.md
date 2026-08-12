# Platforms and limitations

Train Guard uses the same policy on macOS, Linux and Windows, but the process
and sensor adapters differ.

| Platform | `stop` | `gentle` | Battery temperature | Login restart |
|---|---|---|---|---|
| macOS | Suspend and resume through `psutil` | `taskpolicy` background hint | Apple Smart Battery data when exposed | Per-user LaunchAgent |
| Linux | Suspend and resume through `psutil` | Reduced CPU affinity and idle I/O priority | `psutil` or battery sysfs when exposed | systemd user unit |
| Windows | Suspend and resume through `psutil` | Idle process priority | Usually unavailable | Per-user scheduled task |

`gentle` is a scheduling hint. It does not cap power, choose a processor core
class or guarantee a speed. Linux and Windows restore captured scheduling
state when possible.

The macOS adapter cannot read a process's previous `taskpolicy` background
flag. It records that it applied the hint and later clears it, but it cannot
reconstruct a background policy set earlier by another tool.

## Sensor limits

Battery temperature depends on the machine, firmware and driver. If it is
missing, thermal rules cannot run for that observation. `train-guard doctor`
shows the available values and warnings.

Power source and battery percentage come from one snapshot. The portable API
reports whether external power is connected, not charging current. Train Guard
therefore infers `charging` as plugged in below 100 percent.

## Process limits

- Match attachment uses a command-line substring. Prefer `--pid` when an exact
  live process is already available.
- Permissions can prevent inspection or control of another user's process.
- A process can create a child after the final tree snapshot used by
  `stop --kill`.
- An uncatchable supervisor exit can occur between changing a process and
  writing the next recovery record.
- Job-name locks assume a local filesystem with compatible advisory locking.

Read the [failure recovery guide](failure-model.md) for the supported response
to each case.

## How it differs from adjacent tools

These projects solve related problems at different layers:

| Tool | Primary job | Difference from Train Guard |
|---|---|---|
| [TLP](https://linrunner.de/tlp/) | Linux laptop power profiles and battery-care settings | Applies system policy rather than a replayable policy to one named process tree |
| [auto-cpufreq](https://github.com/AdnanHodzic/auto-cpufreq) | Automatic Linux CPU and power optimization | Adjusts CPU and system power behavior rather than tracking reversible changes for one named job |
| [batt](https://github.com/charlie0129/batt) | Charge control on supported Apple Silicon MacBooks | Controls charging hardware; Train Guard does not set charge limits |
| [Pueue](https://github.com/Nukesor/pueue) | Persistent queues for shell commands | Schedules and manages task queues; Train Guard reacts to power, charge and available temperature signals |
| [cpulimit](https://github.com/opsengine/cpulimit) | CPU-usage limits for a process | Enforces a percentage limit; Train Guard selects `full`, `gentle` or `stop` from a stateful policy |

The tools are not interchangeable, and this table does not claim that every
combination is conflict-free. Test interactions on a disposable job before
running more than one process-control or system-power tool together.

## Scientific limits

Offline replay treats the recorded observations as fixed. Different actions
could have changed later temperature and charge values, but the replay does not
model that feedback. Sweep metrics and bounded sensitivity therefore describe
permission accounting on the supplied trace. They are not forecasts of
battery life, temperature, energy use or throughput.
