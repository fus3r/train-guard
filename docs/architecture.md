# Architecture

`train-guard` separates policy from host side effects. The policy engine knows
nothing about `psutil`, files or subprocesses. The surrounding adapters turn
one sensor sample into a process action, then write the input and result to
local state.

## Data flow

```text
CLI
 |
 | writes job specification and starts a detached supervisor
 v
Supervisor -> SensorReader -> Observation -> PolicyEngine -> PolicyDecision
     |                                                     |
     |                                                     v
     +-> ConfigWatcher                              ProcessController
     |                                                     |
     +-> runtime snapshot                                  v
     +-> EventJournal                              worker process tree

Trace JSONL -> SimulationRunner -> Observation -> PolicyEngine -> report
                                      |                 |
Candidate config ---------------------+-----------------+-> comparison delta
User bounds --------------------------+-> exact marginal envelopes and action margin

Candidate grid -> SweepRunner -> PolicyEngine -> nominal Pareto report
User bounds ----------+-> exact marginal envelopes -> interval-front enclosure
```

There is one supervisor per named job. Each cycle follows the same sequence:

1. Check for a `stop` request.
2. Resolve the current root and descendant processes.
3. Reload the policy if its content changed.
4. Read one battery snapshot and the available pack temperature.
5. Pass the immutable observation to the policy state machine.
6. Apply `full`, `gentle` or `stop` to the resolved processes.
7. Atomically replace the runtime snapshot.
8. Append an event when the decision, reason or observation changes.

The event journal is transition-oriented. It does not append the same decision
every polling interval. A handled stop, worker exit, shutdown or failure appends
a terminal copy of the last observation with a fresh timestamp. This closes the
last piecewise-constant interval when a completed journal is replayed.

## Modules

| Module | Responsibility |
|---|---|
| `model.py` | immutable observations, decisions and process identities |
| `config.py` | typed policy parsing, validation and live reload |
| `policy.py` | pure decision state machine and thermal hysteresis |
| `sensors.py` | macOS, Linux and Windows sensor adapters |
| `processes.py` | process discovery, owned suspension and scheduling bands |
| `state.py` | validated names, schemas and atomic JSON state |
| `journal.py` | JSON Lines and readable text events |
| `simulation.py` | strict trace parsing and deterministic policy replay |
| `robustness.py` | exact bounded objective sensitivity and minimum action-change distance |
| `clairvoyant.py` | exact fractional exposure bounds for replayed policies |
| `sweep.py` | candidate-grid evaluation, trace facts, nominal Pareto reporting and interval-front enclosure |
| `native.py` | discovery and checked protocol for the optional C++ replay kernel |
| `platforms.py` | LaunchAgent, systemd and scheduled-task integration |
| `supervisor.py` | lifecycle, signals, polling and recovery records |
| `cli.py` | user commands, detached launch and rendering |

## Process identity

A PID alone is not an identity because operating systems reuse PIDs. A
`ProcessIdentity` stores both the PID and the process creation time reported by
`psutil`.

For `run` and `attach --pid`, the root identity is captured before the detached
supervisor starts. Every later lookup verifies both fields. If the root exits
and its PID is reused, the replacement process is not adopted.

Match-based attachment is different by design. It searches current command
lines on each cycle and can wait while no process matches. The internal
supervisor command is excluded from the match set. Before reporting ready, the
detached supervisor also captures the identity of its launching CLI and
excludes that exact PID-and-creation-time pair. Startup fails closed if this
identity cannot be established, because the CLI command line itself contains
the user-supplied match string.

## Process-change ownership

The process controller keeps separate records for suspensions and scheduling
changes. It follows five rules:

1. A process that is already stopped is not claimed.
2. Only claimed identities are resumed.
3. PID and creation time are checked again before recovery.
4. On Linux and Windows, scheduling state is captured before `gentle` changes
   it; a resource is not changed when its prior value cannot be read.
5. If a live process leaves the resolved target set, its owned suspension or
   scheduling change is released during that cycle.

Both owned sets are copied into `runtime.json` after every action. A replacement
supervisor can adopt them, and `train-guard recover NAME` can use them when the
recorded supervisor is dead.

Linux affinity and I/O priority, and Windows process priority, are restored
when the job returns to full mode, detaches or shuts down. macOS records
ownership of the matching `taskpolicy` background and clear operations, but
the adapter cannot read whether another tool had already placed the process in
that state. Its `-B` operation is therefore a clear, not a reconstruction of an
external prior value. A failed restore or clear retains its record so a later
recovery can retry it.

## Policy ordering

The policy is evaluated in this order:

1. Continue an active thermal cooldown until the resume threshold is reached.
2. Enter thermal cooldown at the pause threshold.
3. Apply unplugged and battery-floor rules.
4. Apply the warm-charging rule.
5. Apply the general warm-AC rule.
6. Use the configured AC band, reported as `no_battery` on a host without one.

This ordering makes every thermal rule independent of power source: a
mains-only host follows the same gentle and pause ladder as an AC host, and
only the final band reason differs. The cooldown bit is part of the runtime
state, so restarting a supervisor does not discard the hysteresis state.

## State layout

The root defaults to `~/.train-guard` and can be changed with
`TRAIN_GUARD_HOME`.

| Path | Contents | Lifetime |
|---|---|---|
| `config.json` | validated policy values | until removed |
| `run/NAME.meta.json` | job mode and root or match data | active or stale job |
| `run/NAME.guard.json` | supervisor PID and creation time | live supervisor |
| `run/NAME.runtime.json` | last action, observation and owned process changes | active or stale job |
| `run/NAME.ready.json` | child-written supervisor identity | startup handshake only |
| `run/NAME.stop` | one pending soft or kill request | until supervisor reads it |
| `run/NAME.lock` | reusable advisory creation lock | retained |
| `persist/NAME.job` | login restart command or match specification | until stopped or unpersisted |
| `logs/NAME.events.jsonl` | structured transitions and failures | retained |
| `logs/NAME.guard.log` | short readable transition log | retained |
| `logs/NAME.log` | launched worker stdout and stderr | retained |

State JSON is written to a temporary file in the destination directory,
flushed, given user-only permissions and atomically replaced. JSON
serialization rejects NaN and infinity. Each named job has one journal writer;
event files therefore use a simple append with user-only creation permissions.

Runtime reads validate the schema, cooling flag, process list, owned
suspensions and every reversible scheduling record before any recovery data is
adopted. A malformed member invalidates the whole runtime snapshot; valid
members are not silently extracted from an ambiguous record. A runtime file
left without job metadata is reported by `doctor`, blocks a new job from
reusing the name and can still be passed to `recover`.

## Scope and risk priorities

A proposed feature or defense is reviewed against a short risk budget:

1. Name the supported user path and concrete failure.
2. Estimate likelihood, user impact and implementation/maintenance cost.
3. Prefer the smallest change that covers a plausible, material failure.
4. Keep high-complexity defenses for hypothetical cases out of the release
   unless a reproduction, incident or supported-platform constraint justifies
   them.

The highest-priority risks are targeting the wrong process, leaving a process
suspended or tuned, claiming startup before supervision works, and losing the
state needed for recovery. Reproducibility and clear diagnostics come next.

The job-name lock is held across availability checking, worker creation,
metadata writes and supervisor creation. Lock files are not deleted after use:
deleting an unlocked path while another process has the old file open can split
future lockers across two inodes. The operating-system lock is authoritative.

## Lifecycle

### Launch

The CLI first validates the configuration and job name. `run` also validates
the command and working directory before launching a worker and recording its
identity; `attach` records an existing PID or a match pattern instead. Both
paths acquire the job-name lock, reject active or stale state, optionally write
a login restart specification and start the detached supervisor. The supervisor
validates its own startup,
records its PID and creation time in a readiness file, and only then lets the
parent CLI return success. For match-based attachment it first records the
parent CLI identity so that process cannot enter the match set. The parent
rejects a missing or mismatched readiness record. If setup fails, it terminates
child processes created by that attempt, removes its metadata and readiness
state, and restores any login restart specification that the command replaced.

### Soft stop

`train-guard stop NAME` writes a request. The supervisor resumes its owned
suspensions, restores its scheduling changes, removes active state and
detaches. The worker continues. If permissions prevent a complete release, it
keeps stale recovery state instead of discarding the remaining identities.

### Kill

`train-guard stop NAME --kill` releases owned process changes first, sends
terminate to the current process tree, waits up to five seconds and kills
survivors.

### Supervisor shutdown or failure

Handled signals release owned process changes and remove active state. A
handled exception also releases them, retains an error runtime snapshot and
removes the guard identity so `doctor` and `recover` can report stale state.

An uncatchable exit, such as `SIGKILL`, can leave the latest runtime snapshot
behind. Recovery is intentionally explicit because resuming a process is a
state-changing operation.

## Offline replay

`simulation.py` accepts either raw observation JSONL or structured event
journals. It validates timestamps and measurements, then passes each immutable
observation through a fresh `PolicyEngine`. It never constructs a sensor reader,
process controller or state directory.

Action time is attributed from each observation timestamp to the next; the last
sample has zero duration. Reports retain both sample counts and time-weighted
shares so irregular sampling remains visible. See
[`policy-replay.md`](policy-replay.md) for the schema and assumptions.

With `--compare-config`, the runner evaluates a baseline and candidate against
the same sequence and reports time-weighted action deltas and disagreements.
Canonical policy and observation fingerprints make two saved reports
auditable without treating a file path as provenance.

Optional user-supplied temperature and charge half-widths run a separate exact
binary64 sensitivity pass over the replay's existing interval durations. Its
dynamic program propagates thermal cooldown, returns tight marginal objective
envelopes and finds the minimum normalized in-box change that alters the
complete action sequence. Neither replay path constructs the state directory.
See [`replay-sensitivity.md`](replay-sensitivity.md) for the numeric contract and
limits.

The sweep reuses the objective-envelope pass for every candidate but omits the
more expensive action-change margin. Its nominal rows may come from verified
`TGK 1`. When `native` is selected explicitly, or `auto` selects an available
kernel, bounded rows use the separate `TGS 1` protocol after Python
independently verifies the baseline envelope. The two engines are reported
separately. Worst-corner versus best-corner interval dominance can certify some
policies as excluded, while overlapping boxes remain in a conservative outer
enclosure rather than being labelled an exact robust Pareto set.

## Login restart

Persistence stores a command or an attach pattern, not memory. The per-user
login integration calls `restart-persisted`. A run specification starts a new
worker. An attach specification waits for its match or invokes its optional
start command first.

The policy is validated before any stored start command runs. A failed retry
keeps the persistence specification for a later login and makes a best-effort
rollback of a start command created by that retry. On Linux, the generated
`Type=oneshot` user unit sets `RemainAfterExit=yes`: the helper may exit while
the unit remains active, so systemd does not immediately stop the worker and
supervisor processes left in its control group.

Current login integration recognizes and removes the earlier `resume` service
names when installing or uninstalling. Only the current `restart` agent remains
after migration.
