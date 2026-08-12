# Architecture

`train-guard` separates policy from host side effects. The policy engine knows
nothing about `psutil`, files or subprocesses. The surrounding adapters turn
one sensor sample into a process action, then write the input and result to
local state.

## Data flow

<div style="overflow-x: auto;">
<figure style="margin: 1.5em 0; width: 100%; min-width: 640px;">
  <svg viewBox="0 0 720 572" role="img" aria-label="Live supervision, offline replay and grid sweeps all send observations through the same pure policy engine; only the adapters around it differ." style="width: 100%; height: auto; display: block;">
    <defs>
      <marker id="tg-ah" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto-start-reverse">
        <path d="M0,0 L8,3.5 L0,7 Z" fill="currentColor"/>
      </marker>
    </defs>
    <g fill="none" stroke="currentColor" stroke-opacity="0.12">
      <line x1="14" y1="268" x2="386" y2="268"/>
      <line x1="510" y1="268" x2="714" y2="268"/>
      <line x1="14" y1="412" x2="386" y2="412"/>
      <line x1="510" y1="412" x2="714" y2="412"/>
    </g>
    <g font-size="12" font-weight="600" letter-spacing="1.2" fill="currentColor" fill-opacity="0.55">
      <text x="14" y="32">LIVE SUPERVISION</text>
      <text x="14" y="296">OFFLINE REPLAY · simulate</text>
      <text x="14" y="440">OFFLINE SWEEP · sweep</text>
    </g>
    <rect x="394" y="120" width="108" height="436" rx="8" fill="var(--md-accent-fg-color)" fill-opacity="0.07" stroke="var(--md-accent-fg-color)" stroke-width="1.8"/>
    <g text-anchor="middle" fill="currentColor">
      <text x="448" y="316" font-size="15" font-weight="700">PolicyEngine</text>
      <text x="448" y="334" font-size="10.5" fill-opacity="0.6">policy.py</text>
      <text x="448" y="354" font-size="11.5" fill-opacity="0.78">pure state machine</text>
      <text x="448" y="370" font-size="10.5" fill-opacity="0.78">no I/O · no processes</text>
    </g>
    <g fill="var(--md-code-bg-color)" stroke="currentColor" stroke-opacity="0.35">
      <rect x="158" y="48" width="146" height="30" rx="6"/>
      <rect x="14" y="132" width="112" height="44" rx="6"/>
      <rect x="158" y="132" width="146" height="44" rx="6"/>
      <rect x="568" y="132" width="146" height="44" rx="6"/>
      <rect x="568" y="212" width="146" height="32" rx="6"/>
      <rect x="158" y="212" width="146" height="34" rx="6"/>
      <rect x="14" y="316" width="112" height="44" rx="6"/>
      <rect x="158" y="316" width="146" height="44" rx="6"/>
      <rect x="568" y="306" width="146" height="64" rx="6"/>
      <rect x="14" y="460" width="112" height="44" rx="6"/>
      <rect x="158" y="460" width="146" height="44" rx="6"/>
      <rect x="568" y="450" width="146" height="64" rx="6"/>
    </g>
    <g text-anchor="middle" fill="currentColor">
      <text x="231" y="67" font-size="13.5" font-weight="600">CLI</text>
      <text x="70" y="150" font-size="12.5" font-weight="600">SensorReader</text>
      <text x="70" y="166" font-size="10" fill-opacity="0.6">sensors.py</text>
      <text x="231" y="150" font-size="13.5" font-weight="600">Supervisor</text>
      <text x="231" y="166" font-size="10" fill-opacity="0.6">supervisor.py</text>
      <text x="641" y="150" font-size="12.5" font-weight="600">ProcessController</text>
      <text x="641" y="166" font-size="10" fill-opacity="0.6">processes.py</text>
      <text x="641" y="232" font-size="12">worker process tree</text>
      <text x="231" y="226" font-size="10.5">runtime.json · events.jsonl</text>
      <text x="231" y="239" font-size="9.5" fill-opacity="0.6">state after every action</text>
      <text x="70" y="228" font-size="12">config.json</text>
      <text x="70" y="241" font-size="9.5" fill-opacity="0.6">live reload</text>
      <text x="70" y="334" font-size="12.5" font-weight="600">Trace JSONL</text>
      <text x="70" y="350" font-size="9" fill-opacity="0.6">observations or events</text>
      <text x="231" y="334" font-size="12.5" font-weight="600">SimulationRunner</text>
      <text x="231" y="350" font-size="10" fill-opacity="0.6">simulation.py</text>
      <text x="641" y="324" font-size="13" font-weight="600">Replay report</text>
      <text x="641" y="340" font-size="9.5" fill-opacity="0.78">time-weighted actions · deltas</text>
      <text x="641" y="354" font-size="9.5" fill-opacity="0.78">bounded envelopes · margin</text>
      <text x="70" y="478" font-size="12.5" font-weight="600">Candidate grid</text>
      <text x="70" y="494" font-size="9" fill-opacity="0.6">+ the same trace</text>
      <text x="231" y="478" font-size="13" font-weight="600">SweepRunner</text>
      <text x="231" y="494" font-size="10" fill-opacity="0.6">sweep.py</text>
      <text x="641" y="468" font-size="13" font-weight="600">Sweep report</text>
      <text x="641" y="484" font-size="10" fill-opacity="0.78">nominal Pareto front</text>
      <text x="641" y="498" font-size="10" fill-opacity="0.78">interval-front enclosure</text>
    </g>
    <g stroke="currentColor" stroke-width="1.4" stroke-opacity="0.8" fill="none" marker-end="url(#tg-ah)">
      <line x1="231" y1="78" x2="231" y2="128"/>
      <line x1="126" y1="154" x2="154" y2="154"/>
      <line x1="304" y1="154" x2="390" y2="154"/>
      <line x1="502" y1="154" x2="564" y2="154"/>
      <line x1="641" y1="176" x2="641" y2="208"/>
      <line x1="231" y1="176" x2="231" y2="208"/>
      <line x1="126" y1="338" x2="154" y2="338"/>
      <line x1="304" y1="338" x2="390" y2="338"/>
      <line x1="502" y1="338" x2="564" y2="338"/>
      <line x1="126" y1="482" x2="154" y2="482"/>
      <line x1="304" y1="482" x2="390" y2="482"/>
      <line x1="502" y1="482" x2="564" y2="482"/>
    </g>
    <g stroke="currentColor" stroke-width="1.2" stroke-opacity="0.65" stroke-dasharray="4 3" fill="none" marker-end="url(#tg-ah)">
      <line x1="112" y1="218" x2="196" y2="180"/>
      <line x1="231" y1="380" x2="231" y2="364"/>
      <line x1="231" y1="524" x2="231" y2="508"/>
    </g>
    <g fill="currentColor" fill-opacity="0.75" font-size="12">
      <text x="223" y="108" text-anchor="end" font-size="11.5">starts a detached supervisor</text>
      <text x="347" y="146" text-anchor="middle">Observation</text>
      <text x="533" y="146" text-anchor="middle">decision</text>
      <text x="633" y="198" text-anchor="end" font-size="11.5">full · gentle · stop</text>
      <text x="347" y="330" text-anchor="middle">observations</text>
      <text x="533" y="330" text-anchor="middle">actions</text>
      <text x="347" y="464" text-anchor="middle" font-size="11">one replay</text>
      <text x="347" y="476" text-anchor="middle" font-size="11">per policy</text>
      <text x="533" y="474" text-anchor="middle" font-size="11.5">objectives</text>
      <text x="231" y="392" text-anchor="middle" font-size="10.5">optional: candidate config · user bounds</text>
      <text x="231" y="536" text-anchor="middle" font-size="10.5">optional: user bounds</text>
    </g>
  </svg>
  <figcaption style="font-size: 0.72rem; opacity: 0.75; margin-top: 0.4em;">Live supervision, offline replay and grid sweeps all send observations through the same pure policy engine; only the adapters around it differ.</figcaption>
</figure>
</div>

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
