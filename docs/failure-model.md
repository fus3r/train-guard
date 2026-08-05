# Failure model

This document states what `train-guard` detects, what it does automatically and
where a user decision is still required.

## Invalid configuration at startup

`run` and `attach` validate the whole JSON file before creating a worker or
supervisor. Invalid JSON, unknown keys, wrong types and inconsistent
temperature thresholds return exit code 2.

Check the file directly:

```bash
train-guard config --check
```

No job is launched on validation failure.

## Detached supervisor startup

The parent CLI does not treat `Popen` as proof of a successful launch. The
detached supervisor must first validate its state and configuration, append its
`started` event and write a readiness record containing its own PID and process
creation time. The parent compares that record with the identity it captured.

If the child exits, times out or reports a different identity, `run` and
`attach` return exit code 2. The command terminates child processes created by
that attempt, removes its metadata and transient readiness file, and restores
the previous login restart specification when one existed. This does not claim
transactional safety against an uncatchable parent crash between individual
filesystem or process operations.

## Invalid configuration during a run

The supervisor fingerprints the file and parses changed content. If a change is
invalid, it keeps the last valid in-memory policy, writes the error to
`runtime.json` and emits `config_rejected`.

After a valid correction, it emits `config_loaded` and uses the new policy on
the next cycle.

## Missing sensor data

Missing battery temperature disables thermal rules for that observation. Power
source and charge rules continue when battery data is available.
An active thermal cooldown is the exception: it stays active until a reading at
or below the resume threshold proves that the pack has cooled.

Non-finite battery percentages and temperatures outside the supported live
range are converted to missing values and accompanied by warnings before
policy evaluation or JSON serialization.

If the host exposes no battery at all, the observation is marked
`no_battery`, the configured AC band is used and a warning is recorded.

Use `train-guard doctor` or `train-guard status --json` to see the raw
observation and warnings.

## PID reuse

Launched jobs and PID attachments record process creation time with the PID.
Every lookup verifies the pair. A different process that later receives the
same PID is treated as absent.

Metadata from the pre-0.3 format contains only a PID. A live supervisor upgrades
the record only when the current process predates the metadata file, then emits
`state_migrated`. If the process is newer, the PID has been reused and startup
fails while preserving the legacy metadata for inspection.

## Worker exits

When the root of a launched job exits, the supervisor releases any owned child
suspensions and scheduling changes, emits `job_exited` and removes active
runtime files. If any release is denied, it instead retains a stale recovery
record.

Match-based jobs do not exit when the match disappears. They enter `waiting`
and keep polling for a later match. Before waiting, the controller releases
owned changes for processes that are no longer in the resolved target set. The
same reconciliation applies when a child leaves a launched process tree.

## Normal supervisor shutdown

`SIGINT` and `SIGTERM` set a shutdown flag. The loop releases only its owned
process changes, emits `shutdown` and removes active state.

The same release step runs after a handled exception.

## Dead supervisor with stale state

An uncatchable termination can leave metadata and a stopped worker. `status`
marks the guard as dead and `doctor` reports the stale job.

Review the record, then recover it:

```bash
train-guard status
train-guard events NAME
train-guard recover NAME
```

Recovery refuses to run while the recorded supervisor identity is alive. For a
dead supervisor, it reads the saved suspension and scheduling-state identities,
verifies PID and creation time, resumes and restores matches, then removes the
stale active state.

If a release is denied, recovery returns a non-zero status and writes
`recovery_incomplete`. Successfully released identities are removed from the
runtime record while denied identities remain available for a later retry.

There is a small interval between a successful suspension or scheduling change
and the next runtime write. If the supervisor is killed in that interval, no
complete recovery record exists. Inspect the worker with the operating system's
process tools.

## Permission failure

Process inspection, suspension or priority changes can fail when the worker
belongs to another user or the host restricts an API. The process report counts
access failures and the event journal retains that report.

`train-guard` does not request elevation. Run it as the same user as the worker
or change the surrounding service permissions. Cleanup does not discard a
denied ownership record merely to make state appear healthy.

## Reboot

Reboot removes all live process state. A login restart specification can start
the command again, but it cannot recover RAM.

Use application-level checkpoints and test their restore path before depending
on restart:

```bash
train-guard run --restart-on-login --name experiment -- \
  python train.py --resume-from-checkpoint latest
train-guard install-agent
```

On macOS and Linux, installing the current login agent unloads and removes the
older `resume` agent first. On Windows, it removes the older
`TrainGuardResume` task. Uninstall checks both generations as well.

`restart-persisted` validates the policy before launching any optional
`--start` command. If a supervisor startup then fails, the restart
specification remains available for a later login and the helper makes a
best-effort attempt to terminate the start command it just created. On Linux,
the generated oneshot unit uses `RemainAfterExit=yes`; without that setting,
the unit would become inactive as soon as the helper exits and systemd could
stop the detached processes remaining in its control group.

## Process-tree races

The controller resolves a point-in-time process tree. A worker can create or
exit children immediately after that snapshot.

The next policy cycle normally catches new children. During `stop --kill`, a
new child created after the final snapshot can escape termination. Verify the
application's own shutdown behavior when complete tree termination matters.

## Concurrent commands

`run` and `attach` acquire a non-blocking advisory lock for the requested job
name before checking existing state or creating a worker. One of two
same-name commands can proceed; the other exits before launching anything.

The lock is local to the state filesystem. A shared directory mounted on
multiple hosts is outside the supported model unless that filesystem preserves
the host's advisory-lock semantics.

## Match-based attachment

The detached supervisor captures the launching CLI as a PID-and-creation-time
identity before it reports ready. That identity and the supervisor itself are
excluded from substring matching. If the parent identity cannot be read,
startup fails and the attach command rolls back instead of risking control of
the CLI that contains the requested pattern.

Other unrelated commands can still contain the same substring. Use
`attach --pid` when the intended process already has a stable PID, and choose a
specific command-line fragment otherwise.

## Invalid replay traces

`simulate` rejects malformed JSON, invalid measurement types, non-finite
numbers, timestamps without offsets and reverse time order. The only tolerated
parse failure is an unterminated final line in a live journal, which may have
been read during append. Other failures report the file and line number and
make no process or state changes.

Event-journal replay skips lifecycle records without an observation. A journal
with no observation records is rejected rather than reported as an empty
successful run.

## Corrupt state

Malformed state is rejected instead of guessed. `status` reports invalid
runtime data where possible, while list operations skip malformed job metadata.
`recover` validates the complete runtime schema and every owned suspension or
scheduling record before changing a process. If one record is malformed, it
leaves the metadata and runtime file untouched for inspection instead of
recovering a subset and erasing the rest.

Process identities require a positive integer PID and a positive finite
creation time. Current job and persistence schemas reject unknown future schema
versions rather than silently interpreting them as version 1. Legacy metadata
without a schema marker remains readable for the documented migration path.

An interrupted cleanup can leave `runtime.json` after `meta.json` has already
been removed. This recovery state is not silently adopted by a later job with
the same name. `doctor` reports it, name reuse is blocked and
`train-guard recover NAME` remains available even without the metadata file.

Set `TRAIN_GUARD_HOME` to a temporary directory when reproducing a state
problem:

```bash
export TRAIN_GUARD_HOME="$(mktemp -d)"
train-guard config --init
```

Do not point this variable at an existing broad directory. `train-guard`
creates and removes files with names reserved for its own state layout.
