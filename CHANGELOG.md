# Changelog

## Unreleased

- Replace the pairwise Pareto-front scan with an exact
  `O(N + U log U)` dominance pass while preserving duplicate and tie behavior.
- Add a reproducible adversarial benchmark checked against a hand-calculated
  corpus and an independent bounded pairwise oracle.

## 0.3.0 - 2026-08-08

Version 0.3.0 hardens the cross-platform Python supervisor with explicit
process ownership and recovery state, and adds offline policy evaluation on
macOS, Linux and Windows. The package now requires Python 3.9 or later.

### Supervision and recovery

- Policy configuration is typed and validated before a worker is started. The
  decision engine is deterministic and uses one coherent battery observation
  per cycle.
- Jobs and supervisors are identified by PID and process creation time.
  Suspensions and scheduling changes are recorded separately so cleanup does
  not undo changes owned by another tool.
- Detached startup uses a readiness handshake. Runtime state is schema-checked
  and replaced atomically, job-name creation is serialized, and incomplete
  recovery evidence is retained instead of being silently discarded.
- Login restart integration supports per-user launchd, systemd and Windows
  Task Scheduler specifications, including migration of the earlier `resume`
  names.

### Observation and offline evaluation

- JSON Lines transition journals, JSON status output and `doctor` expose the
  supervisor's current decision and recovery state.
- `simulate` replays a policy over an immutable observation trace, and
  comparison mode reports every decision disagreement.
- `sweep` evaluates validated policy grids with time-weighted metrics, a Pareto
  front and an exact joint fractional bound at each policy's recorded hot and
  low-battery exposure.
- An optional dependency-free C++17 replay kernel evaluates large grids. Python
  remains the reference, and native output is accepted only after a bit-exact
  aggregate and decision-fingerprint comparison of the reference policy.

### Release verification

- A clean clone on Python 3.13.7 passed Ruff, formatting, mypy, 226 tests and
  90.18% combined line and branch coverage.
- The source archive was extracted, installed and retested. The pure-Python
  wheel was installed in a separate environment and passed a real detached
  process lifecycle smoke test.
- From the installed wheel, the example replay returned `600/300/900` seconds
  for `full/gentle/stop`; the candidate comparison returned `0/+300/-300`
  seconds with 300 seconds of disagreement; and a compiler-free Python sweep
  returned 75% baseline joint efficiency with a 36 C hot-only diagnostic
  cutoff.
- On battery power, the native benchmark evaluated 421 policies over 20,000
  observations with zero mismatches both with one thread and with automatic
  thread selection. Timing is hardware-specific and is not a release gate.
- The release tag is gated on the same commit passing the seven GitHub Actions
  jobs: Ubuntu, macOS and Windows on Python 3.9 and 3.13, plus the quality and
  packaging job.

### Scope and limits

- `train-guard` is a workload policy, not a hardware safety controller.
- Replay and sweep re-weight a recorded, exogenous trace. They do not predict
  temperature, energy use, battery lifetime or throughput under another
  policy.
- The native kernel is optional and is not shipped in the wheel. Platform and
  recovery limitations remain documented in the README and failure model.
