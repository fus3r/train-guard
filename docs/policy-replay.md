# Offline policy replay

`train-guard simulate` runs the same `PolicyEngine` used by a live supervisor,
but performs no sensor reads and makes no process changes. Its purpose is to
make threshold review reproducible: a policy file and an ordered trace produce
the same decisions on every supported platform.

## Input formats

The command accepts UTF-8 JSON Lines in either of two forms.

### Raw observations

Each non-empty line is one observation:

```json
{"source":"ac","percent":60.0,"temperature_c":36.0,"charging":true,"observed_at":"2026-07-26T09:05:00Z","warnings":[]}
```

Required fields:

| Field | Values |
|---|---|
| `source` | `ac`, `battery` or `no_battery` |
| `observed_at` | RFC 3339 timestamp with `Z` or an explicit UTC offset |

Fractional seconds may use any number of digits; digits beyond
microseconds are truncated. Keys outside this table are rejected so a
misspelled field cannot silently disable the signal it carries.

Optional fields:

| Field | Values |
|---|---|
| `percent` | number from 0 to 100, or `null` |
| `temperature_c` | finite number from -100 to 200, or `null` |
| `charging` | `true`, `false` or `null` |
| `warnings` | list of strings |

A `no_battery` observation must use `null` for both `percent` and `charging`.
NaN, infinity, booleans used as numbers and reverse-ordered timestamps are
rejected.

### Event journals

The event journal at `~/.train-guard/logs/NAME.events.jsonl` is also accepted.
Lines with an `observation` object are replayed; lifecycle lines without one
are skipped.

An event journal is transition-oriented rather than a lossless periodic sensor
record. Replaying it reconstructs the intervals represented by its recorded
observations. A handled stop, worker exit, shutdown or failure includes a
terminal copy of the last observation, so a completed journal closes its final
recorded interval. A live journal is complete only through the latest recorded
observation. Replay does not infer unrecorded temperature or charge changes.

A live journal can be read while its writer is mid-append, so an
unparseable final line without a trailing newline is skipped. A torn or
corrupt line anywhere else still fails with its line number.

## Time weighting

For ordered observations at times `t[0]` through `t[n-1]`, the action selected
at `t[i]` is treated as active until `t[i+1]`. Its duration is therefore:

```text
duration[i] = t[i+1] - t[i]    for i < n - 1
duration[n-1] = 0
```

The time assigned to action `a` is the sum of durations whose decision is `a`.
Its percentage is that sum divided by `t[n-1] - t[0]`. Sample counts are
reported separately because irregular sampling can make sample share and time
share materially different.

Equal timestamps are permitted and contribute zero duration. A one-sample
trace has valid decisions and sample counts, but its time percentages are
`null`.

## Commands

Replay the included trace with the default example policy:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json
```

Replay a live job's event journal with the current user policy:

```bash
train-guard simulate ~/.train-guard/logs/experiment.events.jsonl
```

Emit the complete machine-readable report:

```bash
train-guard simulate examples/power-trace.jsonl --json > replay.json
```

Compare a candidate policy with a baseline:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json \
  --compare-config examples/battery-enabled-policy.json
```

The comparison is counterfactual: both engines receive the exact same ordered
observations. It reports candidate-minus-baseline action seconds and percentage
points, transition-count deltas, action-disagreement duration, and the rows
whose decisions differ. It does not estimate energy, temperature or model
throughput.

Human output lists at most 20 decision transitions by default. Change that
display limit with `--transition-limit`; JSON output always contains every
decision and transition.

## Report contract

The JSON report has `schema_version: 1` and includes:

- the complete validated policy used for replay;
- canonical SHA-256 fingerprints of the policy and parsed observations;
- trace and configuration source paths;
- start, end and elapsed time;
- action counts, seconds and percentages for `full`, `gentle` and `stop`;
- counts for every decision reason encountered;
- action-transition and decision-transition counts;
- the initial decision row followed by every decision change, so the
  `transitions` array holds `decision_transitions + 1` rows;
- one decision row per input observation.

The complete policy is embedded so two reports can be compared without relying
on a mutable external configuration file. The fingerprints identify canonical
validated inputs; they are not signatures and do not authenticate who produced
the trace.

Comparison JSON has its own `schema_version: 1`, contains complete baseline and
candidate replay reports, and adds a `delta` object plus transition-level
`disagreements`.

## Included example

`examples/power-trace.jsonl` contains seven five-minute samples:

1. cool AC power selects `full`;
2. warm charging selects `gentle`;
3. the pause threshold enters thermal `stop`;
4. the job remains stopped during cooldown;
5. the resume threshold returns to `full`;
6. unplugging selects `stop` under the default policy;
7. reconnecting AC returns to `full`.

The 30-minute replay assigns 600 seconds to `full`, 300 to `gentle` and 900 to
`stop`. These are policy outputs on a synthetic trace, not measurements of
battery life, temperature reduction or training throughput.
