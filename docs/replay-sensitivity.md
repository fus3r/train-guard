# Bounded replay sensitivity

Recorded temperature and charge are measurements, not exact physical state.
`train-guard simulate` can answer a narrow, reproducible question when the user
supplies closed uncertainty intervals:

> Which actions are possible, and what are the tight marginal ranges of run
> time, hot exposure and low-battery exposure, over every admitted IEEE-754
> binary64 value?

This is a bounded adversarial analysis. It does not infer sensor accuracy, fit a
noise distribution or attach a confidence level. The half-widths should be
justified separately for the measuring device. This follows the distinction in
[NIST measurement-uncertainty guidance](https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty):
display precision alone does not establish measurement accuracy.

## Command

Provide either or both half-widths:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json \
  --temperature-uncertainty-c 0.5 \
  --charge-uncertainty-pct 1 \
  --json
```

A recorded `41.8` C then admits the closed interval `[41.3, 42.3]`; a
recorded `60` percent charge admits `[59, 61]`. Charge is clipped to `[0, 100]`
and temperature to the replay input domain `[-100, 200]` C. Missing fields
remain missing.

Without `--json`, the command prints the declared box, the runnable-time
interval, locally ambiguous action time and the nearest action-change context.
It states explicitly that the result has no confidence level.

Comparison mode applies the same box independently to the baseline and
candidate while preserving their nominal delta:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json \
  --compare-config examples/battery-enabled-policy.json \
  --temperature-uncertainty-c 0.5 \
  --charge-uncertainty-pct 1
```

## Exact method

The policy changes only at configured temperature and charge thresholds. The
exposure metrics also change at their reporting references, 35 C and 20 percent
by default. For each interval, Train Guard evaluates its endpoints, each
relevant threshold and the adjacent representable binary64 values. These
representatives cover every decision region and every linear exposure segment
without choosing a grid resolution.

Thermal cooldown is the policy's only state. A dynamic program propagates both
reachable cooldown values and the cumulative extrema for:

- runnable seconds;
- degree-seconds above the hot reference while running;
- seconds at or below the low-battery reference while running.

The resulting intervals are tight marginal extrema under the declared model.
They are marginal because the trace attaining one endpoint need not attain an
endpoint of another objective.

Replay also computes the nearest admitted trace whose complete action sequence
differs. For each movable axis, displacement is divided by its declared
half-width; the path cost is the largest normalized displacement over all
samples and axes. The result is the minimum such `L-inf` cost in the declared
binary64 box, capped at one to preserve already-admitted rounded endpoints.

A second dynamic program follows prefixes whose actions still equal the
nominal replay and retains the cheapest prefix for each cooldown state. This is
necessary because a perturbation can first change cooldown while leaving the
current action unchanged, then cause an action divergence later.

The report records the sample, timestamp, nominal action and alternative action
at that first divergence. It does not serialize the earlier perturbations, so
the context is not a complete witness or a certificate.

## Report contract

Replay without either option remains `schema_version: 1`. When an uncertainty
option is present, the outer replay or comparison report uses
`schema_version: 3`. Each replay gains a nested `sensitivity` object with its
own `schema_version: 3`; all existing nominal fields retain their schema-1
meaning.

The nested report contains:

- the exact half-widths, `bounded_adversarial`, `ieee_754_binary64` and
  `confidence_level: null`;
- tight marginal minima and maxima for the three objectives;
- seconds and sample counts for which every reachable path agrees on the
  action, and those for which more than one action is reachable;
- `action_change_margin`, including the nullable normalized distance, its
  hexadecimal binary64 form and the critical divergence context.

For the bundled trace at +/-0.5 C and +/-1 charge point, runnable time is
`600..1500` seconds. The nearest action divergence is one binary64 step below
an inclusive thermal threshold:

```json
{
  "minimum_normalized_action_distance": 1.4210854715202004e-14,
  "minimum_normalized_action_distance_hex": "0x1.0000000000000p-46",
  "stable_for_declared_box": false,
  "critical_sample": {
    "sample": 3,
    "observed_at": "2026-07-26T09:10:00Z",
    "nominal_action": "stop",
    "alternative_action": "gentle"
  }
}
```

A null distance means that every admitted representative path preserves the
nominal action sequence throughout the declared box.

## Assumptions and limits

- Temperature and charge intervals form a Cartesian product at each sample.
  Temporal correlation is unmodelled.
- Power source and charging status remain as recorded.
- The trace is exogenous: actions do not feed back into later temperature,
  charge, performance or energy use.
- Bounds are supplied, not inferred or calibrated. They have no probability or
  confidence interpretation.
- Exactness is relative to the current threshold policy, its binary64 numeric
  domain, the validated trace and the declared bounds.
- Objective intervals are marginal and must not be combined into a synthetic
  joint worst-case trajectory.
- The result is not a hardware guarantee, battery-life prediction, causal
  estimate or safety certification.
