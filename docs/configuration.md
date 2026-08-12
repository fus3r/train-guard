# Configuration

Train Guard reads `config.json` from `~/.train-guard` by default. Set
`TRAIN_GUARD_HOME` to use another state directory. For tests, point it at a new
temporary directory, never an existing broad directory.

Create the default file:

```bash
train-guard config --init
```

Validate it without starting a worker:

```bash
train-guard config --check
```

## Policy keys

| Key | Default | Meaning |
|---|---:|---|
| `poll` | `20` | Seconds between sensor observations |
| `run_on_battery` | `false` | Whether work may continue while unplugged |
| `battery_floor_pct` | `30` | Stop at or below this charge |
| `battery_band` | `"gentle"` | Action above the battery floor while unplugged |
| `ac_band` | `"full"` | Normal action on external power |
| `temp_charge_gentle_c` | `35` | Gentle action while warm and charging below the cutoff |
| `charge_cool_until_pct` | `80` | Charge cutoff for the warm-charging rule |
| `temp_gentle_c` | `38` | Gentle action on external power at or above this temperature |
| `temp_pause_c` | `42` | Enter thermal pause at or above this temperature |
| `temp_resume_c` | `36` | Leave thermal pause at or below this temperature |

The action bands are `full`, `gentle` and `stop`. The unplugged rule still
obeys the battery floor. Temperature thresholds must preserve the documented
ordering, including a resume threshold below the pause threshold.

## Evaluation order

The state machine evaluates a sample in this order:

1. Continue an active thermal cooldown until a reading reaches the resume
   threshold.
2. Enter thermal cooldown at the pause threshold.
3. Apply unplugged and battery-floor rules.
4. Apply the warm-charging rule.
5. Apply the general warm rule on external power.
6. Use `ac_band`.

Thermal cooldown is stateful. A temperature reading below the pause threshold
does not end an existing cooldown unless it also reaches the resume threshold.

## Missing values

If the host exposes no battery temperature, temperature rules are skipped for
that observation and the journal records a warning. An active cooldown remains
active until a temperature at or below the resume threshold proves that it can
end.

A host with no battery follows the external-power thermal ladder. The final
decision reason records the no-battery assumption.

## Live reload

The supervisor watches the configuration file. A valid edit becomes active on
the next cycle. An invalid edit is rejected, the last valid in-memory policy
stays active and the event journal records `config_rejected`.

Unknown keys, booleans used as numbers, non-finite values and inconsistent
thresholds are rejected. Version 0.1 temperature keys are accepted only for
the documented migration path; conflicting old and new keys fail validation.

For the complete decision order and state model, read
[architecture and lifecycle](architecture.md).
