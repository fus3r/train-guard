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

A live job whose agent the owner has exempted skips this order; see
[ignored agents](#ignored-agents).

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

## Ignored agents

`run` and `attach` record the agent session that starts a job: the value of
`--agent`, or else the `CLAUDE_CODE_SESSION_ID` or `CODEX_THREAD_ID` variable
that Claude Code and Codex export to the commands they run. Jobs started
before version 0.5.0 have no agent.

The owner can exempt an agent's jobs by listing its id in `ignored-agents` in
the state directory, one id per line:

```text
# Agents whose jobs run at full speed
3f2b8c1e-5d4a-4e9b-9c7d-1a2b3c4d5e6f  # optional note
```

`#` starts a comment to the end of the line, and surrounding spaces and blank
lines are ignored. An id has 1 to 128 characters without whitespace, control
characters or `#`.

For a job whose agent is listed, the supervisor applies `full` on every cycle
with the reason `agent_ignored`, whatever the power source, charge and
temperature. The battery, charge and thermal rules no longer apply to that
job, and an active thermal cooldown is cleared. This changes the workload
policy only; the operating system's own thermal protection is unaffected. The
supervisor reads the list again on every cycle, so a change takes effect at
the next poll. Once the agent leaves the list, the job follows the policy
again from a fresh cooldown state.

A missing file lists no agent. A file that exists but cannot be read also
exempts no agent: the policy applies, the journal records
`ignored_agents_unreadable` once for each distinct error, and `status`,
`list` and `doctor` report it.

The list records the owner's decision, for example one made in a menu bar app
that monitors coding agents. An agent must not add its own id: the exemption
sets aside the power and temperature policy the owner chose for this machine.
`simulate` and `sweep` never read the list; they replay the policy alone.
