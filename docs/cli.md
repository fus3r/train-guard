# Command-line reference

Run `train-guard COMMAND --help` for the parser's complete option list. This
page groups the supported commands by task.

## Live jobs

### `run`

Launch and supervise a command:

```text
train-guard run [--name NAME] [--restart-on-login] [--cwd DIR] -- COMMAND...
```

`--restart-on-login` stores the command so the login helper can start a new
process after reboot. It does not restore memory or application state.

### `attach`

Supervise an existing process identity or a command-line match:

```text
train-guard attach --pid PID --name NAME
train-guard attach --match TEXT --name NAME [--start COMMAND]
```

Prefer `--pid` when the intended process already has a stable PID. Match mode
can include unrelated commands containing the same text.

### `status`, `list` and `events`

```text
train-guard status [--json]
train-guard list [--json]
train-guard events NAME [--limit N] [--json]
```

`status` includes the current sensor observation, policy, active supervisors
and restart specifications. `events` reads the structured transition journal.

### `stop` and `recover`

```text
train-guard stop NAME [--kill]
train-guard recover NAME
```

Without `--kill`, `stop` releases changes owned by Train Guard and detaches.
`recover` is for a dead supervisor with retained runtime state. It refuses to
act while the recorded supervisor identity is alive.

## Offline analysis

### `simulate`

```text
train-guard simulate TRACE [--config FILE] [--compare-config FILE]
  [--transition-limit N]
  [--temperature-uncertainty-c VALUE]
  [--charge-uncertainty-pct VALUE] [--json]
```

Without uncertainty options, the public JSON schema remains version 1. With at
least one bound, the outer report and nested sensitivity report use schema 3.
The user supplies closed half-widths; the command does not infer sensor error
or a confidence level.

### `sweep`

```text
train-guard sweep TRACE --grid FILE [--config FILE]
  [--engine auto|python|native]
  [--hot-ref VALUE] [--low-battery-ref VALUE] [--top N]
  [--temperature-uncertainty-c VALUE]
  [--charge-uncertainty-pct VALUE] [--json]
```

Python remains the reference engine. Native results are accepted only after
the documented differential check.

## Configuration and diagnostics

```text
train-guard config [--init] [--force] [--check]
train-guard doctor [--json]
```

`config --check` validates the active policy without starting a job. `doctor`
checks installation and local state. Its JSON output can contain user paths;
redact them before attaching it to a public issue.

## Login restart

```text
train-guard install-agent
train-guard uninstall-agent
train-guard restart-persisted
train-guard unpersist NAME
```

`restart-persisted` is normally called by the per-user login integration.
`resume` remains a compatibility alias and does not restore RAM.

## Exit status

- `0` means the command completed its requested check or operation.
- `1` means a diagnostic or cleanup operation found an unresolved failure.
- `2` means the request, configuration, state or trace was invalid.
- `130` means the CLI was interrupted from the terminal.
