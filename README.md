# train-guard

Power-aware supervision for one long-running job.

[![CI](https://github.com/fus3r/train-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/fus3r/train-guard/actions/workflows/ci.yml)

`train-guard` watches a laptop's power source, charge level and available
battery temperature, then applies `full`, `gentle` or `stop` to one named
process tree. The same policy can be replayed against a recorded trace before
it controls a live job.

Train Guard is a workload policy, not a hardware safety controller. It does
not set a charge limit or predict battery life, temperature, energy use or
throughput.

## Install 0.4.0

Install the published package with `pipx`:

```bash
pipx install "train-guard==0.4.0"
train-guard doctor
```

If the PyPI publication has not happened yet, validate the tagged source in a
fresh checkout instead:

```bash
git clone --branch v0.4.0 --depth 1 \
  https://github.com/fus3r/train-guard.git
cd train-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
train-guard doctor
```

Missing battery temperature is common on some systems, especially Windows,
and disables temperature rules for that observation.

## First disposable job

```bash
train-guard config --init
train-guard run --name quickstart -- \
  python -c "import time; print('quickstart running', flush=True); time.sleep(120)"
train-guard status
train-guard events quickstart --limit 10
train-guard stop quickstart
```

Without `--kill`, `stop` releases changes owned by Train Guard and detaches.

## Replay and bounded sweeps

The bundled trace can be replayed without controlling a process:

Create a small policy grid as `grid.json`:

```json
{"temp_pause_c": [40, 42, 44], "run_on_battery": [true, false]}
```

Then run the nominal and bounded analyses:

```bash
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json
train-guard simulate examples/power-trace.jsonl \
  --config config.example.json \
  --temperature-uncertainty-c 0.5 \
  --charge-uncertainty-pct 1 --json
train-guard sweep examples/power-trace.jsonl \
  --grid grid.json --engine python \
  --temperature-uncertainty-c 0.5 \
  --charge-uncertainty-pct 1 --json
```

The uncertainty widths are supplied by the user; Train Guard does not infer
sensor accuracy or attach a confidence level. Exactness is limited to the
current threshold policy and the finite IEEE-754 binary64 representatives in
the declared box. The action-change report gives divergence context, not a
complete witness or certificate. Bounded-sweep survivors form a conservative
outer enclosure, not an exact robust Pareto set.

Replay and sweep re-weight an exogenous recorded trace. They do not model how a
different action would have changed later temperature, charge, performance or
energy use.

For large source-checkout sweeps, the optional C++17 kernel keeps nominal
`TGK 1` and bounded `TGS 1` as separate protocols:

```bash
cmake -S native -B native/build
cmake --build native/build --config Release
python tools/bench_sweep.py \
  --kernel native/build/train-guard-kernel --sensitivity
```

Python remains the reference, and native rows are accepted only after the
baseline agrees bit for bit. The wheel is pure Python and does not ship the
kernel. Benchmark timings are hardware-specific and are not release gates.

## Documentation for this tag

- [Architecture and lifecycle](https://github.com/fus3r/train-guard/blob/v0.4.0/docs/architecture.md)
- [Failure recovery](https://github.com/fus3r/train-guard/blob/v0.4.0/docs/failure-model.md)
- [Offline replay](https://github.com/fus3r/train-guard/blob/v0.4.0/docs/policy-replay.md)
- [Replay sensitivity](https://github.com/fus3r/train-guard/blob/v0.4.0/docs/replay-sensitivity.md)
- [Policy sweep](https://github.com/fus3r/train-guard/blob/v0.4.0/docs/policy-sweep.md)
- [Changelog](https://github.com/fus3r/train-guard/blob/v0.4.0/CHANGELOG.md)
- [MIT license](https://github.com/fus3r/train-guard/blob/v0.4.0/LICENSE)

The later documentation portal is built from these canonical Markdown files.
