# Multi-objective policy sweep

`train-guard sweep` evaluates a grid of candidate policies against one
recorded trace and reports, for every candidate, how much the job would
have been allowed to run and how much heat and low-charge exposure that
permission would have carried. The result is a reproducible comparison
over recorded observations.

```bash
train-guard sweep examples/power-trace.jsonl \
  --grid my-grid.json --config config.example.json
```

The grid file maps policy fields to candidate values; the sweep evaluates
their cartesian product merged over the `--config` baseline:

```json
{
  "temp_pause_c": [40, 42, 44],
  "temp_gentle_c": [36, 38],
  "run_on_battery": [true, false]
}
```

Combinations that fail policy validation (for example a resume threshold
at or above a pause threshold) are skipped and counted in the report.
Duplicates that validate to the same policy are evaluated once.

## What is measured

Each candidate replays the same `PolicyEngine` used by a live supervisor
over the same immutable observations, then reports:

| Metric | Meaning |
|---|---|
| `run_seconds` | time the policy permits work (`full` or `gentle`) |
| `hot_run_degc_seconds` | degree-seconds above `--hot-ref` (default 35 °C) while work is permitted |
| `low_battery_run_seconds` | time work is permitted on battery at or below `--low-battery-ref` (default 20 %) |
| `action_transitions` | action changes; a stability indicator |

A candidate is marked Pareto-optimal when no other candidate permits at
least as much run time with no more hot or low-charge exposure and is
strictly better on at least one of the three.

The dominance pass sorts distinct metric triples by decreasing run time,
compresses the heat axis and keeps a prefix minimum of low-charge exposure in
a Fenwick tree. For `N` rows and `U` distinct triples, it uses
`O(N + U log U)` time and `O(N + U)` memory instead of comparing every pair.
Identical triples remain co-optimal because neither is strictly better. This
is an instance of the multidimensional maxima problem studied by
[Kung, Luccio and Preparata (1975)](https://doi.org/10.1145/321906.321910),
not a new optimization method.

`python tools/bench_pareto.py` checks a hand-calculated mixed corpus and a
bounded prefix against an independent pairwise oracle before timing an
all-nondominated workload. Its timings describe only the printed Python,
platform, point count and repeat count; they are not a CI threshold.

## Reading the numbers honestly

The metrics are exposure re-weightings of a recorded trace, not simulated
outcomes. The recorded temperatures and charge levels were
produced while the original run's actions were in effect; replaying a
different policy does not model how those actions would have changed the
pack's temperature or drain. The report can say "this policy would have
permitted 300 fewer degree-seconds above 35 °C"; it cannot say "the pack
would have been cooler" and it never predicts battery life. This is the
standard limitation of trace-driven policy comparison and it is why the
sweep reports exposure, transitions and permitted time rather than
temperatures or capacity.

The reference temperatures are reporting anchors, not physical thresholds.
Lithium-ion aging changes with temperature and dwell at high charge.
[Apple documents 35 °C as the ambient
ceiling](https://www.apple.com/batteries/maximizing-performance/) for its
MacBook hardware, while [Battery University describes temperatures above 30 °C
as elevated](https://batteryuniversity.com/article/bu-808-how-to-prolong-lithium-based-batteries)
and stresses that packs differ. These values are reporting references, not
claims that the policy predicts battery aging.

## Trace facts

The report also includes policy-independent statistics of the trace
itself, each with the share of trace time its input field covered:

- `hot_degc_seconds`: degree-seconds above the hot reference across the
  whole trace;
- `high_soc_seconds` and `hot_and_full_seconds`: dwell at or above 80 %
  charge, and that dwell while also at or above the hot reference;
- `equivalent_full_cycles`: charge moved through the pack,
  `sum(|Δpercent|) / 200`, counting steps of at least 2 points so
  gauge noise is not misread as cycling.

## The clairvoyant bound

Every replayed policy selects, in effect, a subset of the recorded
inter-observation intervals during which work is permitted; its
`run_seconds` and exposure metrics are sums over that subset. Relax the
selection to portions of intervals and give the selector the whole trace in
advance. Let `x_i` be the permitted seconds selected from an interval of
duration `d_i`, `h_i` its hot rate, and `l_i` its binary low-battery
indicator. The bound solves

`max Σx_i`, subject to `Σh_i x_i ≤ H`, `Σl_i x_i ≤ L`, and
`0 ≤ x_i ≤ d_i`.

Here, `H` and `L` are the candidate's recorded hot and low-battery exposure
budgets.

This is a continuous two-budget linear program. Each one-budget
projection is a fractional knapsack ([Dantzig
1957](https://doi.org/10.1287/opre.5.2.266)), but taking the smaller of
those two optima is not generally attainable by one schedule.
The implementation computes the joint optimum exactly. Because `l_i` is
binary, it can reduce each query to sorted hot-cost prefixes for
low-battery and non-low intervals, with binary search on their concave
piecewise-linear frontiers ([Boyd & Vandenberghe 2004,
§5.6.3](https://web.stanford.edu/~boyd/cvxbook/)).

Concretely, let `U(H)` be the hot-only optimum and `m(H)` the least
low-battery time among hot-only optima (non-low intervals win equal-rate
ties). Let `C_low(y)` be the least hot cost of buying `y` low-battery
seconds, and `F_nonlow(h)` the non-low optimum with hot budget `h`. Then

- if `L >= m(H)`, the hot-only schedule already satisfies both budgets, so
  the joint value is `U(H)`;
- otherwise the low-battery constraint binds, its cheapest prefix costs
  `C_low(L)`, and the joint value is
  `L + F_nonlow(H - C_low(L))`.

All four functions are sorted prefix frontiers. Construction is
`O(n log n)` and a candidate query is `O(log n)`; no general LP solver is
required at runtime.

Each candidate (and the baseline) is then reported with:

| Field | Meaning |
|---|---|
| `bound_run_seconds` | exact joint fractional optimum at the candidate's own hot and low-battery exposure |
| `hot_bound_run_seconds` | independent hot-only optimum; diagnostic, not the reported joint schedule |
| `low_battery_bound_run_seconds` | independent low-battery-only optimum; diagnostic, not the reported joint schedule |
| `efficiency` | `run_seconds / bound_run_seconds`, the reciprocal of an empirical competitive ratio against the clairvoyant schedule |
| `gap_seconds` | work a clairvoyant schedule could have added at the same recorded exposure |
| `hot_only_hindsight_threshold_c` | cutoff of the independent hot-only diagnostic |

The hot-only projection has fixed-threshold structure and at most one
fractional interval at the margin. The joint schedule need not have one
temperature threshold: both budgets can bind, with distinct marginal
intervals. An efficiency near 100% means that permitted time is near the
fractional upper bound at both recorded exposure budgets. The example trace
shows
the mechanism: the default policy permits 900 s at 600 degC·s, the
joint bound at that exposure is 1200 s (75 % efficiency), and enabling
`run_on_battery` picks up the recorded battery interval, which is free on
both axes, and closes the whole gap.

The construction uses a standard evaluation device. It does not introduce a
new optimization algorithm.
Offline baselines with full knowledge of a recorded trace have
benchmarked dynamic power management since the late 1990s: Šimunić,
Benini and De Micheli ([DATE
2000](https://doi.org/10.1109/DATE.2000.840869)) report an explicit
"oracle policy" row next to their online policies; Lu, Chung, Šimunić,
Benini and De Micheli ([DATE
2000](https://doi.org/10.1109/DATE.2000.840010)) replay recorded disk
traces through eleven policies against an "off-line" optimum computed
with full knowledge of future requests; Irani, Shukla and Gupta ([TECS
2003](https://doi.org/10.1145/860176.860180)) measure online DPM
strategies as ratios to the offline optimum per trace; the
Benini-Bogliolo-De Micheli [survey
(2000)](https://doi.org/10.1109/92.845896) defines the ideal policy with
complete a priori knowledge of the workload. What is applied here is
that device, transposed from energy-for-a-request-trace to permitted
work under exposure budgets. The July 2026 review for this project found no
laptop supervisor or charge-management tool that reports such a bound. If you
know one, please open an issue.

The joint optimizer is checked against an independently enumerated
two-constraint LP in exact rational arithmetic over randomized compact
instances. The one-axis dominance property is also checked on randomized
stateful policy replays. The report still clamps `gap_seconds`
at zero and `efficiency` at one because float sums can land an ulp on the
wrong side of an exact inequality.

Like every sweep metric, the bound re-weights one fixed recording under
the exogenous-trace assumption. It bounds permission accounting on this
trace; it says nothing about the temperatures, charge levels or battery
life that different actions would have produced.

## The native kernel

A sweep can multiply thousands of candidate policies by tens of
thousands of observations. The optional native kernel
(`native/replay_kernel.cpp`, C++17) evaluates that product in a single
subprocess call. It is a development and CI artifact: wheels stay pure
Python, `--engine python` is always available, and the packaged CLI
works without any compiler.

The kernel has the following correctness contract:

- floats cross the process boundary as C99 hexadecimal literals
  (`float.hex()` in, `printf "%a"` out), so no decimal rounding exists
  in either direction;
- the kernel mirrors the Python engine's arithmetic operation for
  operation, and the build disables floating-point contraction
  (`-ffp-contract=off`, MSVC `/fp:precise`), so every aggregate is
  bit-identical to CPython's, not merely close;
- before any kernel result is used, the baseline policy is replayed by
  both engines and every aggregate, including an order-sensitive
  FNV-1a fingerprint of the full decision sequence, must match
  exactly, or the sweep refuses the kernel;
- CI builds the kernel with GCC, Apple Clang and MSVC and re-verifies
  the bit-exact agreement on every push;
- large grids are spread over worker threads (`train-guard-kernel
  [threads]`, chosen automatically by the sweep, capped at 16). Whole
  policies are distributed: each policy's accumulation remains one
  sequential dependency chain on one thread, rows land in preallocated
  slots and are printed in input order after every join, so the output
  is byte-identical for every thread count. Parallelism never touches
  the arithmetic, and the differential check runs unchanged.

Engine selection: `--engine auto` (default) uses the kernel when
`train-guard-kernel` is on `PATH` or `TRAIN_GUARD_KERNEL` names it,
`--engine python` forces the reference, `--engine native` fails rather
than fall back.

To build and benchmark locally:

```bash
cmake -S native -B native/build
cmake --build native/build --config Release
python tools/bench_sweep.py --kernel native/build/train-guard-kernel
```

`tools/bench_sweep.py` prints its exact workload and kernel thread count,
which is controllable with `--kernel-threads`. It verifies every row
bit for bit, and reports wall-clock time for both engines. Timings are
hardware-specific; publish them only with the printed protocol attached.
