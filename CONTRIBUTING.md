# Contributing to Train Guard

Train Guard controls live process trees, so a small change with a clear
contract is easier to review than a broad redesign.

## Before writing code

Open an issue or discussion before a large feature, new platform adapter,
policy option, sensor backend or public command. Describe the supported user
path, the failure or missing behavior, its likely impact and the smallest
change that would cover it.

Bug reports should include a reproduction. Use a disposable process and a
temporary state directory when the report involves process control or recovery.
Do not publish secrets, private command arguments, full journals or unredacted
user paths.

## Development setup

```bash
git clone https://github.com/fus3r/train-guard.git
cd train-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -r docs/requirements.txt
```

On Windows, activate the environment with its PowerShell or command prompt
script.

Run the relevant focused test while editing, then the public gates before a
pull request:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy trainguard
python -m coverage run -m pytest
python -m coverage report -m
python -m compileall -q trainguard tests
python -m build
python -m mkdocs build --strict
git diff --check
```

The documentation dependencies remain in their own exact requirements file;
they are not runtime dependencies of Train Guard.

## Tests

Add the smallest test set that protects a reproduced regression, explicit
contract, scientific invariant or material failure mode. Derive expected
results from the specification, an independent calculation or a hand-checked
fixture. Do not add near-duplicate cases to increase a count or coverage
percentage.

Never weaken an existing process-safety, recovery, sanitizer, data-integrity
or scientific check to make a change pass.

## Pull requests

A pull request should state:

- the supported path and concrete problem;
- the behavior before and after the change;
- the focused reproduction or independent oracle;
- the validation commands that actually ran;
- platform, process-safety or scientific limits that remain.

Keep unrelated cleanup out of the same patch. Documentation must distinguish
trace re-weighting from prediction and workload policy from hardware safety.

By submitting a contribution, you agree that it is licensed under the
repository's MIT license.
