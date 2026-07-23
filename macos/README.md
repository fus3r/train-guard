# train-guard for macOS

The shell implementation runs without Python. It can pause a process tree on
battery, request background scheduling while the battery pack is warm, pause at
the configured high threshold, and continue the same live processes on AC.

`taskpolicy -b` changes scheduling priority. macOS still decides which cores run
the process, so the script does not promise efficiency-core placement.

```bash
./train-guard.sh status
./train-guard.sh run --name myjob -- python train.py
./train-guard.sh attach --match "python train.py"

./train-guard.sh run --restart-on-login --name myjob -- \
  python train.py --resume-from-checkpoint latest
./train-guard.sh install-agent

./train-guard.sh stop myjob
./train-guard.sh stop myjob --kill
```

Sleep preserves the process and RAM. Reboot does not. The login service starts
the stored command as a new process, so the application needs an on-disk
checkpoint to resume earlier work. `--persist` is kept as an alias for
`--restart-on-login`.

State and logs live under `${TRAIN_GUARD_HOME:-~/.train-guard}`. Copy
`config.env` to `~/.train-guard/config.env` to change the policy. A running
supervisor rereads it on every poll.

The default pack-temperature thresholds are 35, 38 and 42 degrees Celsius, with
resume at 36. They are policy values, not hardware safety limits.

The script resolves its own location. Run it from any directory or add a
symlink:

```bash
ln -s "$PWD/train-guard.sh" ~/.local/bin/train-guard
```

Use the Python package from the repository root for Linux or Windows.
