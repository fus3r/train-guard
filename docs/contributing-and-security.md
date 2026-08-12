# Contributing and security

Train Guard accepts small, testable changes tied to a supported workflow. The
[contributing guide](https://github.com/fus3r/train-guard/blob/main/CONTRIBUTING.md)
contains the development setup, validation commands and scope rules. Discuss a
large feature in an issue before implementing it.

Use the
[hardware report form](https://github.com/fus3r/train-guard/issues/new?template=hardware-report.yml)
for both successful and failed trials on a real machine. Include the Train
Guard version, Python and operating system versions, available sensors and the
scenario you ran. Remove user paths, command arguments, environment variables
and journal content that should not be public.

Do not open a public issue for a vulnerability. Follow the
[security policy](https://github.com/fus3r/train-guard/security/policy) and use
[GitHub private vulnerability reporting](https://github.com/fus3r/train-guard/security/advisories/new).

Train Guard controls processes. Reproduce ordinary bugs with a disposable,
non-critical job and a temporary `TRAIN_GUARD_HOME` whenever possible.
