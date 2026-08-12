# Security policy

Only the latest public Train Guard release receives security fixes. Upgrade to
the current version before reporting a problem that may already be fixed.

## Report a vulnerability privately

Use
[GitHub private vulnerability reporting](https://github.com/fus3r/train-guard/security/advisories/new).
Do not open a public issue for a vulnerability.

Useful reports identify the affected version and operating system, the
supported command or state transition involved, the impact and the smallest
safe reproduction. Remove tokens, environment variables, user paths, private
commands and journal content that is not needed for the report.

Relevant security problems include controlling the wrong process, bypassing
process identity or ownership checks, unsafe state-file permissions, loss of
recovery evidence and release workflow or package publishing weaknesses.

If a report involves live process control, stop using the affected path on
important workloads. Reproduce with a disposable process and an isolated
`TRAIN_GUARD_HOME` when it is safe to do so.

Train Guard is a workload policy, not a hardware safety controller. Hardware
temperature, charging and electrical protection remain the responsibility of
the operating system and firmware.
