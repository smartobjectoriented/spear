---
name: buildlog-triage
scope: [any]
---
## Procedure

Debug a build-system failure systematically:

1. Read the failing task's log under `build/tmp/work/<recipe>/temp/log.do_<task>.<pid>`.
2. Find the exact failure point: a missing file, a failed command, a syntax error.
3. Check that the recipe's patches applied in the work directory.
4. Look for missing dependencies or wrong paths, fix one at a time, and rebuild.
