---
name: compile-failure
scope: [any]
---
## Procedure

When a build fails:

1. Read the first error in the build output, not the last.
2. Open the file and line it names; check the declaration it complains about.
3. For a link error, find which object should define the missing symbol.
4. Rebuild the smallest target that reproduces the failure.
