---
name: c-readability
scope: [any]
---
## Procedure

To improve the quality of a C source file:

1. Read the whole file first.
2. Replace magic numbers with named constants from the relevant header.
3. Check every return value and handle the error.
4. Make file-local functions static, remove dead code and unused includes.
5. Keep each edit small and verify it in context.
