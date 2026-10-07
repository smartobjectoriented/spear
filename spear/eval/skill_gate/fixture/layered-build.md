---
name: layered-build
scope: [firmware-a, firmware-b]
---
## Procedure

Debugging the layered build used by these firmware trees:

1. Read the task log of the failing recipe before changing anything.
2. Recipes live in `build/meta-*/recipes-*/`, configuration in `build/conf/`.
3. Platform patches are selected per machine; check the one for this board.
4. Clean the recipe and rebuild it to confirm the fix.
