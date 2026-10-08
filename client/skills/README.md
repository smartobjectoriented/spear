# skills/

Procedures offered to every turn in a workspace their scope admits; the model
follows one when it fits the task. A skill is a Markdown file with optional
front matter:

```markdown
---
name: short-kebab-case-name
description: one line saying what it is for
scope: [project-name]       # required: the projects it serves, or [any]
version: 1
---

## Procedure

1. ...
```

`scope` names the registered projects a skill serves; `[any]` makes it generic
and must be written. A skill without one is offered nowhere. Skills are read
from this directory on every turn, so editing one here is enough — there is no
separate registration step.

This directory ships empty. A skill describes a particular codebase, build
system or standard, so it belongs with that material rather than with the
platform: keep yours outside the repository and point SPEAR at them, or track
them in your own downstream.
