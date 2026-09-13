# skills/

Procedures the assistant retrieves by similarity and injects when a turn looks
like one they cover. A skill is a Markdown file with optional front matter:

```markdown
---
name: short-kebab-case-name
description: one line, used to decide relevance
scope: [corpus-name]        # optional: restrict to one corpus
version: 1
---

## Procedure

1. ...
```

`scope` limits a skill to the corpora that list it; a skill without one is
offered everywhere. `skill_library` re-embeds a file whose digest changes, so
editing one here is enough — there is no separate registration step.

This directory ships empty. A skill describes a particular codebase, build
system or standard, so it belongs with that material rather than with the
platform: keep yours outside the repository and point SPEAR at them, or track
them in your own downstream.
