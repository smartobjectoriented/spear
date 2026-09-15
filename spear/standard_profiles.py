"""What the provision roles of a particular document mean.

The guard is generic: it knows that a role has a ceiling and that a modal
verb is evidence only up to it. WHICH ceiling belongs to WHICH role is a
statement about one document's drafting conventions, and it belongs here
rather than inside the guard.

Most specification-style documents follow the conventional reading, which is
why `normative_force.CONVENTIONAL` is the default. A profile is written when
a document's own front matter says something different, or -- as below -- when
it is worth recording explicitly that the conventional reading was checked
against the document rather than assumed.
"""

from __future__ import annotations

import normative_force as force

#: ANSI/VITA 49.2. Section 1.5 of the document sets out its own taxonomy:
#: Rules are mandatory, Recommendations are strongly advised but optional,
#: Permissions are allowances, and Observations and Definitions explain --
#: they do not oblige. That last line is the one that matters here: the
#: document contains Observations whose prose uses "must" while describing an
#: obligation a Rule imposes, and reading those as binding evidence made a
#: turn conclude that the Observation itself was the requirement.
VITA_49_2 = force.RoleTaxonomy(
    name="ANSI/VITA 49.2 §1.5",
    ceilings={
        force.RULE: force.REQUIREMENT_FORCE,
        force.REQUIREMENT: force.REQUIREMENT_FORCE,
        force.RECOMMENDATION: force.RECOMMENDATION_FORCE,
        force.PERMISSION: force.PERMISSION_FORCE,
        force.OBSERVATION: force.INFORMATIVE_FORCE,
        force.DEFINITION: force.INFORMATIVE_FORCE,
        force.INFORMATIVE: force.INFORMATIVE_FORCE,
        # Normative prose the document did not number, and its tables. The
        # bit-assignment tables ARE the requirement a Rule points at, so a
        # ceiling here would discard them.
        force.UNLABELLED: force.REQUIREMENT_FORCE,
    },
    default_ceiling=force.REQUIREMENT_FORCE)


def install():
    """Register every declared profile. Idempotent."""
    force.register_taxonomy("ANSI-VITA-49.2", VITA_49_2)


install()
