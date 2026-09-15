"""Where this installation keeps what a session accumulates.

One answer, in one place. Five call sites used to carry their own copy of it,
and when the directory was renamed they did not all move: three were fallbacks
that only mattered when SPEAR_STATE_DIR was unset, and two were unconditional,
so a test silently skipped against a store that was right there under its new
name. A constant repeated five times is a constant that will be wrong in four
of them.

The contract is the environment variable; the default is where a workstation
puts it when nobody has said otherwise.

rag_chat and backend_select deliberately do NOT use this: they fall back to the
application directory rather than to the home directory, because a container
mounts the app and has no home worth writing to. That is a different default
for a different reason, not an oversight.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where a workstation accumulates when SPEAR_STATE_DIR says nothing.
DEFAULT = Path.home() / ".local" / "state" / "spear"


def state_dir() -> Path:
    """The state root: SPEAR_STATE_DIR, else the workstation default."""
    configured = os.environ.get("SPEAR_STATE_DIR")

    return Path(configured) if configured else DEFAULT


def standards_root() -> Path:
    """The normative-standard store inside it.

    Named here rather than joined at each call site: the four readers of this
    directory should not each decide what it is called.

    SPEAR_STANDARDS_ROOT moves the store WITHOUT moving anything else a
    session accumulates. That is what an extraction evaluation needs: a
    second corpus of the same document, read by the same code, with the
    production store untouched and not even opened. Rolling back is not
    setting the variable.
    """
    configured = os.environ.get("SPEAR_STANDARDS_ROOT")

    return Path(configured) if configured else state_dir() / "standards"
