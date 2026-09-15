"""Human corrections to extracted metadata, kept apart from the extraction.

The extractor's content_type and the provision's own words disagree on 24
units of one bound standard: text saying `shall` stored as INFORMATIVE, a
Permission stored inside a unit typed REQUIREMENT. The rule for answering is
already settled and does not change here -- the provision's WORDS are
authoritative, the stored metadata is advisory -- but a reviewer who has read
the page should be able to record what is actually true without anyone editing
the licensed extraction to do it.

So corrections live beside the extraction, never in it:

    extracted metadata  +  human correction overlay  ->  effective metadata

A correction names the provision by identity, carries the reviewer, the date,
the hash of the text it was made against, and a reason. It goes stale the
moment that text changes, because a correction to something that has since
been re-extracted is an opinion about a different document.

Nothing here writes corrections. An empty overlay is the normal state, and
`effective` then returns exactly what the extraction said.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import provision_identity

ACTIVE = "ACTIVE"
STALE = "STALE"
UNKNOWN_PROVISION = "UNKNOWN_PROVISION"


@dataclass(frozen=True)
class Correction:
    key: provision_identity.ProvisionKey
    field: str
    value: str
    reviewer: str
    reviewed_at: str
    provision_text_sha256: str
    reason: str

    def to_dict(self):
        return {"identity": {"section": self.key.section, "kind": self.key.kind,
                             "ordinal": self.key.ordinal},
                "field": self.field, "value": self.value,
                "reviewer": self.reviewer, "reviewed_at": self.reviewed_at,
                "provision_text_sha256": self.provision_text_sha256,
                "reason": self.reason}


#: Only metadata may be corrected. The text itself is the licensed document
#: and is never overlaid -- a correction that could rewrite a clause would be
#: a way of answering from something the standard does not say.
CORRECTABLE = frozenset({"content_type", "modality"})


def load(document):
    """Corrections from a parsed overlay document, or an empty overlay."""
    if hasattr(document, "read_text"):
        document = json.loads(document.read_text())

    found = {}

    for entry in (document or {}).get("corrections", []):
        if entry.get("field") not in CORRECTABLE:
            continue

        identity = entry.get("identity") or {}
        key = provision_identity.ProvisionKey(
            str(identity.get("section") or ""), str(identity.get("kind") or ""),
            identity.get("ordinal"))
        found.setdefault(key, []).append(Correction(
            key=key, field=str(entry["field"]), value=str(entry.get("value") or ""),
            reviewer=str(entry.get("reviewer") or ""),
            reviewed_at=str(entry.get("reviewed_at") or ""),
            provision_text_sha256=str(entry.get("provision_text_sha256") or ""),
            reason=str(entry.get("reason") or "")))

    return found


def status_of(correction, record):
    if record is None:
        return UNKNOWN_PROVISION

    if (correction.provision_text_sha256
            and correction.provision_text_sha256 != record.text_sha256):
        return STALE

    return ACTIVE


def effective(record, overlay):
    """The metadata to act on: extraction, plus any live correction.

    A stale correction is ignored and reported elsewhere; it is never silently
    applied, because the text it was written against is gone.
    """
    values = {"content_type": record.content_type, "modality": record.modality}

    for correction in overlay.get(record.key, ()):
        if status_of(correction, record) == ACTIVE:
            values[correction.field] = correction.value

    return values


def report(overlay, records):
    """Every correction with its status. Empty overlay, empty report."""
    return [{"provision": str(key), "field": correction.field,
             "value": correction.value, "reviewer": correction.reviewer,
             "status": status_of(correction, records.get(key))}
            for key, corrections in sorted(overlay.items(), key=lambda i: str(i[0]))
            for correction in corrections]
