"""Human approval attaches to a provision, never to the unit that carried it.

One extracted unit of the bound standard carries a Permission, an Observation
and a Rule that share a section and an ordinal. A reviewer who approved the
Rule approved one of those three. Keying approval on the parent `source_id` --
or on the bare label both would render -- silently approves the other two.

So an approval names (section, kind, ordinal), and is checked against the
provision text as extracted TODAY. Re-extraction is a normal event and it can
change what a provision says; an approval whose text has moved underneath it
is stale, and saying so is the whole point of recording the hash.

Nothing here approves anything. It reports one of:

    HUMAN_APPROVED       the provision is present and unchanged
    STALE_APPROVAL       present, but its text no longer hashes the same
    EXTRACTION_MISMATCH  the approved provision is not in the store as keyed
    UNREVIEWED           no approval on record
"""

from __future__ import annotations

import json
import re

import provision_identity

HUMAN_APPROVED = "HUMAN_APPROVED"
STALE_APPROVAL = "STALE_APPROVAL"
EXTRACTION_MISMATCH = "EXTRACTION_MISMATCH"
UNREVIEWED = "UNREVIEWED"


def key_of(entry):
    """The evidence key an approval record names.

    Structural evidence first: a table row is identified by its table and its
    own number, a scope block by its section and a local discriminator.
    Neither is a ProvisionKey, and reducing them to one collapsed four
    approved rows of one table onto a single identity.
    """
    identity = entry.get("identity") or {}
    kind = str(identity.get("kind") or identity.get("provision_type") or "")

    if kind == provision_identity.TABLE_ROW and identity.get("table"):
        return provision_identity.TableRowKey(
            str(identity.get("section") or ""), str(identity["table"]),
            int(identity["row"]))

    if kind == provision_identity.SCOPE_PREAMBLE and identity.get("discriminator"):
        return provision_identity.ScopePreambleKey(
            str(identity.get("section") or ""), str(identity["discriminator"]))

    ordinal = identity.get("ordinal")

    if isinstance(ordinal, str) and not ordinal.isdigit():
        # "Table 8.3.1-1 bit 20" -- the row's own key is the ordinal. Reducing
        # it to None collapsed four approved rows onto one identity, which is
        # the very defect this module exists to prevent.
        digits = re.findall(r"\d+", ordinal)
        ordinal = digits[-1] if digits else None

    return provision_identity.ProvisionKey(
        str(identity.get("section") or ""),
        str(identity.get("kind") or identity.get("provision_type") or ""),
        int(ordinal) if ordinal is not None and str(ordinal).isdigit() else None)


def load(path):
    """Approvals as {ProvisionKey: record}, keyed by identity not by unit."""
    document = json.loads(path.read_text()) if hasattr(path, "read_text") else path
    found = {}

    for entry in document.get("approvals", []):
        found[key_of(entry)] = entry

    return found


def verify(approvals, records):
    """Each approval against the provisions as extracted now.

    `records` maps ProvisionKey -> ProvisionRecord, from the live store.
    """
    report = []

    for key, entry in sorted(approvals.items(), key=lambda item: str(item[0])):
        record = records.get(key)
        row = {"provision": str(key), "expected_source_id": entry.get("carrying_source_id"),
               "status": UNREVIEWED, "detail": ""}

        if record is None:
            row["status"] = EXTRACTION_MISMATCH
            row["detail"] = "no provision with this identity in the store"
        elif entry.get("carrying_source_id") and record.source_id != entry["carrying_source_id"]:
            row["status"] = EXTRACTION_MISMATCH
            row["detail"] = f"carried by {record.source_id}, approval names " \
                            f"{entry['carrying_source_id']}"
        elif entry.get("unit_text_sha256") and \
                entry["unit_text_sha256"] != record.unit_text_sha256:
            row["status"] = STALE_APPROVAL
            row["detail"] = "the unit text has changed since approval"
        else:
            row["status"] = HUMAN_APPROVED

        report.append(row)

    return report


def status_of(key, approvals, records):
    """One provision's status. Never inferred from a neighbour."""
    if key not in approvals:
        return UNREVIEWED

    return next(row["status"] for row in verify({key: approvals[key]}, records))
