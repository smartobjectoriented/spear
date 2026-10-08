"""A synthetic standard for MIXED turns: one conditional requirement, one
counted requirement, one permission and one informative observation."""

from __future__ import annotations

from tests.standard_fixture import synthetic_pdf_bytes

STANDARD_ID, REVISION = "SYNTH-MIXED", "M-1"

PAGES = (
    (
        (760, "Synthetic Record Format"),
        (738, "Revision M-1 - TEST_FIXTURE / SYNTHETIC"),
        (700, "4 Record Fields"),
        (670, "4.2 Header Fields"),
        (640, "Rule 4.2.1-1: The mode field shall be 2 when flag X is set."),
        (610, "Rule 4.2.1-2: The count field shall contain exactly four entries."),
        (580, "Permission 4.2.1-3: The metadata field may be absent."),
        (550, "Observation 4.2.1-4: Older readers ignore the metadata field."),
    ),
)


def mixed_pdf_bytes() -> bytes:
    return synthetic_pdf_bytes(PAGES)


def build_store(root):
    """A bound store for the fixture, with its lexical and cross-reference indexes."""
    from standard.standard_crossrefs import rebuild_cross_reference_index
    from standard.standard_ingest import ingest_pdf
    from standard.standard_retrieval import rebuild_lexical_index
    from standard.standard_store import StandardStore

    pdf = root / "mixed.pdf"
    pdf.write_bytes(mixed_pdf_bytes())
    store = StandardStore(root / "standards")
    ingest_pdf(store, pdf, standard_id=STANDARD_ID, revision=REVISION,
               source_origin="TEST_FIXTURE")
    rebuild_lexical_index(store, STANDARD_ID, REVISION)
    rebuild_cross_reference_index(store, STANDARD_ID, REVISION)

    return store
