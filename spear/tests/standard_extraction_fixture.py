"""Synthetic PDFs reproducing real technical-standard layout defects.

Every page is authored as explicitly positioned text so that ``pdftotext -layout``
reconstructs genuine columns, dot leaders and page furniture. No licensed text.
"""

from __future__ import annotations


def positioned_pdf_bytes(pages: tuple[tuple[tuple[float, float, str], ...], ...]) -> bytes:
    """Build a deterministic PDF from explicit (x, y, text) placements."""
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + index * 2} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, placements in enumerate(pages):
        content_object = 4 + index * 2 + 1
        objects.append((
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object} 0 R >>"
        ).encode())
        commands = [b"BT /F1 10 Tf"]
        for x, y, raw in placements:
            escaped = raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"1 0 0 1 {x} {y} Tm ({escaped}) Tj".encode())
        commands.append(b"ET")
        stream = b"\n".join(commands)
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                       + stream + b"\nendstream")
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, content in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(content)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend((f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                   f"startxref\n{xref}\n%%EOF\n").encode())
    return bytes(output)


def _furniture(page: int) -> tuple[float, float, str]:
    """Repeated page furniture: identical on every page but for the page number."""
    return (72, 40, f"Synthetic Consortium - TEST_FIXTURE / SYNTHETIC - page {page}")


_ROW_X = (72, 230, 400)


def _row(y: float, cells: tuple[str, str, str]) -> tuple[tuple[float, float, str], ...]:
    return tuple((x, y, cell) for x, cell in zip(_ROW_X, cells))


EXTRACTION_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    # 1 -- title / front matter
    ((72, 700, "Synthetic Structural Standard"),
     (72, 660, "Revision R1 - TEST_FIXTURE / SYNTHETIC"),
     _furniture(1)),
    # 2 -- table of contents with dot leaders
    ((72, 720, "Table of Contents"),
     (72, 680, "1 Introduction ....................................... 3"),
     (72, 660, "1.1 Scope ............................................ 3"),
     (72, 640, "2 Packet Format ...................................... 4"),
     (72, 620, "2.3.1 Field Rules .................................... 5"),
     (72, 600, "3 Timing ............................................. 6"),
     (72, 580, "4 Encoding Notes ..................................... 7"),
     _furniture(2)),
    # 3 -- genuine headings plus numeric body sentences
    ((72, 720, "1 Introduction"),
     (72, 690, "This clause introduces the synthetic interchange format."),
     (72, 640, "1.1 Scope"),
     (72, 610, "2015 devices were tested during the qualification trial."),
     (72, 590, "187 octets are required for the extended header form."),
     (72, 570, "102.4 MHz is the nominal sampling rate for this profile."),
     _furniture(3)),
    # 4 -- heading, table caption, columnar rows running to the page bottom
    ((72, 720, "2 Packet Format"),
     (72, 680, "Table 1 Reserved Encodings"),
     *_row(650, ("00000100", "Reserved", "4")),
     *_row(630, ("1111", "Invalid", "15")),
     *_row(610, ("255", "Maximum", "0xFF")),
     *_row(590, ("65536", "Size", "bytes")),
     *_row(570, ("0001", "Nominal", "1")),
     _furniture(4)),
    # 5 -- the same table continuing with no caption, then a real subheading
    (*_row(720, ("0010", "Continued", "7")),
     *_row(700, ("0011", "Extended", "8")),
     *_row(680, ("0100", "Deferred", "9")),
     (72, 620, "2.3.1 Field Rules"),
     (72, 590, "Implementations shall reject a reserved encoding."),
     _furniture(5)),
    # 6 -- numbered list items that are not headings, and an indented paragraph
    ((72, 720, "3 Timing"),
     (72, 690, "The following conditions apply in order:"),
     (90, 660, "1 apples are counted before oranges"),
     (90, 640, "2 oranges are counted before pears"),
     (90, 620, "3 pears are counted last of all"),
     (108, 570, "An ordinary indented paragraph that carries no columns at all "
                "and must not be mistaken for a table row."),
     _furniture(6)),
    # 7 -- annex handling
    ((72, 720, "Annex A Informative Material"),
     (72, 680, "This annex is informative and adds no requirements."),
     (72, 640, "4 Encoding Notes"),
     (72, 610, "Refer to Section 2.3.1 for the normative rule."),
     _furniture(7)),
)


def extraction_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(EXTRACTION_PAGES)


# ---------------------------------------------------------------------------
# The pre-STD1E extractor, kept so a repair can be demonstrated against it.
# ---------------------------------------------------------------------------

_LEGACY_SECTION = __import__("re").compile(r"^\s*(\d+(?:\.\d+)*)\s+(.\S.*)$")


def legacy_canonical_units(pages, *, standard_id, revision, pdf_sha256):
    """Reproduce ``poppler-layout-v1``: any digit-led line opened a section."""
    import re

    from standard_schema import (
        StandardContentType, StandardDocumentUnit, StandardModality,
        make_source_id, source_content_sha256,
    )

    table = re.compile(r"^\s*(?:table|tab\.)\s+[A-Za-z0-9.-]+\b", re.I)
    figure = re.compile(r"^\s*(?:figure|fig\.)\s+[A-Za-z0-9.-]+\b", re.I)

    def paragraphs(page):
        result, pending = [], []
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                if pending:
                    result.append(" ".join(pending))
                    pending = []
                continue
            if _LEGACY_SECTION.match(line) or table.match(line) or figure.match(line):
                if pending:
                    result.append(" ".join(pending))
                pending = [line]
            else:
                pending.append(line)
        if pending:
            result.append(" ".join(pending))
        return tuple(value for value in result if value.strip())

    units, position, section = [], 0, None
    for page_number, page in enumerate(pages, 1):
        for text in paragraphs(page):
            position += 1
            heading = _LEGACY_SECTION.match(text)
            if heading:
                section = heading.group(1)
            structured = bool(table.search(text) or figure.search(text))
            units.append(StandardDocumentUnit(
                source_id=make_source_id(
                    standard_id=standard_id, revision=revision, page=page_number,
                    section=section, unit_position=position, text=text),
                standard_id=standard_id, revision=revision, section=section,
                page=page_number, heading_path=(),
                content_type=(StandardContentType.TABLE if structured
                              else StandardContentType.UNKNOWN),
                modality=StandardModality.NONE, text=text,
                source_pdf_sha256=pdf_sha256,
                source_content_sha256=source_content_sha256(text),
                extractor_version="poppler-layout-v1",
                needs_review=structured, needs_structured_review=structured,
                schema_version=1))
    return tuple(units)


# A table whose first column holds numbers that would continue the clause tree,
# set far right under a column header, plus an annex that renumbers from one.
WIDE_COLUMN_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 720, "1     Scope"),
     (72, 680, "1.1     Terms and Definitions"),
     (72, 650, "This clause carries the body definitions for the format."),
     (72, 40, "Synthetic Consortium - TEST_FIXTURE - page 1")),
    ((72, 720, "2     Encoding"),
     (72, 680, "Table 1 Reserved Encodings"),
     (72, 650, "3"), (400, 650, "Reserved for future use"),
     (72, 630, "4"), (400, 630, "Invalid in this revision"),
     (72, 610, "5"), (400, 610, "Deprecated in this revision"),
     (72, 40, "Synthetic Consortium - TEST_FIXTURE - page 2")),
    ((72, 720, "3     Timing"),
     (72, 680, "The timing rules are normative for every implementation."),
     (72, 40, "Synthetic Consortium - TEST_FIXTURE - page 3")),
    ((72, 720, "Annex A Extra Material"),
     (72, 680, "1.1     Annex Subclause"),
     (72, 650, "This annex text renumbers from one in its own namespace."),
     (72, 40, "Synthetic Consortium - TEST_FIXTURE - page 4")),
)


def wide_column_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(WIDE_COLUMN_PAGES)


# A contents table that names each clause's page, and a table row that claims a
# clause number earlier -- and more plausibly -- than the real heading does.
CONTENTS_TARGET_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 720, "Table of Contents"),
     (72, 690, "1 Scope ................................. 1"),
     (72, 670, "2 Encoding .............................. 2"),
     (72, 650, "3 Framing ............................... 2"),
     (72, 630, "6 Limits ................................ 2"),
     (72, 610, "7 Timing ................................ 4")),
    ((72, 720, "1     Scope"),
     (72, 690, "This clause states what the format covers.")),
    ((72, 720, "2     Encoding"),
     (72, 690, "This clause states how values are encoded."),
     (72, 650, "3     Framing"),
     (72, 620, "This clause states how frames are delimited."),
     (72, 580, "6     Limits"),
     (72, 550, "This clause states the permitted limits."),
     (72, 510, "7   Reserved value")),
    ((72, 720, "Filler body text that opens no clause at all."),),
    ((72, 720, "7     Timing"),
     (72, 690, "This clause states the normative timing rules.")),
)


def contents_target_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(CONTENTS_TARGET_PAGES)


# Captions beside sentences that merely refer to them, definitions beside the
# noun "means", and normative paragraphs the document labels for itself.
CLASSIFIER_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 730, "4     Encoding"),
     (72, 700, "Table 4-1: Reserved Encodings"),
     (72, 670, "Table 4-1 lists the reserved encodings used throughout this document."),
     (72, 640, "Figure 4-2: Packet Layout"),
     (72, 610, "Figure 4-2 shows how the packet is laid out across the link."),
     (72, 570, "Definition 4-1: A Frame is a bounded sequence of samples."),
     (72, 540, "reserved value means a value unavailable for ordinary data."),
     (72, 510, "The encoder provides a standard means of conveying the sample rate."),
     (72, 480, "Rule 4-2: Implementations shall reject a reserved encoding."),
     (72, 450, "Observation 4-3: Every implementation shall meet this case in practice."),
     (72, 420, "Permission 4-4: An implementation may omit the trailer."),
     (72, 390, "Recommendation 4-5: Implementations should log rejected values."),
     (72, 360, "The words \"shall\" and \"must\" are reserved for stating rules here.")),
)


def classifier_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(CLASSIFIER_PAGES)


# Front matter whose boilerplate uses modal verbs, a body clause that genuinely
# requires something, and an annex opened in the body as well as listed up front.
SEMANTIC_PAGES: tuple[tuple[tuple[float, float, str], ...], ...] = (
    ((72, 730, "Table of Contents"),
     (72, 700, "1 Scope ................................. 2"),
     (72, 680, "2 Encoding .............................. 2"),
     (72, 660, "3 Framing ............................... 2"),
     (72, 640, "4 Limits ................................ 2"),
     (72, 620, "Annex A Worked Example .................. 3"),
     (72, 560, "No person shall have authority to issue an interpretation of this "
               "document, requests must be addressed to the committee, and members "
               "may respond at their discretion.")),
    ((72, 730, "1     Scope"),
     (72, 700, "Implementations shall reject reserved values."),
     (72, 670, "2     Encoding"),
     (72, 640, "Receivers must reject malformed packets."),
     (72, 610, "3     Framing"),
     (72, 580, "Implementations should retain the identifier."),
     (72, 550, "4     Limits"),
     (72, 520, "Implementations may omit the optional field."),
     (72, 490, "Transport Identifier means the identifier carried in the header."),
     (72, 460, "This approach means that the receiver discards the frame."),
     (72, 430, "Control data is carried by means of an auxiliary channel."),
     (72, 400, "Rule 4-1: The value shall be encoded. Rule 4-2: The other value "
               "shall also be encoded.")),
    ((72, 730, "Annex A Worked Example"),
     (72, 700, "This annex walks through a complete exchange.")),
)


def semantic_pdf_bytes() -> bytes:
    return positioned_pdf_bytes(SEMANTIC_PAGES)
