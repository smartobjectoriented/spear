"""Permissively generated TEST_FIXTURE/SYNTHETIC PDF for STD0 tests."""

from __future__ import annotations


PAGES = (
    (
        (760, "Synthetic Interchange Standard"),
        (738, "Revision TEST-1 - TEST_FIXTURE / SYNTHETIC"),
        (700, "3 General"),
        (670, "3.1 Reserved Values"),
        (640, "Implementations shall reject reserved value 0xF; see 3.2."),
        (600, "reserved value means a value unavailable for ordinary data."),
    ),
    (
        (760, "3.2 Informative Example"),
        (730, "Informative example: value 0x1 is accepted."),
        (690, "Implementations should log rejected values."),
        (650, "Table 1 Fake Values"),
        (625, "Value  Meaning"),
        (605, "0xF    Reserved"),
    ),
)


def synthetic_pdf_bytes(pages=PAGES) -> bytes:
    """Build a tiny deterministic PDF without a test-only PDF dependency."""
    objects: list[bytes] = []
    page_count = len(pages)
    # Object 1: catalog; object 2: pages; object 3: shared Helvetica font.
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + index * 2} 0 R" for index in range(page_count))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, lines in enumerate(pages):
        page_object = 4 + index * 2
        content_object = page_object + 1
        objects.append((
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object} 0 R >>"
        ).encode())
        commands = [b"BT /F1 11 Tf"]
        for y, raw_text in lines:
            escaped = raw_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"1 0 0 1 72 {y} Tm ({escaped}) Tj".encode())
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
