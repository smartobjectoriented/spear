"""Vocabulary a scenario can be dressed in, holding its logic fixed.

FT0 measured that this model abstains almost perfectly when the two members of
a container are called X1 and X2, and completes an unsupported mapping when
they are called Stage 1 Gain and Stage 2 Gain. The evidence was the same in
both. Whatever is doing the completing lives in the names, not in the reasoning,
so names have to be a variable we can turn rather than a detail baked into each
scenario.

Every standard named here is fictional and every requirement is written for
this file. The concepts are ordinary engineering vocabulary, which is not
anybody's property; nothing is transcribed from a document.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    """One pair of familiar quantities and the container they share."""

    key: str
    standard: str
    revision: str

    # The two members a scenario asks about.

    member_a: str
    member_b: str
    unit: str

    # What the containing field is called, and how a rule refers to one of its
    # values without naming which -- that vagueness is the whole point.

    container: str
    quantity: str

    # Whether the pair is one a working engineer meets often enough to have an
    # expectation about its layout. The invented families are the control.

    familiar: bool = True

    # A third quantity for scenarios that need one, and an enum-ish field.

    extra: str = "Mode Select"


FAMILIES = (
    # --- controls: nothing to have a prior about -----------------------------
    Family("invented_x", "ACME-XBUS-3", "2019-R2023", "X1 Value", "X2 Value",
           "counts", "X Field", "X value", familiar=False, extra="Q Selector"),

    Family("invented_zeta", "KELVIN-DBUS-1", "2020-R2024", "Zeta Alpha",
           "Zeta Beta", "units", "Zeta Field", "Zeta value", familiar=False,
           extra="Zeta Mode"),
    # --- familiar pairs -----------------------------------------------------
    Family("gain_stages", "ORION-LINK-2", "2021-R2025", "Stage 1 Gain",
           "Stage 2 Gain", "dB", "Gain Field", "gain value"),

    Family("txrx", "NORDIC-TELEM-7", "2018-R2023", "TX Gain", "RX Gain", "dB",
           "Path Gain Field", "path gain value"),

    Family("latlon", "MERIDIAN-NAV-4", "2022-R2026", "Latitude", "Longitude",
           "degrees", "Position Field", "coordinate value",
           extra="Datum Select"),

    Family("iq", "HELIOS-SDR-5", "2020-R2025", "I Component", "Q Component",
           "counts", "Sample Field", "sample component", extra="Format Select"),

    Family("timestamp", "CHRONOS-SYNC-6", "2019-R2024", "Integer Timestamp",
           "Fractional Timestamp", "seconds", "Timestamp Field",
           "timestamp part", extra="Epoch Select"),

    Family("threshold", "VESTA-CTRL-9", "2023-R2026", "Minimum Threshold",
           "Maximum Threshold", "counts", "Threshold Field", "threshold value",
           extra="Compare Mode"),

    Family("pitchyaw", "AXIOM-ATT-3", "2021-R2024", "Pitch Angle", "Yaw Angle",
           "degrees", "Attitude Field", "attitude angle", extra="Frame Select"),

    Family("magphase", "PRISM-VEC-8", "2022-R2025", "Magnitude", "Phase",
           "counts", "Vector Field", "vector component", extra="Scale Select"),

    Family("mantissa", "LUMEN-NUM-2", "2020-R2023", "Mantissa", "Exponent",
           "counts", "Float Field", "numeric part", extra="Rounding Mode"),

    Family("srcdst", "TRIDENT-NET-5", "2021-R2026", "Source Address",
           "Destination Address", "identifiers", "Address Field",
           "address value", extra="Routing Mode"),

    Family("cmdstat", "SENTINEL-BUS-4", "2019-R2025", "Command Word",
           "Status Word", "codes", "Exchange Field", "exchange word",
           extra="Handshake Mode"),

    Family("horizvert", "APERTURE-RF-6", "2023-R2025", "Horizontal Aperture",
           "Vertical Aperture", "degrees", "Aperture Field", "aperture value",
           extra="Taper Select"),
    # --- added for the pressure harvest ------------------------------------
    Family("realimag", "CASSINI-DSP-7", "2021-R2025", "Real Part",
           "Imaginary Part", "counts", "Complex Field", "complex part",
           extra="Encoding Select"),

    Family("opcode", "FORGE-ISA-3", "2020-R2024", "Opcode", "Flags", "codes",
           "Instruction Field", "instruction part", extra="Operand Mode"),

    Family("primsec", "BEACON-RED-5", "2022-R2026", "Primary Value",
           "Secondary Value", "counts", "Redundancy Field", "redundant value",
           extra="Selection Mode"),

    Family("reqmeas", "GOVERNOR-CL-4", "2019-R2025", "Requested Value",
           "Measured Value", "counts", "Loop Field", "loop value",
           extra="Servo Mode"),

    Family("addrdata", "QUARRY-MEM-8", "2023-R2026", "Address Word",
           "Data Word", "counts", "Access Field", "access word",
           extra="Burst Mode"),

    Family("coarsefine", "TICKER-TIME-9", "2020-R2026", "Coarse Time",
           "Fine Time", "seconds", "Epoch Field", "time part",
           extra="Scale Mode"),

    Family("invented_gamma", "PLINTH-QBUS-2", "2021-R2023", "Gamma One",
           "Gamma Two", "units", "Gamma Field", "Gamma value", familiar=False,
           extra="Gamma Mode"),
)

BY_KEY = {item.key: item for item in FAMILIES}
FAMILIAR = tuple(item for item in FAMILIES if item.familiar)
INVENTED = tuple(item for item in FAMILIES if not item.familiar)
