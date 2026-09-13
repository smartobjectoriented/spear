"""Schema-versioned identities for local canonical technical standards."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


STANDARD_SCHEMA_VERSION = 1

# A v2 unit carries layout evidence. v1 units keep their exact v1 serialization so
# that an already-ingested corpus keeps its fingerprint and stays verifiable.

STANDARD_UNIT_SCHEMA_VERSION = 2
SUPPORTED_UNIT_SCHEMA_VERSIONS = (1, 2)
SUPPORTED_CORPUS_SCHEMA_VERSIONS = (1, 2)
INDEX_SCHEMA_VERSION = 1
DEFAULT_STANDARD_RETRIEVAL_CONFIGURATION = {
    "version": "standard-retrieval-policy-v1",
    "fusion": "rrf-v1", "rrf_constant": 60,
    "lexical_candidate_count": 20, "vector_candidate_count": 20,
    "final_top_k": 5, "neighbor_context_size": 1,
    "cross_reference_expansion_limit": 2,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^std-[0-9a-f]{32}$")


class StandardSchemaError(ValueError):
    pass


class StandardContentType(StrEnum):
    REQUIREMENT = "REQUIREMENT"
    RECOMMENDATION = "RECOMMENDATION"
    DEFINITION = "DEFINITION"
    INFORMATIVE = "INFORMATIVE"
    EXAMPLE = "EXAMPLE"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    FORMULA = "FORMULA"
    FRONT_MATTER = "FRONT_MATTER"
    PAGE_FURNITURE = "PAGE_FURNITURE"
    UNKNOWN = "UNKNOWN"


class StandardLayoutKind(StrEnum):
    """How a unit sat on the page, as evidence for a later geometry-aware pass."""

    PROSE = "PROSE"
    HEADING = "HEADING"
    COLUMNAR = "COLUMNAR"
    FURNITURE = "FURNITURE"
    TOC_ENTRY = "TOC_ENTRY"


class StandardModality(StrEnum):
    SHALL = "SHALL"
    SHOULD = "SHOULD"
    MAY = "MAY"
    MUST = "MUST"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class HumanValidationStatus(StrEnum):
    NOT_REVIEWED = "NOT_REVIEWED"
    PARTIALLY_REVIEWED = "PARTIALLY_REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


def canonical_json(value: object) -> bytes:
    """The one serialization every fingerprint in this system is taken over.

    Sorted keys and fixed separators make the bytes depend on the value alone,
    so a hash of them identifies content rather than a particular dump.
    """

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def source_content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_source_id(
    *, standard_id: str, revision: str, page: int, section: str | None,
    unit_position: int, text: str,
) -> str:
    """Return an index-independent ID whose identity includes canonical text.

    The text is part of the identity, so a re-extraction that changes what a
    unit SAYS produces a different id. Position is included as well, so two
    units with identical text on one page stay distinguishable.
    """

    identity = {
        "standard_id": standard_id,
        "revision": revision,
        "page": page,
        "section": section,
        "unit_position": unit_position,
        "source_content_sha256": source_content_sha256(text),
    }

    return "std-" + sha256_json(identity)[:32]


def _require_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StandardSchemaError(f"{name} must be a lowercase SHA-256")


def _require_optional_sha(value: str | None, name: str) -> None:
    if value is not None:
        _require_sha(value, name)


@dataclass(frozen=True)
class StandardBinding:
    standard_id: str
    revision: str
    pdf_sha256: str
    corpus_manifest_sha256: str
    index_fingerprint: str
    extractor_version: str
    corpus_schema_version: int = STANDARD_SCHEMA_VERSION
    bound_at: str | None = None
    data_origin: str = "LICENSED_STANDARD"
    vector_index_fingerprint: str | None = None
    cross_reference_index_fingerprint: str | None = None
    retrieval_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.pdf_sha256, "pdf_sha256"),
                            (self.corpus_manifest_sha256, "corpus_manifest_sha256"),
                            (self.index_fingerprint, "index_fingerprint")):
            _require_sha(value, name)

        _require_optional_sha(self.vector_index_fingerprint,
                              "vector_index_fingerprint")
        _require_optional_sha(self.cross_reference_index_fingerprint,
                              "cross_reference_index_fingerprint")
        _require_optional_sha(self.retrieval_fingerprint, "retrieval_fingerprint")

        if not self.standard_id or not self.revision or not self.extractor_version:
            raise StandardSchemaError("binding identifiers must not be empty")

        if self.corpus_schema_version not in SUPPORTED_CORPUS_SCHEMA_VERSIONS:
            raise StandardSchemaError("unsupported corpus schema version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardBinding":
        try:
            return cls(**{name: raw[name] for name in cls.__dataclass_fields__
                          if name in raw})
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(f"malformed standard binding: {exc}") from exc


_UNIT_SCHEMA_2_FIELDS = ("layout_kind", "retrievable", "is_front_matter",
                         "is_toc_entry", "possible_table_continuation",
                         "unit_position")


@dataclass(frozen=True)
class StandardDocumentUnit:
    source_id: str
    standard_id: str
    revision: str
    section: str | None
    page: int
    heading_path: tuple[str, ...]
    content_type: StandardContentType
    modality: StandardModality
    text: str
    source_pdf_sha256: str
    source_content_sha256: str
    extractor_version: str
    parent_source_id: str | None = None
    cross_references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    needs_review: bool = False
    needs_structured_review: bool = False
    schema_version: int = STANDARD_UNIT_SCHEMA_VERSION

    # -- schema 2: layout evidence for a later geometry-aware pass ------------

    layout_kind: StandardLayoutKind = StandardLayoutKind.PROSE
    retrievable: bool = True
    is_front_matter: bool = False
    is_toc_entry: bool = False
    possible_table_continuation: bool = False
    unit_position: int = 0

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise StandardSchemaError("malformed source_id")

        if self.parent_source_id is not None and not _SOURCE_ID.fullmatch(self.parent_source_id):
            raise StandardSchemaError("malformed parent_source_id")

        if self.page < 1 or not self.text.strip():
            raise StandardSchemaError("source unit needs a positive page and text")

        if self.schema_version not in SUPPORTED_UNIT_SCHEMA_VERSIONS:
            raise StandardSchemaError("unsupported source unit schema")

        # A v1 unit predates layout evidence, so carrying any non-default
        # layout field means the version and the content disagree.

        if self.schema_version < 2 and (
                self.layout_kind is not StandardLayoutKind.PROSE
                or not self.retrievable or self.is_front_matter
                or self.is_toc_entry or self.possible_table_continuation
                or self.unit_position):
            raise StandardSchemaError("layout evidence needs source unit schema 2")

        _require_sha(self.source_pdf_sha256, "source_pdf_sha256")
        _require_sha(self.source_content_sha256, "source_content_sha256")

        # The recorded hash must be a hash OF THIS text: a unit whose content
        # and fingerprint disagree would make the whole corpus unverifiable.

        if source_content_sha256(self.text) != self.source_content_sha256:
            raise StandardSchemaError("source content hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["content_type"] = self.content_type.value
        value["modality"] = self.modality.value
        value["layout_kind"] = self.layout_kind.value

        if self.schema_version < 2:
            # A v1 unit serializes exactly as it was stored, so its corpus
            # fingerprint survives this schema being extended.

            for name in _UNIT_SCHEMA_2_FIELDS:
                value.pop(name, None)

        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardDocumentUnit":
        try:
            value = dict(raw)
            value["content_type"] = StandardContentType(value["content_type"])
            value["modality"] = StandardModality(value["modality"])

            # An absent version means v1: the field was added with v2.

            value.setdefault("schema_version", 1)

            # A stored v1 unit is read back as v1, layout fields dropped, so it
            # keeps producing the fingerprint it was stored under.

            if int(value["schema_version"]) < 2:
                for name in _UNIT_SCHEMA_2_FIELDS:
                    value.pop(name, None)
            elif "layout_kind" in value:
                value["layout_kind"] = StandardLayoutKind(value["layout_kind"])

            for name in ("heading_path", "cross_references", "warnings"):
                value[name] = tuple(value.get(name, ()))

            return cls(**value)
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(f"malformed source unit: {exc}") from exc


@dataclass(frozen=True)
class StandardIngestionManifest:
    standard_id: str
    revision: str
    source_pdf_sha256: str
    logical_source_filename: str
    ingestion_timestamp: str
    extractor_version: str
    page_count: int
    canonical_unit_count: int
    section_count: int
    requirement_count: int
    recommendation_count: int
    definition_count: int
    table_figure_marker_count: int
    warnings: tuple[str, ...]
    extraction_errors: tuple[str, ...]
    human_validation_status: HumanValidationStatus
    corpus_manifest_sha256: str
    source_origin: str = "LICENSED_STANDARD"
    raw_pdf_retained: bool = False
    corpus_schema_version: int = STANDARD_SCHEMA_VERSION
    schema_version: int = STANDARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha(self.source_pdf_sha256, "source_pdf_sha256")
        _require_sha(self.corpus_manifest_sha256, "corpus_manifest_sha256")

        if self.page_count < 1 or self.canonical_unit_count < 1:
            raise StandardSchemaError("manifest must describe a non-empty document")

        if self.schema_version != STANDARD_SCHEMA_VERSION:
            raise StandardSchemaError("unsupported ingestion manifest schema")

        if self.corpus_schema_version not in SUPPORTED_CORPUS_SCHEMA_VERSIONS:
            raise StandardSchemaError("unsupported corpus schema version")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["human_validation_status"] = self.human_validation_status.value

        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardIngestionManifest":
        try:
            value = dict(raw)
            value["human_validation_status"] = HumanValidationStatus(
                value["human_validation_status"])
            value["warnings"] = tuple(value.get("warnings", ()))
            value["extraction_errors"] = tuple(value.get("extraction_errors", ()))

            return cls(**value)
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(f"malformed ingestion manifest: {exc}") from exc


@dataclass(frozen=True)
class StandardIndexManifest:
    standard_id: str
    revision: str
    source_pdf_sha256: str
    source_corpus_sha256: str
    index_type: str
    index_version: int
    indexer_version: str
    tokenizer_config_fingerprint: str
    index_fingerprint: str
    created_at: str
    schema_version: int = INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in ((self.source_pdf_sha256, "source_pdf_sha256"),
                            (self.source_corpus_sha256, "source_corpus_sha256"),
                            (self.tokenizer_config_fingerprint,
                             "tokenizer_config_fingerprint"),
                            (self.index_fingerprint, "index_fingerprint")):
            _require_sha(value, name)

        if self.index_type != "bm25" or self.index_version < 1:
            raise StandardSchemaError("unsupported lexical index")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardIndexManifest":
        try:
            return cls(**dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(f"malformed index manifest: {exc}") from exc


@dataclass(frozen=True)
class StandardVectorIndexManifest:
    standard_id: str
    revision: str
    source_pdf_sha256: str
    source_corpus_sha256: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_dimension: int
    normalization_policy: str
    vector_index_format: str
    vector_index_version: int
    indexed_source_count: int
    embedding_config_fingerprint: str
    vector_index_fingerprint: str
    created_at: str
    schema_version: int = INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_pdf_sha256, "source_pdf_sha256"),
            (self.source_corpus_sha256, "source_corpus_sha256"),
            (self.embedding_config_fingerprint, "embedding_config_fingerprint"),
            (self.vector_index_fingerprint, "vector_index_fingerprint"),
        ):
            _require_sha(value, name)

        if (not self.embedding_model_id or not self.embedding_model_revision
                or self.embedding_dimension < 1 or self.indexed_source_count < 1
                or self.normalization_policy != "l2-unit"
                or self.vector_index_format != "canonical-json-float-v1"
                or self.vector_index_version < 1
                or self.schema_version != INDEX_SCHEMA_VERSION):
            raise StandardSchemaError("unsupported vector index")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardVectorIndexManifest":
        try:
            return cls(**dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(f"malformed vector index manifest: {exc}") from exc


@dataclass(frozen=True)
class StandardCrossReferenceIndexManifest:
    standard_id: str
    revision: str
    source_pdf_sha256: str
    source_corpus_sha256: str
    resolver_version: str
    resolved_count: int
    ambiguous_count: int
    unresolved_count: int
    cross_reference_index_fingerprint: str
    created_at: str
    schema_version: int = INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha(self.source_pdf_sha256, "source_pdf_sha256")
        _require_sha(self.source_corpus_sha256, "source_corpus_sha256")
        _require_sha(self.cross_reference_index_fingerprint,
                     "cross_reference_index_fingerprint")

        if (not self.resolver_version or self.schema_version != INDEX_SCHEMA_VERSION
                or min(self.resolved_count,
                       self.ambiguous_count,
                       self.unresolved_count) < 0):
            raise StandardSchemaError("unsupported cross-reference index")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object],
    ) -> "StandardCrossReferenceIndexManifest":
        try:
            return cls(**dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardSchemaError(
                f"malformed cross-reference index manifest: {exc}") from exc


@dataclass(frozen=True)
class StandardCitation:
    standard_id: str
    revision: str
    section: str | None
    page: int
    source_id: str

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id) or self.page < 1:
            raise StandardSchemaError("malformed citation identity")

    def render(self) -> str:
        section = f" §{self.section}" if self.section else ""
        return (f"[{self.standard_id} {self.revision}{section}, p.{self.page}, "
                f"source {self.source_id}]")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "rendered": self.render()}
