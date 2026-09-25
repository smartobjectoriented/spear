"""Strict operator-only ``/standard`` control plane."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from standard_ingest import ingest_candidate, ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_crossrefs import rebuild_cross_reference_index
from standard_vector_index import (
    LICENSED_ORIGIN, configured_embedder, offload_disclosures,
    record_offload_disclosure, rebuild_vector_index,
)
from standard_schema import StandardBinding, canonical_json
from standard_store import (
    INDEX_STATE_READY, RETRIEVAL_MODES, StandardStore, StandardStoreError,
)


class StandardCommandError(ValueError):
    pass


class StandardOperator:
    """Owns ingestion and binding; intentionally never registered as a model tool."""

    def __init__(self, store: StandardStore) -> None:
        self.store = store
        self._active_path = store.root / ".active-binding.json"

    def active_binding(self) -> StandardBinding | None:
        if not self._active_path.exists():
            return None

        try:
            raw = json.loads(self._active_path.read_text("utf-8"))

            if raw.get("active") is None:
                return None

            persisted = StandardBinding.from_dict(raw["active"])
            current = self.store.binding(persisted.standard_id, persisted.revision,
                                         bound_at=persisted.bound_at)
        except Exception as exc:
            raise StandardCommandError(f"active standard binding is invalid: {exc}") from exc

        # The binding names the exact bytes it was made against. If the corpus
        # underneath moved, silently rebinding would hide a source change.

        if (persisted.pdf_sha256 != current.pdf_sha256
                or persisted.corpus_manifest_sha256 != current.corpus_manifest_sha256):
            raise StandardCommandError("active standard canonical source changed; rebind refused")

        # Indexes may legitimately be rebuilt over an unchanged corpus, so a
        # moved retrieval fingerprint is refreshed in place rather than refused.

        if ((persisted.retrieval_fingerprint or persisted.index_fingerprint)
                != (current.retrieval_fingerprint or current.index_fingerprint)):
            self._write_active(current)

        return current

    def use(self, standard_id: str, revision: str) -> StandardBinding:
        binding = self.store.binding(
            standard_id, revision,
            bound_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._write_active(binding)

        return binding

    def stale_binding_status(self, exc: StandardCommandError) -> str:
        """Explain a binding the store can no longer honour, without guessing."""

        lines = ["Active standard: unavailable", f"Reason: {exc}"]

        # The binding file is already known to be unusable; if it cannot even be
        # parsed there is nothing further to report.

        try:
            raw = json.loads(self._active_path.read_text("utf-8"))
            persisted = StandardBinding.from_dict(raw["active"])
        except Exception:
            return "\n".join(lines)

        lines.append(f"Bound: {persisted.standard_id} {persisted.revision}")
        lines.append(f"Bound corpus: {persisted.corpus_manifest_sha256}")

        # Whatever the store still holds for that revision is best-effort
        # context for the operator, so a second failure stays silent.

        try:
            current = self.store.load_manifest(persisted.standard_id, persisted.revision)

            lines.append(f"Active corpus: {current.corpus_manifest_sha256}")
            lines.append(f"Extractor: {current.extractor_version}")
            lines.append("Retrieval indexes: " + self.store.index_state(
                persisted.standard_id, persisted.revision))
            lines.append(
                f"Recover with: /standard rebuild {persisted.standard_id} "
                f"{persisted.revision} then /standard use "
                f"{persisted.standard_id} {persisted.revision}")
        except Exception:
            pass

        return "\n".join(lines)

    def candidate_summary(self, standard_id: str, revision: str) -> str:
        names = self.store.list_candidates(standard_id, revision)

        if not names:
            return "Candidates: none"

        return "Candidates: " + ", ".join(names) + " (awaiting operator review)"

    def candidate_report(self, standard_id: str, revision: str,
                         candidate_id: str) -> str:
        """A metadata-only OLD vs CANDIDATE summary for one stored candidate."""

        try:
            manifest, _ = self.store.load_candidate(standard_id, revision, candidate_id)
            diagnostics = self.store.load_candidate_diagnostics(
                standard_id, revision, candidate_id)
        except Exception as exc:
            return f"  {candidate_id}: unreadable ({type(exc).__name__}: {exc})"

        comparison = diagnostics.get("comparison", {})
        old = comparison.get("old", {})
        new = comparison.get("candidate", {})
        ids = comparison.get("source_ids", {})

        def pair(name: str) -> str:
            return f"{old.get(name, '?')} -> {new.get(name, '?')}"

        return "\n".join((
            f"  {candidate_id}",
            f"    corpus: {manifest.corpus_manifest_sha256}",
            f"    extractor: {diagnostics.get('active_extractor_version')} -> "
            f"{manifest.extractor_version}",
            f"    units: {pair('unit_count')}",
            f"    distinct sections: {pair('distinct_sections')}",
            f"    invalid section ids: {pair('invalid_section_ids')}",
            f"    units on invalid sections: {pair('units_on_invalid_sections')}",
            f"    backwards transitions: {pair('backwards_transitions')}",
            f"    duplicate section anchors: {pair('duplicate_section_anchors')}",
            f"    contents entries: {pair('toc_entry_units')}",
            f"    page furniture: {pair('page_furniture_units')}",
            f"    columnar units: {pair('columnar_units')}",
            f"    structured review: {pair('structured_review_units')}",
            f"    retrievable units: {pair('retrievable_units')}",
            f"    source ids retained/removed/added: {ids.get('retained', '?')}/"
            f"{ids.get('removed', '?')}/{ids.get('added', '?')}",
            f"    layout artifact: {diagnostics.get('layout_artifact_present')}",
            "    review status: READY_FOR_OPERATOR_REVIEW",
        ))

    def unbind(self) -> None:
        self._atomic_write(self._active_path, canonical_json({
            "schema_version": 1,
            "active": None,
        }))

    def _write_active(self, binding: StandardBinding) -> None:
        self._atomic_write(self._active_path, canonical_json({
            "schema_version": 1,
            "active": binding.to_dict(),
        }))

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".binding-", dir=path.parent)

        try:
            # Durable before the rename, so a crash leaves either the old
            # binding or the new one, never a truncated file.

            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            # Already gone on the success path, where the rename consumed it.

            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


# Every operator action this handler implements, with the arguments it takes.
# The usage string and the /help listing are both rendered from this one list,
# so an action cannot be added to the handler and go missing from the help --
# which is exactly how approve-bitfield came to be undiscoverable.

STANDARD_ACTIONS: tuple[tuple[str, str], ...] = (
    ("list", ""),
    ("ingest", "<pdf> --id <id> --revision <revision> [--retain-pdf] "
                "[--origin LICENSED_STANDARD|PUBLIC] [--allow-offload]"),
    ("use", "<id> <revision>"),
    ("status", ""),
    ("rebuild", "[<id> <revision>] [--allow-offload]"),
    ("retrieval", "[<id> <revision>] [lexical|vector|hybrid] [--completion <n>]"),
    ("unbind", ""),
    ("verify", "[<id> <revision>]"),
    ("candidates", "[<id> <revision>]"),
    ("promote-candidate", "<id> <revision> <candidate-id>"),
    ("approve", "<id> <revision> [<reviewer>]"),
    ("approve-bitfield",
     "<id> <revision> <candidate-id> <verdict> <role,role,...> "
     "[--accept-link <link-id>[=ROLE]] [--decline-link <link-id>] "
     "[<packet-identity>]"),
    ("approval-history", "<id> <revision> [<candidate-id>]"),
    ("migrate-approval-history", "<id> <revision> [--apply]"),
    ("build-structure", "[<id> <revision>]"),
)


def standard_usage() -> str:
    return "usage: /standard " + "|".join(name for name, _ in STANDARD_ACTIONS)


def standard_help_lines() -> tuple[str, ...]:
    """The operator actions, for the interactive help to print verbatim."""

    return tuple(f"  /standard {name}" + (f" {arguments}" if arguments else "")
                 for name, arguments in STANDARD_ACTIONS)


def retrieval_summary(store, standard_id, revision) -> str:
    """How the bound document is searched right now, and who decided it.

    The document records its own setting; a session variable may still
    override it, and an override nobody can see is how a measured document
    ends up searched some other way.
    """
    from standard_tools import _configured_mode

    try:
        declared = store.load_retrieval_settings(standard_id, revision)
    except StandardStoreError as exc:
        return f"UNAVAILABLE ({exc})"

    mode = _configured_mode(declared.get("mode"))
    env_completion = os.environ.get("SPEAR_STANDARD_EVIDENCE_COMPLETION", "").strip()

    try:
        completion = (max(0, int(env_completion)) if env_completion
                      else int(declared.get("evidence_completion", 0)))
    except ValueError:
        completion = 0

    overrides = [name for name, value in (
        ("SPEAR_STANDARD_RETRIEVAL_MODE",
         os.environ.get("SPEAR_STANDARD_RETRIEVAL_MODE", "").strip()),
        ("SPEAR_STANDARD_EVIDENCE_COMPLETION", env_completion)) if value]
    origin = ("overridden by " + ", ".join(overrides) if overrides
              else "set for this document" if declared else "default")

    return f"{mode}, completion {completion} ({origin})"


def _build_vector_index(operator, standard_id, revision, allow_offload):
    """(manifest, error). Its absence degrades retrieval; it never fails a rebuild.

    The disclosure is written HERE and not inside rebuild_vector_index: it is a
    fact about an operator decision, not about the index, and the index would
    be byte-identical had the same model run on this machine.
    """
    origin = operator.store.load_manifest(standard_id, revision).source_origin

    try:
        embedder = configured_embedder(origin, allow_offload=allow_offload)

        if embedder is None:
            return None, None

        # Embedding a whole corpus on this machine's CPU saturates it for
        # what the GPU host does in seconds. Skipped, like an unconfigured
        # model, unless a device was named for it.

        if (getattr(embedder, "target", None) is None
                and os.environ.get("SPEAR_STANDARD_EMBED_DEVICE", "cpu") == "cpu"):
            return None, ("not built: this would embed the corpus on the local "
                          "CPU (set SPEAR_STANDARD_EMBED_DEVICE, or for a "
                          "licensed standard pass --allow-offload)")

        manifest = rebuild_vector_index(operator.store, standard_id, revision,
                                        embedder)
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)

    target = getattr(embedder, "target", None)

    if target and origin == LICENSED_ORIGIN:
        record_offload_disclosure(
            operator.store, standard_id, revision, target=target,
            unit_count=manifest.indexed_source_count,
            model_id=manifest.embedding_model_id)

    return manifest, None


def handle_standard_command(command: str, operator: StandardOperator) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise StandardCommandError(f"invalid /standard syntax: {exc}") from exc

    if not parts or parts[0] != "/standard":
        raise StandardCommandError("not a /standard command")

    if len(parts) < 2:
        raise StandardCommandError(standard_usage())

    action, args = parts[1], parts[2:]

    if action == "list":
        if args:
            raise StandardCommandError("usage: /standard list")

        values = operator.store.list_standards()

        if not values:
            return "No standards ingested."

        # With the parameters each was ingested with, not just its name.
        #
        # `status` already reports all of this and reports it only for the
        # BOUND document, so the only way to see what an unbound one holds was
        # to bind it -- and the binding is shared by every session on the
        # machine, which makes "let me look" a change of state for everybody.
        try:
            bound = operator.active_binding()
        except Exception:
            bound = None

        active = ((bound.standard_id, bound.revision) if bound is not None
                  else (None, None))
        lines = ["Available standards:"]

        for standard_id, revision in values:
            mark = "  <- bound" if (standard_id, revision) == active else ""
            lines.append(f"\n  {standard_id} {revision}{mark}")

            # A manifest that cannot be read is reported as such rather than
            # skipped: a document in the store whose parameters are unknown is
            # the thing a reader most needs to be told about.
            try:
                manifest = operator.store.load_manifest(standard_id, revision)
            except Exception as exc:
                lines.append(f"    manifest unreadable ({type(exc).__name__})")
                continue

            retained = ("the document itself is retained"
                        if manifest.raw_pdf_retained else "extraction only")
            lines.append(f"    origin      {manifest.source_origin}"
                         f"  \u00b7  {retained}")
            lines.append(f"    extractor   {manifest.extractor_version}")
            lines.append(
                f"    corpus      {manifest.canonical_unit_count} units"
                f"  \u00b7  {manifest.requirement_count} requirements"
                f"  \u00b7  {manifest.recommendation_count} recommendations"
                f"  \u00b7  {manifest.page_count} pages")
            lines.append(f"    validation  {manifest.human_validation_status}")

            if manifest.extraction_errors:
                lines.append(
                    f"    errors      {len(manifest.extraction_errors)} at "
                    f"extraction")

        return "\n".join(lines)

    if action == "status":
        if args:
            raise StandardCommandError("usage: /standard status")

        # A binding the store can no longer honour is reported, not raised:
        # status is the command an operator runs precisely to diagnose that.

        try:
            binding = operator.active_binding()
        except StandardCommandError as exc:
            return operator.stale_binding_status(exc)

        if binding is None:
            return "Active standard: none"

        manifest = operator.store.load_manifest(binding.standard_id, binding.revision)

        # Vector search and cross references are optional companions to the
        # lexical index; each reports its own absence instead of failing status.
        #
        # A missing vector index is the normal state when no embedding model
        # is configured, and must not read like a broken store.

        try:
            vector = operator.store.load_vector_index(
                binding.standard_id, binding.revision)[0]
            vector_status = (f"Vector: READY\nmodel: {vector.embedding_model_id}\n"
                             f"model revision: {vector.embedding_model_revision}")
        except FileNotFoundError:
            # Mirrors configured_embedder without building one: the offloaded
            # embedder checks the remote host over ssh as it is constructed.
            remote = os.environ.get("SPEAR_STANDARD_EMBED_REMOTE", "").strip()
            local_cpu = os.environ.get("SPEAR_STANDARD_EMBED_DEVICE", "cpu") == "cpu"

            if not os.environ.get("SPEAR_STANDARD_EMBED_MODEL"):
                vector_status = ("Vector: NOT CONFIGURED "
                                 "(no embedding model, lexical retrieval)")
            elif not local_cpu or (remote and manifest.source_origin != LICENSED_ORIGIN):
                vector_status = "Vector: NOT BUILT (run /standard rebuild)"
            elif remote:
                vector_status = ("Vector: NOT BUILT (licensed, kept off the local "
                                 "CPU; /standard rebuild --allow-offload would "
                                 f"send its text to {remote})")
            else:
                vector_status = ("Vector: NOT BUILT (no remote embedder, and the "
                                 "local CPU is not used for a corpus)")
        except Exception as exc:
            vector_status = f"Vector: UNAVAILABLE ({type(exc).__name__})"

        try:
            crossrefs = operator.store.load_cross_reference_index(
                binding.standard_id, binding.revision)[0]
            crossref_status = ("Cross references: READY\n"
                               f"resolved: {crossrefs.resolved_count}\n"
                               f"ambiguous: {crossrefs.ambiguous_count}\n"
                               f"unresolved: {crossrefs.unresolved_count}")
        except Exception as exc:
            crossref_status = f"Cross references: UNAVAILABLE ({type(exc).__name__})"

        # Where an auditor looks. The question "was this licensed text ever
        # sent off this machine?" has one answer and it belongs on the status
        # line, not in a file someone would have to know to open.

        disclosures = offload_disclosures(operator.store, binding.standard_id,
                                          binding.revision)
        disclosure_status = "\n".join(
            [f"Origin: {manifest.source_origin}"]
            + [f"⚠ text sent off this machine: {item['target']} on "
               f"{item['at']} ({item['unit_count']} units, "
               f"{item['model_id']}, operator override)"
               for item in disclosures])

        return "\n".join((
            f"Active: {binding.standard_id}", f"Revision: {binding.revision}",
            f"PDF SHA: {binding.pdf_sha256}",
            f"Active corpus: {binding.corpus_manifest_sha256}",
            f"Extractor: {binding.extractor_version}",
            disclosure_status,
            f"Retrieval indexes: {operator.store.index_state(binding.standard_id, binding.revision)}",
            f"Lexical index: {binding.index_fingerprint}",
            vector_status, crossref_status,
            f"Retrieval fingerprint: {binding.retrieval_fingerprint}",
            "Retrieval: " + retrieval_summary(operator.store, binding.standard_id,
                                              binding.revision),
            f"Validation: {manifest.human_validation_status.value}",
            operator.candidate_summary(binding.standard_id, binding.revision),
        ))

    if action == "use":
        if len(args) != 2:
            raise StandardCommandError("usage: /standard use <id> <revision>")

        binding = operator.use(*args)

        return f"Bound {binding.standard_id} revision {binding.revision}."

    if action == "unbind":
        if args:
            raise StandardCommandError("usage: /standard unbind")

        operator.unbind()

        return "Standard binding cleared."

    if action == "rebuild":
        if len([item for item in args if item != "--allow-offload"]) not in {0, 2}:
            raise StandardCommandError(
                "usage: /standard rebuild [<id> <revision>] [--allow-offload]")

        allow_offload = "--allow-offload" in args
        args = [item for item in args if item != "--allow-offload"]

        if args:
            if len(args) != 2:
                raise StandardCommandError(
                    "usage: /standard rebuild [<id> <revision>] [--allow-offload]")

            standard_id, revision = args
        else:
            active = operator.active_binding()

            if active is None:
                raise StandardCommandError("no active standard to rebuild")

            standard_id, revision = active.standard_id, active.revision

        index = rebuild_lexical_index(operator.store, standard_id, revision)
        crossrefs = rebuild_cross_reference_index(
            operator.store, standard_id, revision)

        # The vector index needs an embedder that may not be configured here;
        # its absence degrades retrieval rather than failing the rebuild.

        vector, vector_error = _build_vector_index(
            operator, standard_id, revision, allow_offload)

        # Rebinding refreshes the retrieval fingerprint the new indexes carry,
        # but only for the revision that is actually bound right now.

        active = operator.active_binding()

        if active and (active.standard_id, active.revision) == (standard_id, revision):
            operator.use(standard_id, revision)

        return (f"Rebuilt lexical index {index.index_fingerprint}; cross references "
                f"{crossrefs.cross_reference_index_fingerprint}; vector "
                f"{vector.vector_index_fingerprint if vector else vector_error or 'not configured'}.")

    if action == "retrieval":
        usage = ("usage: /standard retrieval [<id> <revision>] "
                 "[lexical|vector|hybrid] [--completion <n>]")
        completion = None

        if "--completion" in args:
            at = args.index("--completion")

            try:
                completion = int(args[at + 1])
            except (IndexError, ValueError):
                raise StandardCommandError(usage) from None

            if completion < 0:
                raise StandardCommandError(usage)

            args = args[:at] + args[at + 2:]

        mode = None

        if len(args) in {1, 3}:
            mode = args[-1].lower()
            args = args[:-1]

            if mode not in RETRIEVAL_MODES:
                raise StandardCommandError(usage)

        if len(args) == 2:
            standard_id, revision = args
        elif not args:
            active = operator.active_binding()

            if active is None:
                raise StandardCommandError("no active standard; name one: " + usage)

            standard_id, revision = active.standard_id, active.revision
        else:
            raise StandardCommandError(usage)

        if mode is not None or completion is not None:
            try:
                current = operator.store.load_retrieval_settings(standard_id, revision)
            except StandardStoreError:
                current = {}

            try:
                operator.store.save_retrieval_settings(
                    standard_id, revision,
                    mode=mode or str(current.get("mode", "hybrid")),
                    evidence_completion=(
                        completion if completion is not None
                        else int(current.get("evidence_completion", 0))))
            except (FileNotFoundError, StandardStoreError) as exc:
                raise StandardCommandError(str(exc)) from None

        return (f"Retrieval for {standard_id} {revision}: "
                + retrieval_summary(operator.store, standard_id, revision))

    if action == "ingest":
        if not args:
            raise StandardCommandError(
                "usage: /standard ingest <pdf> --id <id> --revision <revision> "
                "[--retain-pdf] [--origin LICENSED_STANDARD|PUBLIC]")

        pdf = args[0]
        options = args[1:]

        standard_id = revision = None
        retain = candidate = layout = False

        # Everything used to be ingested as LICENSED_STANDARD, the safe
        # default, with no way to say otherwise — so a NIST publication in the
        # public domain carried the same restriction as a purchased ANSI
        # standard, and the guards that read this field could not tell them
        # apart. The one that matters here refuses to embed a licensed corpus
        # on the shared GPU host.

        # None, not the default, so "not passed" stays distinguishable from
        # "passed LICENSED_STANDARD": re-ingesting an identical PDF returns the
        # stored manifest untouched, and only an explicit --origin may move a
        # classification that is already recorded.

        origin = None

        # Deliberate, per-command, and never stored: see configured_embedder.
        allow_offload = False

        index = 0

        while index < len(options):
            option = options[index]

            if option == "--retain-pdf":
                retain = True
                index += 1

                continue

            if option == "--allow-offload":
                allow_offload = True
                index += 1

                continue

            if option == "--candidate":
                candidate = True
                index += 1

                continue

            if option == "--layout":
                layout = True
                index += 1

                continue

            if option == "--origin" and index + 1 < len(options):
                origin = options[index + 1].strip().upper()

                if origin not in {"LICENSED_STANDARD", "PUBLIC"}:
                    raise StandardCommandError(
                        "--origin must be LICENSED_STANDARD or PUBLIC")

                index += 2

                continue

            if option in {"--id", "--revision"} and index + 1 < len(options):
                if option == "--id":
                    standard_id = options[index + 1]
                else:
                    revision = options[index + 1]

                index += 2

                continue

            raise StandardCommandError(f"unknown or incomplete ingestion option: {option}")

        if not standard_id or not revision:
            raise StandardCommandError("ingestion requires --id and --revision")

        # A candidate is extracted beside the active corpus and never replaces
        # it, so the options that act on the active corpus are refused here.

        if candidate:
            if retain:
                raise StandardCommandError(
                    "--retain-pdf applies to the active corpus, not a candidate")

            return _render_candidate(operator, standard_id, revision, pdf, layout)

        if layout:
            raise StandardCommandError("--layout is only available for --candidate")

        manifest = ingest_pdf(
            operator.store, pdf, standard_id=standard_id, revision=revision,
            retain_pdf=retain, source_origin=origin or "LICENSED_STANDARD",
        )
        reclassified = None

        if origin is not None and manifest.source_origin != origin:
            reclassified = operator.store.reclassify_origin(
                standard_id, revision, origin)
            manifest = operator.store.load_manifest(standard_id, revision)

        lexical = rebuild_lexical_index(operator.store, standard_id, revision)
        crossrefs = rebuild_cross_reference_index(operator.store, standard_id, revision)

        vector, vector_error = _build_vector_index(
            operator, standard_id, revision, allow_offload)

        return "\n".join((
            f"Ingested {standard_id} revision {revision}.",
            (f"Reclassified: {reclassified} -> {manifest.source_origin}"
             if reclassified else f"Origin: {manifest.source_origin}"),
            f"Pages: {manifest.page_count}", f"Units: {manifest.canonical_unit_count}",
            f"Sections: {manifest.section_count}", f"Warnings: {len(manifest.warnings)}",
            f"PDF SHA: {manifest.source_pdf_sha256}",
            f"Corpus SHA: {manifest.corpus_manifest_sha256}",
            f"Lexical index: {lexical.index_fingerprint}",
            f"Cross references: {crossrefs.cross_reference_index_fingerprint}",
            f"Vector index: {vector.vector_index_fingerprint if vector else vector_error or 'not configured'}",
            f"Raw PDF retained: {'yes' if retain else 'no'}",
        ))

    if action == "candidates":
        if len(args) not in {0, 2}:
            raise StandardCommandError("usage: /standard candidates [<id> <revision>]")

        standard_id, revision = _target(operator, args)
        names = operator.store.list_candidates(standard_id, revision)

        if not names:
            return f"No candidate corpora for {standard_id} {revision}."

        lines = [f"Candidates for {standard_id} {revision}:"]

        for name in names:
            lines.append(operator.candidate_report(standard_id, revision, name))

        lines.append("Promotion replaces the active corpus and requires operator review:")
        lines.append(f"  /standard promote-candidate {standard_id} {revision} "
                     f"{names[0]}")

        return "\n".join(lines)

    if action == "promote-candidate":
        if len(args) != 3:
            raise StandardCommandError(
                "usage: /standard promote-candidate <id> <revision> <candidate-id>")

        standard_id, revision, candidate_id = args

        try:
            previous = operator.store.load_manifest(standard_id, revision)
            manifest = operator.store.promote_candidate(
                standard_id, revision, candidate_id)
        except (StandardStoreError, FileNotFoundError) as exc:
            raise StandardCommandError(f"promotion refused: {exc}") from exc

        # The promoted corpus invalidates every index the old binding named, so
        # the binding is dropped rather than left pointing at stale retrieval.

        operator.unbind()

        return "\n".join((
            f"Promoted {candidate_id} for {standard_id} {revision}.",
            f"Previous corpus: {previous.corpus_manifest_sha256}",
            f"Active corpus:   {manifest.corpus_manifest_sha256}",
            f"Extractor: {previous.extractor_version} -> {manifest.extractor_version}",
            f"Units: {previous.canonical_unit_count} -> {manifest.canonical_unit_count}",
            "Previous generation retained as "
            f"{'gen-' + previous.corpus_manifest_sha256[:16]}.",
            "Retrieval indexes: REBUILD_REQUIRED.",
            "Standard binding cleared; existing sessions fail closed.",
            f"Next: /standard rebuild {standard_id} {revision}"
            f" then /standard use {standard_id} {revision}",
        ))

    if action == "verify":
        if len(args) not in {0, 2}:
            raise StandardCommandError("usage: /standard verify [<id> <revision>]")

        standard_id, revision = _target(operator, args)

        from standard_ingest import verify_against_contents

        report = verify_against_contents(operator.store.load_units(standard_id, revision))

        lines = [f"Contents cross-check for {standard_id} {revision}:",
                 f"  clauses in the contents table : {report['contents_entries_with_a_page']}",
                 f"  clause headings detected      : {report['headings_detected']}",
                 f"  comparable clauses            : {report['clauses_in_both']}",
                 f"  printed-page offset           : {report['printed_page_offset']}",
                 f"  page agreement                : {report['page_agreement']}"
                 f"/{report['clauses_in_both']}"
                 + (f" ({report['agreement_rate'] * 100:.1f}%)"
                    if report["agreement_rate"] is not None else ""),
                 f"  advertised but not detected   : "
                 f"{len(report['advertised_not_detected'])}"
                 + (f" {report['advertised_not_detected']}"
                    if report["advertised_not_detected"] else "")]

        for item in report["disagreements"]:
            lines.append(f"    clause {item['clause']}: contents p{item['contents_page']} "
                         f"(= p{item['expected_page']}) but detected p{item['detected_page']} "
                         f"({item['delta']:+d})")

        return "\n".join(lines)

    if action == "approve-bitfield":
        if len(args) < 5:
            raise StandardCommandError(
                "usage: /standard approve-bitfield <id> <revision> <candidate-id> "
                "<verdict> <role,role,...> [--accept-link <link-id>[=ROLE]] "
                "[--decline-link <link-id>] [<packet-identity>]")

        standard_id, revision, candidate_id, verdict, roles = args[:5]

        from standard_semantic import StandardSemanticError
        from standard_semantic_store import StandardApprovalStore

        # The link options repeat, so they are consumed in order and whatever
        # is left over is the optional packet identity.

        accept, decline, link_roles, rest = [], [], {}, []
        remaining = list(args[5:])

        while remaining:
            token = remaining.pop(0)

            if token in ("--accept-link", "--decline-link"):
                if not remaining:
                    raise StandardCommandError(f"{token} needs a link id")

                # "lnk-...=RESERVED" says what the linked field is, and only
                # FIELD and RESERVED are things a field can be.

                value, _, role = remaining.pop(0).partition("=")
                (accept if token == "--accept-link" else decline).append(value)

                if role:
                    link_roles[value] = role.strip().upper()
            else:
                rest.append(token)

        if len(rest) > 1:
            raise StandardCommandError(
                "usage: /standard approve-bitfield <id> <revision> <candidate-id> "
                "<verdict> <role,role,...> [--accept-link <link-id>[=ROLE]] "
                "[--decline-link <link-id>] [<packet-identity>]")

        try:
            approval = StandardApprovalStore(operator.store).approve(
                standard_id, revision, candidate_id, verdict=verdict,
                span_roles=[value.strip() for value in roles.split(",") if value.strip()],
                packet_identity=(rest[0] if rest else None),
                accept_links=accept, decline_links=decline, link_roles=link_roles)
        except (StandardSemanticError, StandardStoreError) as exc:
            raise StandardCommandError(f"approval refused: {exc}") from exc

        lines = [
            f"Recorded {approval.verdict} for {approval.candidate_id}.",
            ("Transition: REAPPROVAL" if approval.supersedes
             else "Transition: INITIAL_APPROVAL"),
            f"New event: {approval.event_id}",
        ]

        if approval.supersedes:
            lines.append(f"Supersedes: {approval.supersedes}")

        lines += [
            f"Reviewer: {approval.reviewer} at {approval.reviewed_at}",
            f"Geometry: {approval.structure_fingerprint}",
            f"Diagram field roles: {', '.join(approval.span_roles)}",
        ]

        for item in approval.reviewed_links:
            lines.append(
                f"Normative link {item.link_fingerprint}: "
                f"{'ACCEPTED as ' + item.role if item.accepted else 'DECLINED'} "
                f"(bits {item.high_bit}..{item.low_bit}, word {item.word_index}, "
                f"source {item.normative_source_id})")

        lines.append(f"Review evidence: {approval.review_evidence_fingerprint}")
        lines.append(f"Packet identity: {approval.packet_identity or 'none'}")

        return "\n".join(lines)

    if action == "approval-history":
        if len(args) not in {2, 3}:
            raise StandardCommandError(
                "usage: /standard approval-history <id> <revision> "
                "[<candidate-id>]")

        from standard_approval_history import (StandardApprovalHistory,
                                               StandardApprovalHistoryError)
        from standard_semantic import StandardSemanticError
        from standard_semantic_store import StandardApprovalStore

        standard_id, revision = args[0], args[1]
        candidate_id = args[2] if len(args) == 3 else None

        history = StandardApprovalHistory(operator.store)

        try:
            events = history.events(standard_id, revision,
                                    candidate_id=candidate_id)
            approvals, _ = StandardApprovalStore(operator.store).load(
                standard_id, revision)
            problems = history.verify(standard_id, revision, approvals)
        except (StandardApprovalHistoryError, StandardSemanticError,
                StandardStoreError, FileNotFoundError) as exc:
            raise StandardCommandError(f"approval history refused: {exc}") from exc

        active = {item.event_id for item in approvals.values() if item.event_id}
        target = f" for {candidate_id}" if candidate_id else ""

        # No events at all is ambiguous on its own, so say whether the active
        # approvals simply predate the log rather than implying none exist.

        if not events:
            return (f"No recorded approval decisions{target} in "
                    f"{standard_id} {revision}.\n"
                    f"Active approvals: {len(approvals)}; none names an event, "
                    "so every one predates the history log.")

        lines = [f"Approval history{target} for {standard_id} {revision}:"]

        for event in events:
            lines.append(
                f"  {event.event_id}  {event.event_type.value}"
                f"{'  [ACTIVE]' if event.event_id in active else ''}")
            lines.append(
                f"    {event.candidate_id}  {event.verdict}  "
                f"{len(event.span_roles)} role(s)  by {event.reviewer} "
                f"at {event.approved_at}")
            lines.append(
                f"    geometry {event.structure_fingerprint}")
            lines.append(
                f"    evidence {event.review_evidence_fingerprint or '-'} "
                f"({event.evidence_status.value})")

            if event.supersedes:
                lines.append(f"    supersedes {event.supersedes}")

        carried = [key for key, item in sorted(approvals.items())
                   if item.event_id is None
                   and (candidate_id is None or key == candidate_id)]

        if carried:
            lines.append(f"Active approvals predating the history log: "
                         f"{len(carried)} ({', '.join(carried)})")

        lines.append("Consistency: "
                     + ("OK" if not problems else
                        "; ".join(f"{item['candidate_id']}: {item['reason']}"
                                  for item in problems)))

        return "\n".join(lines)

    if action == "migrate-approval-history":
        apply = False
        rest = []

        for token in args:
            if token == "--apply":
                apply = True
                continue

            if token.startswith("--"):
                raise StandardCommandError(
                    f"unknown option {token} for /standard "
                    "migrate-approval-history")

            rest.append(token)

        if len(rest) != 2:
            raise StandardCommandError(
                "usage: /standard migrate-approval-history <id> <revision> "
                "[--apply]")

        from standard_approval_history import (StandardApprovalHistory,
                                               StandardApprovalHistoryError)
        from standard_semantic import StandardSemanticError
        from standard_semantic_store import StandardApprovalStore

        standard_id, revision = rest

        approvals = StandardApprovalStore(operator.store)
        history = StandardApprovalHistory(operator.store)

        # Without --apply the migration only plans, so the same call serves as
        # the dry run the operator reads before committing to it.

        try:
            before = history.events(standard_id, revision)
            active, _ = approvals.load(standard_id, revision)
            planned = approvals.migrate_history(standard_id, revision,
                                                dry_run=not apply)
        except (StandardApprovalHistoryError, StandardSemanticError,
                StandardStoreError, FileNotFoundError) as exc:
            raise StandardCommandError(f"migration refused: {exc}") from exc

        legacy = [item for item in planned
                  if item["review_evidence_fingerprint"] is None]

        if apply:
            return "\n".join((
                f"Migrated approval history for {standard_id} {revision}.",
                f"Initial events created: {len(planned)}",
                f"Active approvals linked: {len(planned)}",
                f"Existing events preserved: {len(before)}",
            ) + (("Nothing to migrate; every approval was already recorded.",)
                 if not planned else ()))

        lines = [
            f"Approval-history migration plan for {standard_id} {revision}:",
            f"  Active approvals                     : {len(active)}",
            f"  Existing historical events           : {len(before)}",
            f"  Approvals requiring migration        : {len(planned)}",
            f"  Events to create                     : {len(planned)}",
            f"  Active pointers to attach            : {len(planned)}",
            f"  Legacy approvals with no review evidence: {len(legacy)}",
        ]

        for item in planned:
            lines.append(
                f"  {item['candidate_id']}  {item['verdict']}  "
                f"{item['roles']} role(s)  by {item['reviewer']} "
                f"at {item['approved_at']}")
            lines.append(
                f"    evidence {item['review_evidence_fingerprint'] or '<none>'}"
                + ("" if item["review_evidence_fingerprint"]
                   else "  (predates evidence binding)"))

        lines.append("Nothing was written. Re-run with --apply to record these."
                     if planned else
                     "Nothing to migrate; every approval is already recorded.")

        return "\n".join(lines)

    if action == "build-structure":
        if len(args) not in {0, 2}:
            raise StandardCommandError(
                "usage: /standard build-structure [<id> <revision>]")

        standard_id, revision = _target(operator, args)

        return _build_structure(operator, standard_id, revision)

    if action == "approve":
        if len(args) not in {2, 3}:
            raise StandardCommandError(
                "usage: /standard approve <id> <revision> [<reviewer>]")

        return _approve(operator, args[0], args[1],
                        args[2] if len(args) == 3 else None)

    raise StandardCommandError(f"unknown /standard action: {action}")


def load_structure_set(store, standard_id: str, revision: str):
    """Re-derive the stored geometry and prove it is the geometry on disk."""

    import json

    from standard_ingest import verify_against_contents  # noqa: F401  (keeps import graph explicit)
    from standard_semantic import StandardSemanticError
    from standard_structure import extract_structures, structure_fingerprint
    from standard_structure_store import StandardStructureStore

    manifest, _ = StandardStructureStore(store).load(standard_id, revision)
    corpus = store.verify_corpus(standard_id, revision)
    layout = json.loads(
        (store.revision_dir(standard_id, revision) / "layout.json").read_text("utf-8"))

    structures = extract_structures(
        store.load_units(standard_id, revision), layout,
        standard_id=standard_id, revision=revision,
        corpus_fingerprint=corpus.corpus_manifest_sha256)

    # Re-extraction must land on the same fingerprint the store recorded;
    # anything else means the geometry on disk no longer describes this corpus.

    if structure_fingerprint(structures) != manifest.structure_fingerprint:
        raise StandardSemanticError(
            "the stored geometry does not reproduce; rebuild it before promoting")

    return manifest, structures, corpus, layout


def _build_structure(operator: StandardOperator, standard_id: str,
                     revision: str) -> str:
    """Promote every human-approved bitfield candidate, and only those."""

    from standard_semantic import (
        StandardSemanticError, build_semantics, validate_semantics,
    )
    from standard_semantic_store import (
        StandardApprovalStore, StandardSemanticStore, build_semantic_manifest,
    )

    try:
        manifest, structures, corpus, layout = load_structure_set(
            operator.store, standard_id, revision)
        approvals, refused = StandardApprovalStore(operator.store).load(
            standard_id, revision)

        semantics = build_semantics(
            structures, approvals, standard_id=standard_id, revision=revision,
            corpus_fingerprint=corpus.corpus_manifest_sha256,
            layout_fingerprint=layout["layout_fingerprint"],
            structure_fingerprint=manifest.structure_fingerprint,
            layout=layout,
            units=operator.store.load_units(standard_id, revision))

        # Validation re-checks the built semantics against the very corpus,
        # layout and geometry fingerprints they claim to have been built from.

        validate_semantics(
            semantics, structures=structures,
            source_ids=frozenset(unit.source_id for unit in
                                 operator.store.load_units(standard_id, revision)),
            corpus_fingerprint=corpus.corpus_manifest_sha256,
            layout_fingerprint=layout["layout_fingerprint"],
            structure_fingerprint=manifest.structure_fingerprint)

        semantic_manifest = build_semantic_manifest(
            semantics, standard_id=standard_id, revision=revision,
            corpus_fingerprint=corpus.corpus_manifest_sha256,
            layout_fingerprint=layout["layout_fingerprint"],
            structure_fingerprint=manifest.structure_fingerprint,
            approved=sum(1 for item in approvals.values() if item.approves))

        StandardSemanticStore(operator.store).save(semantic_manifest, semantics)
    except (StandardSemanticError, StandardStoreError, FileNotFoundError) as exc:
        raise StandardCommandError(f"structure build refused: {exc}") from exc

    # Blocked candidates are reported by reason rather than individually; the
    # operator needs to know what class of gate stopped them, not which ids.

    reasons = {}

    for item in semantics.blocked:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1

    lines = [
        f"Built semantic structure for {standard_id} {revision}.",
        f"Bitfield candidates:    {len(structures.bitfields)}",
        f"Human approvals usable: {len(approvals)}"
        + (f" ({len(refused)} refused)" if refused else ""),
        f"Bitfield definitions:   {semantic_manifest.bitfield_definition_count}",
        f"Field definitions:      {semantic_manifest.field_definition_count}",
        f"Packet definitions:     {semantic_manifest.packet_definition_count}",
        f"Blocked candidates:     {semantic_manifest.blocked_count}",
    ]

    for reason, count in sorted(reasons.items()):
        lines.append(f"  {reason}: {count}")

    lines.append(f"Semantic fingerprint: {semantic_manifest.semantic_fingerprint}")

    return "\n".join(lines)


def _approve(operator: StandardOperator, standard_id: str, revision: str,
             reviewer: str | None) -> str:
    """Record an operator's approval, but only behind every review gate."""

    from standard_review import (
        REVIEW_SCHEMA_VERSION, StandardReview, StandardReviewError,
    )
    from standard_schema import HumanValidationStatus

    try:
        # Loading verifies the corpus and refuses a review of another generation.

        review = StandardReview.load(operator.store, standard_id, revision,
                                     reviewer=reviewer)
    except (StandardReviewError, StandardStoreError, FileNotFoundError) as exc:
        raise StandardCommandError(f"approval refused: {exc}") from exc

    summary = review.summary()

    if not review.is_complete():
        raise StandardCommandError(
            f"approval refused: {summary['UNREVIEWED']} of {summary['TOTAL']} "
            "review rows are not human-reviewed")

    blocking = review.blocking()

    if blocking:
        detail = ", ".join(f"{key}={value}" for key, value in blocking.items())

        raise StandardCommandError(
            f"approval refused: review findings block approval ({detail})")

    manifest = operator.store.load_manifest(standard_id, revision)
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # The recorded evidence names the review file and its schema, so a later
    # reader can tell which gate actually ran.

    updated = operator.store.record_human_validation(
        standard_id, revision, HumanValidationStatus.APPROVED,
        reviewer=review.reviewer, reviewed_at=reviewed_at,
        evidence={"review_file": review.path.name,
                  "review_schema_version": REVIEW_SCHEMA_VERSION,
                  "rows_reviewed": summary["TOTAL"],
                  "summary": summary})

    return "\n".join((
        f"Approved {standard_id} revision {revision}.",
        f"Corpus: {updated.corpus_manifest_sha256}",
        f"Extractor: {updated.extractor_version}",
        f"Validation: {manifest.human_validation_status.value} -> "
        f"{updated.human_validation_status.value}",
        f"Reviewer: {review.reviewer} at {reviewed_at}",
        f"Rows reviewed: {summary['TOTAL']} "
        f"(PASS {summary['PASS']}, ACCEPTABLE_WARNING "
        f"{summary['ACCEPTABLE_WARNING']})",
    ))


def _target(operator: StandardOperator, args: list[str]) -> tuple[str, str]:
    if args:
        return args[0], args[1]

    # Falling back to the active binding is a convenience, so a binding that is
    # merely unreadable is treated the same as none at all.

    try:
        active = operator.active_binding()
    except StandardCommandError:
        active = None

    if active is None:
        raise StandardCommandError("no active standard; name the id and revision")

    return active.standard_id, active.revision


def _render_candidate(operator: StandardOperator, standard_id: str, revision: str,
                      pdf: str, layout: bool) -> str:
    started = datetime.now(timezone.utc)

    candidate_id, diagnostics = ingest_candidate(
        operator.store, pdf, standard_id=standard_id, revision=revision,
        with_layout=layout)

    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    comparison = diagnostics["comparison"]

    return "\n".join((
        f"Extracted candidate {candidate_id} for {standard_id} {revision}.",
        f"Active corpus:    {diagnostics['active_corpus_sha256']}",
        f"Candidate corpus: {diagnostics['candidate_corpus_sha256']}",
        f"Extractor: {diagnostics['active_extractor_version']} -> "
        f"{diagnostics['candidate_extractor_version']}",
        f"Units: {comparison['old']['unit_count']} -> "
        f"{comparison['candidate']['unit_count']}",
        f"Extraction seconds: {seconds:.2f}",
        "The candidate is isolated; the active corpus is unchanged.",
        f"Review with: /standard candidates {standard_id} {revision}",
    ))
