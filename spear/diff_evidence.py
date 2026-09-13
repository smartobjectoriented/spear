"""Grounded, bounded change evidence for independent review."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import asdict, dataclass
from typing import Callable

from checkpoint import Checkpoint, CheckpointManager
from result_store import ResultStore


def _digest(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


@dataclass(frozen=True)
class DiffEvidence:
    source: str
    changed_paths: tuple[str, ...]
    preview: str
    byte_count: int
    truncated: bool
    result_reference: str | None = None
    binary_paths: tuple[str, ...] = ()
    external_change_paths: tuple[str, ...] = ()
    limitation: str | None = None

    def to_dict(self, *, include_preview: bool = True) -> dict[str, object]:
        result = asdict(self)

        if not include_preview:
            result.pop("preview")

        return result


GitDiffProvider = Callable[[], str | bytes]


class DiffEvidenceService:
    """Prefers checkpoint originals; Git is an injected read-only fallback."""

    def __init__(
        self, *, result_store: ResultStore | None = None,
        model_context_chars: int = 16_000,
    ) -> None:
        if model_context_chars < 256:
            raise ValueError("diff context limit is too small")

        self.result_store = result_store
        self.model_context_chars = model_context_chars

    def from_checkpoint(
        self, manager: CheckpointManager, checkpoint: Checkpoint,
    ) -> DiffEvidence:
        """The diff between a checkpoint's originals and the files as they are now."""

        chunks: list[str] = []
        binary: list[str] = []
        external: list[str] = []
        changed = []

        for relative, item in sorted(checkpoint.paths.items()):
            if item.mutation_type is None:
                continue

            changed.append(relative)
            original = manager.original_bytes(checkpoint, relative)
            current = manager.current_bytes(checkpoint, relative)

            # The file no longer matches what the task left behind, so
            # something outside this task changed it. The reviewer is told,
            # because the diff below is then not the whole story.

            if _digest(current) != item.post_mutation_hash:
                external.append(relative)

            # Binary files get their hashes rather than a diff. The NUL check
            # is deliberate: bytes can decode cleanly and still not be text.

            try:
                before = (original or b"").decode("utf-8")
                after = (current or b"").decode("utf-8")

                if b"\0" in (original or b"") or b"\0" in (current or b""):
                    raise UnicodeError("binary content")
            except UnicodeError:
                binary.append(relative)
                chunks.append(
                    f"Binary change {relative}: {_digest(original) or 'absent'} -> "
                    f"{_digest(current) or 'absent'}\n"
                )

                continue

            # /dev/null is how a unified diff spells a creation or a deletion.

            from_name = "/dev/null" if original is None else f"a/{relative}"
            to_name = "/dev/null" if current is None else f"b/{relative}"
            chunks.extend(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=from_name, tofile=to_name, n=3,
            ))

        return self._bounded(
            "checkpoint", tuple(changed), "".join(chunks), checkpoint.task_id,
            tuple(binary), tuple(external),
        )

    def from_git(
        self, task_id: str, provider: GitDiffProvider,
        *, changed_paths: tuple[str, ...] = (),
    ) -> DiffEvidence:
        """Use a caller-owned SAFE command boundary; never spawn Git directly."""

        output = provider()
        payload = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)

        return self._bounded("git", changed_paths, payload, task_id, (), ())

    def unavailable(self, changed_paths: tuple[str, ...], reason: str) -> DiffEvidence:
        return DiffEvidence(
            "unavailable", changed_paths, "", 0, False,
            limitation=reason[:500],
        )

    def _bounded(
        self, source: str, changed_paths: tuple[str, ...], content: str,
        task_id: str, binary_paths: tuple[str, ...],
        external_paths: tuple[str, ...],
    ) -> DiffEvidence:
        """Bound a diff for the model, keeping the whole of it in the store."""

        encoded = content.encode("utf-8", errors="replace")
        truncated = len(content) > self.model_context_chars
        reference = None

        if truncated and self.result_store is not None:
            reference = self.result_store.put(
                encoded, task_id=task_id,
                metadata={"kind": "review_diff", "source": source,
                          "changed_file_count": len(changed_paths)},
            ).reference

        preview = content[:self.model_context_chars]

        if truncated:
            preview += (
                f"\n[diff bounded: {len(encoded)} bytes; "
                + (f"full evidence {reference}]" if reference else "no result store]")
            )

        return DiffEvidence(
            source, changed_paths, preview, len(encoded), truncated, reference,
            binary_paths, external_paths,
        )

