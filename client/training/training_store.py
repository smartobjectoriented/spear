"""Crash-safe local filesystem storage for canonical training episodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from training.training_data import TrainingDataError, TrainingEpisode


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class TrainingStoreError(RuntimeError):
    pass


class TrainingStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.episodes = self.root / "episodes"
        self.drafts = self.root / "drafts"
        self.manifests = self.root / "manifests"

        for directory in (self.root, self.episodes, self.drafts, self.manifests):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(directory, 0o700)

    @staticmethod
    def _validate_id(episode_id: str) -> str:
        if not isinstance(episode_id, str) or not _SAFE_ID.fullmatch(episode_id):
            raise TrainingStoreError("unsafe episode id")

        return episode_id

    def _path(self, directory: Path, episode_id: str, suffix: str) -> Path:
        """The file an episode id names, confined to its directory."""

        name = self._validate_id(episode_id) + suffix
        path = (directory / name).resolve()

        if path.parent != directory.resolve():
            raise TrainingStoreError("episode path escaped training store")

        return path

    @staticmethod
    def _serialize(value: object) -> bytes:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("utf-8")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)

        try:
            os.fchmod(descriptor, 0o600)

            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)

            # The directory is fsynced too: without it the rename itself can be
            # lost in a crash, leaving the episode written but not present.

            directory_fd = os.open(path.parent, os.O_RDONLY)

            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_draft(self, episode_id: str, payload: Mapping[str, object]) -> Path:
        path = self._path(self.drafts, episode_id, ".json")
        self._atomic_write(path, self._serialize(payload))

        return path

    def discard_draft(self, episode_id: str) -> None:
        path = self._path(self.drafts, episode_id, ".json")

        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def load_draft(self, episode_id: str) -> Mapping[str, object] | None:
        path = self._path(self.drafts, episode_id, ".json")

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingStoreError(f"could not load training draft: {exc}") from exc

        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise TrainingStoreError("malformed training draft")

        return value

    def save_episode(self, episode: TrainingEpisode) -> Path:
        """Write an episode once, immutably, and index it exactly once."""

        # Round-tripped first: an episode that cannot be read back is refused
        # before anything is written, not discovered later on load.

        TrainingEpisode.from_dict(episode.to_dict())

        path = self._path(self.episodes, episode.episode_id, ".json")
        checksum_path = self._path(self.episodes, episode.episode_id, ".sha256")
        content = self._serialize(episode.to_dict())
        checksum = hashlib.sha256(content).hexdigest()
        manifest = self.manifests / "episodes.jsonl"
        metadata = {
            "episode_id": episode.episode_id, "task_id": episode.task_id,
            "timestamp": episode.timestamp, "checksum": checksum,
            "eligibility": episode.training_metadata.get("eligibility"),
        }

        # The lock covers both the episode file and the manifest append, so
        # concurrent writers cannot interleave and double-index an episode.

        lock_path = self.manifests / ".store.lock"
        lock = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)

        try:
            fcntl.flock(lock, fcntl.LOCK_EX)

            # Episodes are immutable: re-saving identical content is a no-op,
            # while the same id with different content is a bug worth raising.

            if path.exists():
                existing = path.read_bytes()

                if hashlib.sha256(existing).hexdigest() != checksum:
                    raise TrainingStoreError(
                        "immutable episode already exists with other content"
                    )

                if not checksum_path.exists():
                    self._atomic_write(
                        checksum_path, (checksum + "\n").encode("ascii")
                    )
            else:
                self._atomic_write(path, content)
                self._atomic_write(checksum_path, (checksum + "\n").encode("ascii"))

            # The manifest is an append-only index that a crash may have left
            # without its entry, so it is scanned before appending.

            indexed = False

            if manifest.exists():
                with manifest.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if item.get("episode_id") == episode.episode_id:
                            indexed = True
                            break

            if not indexed:
                descriptor = os.open(
                    manifest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600,
                )

                try:
                    os.write(descriptor, self._serialize(metadata))
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)

        return path

    def exists(self, episode_id: str) -> bool:
        return self._path(self.episodes, episode_id, ".json").is_file()

    def load_episode(self, episode_id: str) -> TrainingEpisode:
        path = self._path(self.episodes, episode_id, ".json")
        checksum_path = self._path(self.episodes, episode_id, ".sha256")

        # The sidecar checksum is verified on every load: a training episode
        # is evidence, and silently reading a corrupted one is worse than failing.

        try:
            content = path.read_bytes()
            expected = checksum_path.read_text(encoding="ascii").strip()

            if hashlib.sha256(content).hexdigest() != expected:
                raise TrainingStoreError("training episode checksum mismatch")

            return TrainingEpisode.from_dict(json.loads(content))
        except (OSError, json.JSONDecodeError, TrainingDataError) as exc:
            if isinstance(exc, TrainingStoreError):
                raise

            raise TrainingStoreError(f"could not load training episode: {exc}") from exc

    def record_verdict(self, episode_id: str, verdict: str) -> None:
        """Append an operator's /good or /bad on an episode.

        Episodes are sealed by checksum, so a verdict given after the fact is
        kept beside them, in its own append-only journal, and the dataset
        builders read both.
        """

        if verdict not in {"good", "bad"}:
            raise TrainingStoreError(f"unknown verdict: {verdict}")

        self._append(episode_id, "operator-verdicts.jsonl",
                     {"episode_id": episode_id, "verdict": verdict})

    def operator_verdicts(self) -> dict[str, str]:
        """The latest operator verdict on each episode that has one."""

        verdicts: dict[str, str] = {}

        for item in self._journal("operator-verdicts.jsonl"):
            if item.get("verdict") in {"good", "bad"}:
                verdicts[str(item.get("episode_id"))] = item["verdict"]

        return verdicts

    #: Longest correction kept: a whole answer, not a document.
    MAX_CORRECTION_CHARS = 64_000

    def record_correction(self, episode_id: str, text: str) -> None:
        """Append an operator's corrected answer to an episode (/fix).

        The answer the model gave stays in the sealed episode; the correction
        goes beside it, like a verdict, and the dataset builders turn the two
        into a target and a preference pair over the same context.
        """

        text = (text or "").strip()

        if not text or len(text) > self.MAX_CORRECTION_CHARS:
            raise TrainingStoreError("a correction is a non-empty answer of at most "
                                     f"{self.MAX_CORRECTION_CHARS} characters")

        self._append(episode_id, "operator-corrections.jsonl",
                     {"episode_id": episode_id, "correction": text})

    def operator_corrections(self) -> dict[str, str]:
        """The latest operator correction of each episode that has one."""

        corrections: dict[str, str] = {}

        for item in self._journal("operator-corrections.jsonl"):
            if isinstance(item.get("correction"), str) and item["correction"].strip():
                corrections[str(item.get("episode_id"))] = item["correction"]

        return corrections

    def _append(self, episode_id: str, name: str, entry: Mapping[str, object]) -> None:
        self._path(self.episodes, episode_id, ".json")
        line = json.dumps({**entry, "at": datetime.now(timezone.utc).isoformat(
            timespec="seconds")}, sort_keys=True, ensure_ascii=False)
        lock = os.open(self.manifests / ".store.lock", os.O_RDWR | os.O_CREAT, 0o600)

        try:
            fcntl.flock(lock, fcntl.LOCK_EX)

            with (self.manifests / name).open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)

    def _journal(self, name: str) -> Iterator[Mapping[str, object]]:
        journal = self.manifests / name

        if not journal.exists():
            return

        with journal.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(item, dict):
                    yield item

    def latest_episode_id(self, session_id: str | None = None) -> str | None:
        """The most recently stored episode, of this session when one is named."""

        for item in reversed(list(self.iterate_metadata())):
            episode_id = str(item.get("episode_id", ""))

            if session_id is None:
                return episode_id

            try:
                if self.load_episode(episode_id).session_id == session_id:
                    return episode_id
            except TrainingStoreError:
                continue

        return None

    def iterate_metadata(self, *, eligibility: str | None = None) -> Iterator[Mapping[str, object]]:
        manifest = self.manifests / "episodes.jsonl"

        if not manifest.exists():
            return

        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                # A torn final line from an interrupted append is skipped: the
                # episodes themselves are intact, and the index is rebuildable.

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if eligibility is None or item.get("eligibility") == eligibility:
                    yield item
