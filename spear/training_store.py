"""Crash-safe local filesystem storage for canonical training episodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import fcntl
from pathlib import Path
from typing import Iterator, Mapping

from training_data import TrainingDataError, TrainingEpisode


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
