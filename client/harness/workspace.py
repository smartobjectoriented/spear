"""The workspace boundary and where the sandbox exposes it.

``Workspace`` confines filesystem tool paths to declared roots;
``SandboxSpec`` and the mount helpers decide the path those roots answer to
inside the sandbox, so the prompt and the sandbox agree on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from harness.tool_primitives import PathNotFoundError, PathPolicyError


# Paths the sandbox builds its own runtime from. A workspace mounted at, or
# containing, one of these would either be shadowed by it or make it writable,
# so such a workspace keeps the neutral /workspace mount instead.

RUNTIME_MOUNTS = ("/usr", "/bin", "/lib", "/lib64", "/etc", "/proc", "/dev",
                  "/sys", "/run", "/boot")


def identity_mount_for(root: "Path | str",
                       reserved: "tuple[str, ...]" = ()) -> str | None:
    """Where to mount `root` so that its path is the SAME inside and outside.

    Tools that record absolute paths — CMake caches, generated Makefiles,
    compile_commands.json, object files — break when the tree answers to a
    different name inside the sandbox. A build configured on the host then
    fails in the sandbox with "cd /home/.../build: No such file or directory"
    and a CMakeCache pointing at a directory that no longer exists. Binding the
    tree at its own path removes that entire class of failure rather than
    papering over each instance.

    Returns None when the identity path is unsafe, and the caller falls back to
    the neutral mount.
    """
    r = os.path.realpath(str(root))

    if r == "/":
        return None                        # would shadow the whole runtime

    for m in RUNTIME_MOUNTS:
        # At or under a runtime mount: the bind would layer on top of it, and
        # under /usr it would make part of the read-only runtime writable.

        if r == m or r.startswith(m + "/"):
            return None

        # An ancestor of one: mounting here hides the runtime beneath it.

        if m.startswith(r.rstrip("/") + "/"):
            return None

    for res in reserved:
        # The sandbox home and tmpdir are created before this bind; a mount at
        # or above them would hide them and leave HOME/TMPDIR dangling.

        if r == res or res.startswith(r.rstrip("/") + "/"):
            return None

    return r


@dataclass(frozen=True)
class SandboxSpec:
    """Fixed, intentionally small filesystem and environment contract."""

    workspace_mount: str = "/workspace"
    home: str = "/home/sandbox"
    tmpdir: str = "/tmp"

    # /usr/local/bin FIRST, and it is not an afterthought: ib.md tells the
    # operator to symlink each cross-toolchain's whole bin/ there, so leaving
    # it off PATH meant `make so3` died on "aarch64-none-elf-gcc: No such file
    # or directory" while the binary sat in the sandbox all along, mounted and
    # unreachable. It costs no new mount -- /usr is already bound read-only.

    path: str = "/usr/local/bin:/usr/bin:/bin"

    # bitbake refuses to start without a UTF-8 locale, and --clearenv leaves
    # none. ib.md lists it among the host prerequisites for exactly this
    # reason; the failure is "Please make sure locale 'en_US.UTF-8' is
    # available", which reads like a host problem rather than a sandbox one.

    locale: str = "en_US.UTF-8"

    # World-readable identity data a build resolves uids and gids through.

    identity_files: tuple = ("/etc/passwd", "/etc/group")

    # Directory of symlinks that /usr/bin/cc and friends resolve through.

    alternatives: str = "/etc/alternatives"

    # Where cross-toolchains are unpacked. Bound read-only when present; the
    # /usr/local/bin symlinks that name them are useless without their target.

    toolchain_dirs: tuple = ("/opt/toolchains",)


def effective_mount_root(root: "Path | str",
                         spec: "SandboxSpec | None" = None) -> str:
    """Where a workspace rooted at ``root`` is exposed inside the sandbox.

    The prompt must tell the model the same path the sandbox actually binds,
    so both go through this one function. SPEAR_SANDBOX_IDENTITY_MOUNT=0
    restores the historical /workspace mount for debugging.
    """
    spec = spec or SandboxSpec()

    if os.environ.get("SPEAR_SANDBOX_IDENTITY_MOUNT", "1") == "0":
        return spec.workspace_mount

    return identity_mount_for(root, (spec.home, spec.tmpdir)) or spec.workspace_mount


@dataclass(frozen=True)
class Workspace:
    """A canonical root that all filesystem tool paths must stay within."""

    root: Path
    allow_absolute_paths: bool = False
    #: Additional writable trees, declared by the caller.  The launch directory
    #: stays the *primary* root: relative paths resolve there and nowhere else,
    #: so one string never denotes two files.  A secondary root is reached by
    #: absolute path only — which is also what makes the intent explicit in the
    #: audit trail.
    extra_roots: tuple[Path, ...] = ()

    @classmethod
    def from_path(cls, path: str | Path, *, allow_absolute_paths: bool = False,
                  extra_roots: Sequence[str | Path] = ()) -> "Workspace":
        root = Path(path).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise PathPolicyError(f"workspace is not a directory: {root}")

        return cls(root=root, allow_absolute_paths=allow_absolute_paths,
                   extra_roots=cls._canonical_extra_roots(root, extra_roots))

    @staticmethod
    def _canonical_extra_roots(root: Path,
                               candidates: Sequence[str | Path]) -> tuple[Path, ...]:
        """Canonical, deduplicated, non-nested, existing directories.

        A root nested in another would be mounted twice and make a path's label
        ambiguous.  A root that no longer exists is dropped rather than fatal:
        a stale entry in a corpus registry must not stop the assistant from
        starting.
        """
        kept: list[Path] = []

        for candidate in candidates:
            try:
                resolved = Path(candidate).expanduser().resolve(strict=True)
            except OSError:
                continue

            if not resolved.is_dir():
                continue

            if resolved == root or resolved.is_relative_to(root):
                continue

            if any(resolved.is_relative_to(seen) for seen in kept):
                continue

            kept = [seen for seen in kept if not seen.is_relative_to(resolved)]
            kept.append(resolved)

        return tuple(kept)

    @property
    def roots(self) -> tuple[Path, ...]:
        """Every writable tree, primary first."""
        return (self.root,) + self.extra_roots

    def boundary_hint(self, *, readable_elsewhere: bool = True) -> str:
        """Name the trees a refusal leaves available, and the way out of it.

        A refusal that only states the rule is a dead end: the model re-ran the
        same denied command, then narrated "let me try a relative path" and
        issued the same absolute one. What it lacked was the two facts this
        sentence carries -- which trees it may write, and that looking at a
        file elsewhere is a different, permitted thing.
        """
        shown = [str(root) for root in self.roots[:3]]
        more = len(self.roots) - len(shown)
        listed = ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")
        hint = f" Writable trees: {listed}."

        if readable_elsewhere:
            hint += (" Anywhere else is READ-ONLY and reachable from bash by "
                     "absolute path (cat/sed/ls/grep), not from the file tools."
                     " To make a tree writable, declare it as a corpus.")

        return hint

    def mount_map(self, primary_mount: str = "/workspace",
                  identity: bool = False) -> dict[str, str]:
        """Host root → path it is exposed at inside the sandbox.

        The model cannot use a secondary tree from bash without this: on the
        host it is /home/…/so3, inside the sandbox it is /workspaces/so3.
        Names are made unique because two roots may share a basename.
        """

        if identity:
            # Every root keeps its own path; nothing has to be renamed, so the
            # uniquifying below is unnecessary and the map is the identity.

            out = {str(self.root): primary_mount}

            for extra in self.extra_roots:
                mount = effective_mount_root(extra)
                out[str(extra)] = (mount if mount != SandboxSpec().workspace_mount
                                   else f"{primary_mount}s/{extra.name}")

            return out

        mounts = {str(self.root): primary_mount}
        used: set[str] = set()

        for extra in self.extra_roots:
            name, suffix = extra.name, 2

            while name in used:
                name = f"{extra.name}-{suffix}"
                suffix += 1

            used.add(name)
            mounts[str(extra)] = f"{primary_mount}s/{name}"

        return mounts

    def _from_mount_path(self, candidate: Path) -> tuple[Path, bool]:
        """Translate a sandbox mount path back to its host root.

        bash sees a secondary root at /workspaces/<name>; the file tools act on
        the host. Accepting the mount path everywhere gives the model a single
        addressing scheme instead of two that silently disagree. Translation is
        done BEFORE containment, so `.. ` out of a mount is still caught.
        """

        if not candidate.is_absolute():
            return candidate, False

        # Both namings are accepted: the /workspace literals, and the mounts
        # actually in force. Under the identity mount the latter are the host
        # paths themselves — which the prompt now tells the model to use, so
        # refusing them here would put bash and edit_file back in disagreement.

        effective = effective_mount_root(self.root)
        pairs = list(self.mount_map().items())

        if effective != SandboxSpec().workspace_mount:
            # Both maps are keyed by host root, so they are chained as pairs:
            # merging the dicts would drop one naming for the other.

            pairs += list(self.mount_map(effective, identity=True).items())

        for host, mount in pairs:
            mount_path = Path(mount)

            if candidate == mount_path:
                return Path(host), True

            if candidate.is_relative_to(mount_path):
                return Path(host) / candidate.relative_to(mount_path), True

        return candidate, False

    def _containing_root(self, resolved: Path) -> Path | None:
        for candidate in self.roots:
            if resolved.is_relative_to(candidate):
                return candidate

        return None

    def resolve(self, raw_path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a user path and reject traversal, symlink, or root escapes.

        ``Path.resolve(strict=False)`` follows every existing symlink component,
        including the parent of a target file that does not yet exist.  Checking
        the resulting canonical path before opening it prevents symlink escapes.
        """
        value = str(raw_path).strip()

        if not value:
            raise PathPolicyError("path is empty")

        candidate, from_mount = self._from_mount_path(Path(value).expanduser())
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve(
            strict=False
        )

        if candidate.is_absolute():
            # An absolute path is how a secondary root is reached at all.  Into
            # the PRIMARY root it stays gated by the legacy flag, so declaring
            # extra roots never silently widens what the launch directory
            # accepts.

            containing = self._containing_root(resolved)

            if containing is None:
                raise PathPolicyError(
                    "path escapes the workspace." + self.boundary_hint())

            if (containing == self.root and not self.allow_absolute_paths
                    and not from_mount):

                # The legacy flag gates HOST absolute paths into the launch
                # directory. A /workspace/... path is the sandbox's own naming,
                # which the harness itself told the model to use.

                raise PathPolicyError(
                    "absolute paths are not allowed for the launch directory — "
                    f"pass it relative to {self.root}, e.g. "
                    f"'{resolved.relative_to(self.root).as_posix() or '.'}'")
        elif not resolved.is_relative_to(self.root):
            # Relative paths resolve against the primary root only.

            raise PathPolicyError(
                "path escapes the workspace." + self.boundary_hint())

        if must_exist and not resolved.exists():
            raise PathNotFoundError(f"path does not exist: {value}")

        return resolved

    def relative(self, path: str | Path) -> str:
        # Audit callers may hold an already-resolved absolute Path even when
        # user-supplied absolute paths are disabled.  This helper only formats
        # a checked workspace-relative label; it never grants path access.

        candidate = Path(path).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve(
            strict=False
        )
        containing = self._containing_root(resolved)

        if containing is None:
            raise PathPolicyError("path escapes the workspace")

        label = resolved.relative_to(containing).as_posix() or "."

        # Qualify secondary roots so a label can never be mistaken for a file
        # of the launch directory — in the audit trail above all.

        return label if containing == self.root else f"{containing.name}:{label}"
