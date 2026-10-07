"""What a turn asked about, and where its work is allowed to land.

Three boundaries, all read from the operator's own words and the tree, never
from what the model found along the way:

  a question is not a write      -- see request_intent.advisory
  a named target is not the tree -- sibling_write_refusal
  this project is not the disk   -- outside_project_refusal

The second and third exist because of the same run. Asked whether an artifact
of ONE named target could live in another directory, a turn grepped, found the
same line in seven analogous files for seven other targets, and edited all
seven: similarity was read as instruction. Its build command then did not exist
at the root it guessed, so it searched the whole home directory, found a
build.sh in an unrelated repository, and read that project's scripts as if
they described this one.

Investigation is not bounded here: reading a sibling target to understand a
shared mechanism is how the mechanism is understood. Only WRITING to a sibling,
and SEARCHING outside the project, are.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import request_intent

# ---------------------------------------------------------------------------
# Scope anchors.

# The words a request names things with. Deliberately permissive: a token only
# ever matters when it is ALSO the distinguishing part of a file name in the
# tree, so ordinary words cost nothing.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*[A-Za-z0-9]|[A-Za-z0-9]")

# The operator widened the scope in so many words. A target family named in
# the plural with a quantifier, or "everywhere": these, and only these, are
# permission to touch the siblings.
_BROADENED_RE = re.compile(
    r"\b(?:all|every|each)\s+(?:of\s+the\s+|the\s+)?(?:\w+[\s-]){0,2}?"
    r"(?:platforms?|targets?|boards?|variants?|configurations?|configs?|"
    r"modules?|machines?|architectures?|archs?|files?|places?|occurrences?|"
    r"instances?|backends?|drivers?|flavou?rs?|builds?)\b"
    r"|\beverywhere\b|\bacross\s+(?:the\s+)?(?:board|tree|codebase|repo)\b"
    r"|\b(?:toutes?|tous)\s+les\s+\w+|\bpartout\b",
    re.I)

# Directory or file names that denote shared implementation by convention,
# not a member of a target family. A target called "common" is the place the
# named target's shared code lives.
_SHARED_NAMES = frozenset({"common", "shared", "generic", "default", "base",
                           "core", "include", "lib", "all", "util", "utils"})

# How many analogous entries make a family. Two could be a coincidence of
# naming; three is a pattern somebody set up.
_FAMILY_MIN = 3

# Where the harness writes the tool results of a turn into its own history.
# Anchors are read from the operator's words, not from what grep printed.
_HARNESS_SUFFIX = re.compile(r"\n\s*\[Tools executed during this turn", re.S)


def operator_text(text: str) -> str:
    """The operator's words, without the tool transcript history appends."""
    match = _HARNESS_SUFFIX.search(text or "")

    return (text or "")[:match.start()] if match else (text or "")


# Words that name nothing, whatever a tree calls its files.
_STOPWORDS = frozenset("""
a an and are as at be by can could do does for from have how i if in instead
is it its make me my no not of on or our please should so that the then this
to us we what when where which why will with would yes you
""".split())


def anchors(text: str) -> frozenset[str]:
    """The tokens a request could be naming a target with, case-folded."""
    return frozenset(token.casefold()
                     for token in _TOKEN_RE.findall(operator_text(text))
                     if len(token) > 1 and token.casefold() not in _STOPWORDS)


def broadened(text: str) -> bool:
    """Did the operator ask for the change on every member of a family?"""
    return bool(_BROADENED_RE.search(operator_text(text)))


@dataclass(frozen=True)
class RequestScope:
    """What this turn asked about, in the tree it runs in."""

    root: str = ""
    text: str = ""
    anchors: frozenset[str] = field(default_factory=frozenset)
    broadened: bool = False
    extra_roots: tuple[str, ...] = ()

    @classmethod
    def of(cls, current: str, *, previous: str = "", root: str = "",
           extra_roots=()) -> "RequestScope":
        """The scope of the current turn.

        A follow-up that only says "yes, do it" names nothing: what it agrees
        to was named by the turn before, so that turn's anchors carry. Any
        other turn stands on its own words -- the previous question is not a
        second instruction.
        """
        current = operator_text(current)
        words = current

        if previous and request_intent.follow_up(current):
            words = operator_text(previous) + "\n" + current

        return cls(root=str(root or ""), text=words, anchors=anchors(words),
                   broadened=broadened(words),
                   extra_roots=tuple(str(item) for item in extra_roots or ()))

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(item for item in (self.root, *self.extra_roots) if item)


def _split_name(name: str) -> tuple[str, str]:
    """(stem, extension) of a directory entry; directories have no extension."""
    stem, dot, ext = name.partition(".")

    return (stem, dot + ext) if stem else (name, "")


def _entries(directory: Path):
    try:
        return [entry for entry in os.scandir(directory)
                if not entry.name.startswith(".")]
    except OSError:
        return []


def _affix(sibling: str, anchor: str, candidate: str):
    """(head, tail) when `candidate` is `sibling` with `anchor` swapped out.

    alpha.cfg   -> beta.cfg          ("", ".cfg")
    a_guest.cfg -> b_x_guest.cfg     ("", "_guest.cfg")
    net-x.mk    -> app.mk            None: the rest of the name differs too
    """
    at = sibling.casefold().find(anchor)

    if at < 0:
        return None

    head, tail = sibling[:at].casefold(), sibling[at + len(anchor):].casefold()
    member = _member(candidate, head, tail)

    # The member must be a name in its own right, and not the anchor wearing
    # a suffix: "alpha_guest" is a variant of "alpha", not a sibling.
    if not member or anchor in member \
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", member) is None:
        return None

    return head, tail


def _member(name: str, head: str, tail: str) -> str:
    """The part of `name` that differs within the family, or ""."""
    name = name.casefold()

    if len(name) <= len(head) + len(tail) \
            or not (name.startswith(head) and name.endswith(tail)):
        return ""

    return name[len(head):len(name) - len(tail)]


def _children(path: str) -> frozenset[str]:
    return frozenset(entry.name for entry in _entries(Path(path)))


def _alike(first: str, second: str) -> bool:
    """Do two directories hold the same kind of thing?

    Two members of one family are laid out alike -- each target directory
    carries the same handful of files. A directory that merely sits next to
    one is not its analogue: a project's build/ is not a sibling of its
    linux/ because both are top-level names. Nor are two directories that
    share only sub-directories: BitBake layers all hold classes/ and conf/,
    and what makes each one is its own recipes-<name>/. Targets share files.
    """
    a, b = _children(first), _children(second)

    if not a or not b:
        return False

    shared = a & b

    if not any(os.path.isfile(os.path.join(first, name)) for name in shared):
        return False

    return len(shared) * 2 >= len(a | b)


def _sibling_of(directory: Path, component: str, scope: RequestScope):
    """The anchor-bearing entry `component` is an analogue of, or None."""
    stem, ext = _split_name(component)

    if stem.casefold() in _SHARED_NAMES:
        return None

    path = directory / component
    candidate_is_dir = path.is_dir() or (not path.exists() and not ext)
    analogous = [entry for entry in _entries(directory)
                 if entry.name != component
                 and entry.is_dir() == candidate_is_dir
                 and _split_name(entry.name)[1] == ext]

    if len(analogous) + 1 < _FAMILY_MIN:
        return None

    for entry in analogous:
        entry_stem = _split_name(entry.name)[0].casefold()

        # Only a whole segment of the neighbour's name is an anchor: "a"
        # inside "bsp-linux" names nothing.
        segments = set(re.split(r"[_.-]", entry_stem)) | {entry_stem}

        for anchor in scope.anchors & segments:
            affix = _affix(entry.name, anchor, component)

            if affix is None:
                continue

            # A request that names two members of a family is not narrowed
            # to either: it spans the family. "move it from a/x to b/y" names
            # the layers a and b as places, and neither is a target.
            members = {_member(item.name, *affix) for item in analogous}
            members.add(_member(component, *affix))

            if len(scope.anchors & members) > 1:
                continue

            # Directories are a family only when they hold the same kind of
            # content: meta-a and meta-b are two layers, not two targets, and
            # a name shared around them is not a reason to call them one.
            if candidate_is_dir and not _alike(entry.path, str(path)):
                continue

            return entry.name, anchor

    return None


def sibling_write_refusal(scope: RequestScope | None, target: str) -> str:
    """Why a write to `target` is outside what the turn asked for, or "".

    A shared file the named target depends on passes: nothing in its name is
    the analogue of an anchor. A file whose name is another member of the
    family the anchor belongs to is refused, unless the operator widened the
    request to the whole family. The refusal is not permanent: a write that
    states in `scope_reason` why the named target cannot be changed without
    it (a shared source that would otherwise leave the tree inconsistent) is
    recorded and let through by the caller.
    """
    if scope is None or scope.broadened or not scope.anchors or not target:
        return ""

    root = Path(scope.root or ".")
    path = Path(target)

    if not path.is_absolute():
        path = root / path

    try:
        relative = path.resolve(strict=False).relative_to(root.resolve())
    except (ValueError, OSError):
        return ""

    directory = root.resolve()

    for component in relative.parts:
        found = _sibling_of(directory, component, scope)

        if found is not None:
            neighbour, anchor = found

            return (
                f"the request is about '{anchor}', and '{relative}' belongs to "
                f"another member of the same family ('{component}' is the "
                f"analogue of '{neighbour}'). A similar line in a sibling is "
                f"not a reason to change it: keep the change on '{anchor}'. "
                f"If this file genuinely has to change for '{anchor}' to work "
                f"-- a shared source that would otherwise leave the tree "
                f"inconsistent or unbuildable -- call again with scope_reason "
                f"saying exactly why. If the user wants every target changed, "
                f"that is theirs to ask for: name the other files in your "
                f"answer instead.")

        directory = directory / component

    return ""


# ---------------------------------------------------------------------------
# Shell writes.

# Programs whose file operands are what they write. The destination of cp,
# mv, install and ln is the last operand; every other operand of these is a
# target in its own right.
_LAST_OPERAND_WRITERS = frozenset({"cp", "mv", "install", "ln", "rsync"})
_ALL_OPERAND_WRITERS = frozenset({"rm", "rmdir", "touch", "mkdir", "truncate",
                                  "chmod", "chown", "tee", "unlink", "shred"})

# An interpreter handed its program inline: what it writes is not visible to
# any parser here. It is judged by the paths its text mentions.
_INLINE_CODE = {"python": {"-c"}, "python3": {"-c"}, "perl": {"-e", "-i", "-pi",
                "-pie", "-ne"}, "ruby": {"-e"}, "node": {"-e"},
                "sh": {"-c"}, "bash": {"-c"}, "zsh": {"-c"}, "awk": set(),
                "gawk": set()}
_REDIRECT = re.compile(r"^\d*>{1,2}\|?(.*)$")


def _shell_destinations(argv):
    """The paths one shell stage writes, as far as its words say."""
    binary = Path(argv[0]).name
    operands = [arg for arg in argv[1:] if not arg.startswith("-")]
    found = []

    for index, arg in enumerate(argv):
        match = _REDIRECT.match(arg)

        if match and index:
            target = match.group(1) or (argv[index + 1]
                                        if index + 1 < len(argv) else "")

            if target and not target.startswith("&"):
                found.append(target)

    if binary in _LAST_OPERAND_WRITERS and len(operands) >= 2:
        found.append(operands[-1])
        if binary == "mv":
            found.extend(operands[:-1])    # a move also writes its source away
    elif binary in _ALL_OPERAND_WRITERS:
        found.extend(operands[1:] if binary in {"chmod", "chown"} else operands)
    elif binary == "sed" and any(arg.startswith("-i") or arg == "--in-place"
                                 for arg in argv[1:]):
        found.extend(operands[1:])
    elif binary == "dd":
        found.extend(arg[3:] for arg in argv[1:] if arg.startswith("of="))

    return [item for item in found if item and not item.startswith("/dev/")]


def _opaque(argv):
    binary = Path(argv[0]).name.rstrip("0123456789.") or Path(argv[0]).name
    flags = _INLINE_CODE.get(Path(argv[0]).name, _INLINE_CODE.get(binary))

    if flags is None:
        return False

    return not flags or any(arg in flags or (binary == "perl"
                                             and arg.startswith("-i"))
                            for arg in argv[1:])


def _expand_in(scope: RequestScope, token: str) -> list[str]:
    """`token` as the project paths it denotes: globs expanded, or itself."""
    if not any(char in token for char in "*?["):
        return [token]

    import glob

    base = scope.root or "."
    pattern = token if token.startswith("/") else os.path.join(base, token)

    return sorted(glob.glob(pattern))[:200] or [token]


def _children_matching(scope: RequestScope, path: str, patterns) -> list[str]:
    """The entries under a directory a find over it could hand to a writer.

    At any depth, as find walks: `find board -name post_image.sh | xargs
    sed -i` reaches board/<platform>/post_image.sh, two levels down.
    """
    import fnmatch

    full = path if os.path.isabs(path) else os.path.join(scope.root or ".", path)

    if not os.path.isdir(full):
        return []

    found = []

    for directory, subdirectories, files in os.walk(full):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]

        for name in subdirectories + files:
            if not patterns or any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                found.append(os.path.join(directory, name))

                if len(found) >= 200:
                    return found

    return found


def _writes_fed(argv) -> bool:
    """A writer whose files arrive from stdin or from find: xargs, -exec."""
    words = [Path(arg).name for arg in argv]

    if words[0] == "xargs" or "-exec" in argv or "-execdir" in argv:
        rest = argv[1:]

        return any(Path(arg).name in (_ALL_OPERAND_WRITERS
                                      | _LAST_OPERAND_WRITERS | {"sed", "perl"})
                   for arg in rest) and (
            "-i" in rest or any(arg.startswith("-i") for arg in rest)
            or any(Path(arg).name in _ALL_OPERAND_WRITERS
                   | _LAST_OPERAND_WRITERS for arg in rest))

    return False


_WORD_PATH = re.compile(r"[\w./-]*/[\w./-]+|[\w-]+\.[A-Za-z]\w{0,6}")


def _stage_directories(stages, start: str) -> list[str]:
    """The directory each stage runs in: `start`, moved by every `cd`."""
    directories = []
    directory = start

    for argv in stages:
        directories.append(directory)

        if Path(argv[0]).name in ("cd", "pushd") and len(argv) > 1 \
                and "$" not in argv[1] and argv[1] != "-":
            directory = os.path.normpath(os.path.join(directory, _expand(argv[1])))

    return directories


def _at(directory: str, token: str) -> str:
    """A path operand as the stage running in `directory` means it.

    A redirection written against its file (`>>x`) and dd's `of=x` keep
    their prefix. A word that is not a path -- inline code, quoted text --
    is left as it is.
    """
    redirect = _REDIRECT.match(token)

    if redirect:
        target = redirect.group(1)

        if not target or target.startswith("&"):
            return token

        return token[:len(token) - len(target)] + _at(directory, target)

    if token.startswith("of="):
        return "of=" + _at(directory, token[3:])

    if (not token or token.startswith(("/", "$", "~", "-", ">", "<"))
            or re.search(r"[\s'\"()=;]", token)):
        return token

    return os.path.join(directory, token)


def shell_write_targets(command: str, scope: RequestScope, cwd: str | None = None) -> list[str]:
    """Every path a shell command may write: a redirection, the destination
    of cp/mv/install/ln, the operands of rm/touch/mkdir/tee/truncate/sed -i.
    An interpreter given its program inline (python -c, perl -i, sh -c, awk)
    cannot be parsed for its writes, so every path its text mentions counts
    as one -- and so does every path a command names when its destination is
    only decided at run time (`$f` in a loop, `find -exec ... {}`, xargs).

    A relative operand is taken from the directory its stage runs in: `cwd`
    (where the session stands, or the command's workdir), moved by every
    `cd` before it. Read from the project root instead, `cd board && echo >
    rpi4/x` wrote to a sibling the same redirect by full path could not.
    """
    if not command:
        return []

    words = list(_split_stages(command))
    patterns = [argv[index + 1] for argv in words
                for index, arg in enumerate(argv[:-1])
                if arg in ("-name", "-iname")]
    directories = _stage_directories(words, cwd or scope.root or ".")
    stages = [[argv[0]] + [token if index and argv[index - 1] in ("-name", "-iname")
                           else _at(directory, token)
                           for index, token in enumerate(argv) if index]
              for argv, directory in zip(words, directories)]
    # What the whole command names, globs expanded in the project: the
    # candidates when a write's destination is only known at run time.
    mentioned = [path for argv in stages for token in argv[1:]
                 for path in _expand_in(scope, token)]
    mentioned += [path for item in list(mentioned)
                  for path in _children_matching(scope, item, patterns)]
    targets = []

    for argv in stages:
        destinations = _shell_destinations(argv)
        targets += [path for item in destinations for path in _expand_in(scope, item)]

        if _opaque(argv) or any("$" in item or "{}" in item
                                for item in destinations) or (
                _writes_fed(argv) and not destinations):
            targets += mentioned
            targets += _WORD_PATH.findall(" ".join(argv[1:]))

    return list(dict.fromkeys(targets))


def shell_write_refusal(command: str, scope: RequestScope | None,
                        cwd: str | None = None) -> str:
    """Why this command writes outside the requested target, or "".

    The same judgement as sibling_write_refusal, applied to every path the
    command writes (shell_write_targets). Builds and tests name no sibling
    destination and pass.
    """
    if scope is None or scope.broadened or not scope.anchors or not command:
        return ""

    for target in shell_write_targets(command, scope, cwd):
        refusal = sibling_write_refusal(scope, target)

        if refusal:
            return (refusal.split(" If this file genuinely")[0]
                    + " A shell command is not a way "
                    "around this: make the change with edit_file, on "
                    "the requested target, or give scope_reason there.")

    return ""


# ---------------------------------------------------------------------------
# Workspace discipline.

# Commands that walk a tree. Pointed outside the project they are a search of
# somebody else's disk, which is what the observed run did.
_SEARCHERS = frozenset({"find", "locate", "mlocate", "plocate", "fd", "fdfind",
                        "tree", "du", "rg", "ag", "ack"})
_RECURSIVE_FLAG = re.compile(r"^-(?:[a-zA-Z]*[rR][a-zA-Z]*|-recursive)$")

_PATH_TOKEN = re.compile(r"(?:~|\$HOME|\$\{HOME\})(?:/[^\s'\";|&<>()]*)?"
                         r"|/[^\s'\";|&<>()]*")


# Words that open a compound command; the program is the word after them.
_SHELL_KEYWORDS = frozenset({"do", "then", "else", "elif", "time", "!", "{",
                             "(", "exec", "nohup", "env", "sudo", "command"})


def _split_stages(command: str):
    # find's `-exec ... \;` terminator is not a command separator; `+` ends
    # the same -exec without looking like one.
    command = command.replace("\\;", "+").replace("';'", "+")

    for stage in re.split(r"&&|\|\||[;|\n]", command):
        try:
            argv = shlex.split(stage, posix=True)
        except ValueError:
            argv = stage.split()

        while argv and argv[0] in _SHELL_KEYWORDS:
            argv = argv[1:]

        if argv:
            yield argv


def _expand(token: str) -> str:
    home = str(Path.home())

    for prefix in ("${HOME}", "$HOME", "~"):
        if token.startswith(prefix):
            return os.path.normpath(home + token[len(prefix):])

    return os.path.normpath(token)


def _inside(path: str, roots) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/")
               for root in roots)


def _named_by_operator(path: str, text: str) -> bool:
    folded = (text or "").casefold()

    return bool(path) and (path.casefold() in folded
                           or path.replace(str(Path.home()), "~").casefold()
                           in folded)


def outside_project_refusal(command: str, scope: RequestScope | None) -> str:
    """Why this command leaves the project it runs in, or "".

    Allowed: anything under the project's own roots, and system paths read
    by name (a header, a toolchain binary). Refused, unless the operator
    named the path this turn:

      - a search (find, locate, grep -r, ...) rooted outside the project,
        or with no root at all when it is a database search like locate;
      - any path in the operator's home directory outside the project --
        another checkout is another project, not a source of this one's
        build commands.
    """
    if scope is None or not scope.roots or not command:
        return ""

    roots = [os.path.normpath(root) for root in scope.roots]
    roots += ["/workspace"]
    home = str(Path.home())

    for argv in _split_stages(command):
        binary = Path(argv[0]).name
        searching = binary in _SEARCHERS or (
            binary in {"grep", "egrep", "fgrep", "ls"}
            and any(_RECURSIVE_FLAG.match(arg) for arg in argv[1:]))

        if binary in {"locate", "mlocate", "plocate"}:
            return (f"'{binary}' searches the whole machine, not this "
                    f"project. {discovery_hint(scope)}")

        for token in argv[1:]:
            if not _PATH_TOKEN.fullmatch(token):
                continue

            path = _expand(token)

            if _inside(path, roots) or _named_by_operator(token, scope.text) \
                    or _named_by_operator(path, scope.text):
                continue

            above_project = any(_inside(root, [path]) for root in roots)
            in_home = _inside(path, [home])

            if searching and (above_project or in_home
                              or not path.startswith(("/usr/include",
                                                      "/usr/share/doc"))):
                return (f"'{token}' is outside the project, and a search there "
                        f"looks through other people's trees for this one's "
                        f"answers. {discovery_hint(scope)}")

            if in_home:
                return (f"'{token}' is in another tree than this project "
                        f"({roots[0]}). What is there describes that project, "
                        f"not this one. {discovery_hint(scope)}")

    return ""


# ---------------------------------------------------------------------------
# Build discovery.

# The places a tree says how it is built, in the order they are worth reading.
_TOP_LEVEL_BUILD = ("Makefile", "makefile", "GNUmakefile", "CMakeLists.txt",
                    "meson.build", "build.sh", "Cargo.toml", "go.mod",
                    "package.json", "pyproject.toml", "setup.py", "configure",
                    "env.sh")
_SCRIPT_DIRS = ("scripts", "script", "tools", "bin", "build")
_DOCS = ("README.md", "README.rst", "README", "README.txt", "BUILDING.md",
         "INSTALL.md", "CONTRIBUTING.md", "doc", "docs")
_CI = (".gitlab-ci.yml", ".github/workflows", "Jenkinsfile", ".travis.yml",
       "azure-pipelines.yml")
_ENTRY_RE = re.compile(r"(?:^|[-_])(?:build|make|compile|test|check|run)"
                       r"(?:[-_.]|$)", re.I)


def discover_build_entrypoints(root: str, limit: int = 12) -> list[str]:
    """Project-local files that say how this tree is built, bounded.

    Reads only the root and the conventional first-level script directories;
    never walks, never leaves the root. Order: top-level build files, script
    entry points, CI, docs.
    """
    base = Path(root or ".")
    found: list[str] = []

    def add(relative):
        if relative not in found and len(found) < limit:
            found.append(relative)

    for name in _TOP_LEVEL_BUILD:
        if (base / name).is_file():
            add(name)

    for directory in _SCRIPT_DIRS:
        for entry in sorted(_entries(base / directory), key=lambda e: e.name):
            if entry.is_file() and _ENTRY_RE.search(entry.name) \
                    and (entry.name.endswith((".sh", ".py", ".mk"))
                         or os.access(entry.path, os.X_OK)):
                add(f"{directory}/{entry.name}")

    for name in _CI + _DOCS:
        if (base / name).exists():
            add(name)

    return found


def discovery_hint(scope: RequestScope | None) -> str:
    """The bounded order to find this project's commands in, with results."""
    root = scope.root if scope is not None else ""
    found = discover_build_entrypoints(root) if root else []
    listed = (" Found in this project: " + ", ".join(found) + "."
              if found else "")

    return ("Do not guess another command and do not search outside the "
            f"project. Find its own: `git rev-parse --show-toplevel`, then "
            f"the top level, scripts/, README/docs and CI configuration, and "
            f"choose the command from what they say.{listed}")


# The shell's own words for "that program is not there".
_MISSING_RE = re.compile(r"(?:command not found|No such file or directory|"
                         r"not found)\s*$", re.M)


def missing_command_hint(command: str, output: str, exit_code,
                         scope: RequestScope | None) -> str:
    """What to say when the command a turn guessed does not exist, or "".

    Exit 127 is the shell's "no such program"; 126 with ENOENT is a script
    path that is not there. Anything else is the program's own failure and
    is left to its own output.
    """
    if exit_code not in (126, 127) and not (
            exit_code not in (0, None) and _MISSING_RE.search(output or "")
            and _first_program(command) in (output or "")):
        return ""

    return f"\n'{_first_program(command)}' does not exist here. " \
        + discovery_hint(scope)


def _first_program(command: str) -> str:
    for argv in _split_stages(command or ""):
        if argv[0] in {"cd", "export", "set", "source", "."}:
            continue

        return argv[0]

    return ""
