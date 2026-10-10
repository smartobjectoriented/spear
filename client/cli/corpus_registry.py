"""The corpus registry (projects.json), its discovery and /corpus commands."""

import os
import re
import sys
import json
from cli.chat_settings import APP_DIR, ROOT_DIR
from context.workspace_context import FAMILY_PREFIX, project_families, reserved_project_id
from cli.terminal_ui import C_ACCENT, C_DIM, C_ERR, C_OK, C_RST, C_WARN


# ── Projects ─────────────────────────────────────────────────────────
# One unified concept: a PROJECT is a working tree with its own corpus
# (ChromaDB collection), history and memories.
#
# What a corpus DOES is declared, not inferred from its type. `kind` is a
# label -- it scopes skills and prints in the listing -- and each behaviour
# has its own key:
#   collection   name an existing index instead of deriving one from the path
#   indexer      "buildsystem" (curated BitBake/Yocto walk) or "generic"
#   autoindex    build a missing index on first sight (default: no)
#   prompt_file  a domain prompt for this corpus (default: the generic one)
#   public       the tree may be named in a public image (default: no)
#
# One key used to imply all four. It meant a registry entry did not say what
# its corpus did, and that the four could not be used apart.
# The registry lives in projects.json: {name: {"path": ..., "kind": ...}}.
# Manage it with `spear-corpus add/list/rm`.

PROJECTS_FILE = f"{APP_DIR}/projects.json"

# No corpus is built in. Two used to be -- a pair of product checkouts under
# one organisation's home directory, registered automatically wherever the
# directory happened to exist -- and a platform that ships somebody's tree as
# a first-class corpus is a platform shipping their tree. Corpora are declared
# by whoever owns them, with `/corpus add` or projects.json, and the registry
# below does not care where they came from.


# Where a RELATIVE corpus path is rooted. Absolute paths are untouched, so the
# workstation registry keeps working exactly as before; a container ships a
# registry of relative paths instead and mounts the trees under one root. That
# is what makes the same projects.json usable by someone whose checkouts are
# not under /home/operator.
# NB: not CORPUS_ROOT -- that name is already the *current* corpus's root,
# rebound by set_project() on every switch. This one is a fixed prefix for the
# registry and must never move.
# Defaults to the REPOSITORY root, so a corpus that lives inside the repo is
# registered as `llama.cpp-next`, not `/opt/llm/spear/llama.cpp-next`, and
# a clone anywhere finds it. A container overrides it with the mount root.
# Absolute paths are untouched either way: the trees outside the repo are
# genuinely machine-specific and stay spelled out.

CORPORA_ROOT = os.environ.get("SPEAR_CORPUS_ROOT") or ROOT_DIR


def resolve_corpus_path(path):
    """Absolute path of a registered corpus, honouring SPEAR_CORPUS_ROOT."""

    if os.path.isabs(path) or not CORPORA_ROOT:
        return path

    return os.path.normpath(os.path.join(CORPORA_ROOT, path))


_WARNED: set[str] = set()


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print(message, file=sys.stderr)


def load_projects():
    """Return {name: {"path","kind", ...}}. Tolerates the legacy {name: path} format (treated as
    generic). Extra keys (exclude, include_build — see reindex_options) are
    PRESERVED: rebuilding specs from path/kind alone erased them at the first
    save_projects()."""

    try:
        with open(PROJECTS_FILE) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        raw = {}

    projects = {}

    for name, val in raw.items():
        # A name that reads as a family in a scope ("family:x") is refused
        # here, where the registry is read, rather than silently matching
        # every rule written for that family.

        if reserved_project_id(name):
            _warn_once(f"projects.json: '{name}' is not a usable project name "
                       f"(names starting with '{FAMILY_PREFIX}' are reserved for "
                       f"project families); the entry is ignored")
            continue

        if not isinstance(val, str):
            bad = project_families(val)[1]

            if bad:
                _warn_once(f"projects.json: '{name}' declares unusable families "
                           f"{', '.join(bad)} (lower-case names, no ':'); they are ignored")

        if isinstance(val, str):
            projects[name] = {"path": val, "kind": "generic"}
        else:
            spec = dict(val)
            spec["kind"] = spec.get("kind", "generic")
            projects[name] = spec

    for spec in projects.values():
        spec["path"] = resolve_corpus_path(spec["path"])

    return projects


def save_projects(projects):
    try:
        with open(PROJECTS_FILE, "w") as f:
            json.dump(projects, f, indent=2)
    except OSError as exc:
        # The registry is shipped configuration, not session state: in a
        # container it lives in the image and is not the user's to change.
        # Say where it has to be done instead of raising a stack trace.

        raise SystemExit(
            f"cannot write the corpus registry ({PROJECTS_FILE}): {exc}\n"
            f"Registering a corpus is a host operation — do it there, then "
            f"rebuild the image (scripts/docker/build.sh). A one-off tree can be "
            f"reached with --corpora instead.") from exc


def detect_project(projects):
    """Project whose path contains the cwd, or None."""
    cwd = os.path.realpath(os.getcwd())
    best = None

    for name, spec in projects.items():
        p = os.path.realpath(spec["path"])

        if cwd == p or cwd.startswith(p + "/"):
            if best is None or len(p) > len(projects[best]["path"]):
                best = name

    return best


# Snapshot trees a build system leaves beside the live ones. Registered,
# indexed, and never what a question is about.

_SNAPSHOT_SUFFIXES = (".back", ".pristine", ".0")


def corpora_below(projects, cwd=None):
    """Every registered corpus sitting under the cwd.

    Launching in an umbrella checkout used to pick ONE child and chdir into it:
    from ~/soo/so3 the session moved into ~/soo/so3/so3 and lost avz/, build/,
    doc/ and the scripts -- half of what the questions are about -- while the
    orientation map, written for the umbrella, pointed at paths that no longer
    existed from there. The tools belong at the umbrella root, where every part
    is reachable, and the parts belong in the federation, where each keeps its
    own index and contributes its own best hits.
    """
    cwd = os.path.realpath(cwd or os.getcwd())
    found = {}

    for name, spec in projects.items():
        path = os.path.realpath(spec.get("path", ""))

        if path == cwd or not path.startswith(cwd + os.sep):
            continue

        if name.endswith(_SNAPSHOT_SUFFIXES) or path.endswith(_SNAPSHOT_SUFFIXES):
            continue

        found[name] = path

    return found


def enclosing_corpus(projects, cwd=None):
    """Registered corpus sitting under the cwd, when there is an obvious one.

    detect_project answers "is the cwd INSIDE a corpus". The opposite happens
    just as often and used to leave the session with no RAG at all: launched
    from ~/soo/so3, whose corpus is ~/soo/so3/so3, the chat fell back to an
    ad-hoc index of the parent — which does not exist — and the model answered
    from memory. A passive warning was printed and routinely missed.

    Ambiguity is the reason this cannot be a plain "first child wins":
    ~/soo/so3 holds six registered corpora (so3, u-boot, avz, qemu, atf,
    avz.back). Two signals are unambiguous enough to act on, and nothing else
    is:
      - exactly ONE registered corpus directly below the cwd;
      - or one whose basename matches the cwd's, the umbrella-checkout shape
        (~/soo/so3 -> ~/soo/so3/so3).
    Anything else returns None with the candidates, so the caller can name
    them instead of guessing.
    """
    cwd = os.path.realpath(cwd or os.getcwd())
    children = {}

    for name, spec in projects.items():
        p = os.path.realpath(spec["path"])

        if p != cwd and p.startswith(cwd + os.sep):
            children[name] = p

    direct = {n: p for n, p in children.items()
              if os.path.dirname(p) == cwd}

    if len(direct) == 1:
        return next(iter(direct)), sorted(direct)

    same_name = [n for n, p in direct.items()
                 if os.path.basename(p) == os.path.basename(cwd)]

    if len(same_name) == 1:
        return same_name[0], sorted(direct)

    return None, sorted(direct or children)


_SCAN_EXTS = {".c", ".h", ".S", ".cpp", ".hpp", ".cc", ".py", ".sh", ".md",
              ".rst", ".dts", ".dtsi", ".cfg", ".conf", ".bb", ".inc",
              ".yaml", ".yml", ".mk"}
_SCAN_SKIP = {".git", "build", "out", "tmp", "__pycache__", "node_modules"}


def _scan_count(path, cap=20000):
    """(files, capped) — capped says the walk stopped early, so `files` is a
    floor and not a count. It matters in what the split prints: agency was
    announced as "20017 files" when it holds 66246, a whole kernel tree, and
    that number is what a reader uses to decide whether it fits in one index.
    """
    c = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SCAN_SKIP]
        c += sum(1 for f in files
                 if f in ("Makefile", "Kconfig", "CMakeLists.txt")
                 or os.path.splitext(f)[1] in _SCAN_EXTS)

        if c > cap:
            return c, True

    return c, False


def workspace_components(path, projects, min_files=300):
    """The sub-trees of `path` big enough to deserve a corpus of their own.

    One (name, abspath, files, registered_as) per component, `registered_as`
    naming the corpus that already owns the tree, or None. Shared by the
    launch-time split offer and by `/corpus scan`, which used to carry a second
    copy of this walk inside a shell script — a copy whose skip list and
    extension set had already drifted from this one.

    An already-registered tree is reported without being counted: it is listed
    for information, and walking a kernel tree to say "already: agency" would
    cost seconds at every launch.
    """
    known = {os.path.realpath(s["path"]): n for n, s in projects.items()}
    comps = []

    for sub in sorted(os.listdir(path)):
        p = os.path.join(path, sub)

        if not os.path.isdir(p) or sub in _SCAN_SKIP:
            continue

        rp = os.path.realpath(p)
        owner = known.get(rp)

        if owner:
            comps.append((sub, rp, None, owner))
            continue

        n, capped = _scan_count(p)

        if n >= min_files:
            comps.append((sub, rp, f"{n}+" if capped else str(n), None))

    return comps


def offer_workspace_split(path, projects, min_files=300):
    """When an unregistered dir is really a multi-component workspace (several
    big sub-trees), offer to register each component as its own corpus — so
    a huge upstream tree (u-boot, qemu...) never dilutes the index. Returns a
    chosen project spec dict, or None to fall through to plain ad-hoc."""

    if not sys.stdin.isatty():
        return None

    # A virtualenv root is not a multi-component workspace: its lib/ and lib64/
    # are site-packages, and offering to index them as corpora is noise.

    if os.path.exists(os.path.join(path, "pyvenv.cfg")):
        return None

    comps = [c for c in workspace_components(path, projects, min_files)
             if c[3] is None]

    if len(comps) < 2:
        return None

    print(f"{C_WARN}This looks like a multi-component workspace "
          f"({len(comps)} big sub-trees):{C_RST}")

    for sub, _, n, _owner in comps:
        print(f"  {C_DIM}·{C_RST} {sub} ({n} files)")

    print("Register each as its own corpus (recommended — keeps indexes "
          "focused)?")
    print(f"  {C_ACCENT}❯{C_RST} 1. Yes, register them {C_DIM}(Enter){C_RST}")
    print(f"    2. No, index this whole directory as one")

    try:
        ans = input("  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if ans not in ("", "1", "o", "y", "oui", "yes"):
        return None

    base = os.path.basename(path.rstrip("/"))
    reg = load_projects()
    names = []

    for sub, p, _n, _owner in comps:
        name = sub if sub not in reg else f"{base}-{sub}"
        reg[name] = {"path": p, "kind": "generic"}
        names.append(name)

    save_projects(reg)
    print(f"{C_OK}Registered: {', '.join(names)}{C_RST}")
    print("Which one now? (number, or Enter to pick later)")

    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")

    try:
        a = input("  ").strip()
    except (EOFError, KeyboardInterrupt):
        a = ""

    if a.isdigit() and 1 <= int(a) <= len(names):
        n = names[int(a) - 1]
        return {"name": n, **reg[n]}

    return None


def resolve_project_at_startup():
    """The CURRENT directory decides everything — no prompt:
      1. explicit --corpus <name> (or --here) wins;
      2. else if the cwd is inside a registered corpus, use it;
      3. else ad-hoc on the cwd (offering a workspace split if it is a
         multi-component tree).
    Tools always run in the cwd regardless; the corpus only picks the RAG
    index/history/memories."""
    projects = load_projects()

    if "--here" in sys.argv[1:]:
        split = offer_workspace_split(os.path.realpath(os.getcwd()), projects)

        if split:
            return split

        return {"name": "adhoc:" + os.path.basename(os.getcwd().rstrip("/")),
                "path": os.getcwd(), "kind": "generic"}

    # --corpus (and the --project/--checkout aliases) forces a registered corpus

    for flag in ("--corpus", "--project", "--checkout"):
        if flag in sys.argv[1:]:
            i = sys.argv.index(flag)

            if i + 1 < len(sys.argv) and sys.argv[i + 1] in projects:
                n = sys.argv[i + 1]
                return {"name": n, **projects[n]}

            print(f"{C_ERR}Unknown corpus (registered: "
                  f"{', '.join(projects) or 'none'}){C_RST}")
            sys.exit(1)

    # cwd inside a registered corpus → use it

    found = detect_project(projects)

    if found:
        return {"name": found, **projects[found]}

    # cwd ABOVE registered corpora → an umbrella session: tools stay HERE, and
    # every part below is federated. The prefix machinery in federated_corpora
    # rewrites each chunk's path so what the model reads is what bash can open,
    # which is what used to force the chdir.

    below = corpora_below(projects)

    if below:
        cwd = os.path.realpath(os.getcwd())
        print(f"{C_DIM}  ⎿  {len(below)} registered corpora sit under this "
              f"directory ({', '.join(sorted(below))}) — tools run here, "
              f"retrieval spans them. --corpus <name> narrows to one.{C_RST}")

        return {"name": "workspace:" + (os.path.basename(cwd) or "root"),
                "path": cwd, "kind": "generic", "corpora": sorted(below),
                # Discovered, not declared. Declared federations and shared
                # corpora are a deliberate choice and stay attached every turn;
                # these eight are whatever happened to be registered under the
                # cwd, so they are selected per question instead.
                "auto_corpora": sorted(below)}

    # cwd outside any corpus → ad-hoc on the cwd (no picker). Offer a split
    # if it is a big multi-component workspace.

    if sys.stdin.isatty():
        split = offer_workspace_split(os.path.realpath(os.getcwd()), projects)

        if split:
            return split

    return {"name": "adhoc:" + os.path.basename(os.getcwd().rstrip("/")) or "cwd",
            "path": os.getcwd(), "kind": "generic"}


# No "/corpus" prefix in it: the same string answers the slash command, where
# the user has already typed it, and the spear-corpus CLI, where it is wrong.
CORPUS_USAGE = ("usage: list | add <name> [path] [--kind K] [--indexer I] "
                "[--autoindex] [--public] [--prompt-file F] | rm <name> | "
                "scan [path] [--min N]")

# What a corpus name may hold. Every existing one passes (sye_sol,
# llama.cpp-next, avz.back, opencn-u-boot); a slash or a space does not.
CORPUS_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _save_registry(projects):
    """save_projects(), without taking the session down. Returns an error
    string, or None.

    save_projects raises SystemExit when the registry is read-only — a
    container ships it in the image. That is the right answer for a one-shot
    CLI and the wrong one inside a chat loop, where a mistyped /corpus add
    would take the whole conversation with it.
    """
    try:
        save_projects(projects)
    except SystemExit as exc:
        return str(exc)

    return None


def handle_corpus_command(args, current=None):
    """The corpus registry from inside the session: list / add / rm / scan.

    Same registry as the `spear-corpus` CLI and, now, the same code. That
    script carried its own copy of the load/save, of the seeds and of the
    component walk, against a hardcoded projects.json — so it ignored
    SPEAR_CORPUS_ROOT, and its skip list and extension set had already drifted
    from the indexer's. It is a wrapper around this function instead.

    Registering does NOT move the session: the corpus decides the index, the
    history and the memories, and all three are bound at launch.
    `spear-chat --corpus <name>` is still how you switch. What does take effect
    at once is the exclusion — reindex_options() re-reads the registry — so
    registering a vendored sub-tree and running /reindex right after leaves it
    out of the corpus above it, which is the usual reason to register one.
    """
    args = list(args)
    cmd = args[0] if args else "list"
    flags, rest, opts = [], [], {}
    it = iter(args[1:])

    # These take a VALUE, so they cannot be filtered out by a
    # `startswith("-")` comprehension: the value would fall through to the
    # positionals and `scan --min 500` would read 500 as the directory to
    # scan, or `add c /tree --kind k` would read k as a second path.

    valued = {"--min", "--kind", "--indexer", "--prompt-file"}

    for a in it:
        head, sep, tail = a.partition("=")

        if head in valued:
            opts[head.lstrip("-").replace("-", "_")] = tail if sep else next(it, "")
        elif a.startswith("-"):
            flags.append(a)
        else:
            rest.append(a)

    projects = load_projects()

    def abspath(p):
        return os.path.realpath(os.path.expanduser(p))

    if cmd == "list":
        if not projects:
            return "(no corpus registered)"

        rows = [f"  {'*' if n == current else ' '} {n:18s} {s['kind']:8s} "
                f"{s['path']}" for n, s in sorted(projects.items())]

        return "\n".join(rows) + f"\n  ({len(projects)} corpora"  \
               + (", * = this session)" if current in projects else ")")

    if cmd == "add" and rest:
        name = rest[0]

        # A corpus NAME is not a path. `/corpus add agency/linux`, typed from
        # the tree above it, registered the name "agency/linux" pointing at the
        # CWD — the path defaults to it, and nothing said the first argument
        # had been read as a name. The session then displayed
        # "corpus: agency/linux" while indexing the parent, which is exactly
        # the kind of wrong that looks right.

        if not CORPUS_NAME_RE.fullmatch(name):
            hint = ""

            if os.path.isdir(abspath(name)):
                hint = (f" You gave a path: `add "
                        f"{os.path.basename(name.rstrip('/')) or 'name'} "
                        f"{name}` registers that tree instead.")

            return (f"'{name}' is not a corpus name — a name goes into "
                    f"`--corpus <name>` and into the banner, so it holds "
                    f"letters, digits and . _ + - only.{hint}")

        path = abspath(rest[1]) if len(rest) > 1 else abspath(os.getcwd())

        if not os.path.isdir(path):
            return f"not a directory: {path}"

        known = projects.get(name)

        if known and os.path.realpath(known["path"]) != path:
            return (f"'{name}' already points at {known['path']} — remove it "
                    f"first (/corpus rm {name})")

        # Re-registering the same tree must not drop what was declared about
        # it: exclude, collection and include_build are hand-written and would
        # be silently lost by rebuilding the spec from name and path.

        spec = dict(known or {})
        spec["path"] = path
        spec["kind"] = opts.get("kind") or spec.get("kind", "generic")

        # Each behaviour named separately. One flag that switched four of them
        # at once is how they became invisible in the first place.

        for option, key in (("indexer", "indexer"), ("prompt_file", "prompt_file")):
            if opts.get(option):
                spec[key] = opts[option]

        if "--autoindex" in flags:
            spec["autoindex"] = True

        if "--public" in flags:
            spec["public"] = True
        projects[name] = spec
        err = _save_registry(projects)

        if err:
            return err

        return (f"registered '{name}' ({spec['kind']}) -> {path}\n"
                f"  index it:  spear-chat --corpus {name}   then /reindex\n"
                f"  a tree inside another corpus is excluded from it on the "
                f"next /reindex")

    if cmd == "rm" and rest:
        name = rest[0]

        if name == current:
            return (f"'{name}' is this session's own corpus — its index, "
                    f"history and memories are bound to it; relaunch "
                    f"elsewhere to remove it")

        if projects.pop(name, None) is None:
            return f"no such corpus: {name}"

        err = _save_registry(projects)

        return err or (f"removed '{name}' — its index is left in the store "
                       f"(nothing reads it now)")

    if cmd == "scan":
        root = abspath(rest[0]) if rest else abspath(os.getcwd())

        try:
            mn = int(opts.get("min", 300))
        except ValueError:
            return CORPUS_USAGE

        if not os.path.isdir(root):
            return f"not a directory: {root}"

        comps = workspace_components(root, projects, mn)

        if not comps:
            return f"no component of {root} reaches {mn} indexable files"

        base = os.path.basename(root.rstrip("/"))
        lines = [f"Scanning {root} (components with >= {mn} indexable files):"]
        added = []

        for name, path, n, owner in comps:
            if owner:
                lines.append(f"  {'':>8}  {name:16s} already: {owner}")
                continue

            new_name = name if name not in projects else f"{base}-{name}"
            projects[new_name] = {"path": path, "kind": "generic"}
            added.append(new_name)
            lines.append(f"  {n:>8}  {name:16s} -> corpus '{new_name}'")

        if not added:
            return "\n".join(lines + ["Nothing new to register."])

        err = _save_registry(projects)

        if err:
            return err

        return "\n".join(lines + [f"Registered {len(added)}: "
                                  f"{', '.join(added)}"])

    return CORPUS_USAGE
