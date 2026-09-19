"""The lifecycle of a turn that changes code to meet an authoritative source.

Two kinds of evidence decide such a turn, and they live in different places.
What is REQUIRED comes from a document nobody in the repository controls; what
is CURRENTLY TRUE comes from the repository itself. Neither one alone says
what to change: a requirement without the code names no symbol, and the code
without the requirement names no obligation. The change is the difference
between them, and the difference cannot be computed until both halves are in
hand.

A turn that starts editing before it holds both halves is not working from a
gap. It is working from an impression of one, and the repair that follows is
an impression of a repair -- which is what the write gate below exists to
stop. The gate is not a budget and not a tool count: it asks whether the
session can state, for at least one requirement, what the source demands, what
the implementation does, where the two differ, what will be changed, and what
will demonstrate the change. Until it can, nothing writes.

None of this is specific to any document, any repository or any domain. What
makes a source authoritative is that the session is bound to it; what makes a
file implementation evidence is that a reading tool put it in front of the
model. Both are facts the harness already measures for other reasons, and the
ledgers here consume them rather than re-deriving them.

Three properties are deliberate:

* **The gate is about evidence, not about wording.** A plan item is accepted
  because the requirement it cites was retrieved this turn and the file it
  names was read this turn, not because it is well phrased. A fluent item with
  a citation nobody opened is refused exactly like an empty one.

* **Refusal is visible and tells the model what to do.** A silently withheld
  tool reads as a model that would not act; a refusal that names the missing
  half reads as an instruction. The message is returned through the tool
  channel so it arrives where the model is already looking.

* **A plan is not a promise.** New evidence that contradicts a plan item sends
  the turn back to PLAN rather than letting it improvise, and the reason, the
  invalidated item and the evidence that invalidated it are all recorded.

The phases are ordinary: INVESTIGATE, PLAN, EDIT, TEST, REVIEW, DONE. They
exist so that "when did the first write happen?" has an answer that is not a
round number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Phase(StrEnum):
    INVESTIGATE = "investigate"
    PLAN = "plan"
    EDIT = "edit"
    TEST = "test"
    REVIEW = "review"
    DONE = "done"


#: The first sentence of every refusal. Fixed wording, because it is also what
#: a diagnostic reads to tell a gated refusal from any other denial.
WRITE_BLOCKED = ("Complete investigation and establish an evidence-backed "
                 "implementation plan before modifying files.")

#: The tool that records that plan. Named here, beside the refusal that asks
#: for it, so the two cannot drift: a refusal that says "record a plan"
#: without naming the call leaves the model to guess, and a run that guesses
#: spends its rounds guessing. The registry spells the same name.
PLAN_TOOL = "plan_change"

#: Why a write was refused, as its own token: the phase alone does not
#: distinguish "nothing has been read" from "everything has been read and
#: nothing has been planned", and those need different things from the model.
NO_AUTHORITY = "authority_evidence_missing"
NO_IMPLEMENTATION = "implementation_evidence_missing"
NO_PLAN = "plan_missing"
REVIEW_IS_READ_ONLY = "review_is_read_only"

#: How many equivalent observations in a row mean the session is no longer
#: learning anything. Small on purpose. The failure this bounds ran to dozens
#: of calls before anything noticed; a threshold set where the damage becomes
#: visible is a threshold set far too late.
REPETITION_LIMIT = 3

#: And how many times the turn is told so. A demand repeated eight times in
#: one turn is not a demand, it is wallpaper -- and every copy of it is paid
#: for out of the context window. After this many, the harness stops asking
#: and narrows the round instead.
MAX_SYNTHESES = 3

#: And how many times the same validation command may fail unchanged before
#: re-running it stops being a test and becomes a wish.
FAILED_VALIDATION_LIMIT = 2

#: How many further calls a turn may spend once it holds both halves of the
#: gap and has planned nothing. Generous rather than tight -- a five-file
#: change is understood by reading five files, and cutting that short is the
#: opposite fault -- but finite, because "read a bit more" is always available.
PLAN_PATIENCE = 8

#: And how many times the turn is reminded. Twice: once is easy to miss in a
#: long round, and a third would be nagging.
MAX_PLAN_DEMANDS = 2

#: A clause identifier as documents write them -- "Rule 5.2.1-2", "§5.2",
#: "Section 5.1.2", "Table 3.2-1" -- reduced to the section number, which is
#: what a retrieval is recorded under. The trailing ordinal is dropped: a
#: session that read §5.2.1 has read the text of every provision in it.
_SECTION = re.compile(r"\b(\d+(?:\.\d+)+)(?:-\d+)?\b")

#: An opaque retrieval handle, whatever the store calls them.
_HANDLE = re.compile(r"\b[a-z]{2,4}-[0-9a-f]{8,}\b")

#: A path, or the tail of one. Matched loosely because a plan item may write
#: `src/a/b.c`, `./src/a/b.c` or just `b.c` for a file the turn plainly read.
_PATH = re.compile(r"[\w.\-/]*\w\.[A-Za-z0-9_+]{1,8}\b")


def _sections(text):
    return {found for found in _SECTION.findall(text or "")}


def _handles(text):
    return {found.lower() for found in _HANDLE.findall(text or "")}


def _basename(path):
    return (path or "").rstrip("/").rsplit("/", 1)[-1]


@dataclass
class EvidenceState:
    """What this turn has actually been shown, on each side of the gap.

    Filled by the harness from the tool results themselves, never from what a
    call asked for: a search that returned nothing establishes nothing, and a
    plan item may not cite the query that failed to find it.
    """

    #: Section numbers and retrieval handles the authoritative source returned.
    authority_keys: set = field(default_factory=set)
    #: How many units of authoritative text were put in front of the model.
    authority_units: int = 0
    #: Files a reading tool put in front of the model, as it named them.
    implementation_files: set = field(default_factory=set)

    #: The tree this session is about. Absolute paths outside it are not
    #: THIS implementation, whatever they contain: a session with several
    #: corpora attached can read a great deal of C that has nothing to do with
    #: the task, and a gate that accepted it would open on the wrong evidence.
    #: Empty means "no boundary stated", and then every path counts.
    root: str = ""

    def observe_authority(self, keys=(), units=0):
        self.authority_keys |= {str(key).lower() for key in keys if key}
        self.authority_units += max(0, int(units or 0))
        return self

    def observe_implementation(self, paths=()):
        self.implementation_files |= {path for path in
                                      (str(item) for item in paths if item)
                                      if self._inside(path)}
        return self

    def _inside(self, path):
        """A relative path is in the tree by construction; an absolute one is
        in it only if it says so."""
        if not self.root or not path.startswith("/"):
            return True

        root = self.root.rstrip("/")

        return path == root or path.startswith(root + "/")

    @property
    def has_authority(self):
        return bool(self.authority_keys) or self.authority_units > 0

    @property
    def has_implementation(self):
        return bool(self.implementation_files)

    def backs_requirement(self, citation):
        """Was the cited provision actually retrieved this turn?

        A handle matches exactly. A section matches when the session read that
        section, or read something inside it: a turn that opened §5.2.1 may
        cite §5.2, because it has the text. The reverse is not true and is not
        allowed -- reading a chapter heading is not reading the rule under it.
        """
        wanted = _handles(citation)

        if wanted & self.authority_keys:
            return True

        read = {key for key in self.authority_keys if key and key[0].isdigit()}

        for section in _sections(citation):
            if section in read:
                return True

            if any(deeper.startswith(section + ".") for deeper in read):
                return True

        return False

    def backs_implementation(self, citation):
        """Was a file this item names actually read this turn?"""
        names = {_basename(path) for path in self.implementation_files}
        names.discard("")

        for candidate in _PATH.findall(citation or ""):
            if _basename(candidate) in names:
                return True

        return False

    def summary(self, *, implementation=False, limit=12):
        """What this session actually holds, in the spelling it holds it."""
        found = sorted(self.implementation_files if implementation
                       else self.authority_keys)

        if not found:
            return "(nothing read yet)"

        shown = ", ".join(found[:limit])

        return shown + (f" and {len(found) - limit} more"
                        if len(found) > limit else "")

    def to_dict(self):
        return {"authority_units": self.authority_units,
                "authority_keys": sorted(self.authority_keys),
                "implementation_files": sorted(self.implementation_files)}


@dataclass(frozen=True)
class GapItem:
    """One requirement, what the code does about it, and what will be done.

    Seven fields, and every one of them is load-bearing. Drop the evidence and
    the item is an assertion; drop the current behaviour and it is a feature
    request; drop the validation and nothing will ever say whether the change
    worked.
    """

    requirement: str = ""
    requirement_evidence: str = ""
    current_behaviour: str = ""
    implementation_evidence: str = ""
    gap: str = ""
    correction: str = ""
    validation: str = ""

    FIELDS = ("requirement", "requirement_evidence", "current_behaviour",
              "implementation_evidence", "gap", "correction", "validation")

    @classmethod
    def from_mapping(cls, raw):
        raw = raw if isinstance(raw, dict) else {}

        return cls(**{name: str(raw.get(name) or "").strip()
                      for name in cls.FIELDS})

    def missing(self):
        """The fields this item left empty, in declaration order."""
        return tuple(name for name in self.FIELDS if not getattr(self, name))

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}


@dataclass(frozen=True)
class ItemVerdict:
    item: GapItem
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class PlanAcceptance:
    accepted: tuple = ()
    rejected: tuple = ()

    @property
    def any_accepted(self):
        return bool(self.accepted)

    def report(self):
        """What to tell the model, when something was refused."""
        if not self.rejected:
            return ""

        lines = [f"{len(self.rejected)} plan item(s) were not accepted:"]

        for index, verdict in enumerate(self.rejected, 1):
            head = verdict.item.requirement or "(unnamed requirement)"
            lines.append(f"  {index}. {head[:90]} — {verdict.reason}")

        return "\n".join(lines)


@dataclass(frozen=True)
class WriteDecision:
    allowed: bool
    reason: str = ""
    message: str = ""


@dataclass
class Replan:
    reason: str
    invalidated: str
    evidence: str


@dataclass
class WorkPhaseLedger:
    """One turn's phase, its gap model, its plan and why it changed.

    Inert unless the turn is one this governs. `engaged` is set once, from two
    facts the caller already knows -- the turn is bound to an authoritative
    source, and the user asked for the code to change -- so a pure question and
    a pure code task both take exactly the path they took before.
    """

    engaged: bool = False
    phase: Phase = Phase.INVESTIGATE
    evidence: EvidenceState = field(default_factory=EvidenceState)

    items: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    replans: list = field(default_factory=list)

    #: Where the first write landed, as a phase. The whole diagnostic rests on
    #: this one field: "the first edit happened during INVESTIGATE" is the
    #: failure, stated without reference to any round number.
    first_write_phase: str = ""
    writes: int = 0
    refusals: int = 0

    validations: list = field(default_factory=list)
    failed_validations: dict = field(default_factory=dict)

    consecutive_without_evidence: int = 0
    syntheses_forced: int = 0

    #: Calls made since both halves of the gap were in hand and no plan had
    #: been accepted. INVESTIGATE has an end; without something that notices
    #: it, "read a bit more" is always available and a turn takes it.
    calls_since_investigated: int = 0
    plan_demands: int = 0

    # -- engagement -------------------------------------------------------

    def engage(self, *, authority_bound, write_requested, root="",
               prior_authority=()):
        """Does this lifecycle govern the turn at all?

        Both halves, and only both. A normative question with no change asked
        for is answered, not planned; a change with no authoritative source is
        the ordinary editing workflow and keeps it.
        """
        self.engaged = bool(authority_bound) and bool(write_requested)
        self.evidence.root = str(root or "")

        # Evidence the SESSION holds, not only what this turn re-read. On a
        # two-prompt shape the requirements were established when the question
        # was answered, and the change is asked for next; making the second
        # turn re-retrieve them is how a run spends its budget proving what it
        # already knows. It settles the authority half only -- the
        # implementation still has to be read in THIS turn before anything
        # writes, which is the half that was missing in the failure.
        self.evidence.observe_authority(prior_authority, 0)

        return self.engaged

    # -- observation ------------------------------------------------------

    def observe_call(self, *, authority_keys=(), authority_units=0,
                     implementation_paths=()):
        """One tool call, and what it added to either side of the gap.

        ONE call, one count. A call that established nothing on either side is
        counted as barren rather than judged: one of those is ordinary
        navigation -- a search reworded, a directory listed -- and only a run
        of them means the session has stopped learning and is circling.

        The totals are passed rather than the deltas, because the harness's
        ledgers are cumulative and the caller should not have to diff them.
        """
        before = (len(self.evidence.authority_keys),
                  self.evidence.authority_units,
                  len(self.evidence.implementation_files))

        self.evidence.observe_authority(authority_keys, 0)
        self.evidence.authority_units = max(self.evidence.authority_units,
                                            int(authority_units or 0))
        self.evidence.observe_implementation(implementation_paths)

        after = (len(self.evidence.authority_keys),
                 self.evidence.authority_units,
                 len(self.evidence.implementation_files))

        self._count_progress(before != after)

        # Counted only when the turn was ALREADY past investigating: the call
        # that completes it is the one that earned the right to plan, not a
        # call spent putting the plan off.
        planning = self.phase == Phase.PLAN
        self._open_plan()

        if planning:
            self.calls_since_investigated += 1

        return self

    def _count_progress(self, gained):
        if gained:
            self.consecutive_without_evidence = 0
        else:
            self.consecutive_without_evidence += 1

    def _open_plan(self):
        if (self.engaged
                and self.phase == Phase.INVESTIGATE
                and self.evidence.has_authority
                and self.evidence.has_implementation):
            self.phase = Phase.PLAN

    # -- investigation ----------------------------------------------------

    def investigation_gaps(self):
        """Which half of the question is still unanswered."""
        missing = []

        if not self.evidence.has_authority:
            missing.append(NO_AUTHORITY)

        if not self.evidence.has_implementation:
            missing.append(NO_IMPLEMENTATION)

        return tuple(missing)

    @property
    def investigation_complete(self):
        return not self.investigation_gaps()

    # -- planning ---------------------------------------------------------

    def record_plan(self, raw_items, *, supersedes="", reason="",
                    new_evidence=""):
        """Take a plan, keeping only the items the evidence actually backs.

        A rejected item is kept too, with its reason: the model is told what
        was wrong with it, and a diagnostic can see that a turn tried to plan
        from something it had not read.
        """
        verdicts = [self._judge(GapItem.from_mapping(raw))
                    for raw in (raw_items or [])]
        accepted = tuple(v for v in verdicts if v.accepted)
        rejected = tuple(v for v in verdicts if not v.accepted)

        if supersedes or reason:
            self.replans.append(Replan(reason or "replanned",
                                       supersedes, new_evidence))

        if supersedes:
            self.items = [item for item in self.items
                          if item.requirement != supersedes]

        self.items.extend(v.item for v in accepted)
        self.rejected.extend(rejected)

        if self.items and self.phase in (Phase.INVESTIGATE, Phase.PLAN):
            self.phase = Phase.EDIT
        elif self.items and self.phase == Phase.REVIEW:
            self.phase = Phase.EDIT

        self.consecutive_without_evidence = 0

        return PlanAcceptance(accepted, rejected)

    def _judge(self, item):
        empty = item.missing()

        if empty:
            return ItemVerdict(item, False,
                               "missing: " + ", ".join(empty))

        if not self.evidence.backs_requirement(item.requirement_evidence):
            # Naming what IS held, not only what is missing. A refusal that
            # says "not retrieved" and stops leaves the model to guess whether
            # the fault is the spelling, the section or the retrieval, and a
            # run spent nine minutes re-searching for a provision it had
            # already read under a number the item did not use.
            return ItemVerdict(
                item, False,
                f"the cited provision {item.requirement_evidence!r} is not "
                f"among the evidence this session holds. Retrieve it, or cite "
                f"one of: {self.evidence.summary()}")

        if not self.evidence.backs_implementation(item.implementation_evidence):
            return ItemVerdict(
                item, False,
                f"no file named by {item.implementation_evidence!r} was read "
                f"this turn. Read the implementation before planning a change "
                f"to it, or name one of: "
                f"{self.evidence.summary(implementation=True)}")

        return ItemVerdict(item, True)

    @property
    def has_plan(self):
        return bool(self.items)

    # -- replanning -------------------------------------------------------

    def invalidate(self, *, reason, invalidated="", evidence=""):
        """New information has made the current plan wrong.

        The turn goes back to PLAN rather than carrying on: an edit made
        against a plan that is known to be wrong is an edit nobody chose.
        """
        self.replans.append(Replan(reason, invalidated, evidence))

        if invalidated:
            self.items = [item for item in self.items
                          if item.requirement != invalidated]

        if not self.items or invalidated:
            self.phase = Phase.PLAN

        self.consecutive_without_evidence = 0

        return self

    # -- the write gate ---------------------------------------------------

    def may_write(self):
        """Whether a mutating action may run right now, and why not."""
        if not self.engaged:
            return WriteDecision(True)

        if self.phase in (Phase.EDIT, Phase.TEST):
            return WriteDecision(True)

        if self.phase == Phase.REVIEW:
            return WriteDecision(
                False, REVIEW_IS_READ_ONLY,
                "The normative review is read-only. If it found something the "
                "implementation must answer, record the replan and the new "
                "evidence first; the write gate reopens with the plan.")

        missing = self.investigation_gaps()

        if NO_AUTHORITY in missing and NO_IMPLEMENTATION in missing:
            detail = ("Neither the authoritative requirements nor the current "
                      "implementation has been read this turn.")
        elif NO_AUTHORITY in missing:
            detail = ("The implementation has been read, but no requirement "
                      "has been retrieved from the authoritative source, so "
                      "there is nothing to be compliant with.")
        elif NO_IMPLEMENTATION in missing:
            detail = ("Requirements have been retrieved, but no source file "
                      "has been read, so what the code does today is unknown "
                      "and the gap cannot be stated.")
        else:
            detail = (f"Investigation is complete and no plan has been "
                      f"accepted. Call `{PLAN_TOOL}` once for the requirement "
                      f"you intend to satisfy, giving all seven fields: the "
                      f"provision and where you read it, what the code does "
                      f"now and in which file you read it, the gap between "
                      f"them, the change you intend, and the test that will "
                      f"prove it. One requirement per call.")

        reason = missing[0] if missing else NO_PLAN

        return WriteDecision(
            False, reason,
            WRITE_BLOCKED + " " + detail + " " + self._rejection_hint())

    def _rejection_hint(self):
        if not self.rejected:
            return ""

        last = self.rejected[-1]

        return f"(Last plan item refused: {last.reason})"

    def refuse_write(self):
        """Record that the gate turned a write away."""
        self.refusals += 1

        return self

    def note_write(self):
        """A mutation that reached the tree, and the phase it landed in.

        The phase is recorded as it WAS, never as it should have been. If a
        write ever lands during INVESTIGATE, the diagnostic must say so rather
        than tidy it away -- that reading is the whole point of the field.
        """
        self.writes += 1

        if not self.first_write_phase:
            self.first_write_phase = str(self.phase)

        return self

    # -- validation -------------------------------------------------------

    def note_validation(self, command, status, *, covers=()):
        """One validation action: what ran, what it said, what it covers."""
        self.validations.append({"command": command, "status": status,
                                 "covers": [str(item) for item in covers]})

        if status != "passed":
            self.failed_validations[command] = (
                self.failed_validations.get(command, 0) + 1)
        else:
            self.failed_validations.pop(command, None)

        if self.phase == Phase.EDIT:
            self.phase = Phase.TEST

        self.consecutive_without_evidence = 0

        return self

    def validation_exhausted(self, command):
        """Has this exact command failed unchanged often enough to stop?"""
        return self.failed_validations.get(command, 0) >= FAILED_VALIDATION_LIMIT

    def unresolved_items(self):
        """Plan items nothing that ran claims to cover.

        Reported rather than enforced: a requirement whose validation is "the
        existing suite already covers it" is a legitimate answer, and one that
        nothing covers is a fact the turn should say out loud.

        An item counts as covered when a validation named it outright, or when
        a command that PASSED answers the description the item gave of its own
        validation -- "the project's own tests" against a `ctest` that ran is
        the same claim in two vocabularies. Only a pass counts: a suite that
        failed has covered the requirement in the sense that matters least.
        """
        covered = set()
        passed = [run["command"] for run in self.validations
                  if run["status"] == "passed"]

        for run in self.validations:
            covered |= set(run["covers"])

        # Compared as WORDS, not as substrings: "test" is inside "latest",
        # "context" and "greatest", and a coverage check that matched those
        # would report everything as covered and mean nothing.
        ran = set()

        for command in passed:
            ran |= {word for word in re.split(r"\W+", command.lower()) if word}

        def answered(item):
            if item.requirement in covered:
                return True

            words = {word for word in re.split(r"\W+", item.validation.lower())
                     if len(word) >= 4}

            return bool(words & ran)

        return tuple(item for item in self.items if not answered(item))

    @property
    def validated(self):
        """A project-level validation ran and passed since the last write."""
        return any(run["status"] == "passed" for run in self.validations)

    # -- repetition -------------------------------------------------------

    @property
    def repeating(self):
        return (self.consecutive_without_evidence >= REPETITION_LIMIT
                and self.syntheses_forced < MAX_SYNTHESES)

    def owes_a_plan(self, *, after=PLAN_PATIENCE):
        """Both halves are in hand, nothing is planned, and reading goes on.

        Asked once per interval and answered at most twice: this is a reminder
        that INVESTIGATE has an end, not a second gate. A turn that has good
        reason to keep reading says so and keeps reading; nothing here stops
        it, and the write gate is what actually holds the line.
        """
        if (not self.engaged or self.phase != Phase.PLAN
                or self.plan_demands >= MAX_PLAN_DEMANDS
                or self.calls_since_investigated < after):
            return False

        self.plan_demands += 1
        self.calls_since_investigated = 0

        return True

    @property
    def plan_demand_ignored(self):
        """The turn has been asked for a plan and gone on reading instead.

        The answer to that is not a third request. The harness has one lever
        that is not a request -- the round's tool list -- and this is what
        says to pull it.
        """
        return self.engaged and self.plan_demands >= MAX_PLAN_DEMANDS \
            and self.phase == Phase.PLAN

    def force_synthesis(self):
        """Stop exploring. Either the plan stands or it needs replacing."""
        self.syntheses_forced += 1
        self.consecutive_without_evidence = 0

        return self.phase

    # -- phase movement the caller drives ---------------------------------

    def begin_review(self):
        if self.engaged and self.phase in (Phase.EDIT, Phase.TEST):
            self.phase = Phase.REVIEW

        return self

    def finish(self):
        if self.engaged:
            self.phase = Phase.DONE

        return self

    # -- diagnostics ------------------------------------------------------

    def to_dict(self):
        return {
            "schema_version": 1,
            "engaged": self.engaged,
            "phase": str(self.phase),
            "evidence": self.evidence.to_dict(),
            "plan_items": [item.to_dict() for item in self.items],
            "rejected_items": [{"reason": v.reason, **v.item.to_dict()}
                               for v in self.rejected],
            "replans": [{"reason": r.reason, "invalidated": r.invalidated,
                         "evidence": r.evidence} for r in self.replans],
            "first_write_phase": self.first_write_phase,
            "writes": self.writes,
            "write_refusals": self.refusals,
            "validations": list(self.validations),
            "unresolved_items": [item.requirement
                                 for item in self.unresolved_items()],
            "syntheses_forced": self.syntheses_forced,
        }
