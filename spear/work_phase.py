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
from dataclasses import dataclass, field, replace
from enum import StrEnum

import requirement_set
from requirement_set import Disposition, RequirementSet


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
#: Every carried requirement needs a disposition before anything is written.
#: Distinct from NO_PLAN: a turn here HAS a plan, and it does not yet cover
#: the whole of what the previous turn established.
PLAN_INCOMPLETE = "plan_does_not_cover_the_requirements"
REVIEW_IS_READ_ONLY = "review_is_read_only"

#: How many equivalent observations in a row mean the session is no longer
#: learning anything. Small on purpose. The failure this bounds ran to dozens
#: of calls before anything noticed; a threshold set where the damage becomes
#: visible is a threshold set far too late.
#:
#: Two, not three. At three it fired in every one of four measured runs and
#: fired three times in each, which is a threshold being reached rather than
#: a threshold being respected: the turn was allowed to restart exploring
#: after each intervention. One early synthesis beats three late ones.
REPETITION_LIMIT = 2

#: And how many times the turn is told so. A demand repeated eight times in
#: one turn is not a demand, it is wallpaper -- and every copy of it is paid
#: for out of the context window. After this many, the harness stops asking
#: and narrows the round instead.
MAX_SYNTHESES = 2

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

#: A correction that names somewhere in the code: a file, or a function call.
#: Weak on purpose -- this is not a proof that the lifecycle point is right,
#: only a refusal to accept a plan that names no point at all.
_NAMES_A_PATH = re.compile(r"[\w.\-/]*\w\.[A-Za-z0-9_+]{1,8}\b|\b\w+\(\)")

#: A correction that MOVES work: calling something that already exists from
#: somewhere it is not called from today. The generic risk is never the call
#: itself, it is everything the old site guaranteed and the new one may not --
#: which thread owns it, what lock is held, how long the state it touches
#: lives, who produced that state. A function that is safe where it is called
#: now is not thereby safe from anywhere.
_MOVES_A_CALL = re.compile(
    r"\b(?:call|calling|invoke|invoking|move|moving|relocate|hoist|reuse|"
    r"trigger)\b[^.;]{0,80}\b(?:from|into|in|at|inside|within|to)\b",
    re.I)

#: A plan item saying plainly that no automated test can reach this. Narrow
#: on purpose: it is an admission, and an admission has to be made, not
#: inferred from a vague validation field.
_NO_TEST_POSSIBLE = re.compile(
    r"\bno\s+(?:automated\s+|executable\s+)?test\b"
    r"|\bcannot\s+be\s+(?:automatically\s+)?tested\b"
    r"|\bnot\s+testable\b|\buntestable\b"
    r"|\brequires?\s+(?:real\s+)?hardware\b", re.I)

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

    #: What is to become of this requirement. Not one of FIELDS: the seven
    #: are what must be SAID, and this is what is being said about them. It
    #: defaults to the ordinary case, so a turn that says nothing about it
    #: has planned a change, which is what calling this tool usually means.
    disposition: str = str(Disposition.CHANGE_PLANNED)

    #: The canonical provision this item answers, resolved by the ledger from
    #: whatever citation the model wrote. It is the item's IDENTITY: two calls
    #: that resolve to the same provision are one item, however differently
    #: they are worded. Empty when the item answers nothing carried, and then
    #: the citation itself has to serve -- a turn may discover a requirement
    #: of its own, and it should not thereby acquire a duplicate every time
    #: it revises it.
    bound_to: str = ""

    FIELDS = ("requirement", "requirement_evidence", "current_behaviour",
              "implementation_evidence", "gap", "correction", "validation")

    @classmethod
    def from_mapping(cls, raw):
        raw = raw if isinstance(raw, dict) else {}
        stated = str(raw.get("disposition") or "").strip().lower()

        return cls(disposition=(stated if stated in
                                {str(value) for value in requirement_set.STATED}
                                else str(Disposition.CHANGE_PLANNED)),
                   **{name: str(raw.get(name) or "").strip()
                      for name in cls.FIELDS})

    def missing(self):
        """The fields this item left empty, in declaration order."""
        return tuple(name for name in self.FIELDS if not getattr(self, name))

    def settled_disposition(self):
        """The label, corrected by the item's own content.

        An item that says it could not be determined, and then names the
        function to change and the test that will prove it, has determined it.
        Measured: two runs of one workflow answered `undetermined` to every
        requirement they carried while describing concrete corrections for
        each -- the label is cheap and the seven fields are not, so the fields
        decide. It is the same principle as everywhere else here: what the
        turn DID beats what the turn called it.
        """
        if self.disposition != str(Disposition.UNDETERMINED):
            return self.disposition

        names = _PATH.findall(self.correction) or re.findall(
            r"\b\w+\(\)", self.correction)

        return (str(Disposition.CHANGE_PLANNED) if names and self.gap
                else self.disposition)

    @property
    def identity(self):
        """What makes two plan items the same item.

        The provision, when one was resolved; otherwise the citation the item
        gave, normalised. NEVER the free-text requirement statement: a model
        rewords its own sentence between calls, and keying on that is how one
        rule became eight logical requirements in a single turn.
        """
        return self.bound_to or " ".join(
            self.requirement_evidence.lower().split())[:120]

    def to_dict(self):
        return {**{name: getattr(self, name) for name in self.FIELDS},
                "disposition": self.disposition, "bound_to": self.bound_to}


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

    #: What the previous grounded turn established and this one inherited.
    #: Empty on a turn that established its own scope, and then the coverage
    #: gate below has nothing to ask for.
    requirements: RequirementSet = field(default_factory=RequirementSet)

    items: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    replans: list = field(default_factory=list)
    #: Plan items replaced by a later call for the same provision.
    revised: list = field(default_factory=list)

    #: Where the first write landed, as a phase. The whole diagnostic rests on
    #: this one field: "the first edit happened during INVESTIGATE" is the
    #: failure, stated without reference to any round number.
    first_write_phase: str = ""
    writes: int = 0
    refusals: int = 0

    validations: list = field(default_factory=list)
    failed_validations: dict = field(default_factory=dict)
    #: Every path this turn actually wrote. A test named in a plan item
    #: counts as evidence only if it is among these: a suite the turn ran but
    #: did not touch cannot have gained a check for behaviour invented today.
    written: set = field(default_factory=set)

    consecutive_without_evidence: int = 0
    syntheses_forced: int = 0

    #: Calls made since both halves of the gap were in hand and no plan had
    #: been accepted. INVESTIGATE has an end; without something that notices
    #: it, "read a bit more" is always available and a turn takes it.
    calls_since_investigated: int = 0
    plan_demands: int = 0

    # -- engagement -------------------------------------------------------

    def engage(self, *, authority_bound, write_requested, root="",
               prior_authority=(), requirements=None):
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

        # The contract, when the turn pointed back at one. Its provisions
        # count as authority evidence for planning, for the same reason the
        # carried clauses do: they were read, in this session, out of the
        # bound document.
        if requirements is not None and self.engaged:
            self.requirements = requirements
            carried_keys = set()

            for item in requirements:
                carried_keys |= requirement_set.citations_in(item.key)

                if item.section:
                    carried_keys.add(item.section)

                # The handle the requirement was retrieved under. A plan item
                # that cites a carried requirement by its own source id was
                # refused as unread -- the one citation form that is exactly
                # and only this provision, and the only one the evidence state
                # did not hold.
                if item.source_id:
                    carried_keys.add(item.source_id)

            # ...and the section each printed citation sits in. "Rule
            # 8.4.1.1-3" identifies a provision; the evidence state keys on
            # sections, and without the ordinal stripped off, a plan item
            # citing the very rule it inherited was refused as unread.
            carried_keys |= {key.rsplit("-", 1)[0]
                             for key in list(carried_keys) if "-" in key}
            self.evidence.observe_authority(carried_keys, 0)

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
        verdicts = [self._judge(self._bound(GapItem.from_mapping(raw)))
                    for raw in (raw_items or [])]
        accepted = tuple(v for v in verdicts if v.accepted)
        rejected = tuple(v for v in verdicts if not v.accepted)

        if supersedes or reason:
            self.replans.append(Replan(reason or "replanned",
                                       supersedes, new_evidence))

        if supersedes:
            self.items = [item for item in self.items
                          if item.requirement != supersedes]

        # One item per provision. A second call for the same requirement is a
        # REVISION of it, not a second requirement: the run that made this
        # necessary dispositioned one rule eight times and was told each time
        # that another planned change now stood, until seventeen of them stood
        # for four requirements and the closing report listed six things that
        # were mostly not requirements at all.

        for verdict in accepted:
            known = {item.identity: index
                     for index, item in enumerate(self.items)}
            at = known.get(verdict.item.identity)

            if at is None:
                self.items.append(verdict.item)
            else:
                self.revised.append(
                    {"requirement": verdict.item.identity,
                     "from": self.items[at].disposition,
                     "to": verdict.item.disposition})
                self.items[at] = verdict.item

        self.rejected.extend(rejected)

        # Bind each accepted item to the carried requirement it answers, and
        # record what the turn said would become of it. An item that answers
        # nothing carried is fine -- a turn may discover a requirement of its
        # own -- but it closes nothing either.

        for verdict in accepted:
            self.requirements.dispose(
                verdict.item.requirement_evidence,
                verdict.item.settled_disposition(),
                note=verdict.item.correction[:200])

        if self.items and self.phase in (Phase.INVESTIGATE, Phase.PLAN):
            self.phase = Phase.EDIT
        elif self.items and self.phase == Phase.REVIEW:
            self.phase = Phase.EDIT

        self.consecutive_without_evidence = 0

        return PlanAcceptance(accepted, rejected)

    def _bound(self, item):
        """Resolve the item's citation to a carried provision, once.

        Done before the item is judged, so identity is settled by the ledger
        rather than by whatever the model typed -- and so a citation that
        resolves to a provision is recorded under the provision's own name.
        """
        found = self.requirements.match(item.requirement_evidence)

        return item if found is None else replace(item, bound_to=found.key)

    def _judge(self, item):
        empty = item.missing()

        if empty:
            return ItemVerdict(item, False,
                               "missing: " + ", ".join(empty))

        # An item the ledger resolved to a carried provision is backed by
        # construction: that provision was retrieved, in this session, out of
        # the bound document, which is the whole question this check asks.
        # Without this, a citation the ledger had just matched could still be
        # refused as unread, and the requirement it answered stayed silent.
        if item.bound_to:
            pass
        elif not self.evidence.backs_requirement(item.requirement_evidence):
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

        # A requirement that says WHEN is not satisfied by code that emits the
        # right thing eventually. The provision's own condition is quoted back
        # and the item has to say where in the code that point is reached --
        # a run produced acknowledgements at the wrong lifecycle stage and
        # every field of its plan was filled in correctly.
        carried = self.requirements.match(item.requirement_evidence)

        if (carried is not None and carried.trigger
                and not _NAMES_A_PATH.search(item.correction)):
            return ItemVerdict(
                item, False,
                f"this provision conditions its obligation — \"{carried.trigger}\" "
                f"— so the correction has to name the point in the code where "
                f"that condition is reached: the function or file the change "
                f"lands in, not only what it will do")

        # Two sites, two readings. An item that moves an existing call has to
        # show it read BOTH ends -- the thing being called and the place it is
        # to be called from -- because the constraints that matter are never
        # written at the call site being added.
        if _MOVES_A_CALL.search(item.correction):
            sites = {_basename(found) for found
                     in _PATH.findall(item.implementation_evidence)}

            if len(sites) < 2:
                return ItemVerdict(
                    item, False,
                    "this change calls something from a place it is not "
                    "called from today. Name BOTH files in "
                    "`implementation_evidence` -- where the thing being "
                    "called is defined and where the new call goes -- and say "
                    "in `current_behaviour` what the existing callers assume "
                    "about thread, lock or lifetime. A function that is safe "
                    "where it is called now is not safe from anywhere.")

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

        # Always back to PLAN, whether or not the caller could name the item
        # at fault. A general invalidation is the case where it CANNOT -- the
        # project's build failing the same way twice says the understanding is
        # wrong without saying which part of it -- and leaving such a turn in
        # EDIT let it write its third patch against the same wrong plan.
        self.phase = Phase.PLAN

        self.consecutive_without_evidence = 0

        return self

    # -- the write gate ---------------------------------------------------

    def uncovered_requirements(self):
        """Carried requirements this turn has said nothing about yet.

        Not the same question as `open_items`. This one is about SILENCE, and
        it is what the write gate asks: a requirement the turn has declared
        undeterminable has been answered and does not hold the gate shut, even
        though it does keep the turn from being finished.
        """
        return self.requirements.unstated()

    def may_write(self):
        """Whether a mutating action may run right now, and why not."""
        if not self.engaged:
            return WriteDecision(True)

        # The coverage gate, and it is checked BEFORE the phase: a plan that
        # covers one of five carried requirements is a plan for one fifth of
        # the task, and letting it write is how the other four stop being
        # mentioned. Four runs of the same two turns implemented between one
        # and eight of the behaviours the first turn had established; the
        # ones that dropped requirements dropped them here, silently.
        #
        # Every disposition is accepted, including "I could not tell". What
        # is refused is saying nothing at all.

        open_items = self.uncovered_requirements()

        if open_items and self.phase != Phase.REVIEW:
            # A requirement that names several cases is not closed by a
            # disposition that answers one of them, so the cases are put in
            # front of the turn with the requirement. Which cases they are is
            # read out of the provision's own words -- see requirement_set.
            listed = "; ".join(
                f"{item.key} — {item.statement[:110]}"
                + (f" [covers each of: {', '.join(item.members)}]"
                   if item.members else "")
                for item in open_items[:6])
            more = (f" and {len(open_items) - 6} more"
                    if len(open_items) > 6 else "")

            return WriteDecision(
                False, PLAN_INCOMPLETE,
                f"{WRITE_BLOCKED} The previous turn established "
                f"{len(self.requirements)} requirement(s) and "
                f"{len(open_items)} of them still have no disposition. Call "
                f"`{PLAN_TOOL}` once for each, citing it in "
                f"`requirement_evidence`, with `disposition` set to "
                f"change_planned, satisfied_already, explicitly_out_of_scope "
                f"or undetermined. Outstanding: {listed}{more}")

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

    def note_write(self, paths=()):
        """A mutation that reached the tree, and the phase it landed in.

        The phase is recorded as it WAS, never as it should have been. If a
        write ever lands during INVESTIGATE, the diagnostic must say so rather
        than tidy it away -- that reading is the whole point of the field.

        A requirement moves from planned to implemented HERE, from the paths
        the write actually touched, and not when the model says it is done:
        whether a file changed is a fact about the tree.
        """
        self.writes += 1

        if not self.first_write_phase:
            self.first_write_phase = str(self.phase)

        touched = {_basename(str(path)) for path in paths if path}
        self.written |= {str(path) for path in paths if path}

        if not touched:
            return self

        for item in self.items:
            named = {_basename(found) for found
                     in _PATH.findall(item.implementation_evidence + " "
                                      + item.correction)}

            if not (named & touched):
                continue

            carried = self.requirements.match(item.requirement_evidence)

            # Promoted from UNDETERMINED too: the file was written, which
            # settles what a label could only claim.
            if carried is not None and carried.disposition in (
                    str(Disposition.CHANGE_PLANNED),
                    str(Disposition.UNDETERMINED)):
                self.requirements.dispose(
                    carried.key, Disposition.CHANGE_IMPLEMENTED,
                    note=carried.note, by="harness")

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

    def unvalidated_requirements(self):
        """Requirements whose behaviour changed and nothing exercised it.

        A green suite is not evidence about a branch it never reaches. The
        run that made this necessary changed acknowledge emission, added no
        test, ran 198 existing checks, and reported that all of them passed --
        which was true, and said nothing whatever about the new behaviour.

        Two ways out, and both are honest: a validation that ran and PASSED
        naming something the turn also wrote (a test it added or changed), or
        an explicit statement in the plan item that no automated test can
        reach it. What is refused is the third way, where an untouched suite
        is offered as proof.
        """
        if not self.engaged or not len(self.requirements):
            return ()

        proven, excused = set(), set()

        for item in self.items:
            if item.identity in self._validated_identities():
                proven.add(item.bound_to or item.identity)

            if _NO_TEST_POSSIBLE.search(item.validation):
                excused.add(item.bound_to or item.identity)

        changed = {found.key for found in self.requirements
                   if found.disposition == str(Disposition.CHANGE_IMPLEMENTED)}

        return tuple(found for found in self.requirements
                     if found.key in changed - proven - excused)

    def _validated_identities(self):
        """Plan items a validation that passed actually reached.

        The link is the test the item named AND the turn touched: a test file
        the turn wrote or changed is a test that can reach the new path, and
        a pre-existing one it merely ran is not evidence that it does.
        """
        if not self.validated:
            return set()

        touched = {_basename(path) for path in self.written}
        proven = set()

        for item in self.items:
            named = {_basename(found)
                     for found in _PATH.findall(item.validation)}

            if named & touched:
                proven.add(item.identity)

        return proven

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

    def contract_closed(self):
        """Every in-scope requirement disposed, none open, none unvalidated.

        The signal to land. A real run reached this state and then went on
        revisiting the ledger, re-dispositioning rules it had already closed
        and re-reading files it had already read, for another several minutes.
        There is nothing left to find once the contract is closed; what is
        left is to say so.
        """
        if not self.engaged or not len(self.requirements):
            return False

        return (not self.requirements.open_items()
                and not self.requirements.unstated()
                and not self.unvalidated_requirements())

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
            "requirements": self.requirements.to_dict(),
            "requirement_matrix": self.requirements.matrix(),
            "requirements_open": [item.key
                                  for item in self.requirements.open_items()],
        }
