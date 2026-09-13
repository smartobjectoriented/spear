"""What may speak for the standard, and what may only speak for the code.

A bound session read a source comment claiming the standard allowed two
response selectors together, and repeated it as a description of the standard.
The bound clause says the opposite. The comment was wrong; the answer was
wrong because it took the comment as evidence about the document.

The standard here is invented -- see `synthetic_standard` -- because the
clause text of a real one is its publisher's. What is under test is the
precedence rule, which does not care which document the evidence came from.

Two rules have to hold at once and they pull in opposite directions:

  precedence  the standard outranks every implementation artefact
  scope       the standard is not a list of work items

The second is the task-scope fix, already committed. These tests hold both,
because the cheapest way to satisfy precedence is to turn every clause into
an obligation, and that is the bug we just removed.
"""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import conformance_guard
import normative_precedence
import synthetic_standard
import standard_answer_policy as sap
from conformance_guard import ClauseLedger

BINDING = synthetic_standard.BINDING


def executable_text(module):
    """Everything a module DOES, with its prose left out.

    Docstrings and comments are dropped; string literals, identifiers and
    attribute names are kept. `ast` discards comments on its own, so only
    the docstrings have to be identified.
    """
    tree = ast.parse(inspect.getsource(module))
    prose = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue

        first = node.body[0] if node.body else None

        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            prose.add(id(first.value))

    parts = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in prose:
                parts.append(node.value)
        elif isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            parts.append(node.name)

    return "\n".join(parts)

# The clause the turn retrieves, and the comment in the tree that contradicts
# it. Both come from the synthetic standard: what is tested is precedence, and
# no rule of any real document is hardcoded in the module under test.
CLAUSE = synthetic_standard.SECTION
COMMENT = synthetic_standard.CODE_COMMENT


def guard(answer, sections=(), binding=BINDING):
    ledger = ClauseLedger(sections=set(sections))

    return normative_precedence.guard(
        answer, ledger, standard_id="ANSI-VITA-49.2", revision="2017-R2024",
        binding=binding)


class ACommentIsNotTheStandard(unittest.TestCase):
    """CASE 1 -- asked what the standard requires, with a comment in the way."""

    REPEATED = synthetic_standard.REPEATED_COMMENT
    GROUNDED = synthetic_standard.GROUNDED_CLAIM

    def test_the_comment_repeated_as_the_standard_is_not_entitled(self):
        _, findings, _, fired = guard(self.REPEATED, sections=[CLAUSE])

        self.assertTrue(fired)
        self.assertEqual([item["kind"] for item in findings],
                         [normative_precedence.UNGROUNDED_REQUIREMENT])

    def test_reading_the_comment_does_not_ground_it(self):
        """The ledger is fed by the normative tools and by nothing else."""
        policy = sap.policy_for(BINDING, "what does the standard require?")
        policy.observe_code_read("bash", {"command": "sed -n 1,40p src/ack.c"},
                                 COMMENT)

        self.assertEqual(policy.clauses.sections, set())
        self.assertEqual(sorted(policy.code.files), ["src/ack.c"])

    def test_the_answer_from_the_document_stands(self):
        answer, findings, replaced, fired = guard(self.GROUNDED,
                                                  sections=[CLAUSE])

        self.assertFalse(fired)
        self.assertEqual((findings, replaced), ([], []))
        self.assertEqual(answer, self.GROUNDED)

    def test_the_same_answer_without_the_retrieval_is_not(self):
        # Same words, nothing read: what makes it evidence is the retrieval.

        _, _, _, fired = guard(self.GROUNDED, sections=[])

        self.assertTrue(fired)

    def test_what_replaces_it_says_where_a_requirement_comes_from(self):
        answer, _, _, _ = guard(self.REPEATED, sections=[CLAUSE])

        self.assertNotIn("allows SelA", answer)
        self.assertIn(f"§{synthetic_standard.SECTION}", answer)
        self.assertIn("code, comments and notes describe the implementation",
                      answer)


class DeviationsAreReportable(unittest.TestCase):
    """CASE 2 -- the implementation contradicts the standard."""

    REPORT = (synthetic_standard.GROUNDED_CLAIM + " widgetEncodeReply() sets SelA and "
              "SelB together at reply.c:212, which deviates from it.")

    def test_a_deviation_report_survives_intact(self):
        answer, _, _, fired = guard(self.REPORT, sections=[CLAUSE])

        self.assertFalse(fired)
        self.assertEqual(answer, self.REPORT)

    def test_existing_behaviour_is_not_a_clearance(self):
        """The old guard still owns the other half: a verdict needs a clause."""
        cleared = "The implementation correctly follows the standard."
        _, findings, _, fired = conformance_guard.guard(
            cleared, ClauseLedger(sections={CLAUSE}),
            standard_id="ANSI-VITA-49.2", revision="2017-R2024",
            binding=BINDING)

        self.assertTrue(fired)
        self.assertEqual([item["kind"] for item in findings],
                         [conformance_guard.UNSCOPED_VERDICT])


class UnsupportedIsNotNonCompliant(unittest.TestCase):
    """CASE 3 -- the standard defines more than the code claims to support."""

    UNSUPPORTED = (f"{synthetic_standard.OPTIONAL_FEATURE.capitalize()} is not supported "
                   "by this implementation: widgetEncodeReply() never "
                   "writes it.")

    def test_saying_a_feature_is_unsupported_is_not_guarded(self):
        answer, findings, _, fired = guard(self.UNSUPPORTED, sections=[CLAUSE])

        self.assertFalse(fired)
        self.assertEqual(findings, [])
        self.assertEqual(answer, self.UNSUPPORTED)

    def test_a_retrieved_clause_creates_no_work(self):
        """Precedence and scope are orthogonal, and this is where they meet.

        The cheapest way to make the standard outrank the code is to demand
        the code satisfy every clause retrieved. That is the bug the
        task-scope fix removed, and nothing here may bring it back.
        """
        from agent_runtime import carried_obligations

        class Context:
            prior_clauses = (synthetic_standard.SECTION, "4.4.2", "4.1.1")
            read_only = False

        self.assertEqual(
            carried_obligations(Context(), "does this implementation comply "
                                           "with the bound standard?"), ())

    def test_the_rule_says_so_in_the_turns_own_words(self):
        rule = normative_precedence.PRECEDENCE_RULE

        self.assertIn("A clause is not a work item", rule)
        self.assertIn("not a defect", rule)


class TheCodeAnswersForItself(unittest.TestCase):
    """CASE 4 -- asked how the implementation behaves."""

    BEHAVIOUR = ("widgetEncodeReply() writes the SelA bit when SelA was "
                 "asked for and the SelB bit when SelB was, in the same word, "
                 "and emits one reply. The caller must provide a request id.")

    def test_a_description_of_the_code_is_left_alone(self):
        answer, findings, _, fired = guard(self.BEHAVIOUR, sections=[])

        self.assertFalse(fired)
        self.assertEqual(findings, [])
        self.assertEqual(answer, self.BEHAVIOUR)

    def test_the_two_statements_stay_apart(self):
        both = ("The implementation emits one reply for both selectors. "
                "Rule 4.2.1.1-3 requires one reply per selector asked for.")
        answer, _, _, fired = guard(both, sections=[synthetic_standard.SECTION])

        self.assertFalse(fired)
        self.assertEqual(answer, both)


class RepairFollowsTheStandard(unittest.TestCase):
    """CASE 5 -- the target behaviour comes from the clause, not the code."""

    def test_a_bound_turn_is_told_which_artefact_is_suspect(self):
        rule = normative_precedence.PRECEDENCE_RULE

        self.assertIn("the artefact is the suspect one", rule)
        self.assertIn("do not reinterpret the clause", rule)

    def test_an_unresolved_conflict_is_reported_not_decided(self):
        self.assertIn("report the uncertainty rather than choosing a "
                      "behaviour", normative_precedence.PRECEDENCE_RULE)

    def test_the_rule_reaches_every_bound_turn(self):
        """Attached to the binding, not to a project's system prompt.

        A system prompt is chosen by corpus kind; a binding is a property of
        the turn. A rule that lived in a prompt would miss exactly the
        sessions that bind a standard from any other kind of corpus -- which
        is what happened to the evidence half of this contract until it moved
        here too.
        """
        import task_controller

        source = inspect.getsource(task_controller.TaskController.run)

        self.assertIn("normative_precedence.PRECEDENCE_RULE", source)
        self.assertIn("normative_precedence.EVIDENCE_RULE", source)
        self.assertIn("BOUND STANDARD", source)


class ProvenanceDecidesNormativity(unittest.TestCase):
    """CASE 6 -- what makes a statement normative is where it came from."""

    CLAIM = "The standard requires the Message ID to be echoed."

    def test_no_source_of_evidence_but_the_document_grounds_a_claim(self):
        policy = sap.policy_for(BINDING, "what does the standard require?")

        # Everything an implementation can offer, in the voice of the
        # standard, through every channel that is not a normative tool.
        for name, arguments, text in (
            ("bash", {"command": "grep -rn 'Rule 8.4.1.1' src/"},
             "src/ack.c:12: /* Rule 8.4.1.1-2 permits AckS with AckX */"),
            ("read_file", {"path": "doc/design.md"},
             "The standard requires AckS and AckX to travel together."),
            ("search_corpus", {"query": "acknowledge"},
             "skill: the standard allows combined acknowledgement bits."),
        ):
            policy.observe_code_read(name, arguments, text)

        self.assertEqual(policy.clauses.sections, set())

        _, findings, _, fired = normative_precedence.guard(
            self.CLAIM, policy.clauses, standard_id="ANSI-VITA-49.2",
            revision="2017-R2024", binding=BINDING)

        self.assertTrue(fired)
        self.assertEqual(len(findings), 1)

    def test_only_the_normative_tools_write_the_ledger(self):
        """Not a property of today's wiring: of where the runtime calls it."""
        from agent_runtime import AgentRuntime

        source = inspect.getsource(AgentRuntime.run)

        self.assertIn("call.name in policy.STANDARD_TOOLS", source)
        self.assertIn("policy.observe_tool_result", source)

    def test_the_rule_names_the_artefacts_that_are_not_evidence(self):
        rule = normative_precedence.PRECEDENCE_RULE.lower()

        for artefact in ("code", "comments", "tests", "documentation",
                         "skills", "memories", "earlier answers"):
            with self.subTest(artefact=artefact):
                self.assertIn(artefact, rule)


class ProseKeepsItsCitation(unittest.TestCase):
    """What the real backend wrote, and what sentence-local grounding did to it.

    Qwen3-Coder-Next, bound to the standard, quoted the rule in full, concluded
    from it, and closed with a restatement carrying no citation of its own.
    The first version of this guard replaced that true, retrieved, correctly
    cited claim with a note saying nothing backed it -- a worse sentence than
    the one it removed. Prose does not repeat its citation in every clause.
    """

    ANSWER = (
        f"According to the bound {synthetic_standard.SID} standard "
        f"(revision {synthetic_standard.REV}):\n\n"
        "**Normative requirement:**\n"
        f"- {synthetic_standard.RULE} [\u00a7{synthetic_standard.SECTION}, p.{synthetic_standard.PAGE}]\n"
        "- The subtype rules further specify that the other two selectors "
        "are clear in each case.\n\n"
        "Therefore, SelA and SelB may NOT be set simultaneously in a single "
        "Widget Reply. The standard requires exactly one of the three "
        "selectors to be set.")

    def test_a_restatement_of_a_cited_rule_is_grounded(self):
        answer, findings, _, fired = guard(self.ANSWER, sections=[CLAUSE])

        self.assertEqual(findings, [])
        self.assertFalse(fired)
        self.assertEqual(answer, self.ANSWER)

    def test_the_same_restatement_with_nothing_read_is_not(self):
        _, findings, _, fired = guard(self.ANSWER, sections=[])

        self.assertTrue(fired)
        self.assertTrue(findings)

    def test_the_citation_does_not_carry_indefinitely(self):
        """Bounded, or one citation licenses a whole answer."""
        stray = "The standard permits SelA to be combined with SelB."
        far = (self.ANSWER + " The converter emits one reply per request. "
               "The journal records each one. The link layer retries on "
               "timeout. Each entry carries a request id. " + stray)
        _, findings, _, fired = guard(far, sections=[CLAUSE])

        self.assertTrue(fired)
        self.assertEqual([item["sentence"] for item in findings], [stray])

    def test_the_window_is_named_once(self):
        self.assertEqual(normative_precedence._CONTEXT_SENTENCES, 3)


class TheGuardEventReachesTheTrace(unittest.TestCase):
    """A guard that fires invisibly cannot be diagnosed.

    The runtime filters the policy trace to a fixed key list. The new keys
    were missing from it, so the removed sentence had to be recovered by
    re-running the whole turn under --record.
    """

    def test_the_runtime_emits_the_precedence_keys(self):
        from agent_runtime import AgentRuntime

        source = inspect.getsource(AgentRuntime._result)

        self.assertIn('"normative_precedence_triggered"', source)


class TheGuardIsGeneric(unittest.TestCase):
    """No standard, no clause and no packet field is spelled into the code."""

    def test_nothing_in_the_module_names_a_standard_or_a_clause(self):
        """The EXECUTABLE module: literals, names, patterns.

        Prose is exempt and deliberately so -- every guard in this codebase
        states the failure that produced it, in the words of the session
        that failed. What must not name a standard is the code: a pattern
        or a constant that spells one out works for that standard alone and
        silently passes the same sentence about every other.
        """
        body = executable_text(normative_precedence)

        for token in ("VITA", "vita", "AckV", "AckX", "AckS", "8.4", "49.2"):
            with self.subTest(token=token):
                self.assertNotIn(token, body)

    def test_the_subject_is_built_from_the_binding(self):
        other = {"standard_id": "NIST-RS274NGC", "revision": "v3"}
        claim = "NIST-RS274NGC requires the feed rate to precede the move."

        self.assertTrue(guard(claim, sections=[], binding=other)[3])
        # ...and the same sentence is not about the standard bound elsewhere.
        self.assertFalse(guard("NIST-RS274NGC requires the feed rate to "
                               "precede the move.", sections=[],
                               binding=BINDING)[3])

    def test_an_unbound_turn_has_no_policy_at_all(self):
        self.assertIsNone(sap.policy_for(None, "what does the standard say?"))


class ThePolicyRunsIt(unittest.TestCase):
    """The guard is in the chain, not merely importable."""

    def test_an_unbacked_requirement_does_not_reach_the_reader(self):
        policy = sap.policy_for(BINDING, "what does the standard require?")
        answer = policy.finalize(
            "The standard permits AckS to be combined with AckX.",
            stopped_by=sap.STOPPED_BY_MODEL)

        self.assertTrue(policy.precedence_fired)
        self.assertNotIn("permits AckS", answer)

    def test_the_trace_records_what_was_removed(self):
        policy = sap.policy_for(BINDING, "what does the standard require?")
        policy.finalize("The standard allows both bits to be set.",
                        stopped_by=sap.STOPPED_BY_MODEL)
        trace = policy.trace()

        self.assertTrue(trace["normative_precedence_triggered"])
        self.assertEqual(trace["ungrounded_requirements"][0]["kind"],
                         normative_precedence.UNGROUNDED_REQUIREMENT)

    def test_a_grounded_answer_passes_the_whole_chain(self):
        policy = sap.policy_for(BINDING, "what does the standard require?")
        policy.clauses.sections.add(CLAUSE)
        grounded = synthetic_standard.GROUNDED_CLAIM
        self.assertEqual(
            policy.finalize(grounded, stopped_by=sap.STOPPED_BY_MODEL),
            grounded)
        self.assertFalse(policy.precedence_fired)

    def test_the_reader_is_told_when_nothing_survived(self):
        policy = sap.policy_for(BINDING, "what does the standard require?")
        policy.finalize("The standard permits AckS with AckX.",
                        stopped_by=sap.STOPPED_BY_MODEL)

        self.assertIn("normative precedence", policy._withheld_note())


if __name__ == "__main__":
    unittest.main()
