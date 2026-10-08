"""Coverage: what a bound change answers to, beyond what the model cited.

A normative pass once cited five of the six numbered rules of one list, and
the sixth was neither implemented nor checked. Coverage closes a cited
provision over the groups the store itself records -- its numbered list, its
table, the provisions it names -- and nothing wider; applicability is then
decided from deterministic facts only.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from normative import normative_constraints as nc
from normative import normative_coverage as cov
from normative import provision_identity as pi
from tests.standard_fixture import synthetic_pdf_bytes

STANDARD_ID, REVISION = "SYNTH-LIST", "L-1"

PAGES = ((
    (760, "Synthetic List Standard"),
    (738, "Revision L-1 - TEST_FIXTURE / SYNTHETIC"),
    (700, "5 Frames"),
    (670, "5.1 Frame Header"),
    (640, "Rule 5.1-1: The header shall carry a version field."),
    (610, "Rule 5.1-2: The version field shall be 3."),
    (580, "Rule 5.1-3: A receiver shall reject a frame whose version differs."),
    (550, "Observation 5.1-1: Older senders used version 2."),
    (520, "Permission 5.1-1: A sender may pad the header."),
    (490, "5.2 Frame Trailer"),
    (460, "Rule 5.2-1: The trailer shall repeat the version, as stated in Rule 5.1-2."),
    (430, "Rule 5.2-2: The trailer shall end with a zero word."),
),)


def build_store(root):
    from standard.standard_ingest import ingest_pdf
    from standard.standard_retrieval import rebuild_lexical_index
    from standard.standard_store import StandardStore

    pdf = root / "list.pdf"
    pdf.write_bytes(synthetic_pdf_bytes(PAGES))
    store = StandardStore(root / "standards")
    ingest_pdf(store, pdf, standard_id=STANDARD_ID, revision=REVISION,
               source_origin="TEST_FIXTURE")
    rebuild_lexical_index(store, STANDARD_ID, REVISION)

    return store


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            store = build_store(Path(directory))
            units = [unit.to_dict() for unit in store.load_units(STANDARD_ID, REVISION)]

        cls.units = units
        cls.records = pi.records_for_units(units)
        cls.universe = cov.Universe(cls.records, units)

    def record(self, label):
        return next(item for item in self.records if str(item.key) == label)

    def close(self, *labels):
        return cov.close([self.record(label) for label in labels], self.universe)


class Closure(Fixture):
    def test_a_cited_rule_brings_the_rest_of_its_numbered_list(self):
        coverage = self.close("Rule 5.1-2")
        added = {str(item.record.key): item for item in coverage.added}

        self.assertEqual(sorted(added), ["Rule 5.1-1", "Rule 5.1-3"])
        self.assertEqual(added["Rule 5.1-3"].source_relation, cov.SAME_NORMATIVE_LIST)
        self.assertEqual(added["Rule 5.1-3"].originating,
                         cov.instance_of(self.record("Rule 5.1-2")))
        self.assertIn("§5.1", added["Rule 5.1-3"].parent)
        self.assertTrue(coverage.complete)

    def test_nothing_from_another_list_kind_or_section(self):
        added = {str(item.record.key) for item in self.close("Rule 5.1-2").added}

        self.assertNotIn("Permission 5.1-1", added)     # its own list
        self.assertNotIn("Observation 5.1-1", added)    # informative
        self.assertNotIn("Rule 5.2-2", added)           # another section

    def test_a_provision_named_outright_is_one_step_away(self):
        coverage = self.close("Rule 5.2-1")
        added = {str(item.record.key): item.source_relation for item in coverage.added}

        self.assertEqual(added["Rule 5.1-2"], cov.EXPLICIT_REFERENCE)
        self.assertEqual(added["Rule 5.2-2"], cov.SAME_NORMATIVE_LIST)
        self.assertNotIn("Rule 5.1-3", added)            # not onward from the named one

    def test_a_group_over_the_limit_leaves_coverage_incomplete(self):
        with mock.patch.object(cov, "GROUP_LIMIT", 2):
            coverage = self.close("Rule 5.1-2")

        self.assertFalse(coverage.complete)
        self.assertIn("over the limit", coverage.incomplete_reason)
        self.assertEqual(coverage.added, ())

    def test_nothing_cited_is_incomplete(self):
        coverage = cov.close([], self.universe)

        self.assertFalse(coverage.complete)


class QuotedCitations(Fixture):
    def test_a_section_and_its_verbatim_words_name_the_provision(self):
        found = cov.quoted('Per §5.1: "A receiver shall reject a frame whose version '
                           'differs." It also says more.', self.universe)

        self.assertEqual([str(item.key) for item in found], ["Rule 5.1-3"])

    def test_a_shortened_quote_counts_by_its_fragments(self):
        found = cov.quoted('[§5.2] "The trailer shall repeat the version, as stated ... '
                           'in Rule 5.1-2"', self.universe)

        self.assertEqual([str(item.key) for item in found], ["Rule 5.2-1"])

    def test_a_paraphrase_or_the_wrong_section_names_nothing(self):
        for answer in ('§5.1: "receivers must drop frames with another version number"',
                       '§5.2: "A receiver shall reject a frame whose version differs."',
                       'A receiver shall reject a frame whose version differs.'):
            with self.subTest(answer=answer):
                self.assertEqual(cov.quoted(answer, self.universe), [])


class Applicability(Fixture):
    def decide(self, label, objective="o", declared=None, bound=None, origin=cov.CITED):
        entry = cov.CoverageEntry(self.record(label), origin)

        return cov.decide(entry, objective, declared or {}, bound or {})

    def test_the_request_naming_it_or_its_section(self):
        self.assertEqual(self.decide("Rule 5.1-3", "Make the header follow Rule 5.1-3.").status,
                         cov.APPLICABLE)
        self.assertEqual(self.decide("Rule 5.1-3", "Implement the Rule 5.1 provisions.").status,
                         cov.APPLICABLE)
        self.assertEqual(self.decide("Rule 5.2-1", "Implement the Rule 5.1 provisions.").status,
                         cov.UNRESOLVED)
        self.assertEqual(self.decide("Rule 5.1-3", "Implement section 5.11 framing.").status,
                         cov.UNRESOLVED)

    def test_being_cited_is_advice_only(self):
        decision = self.decide("Rule 5.1-3")

        self.assertEqual(decision.status, cov.UNRESOLVED)
        self.assertIn("POSSIBLY_APPLICABLE", decision.advisory)

    def test_the_project_declares_it(self):
        instance = cov.instance_of(self.record("Rule 5.1-3"))
        out = self.decide("Rule 5.1-3", declared={instance: [(cov.NOT_APPLICABLE,
                                                               "receive-only", "projects.json")]})
        both = self.decide("Rule 5.1-3", declared={instance: [
            (cov.NOT_APPLICABLE, "a", "projects.json"), (cov.APPLICABLE, "b", "projects.json")]})
        bound = self.decide("Rule 5.1-3", bound={instance: {"version-check"}})

        self.assertEqual((out.status, out.basis), (cov.NOT_APPLICABLE, "projects.json: receive-only"))
        self.assertEqual(both.status, cov.UNRESOLVED)
        self.assertEqual(bound.status, cov.APPLICABLE)
        self.assertIn("version-check", bound.basis)

    def test_an_identity_must_name_one_instance(self):
        record = self.record("Rule 5.1-2")

        self.assertEqual(self.universe.resolve(cov.instance_of(record)), [record])
        self.assertEqual(self.universe.resolve("Rule 5.1-2"), [record])
        self.assertEqual(self.universe.resolve("Rule 5.1-2@000000000000"), [])
        self.assertEqual(self.universe.resolve("the version rule"), [])


class Packet(Fixture):
    def packet(self, objective, declared=None):
        coverage = self.close("Rule 5.1-2")
        decisions = {item.instance: cov.decide(item, objective, declared or {}, {})
                     for item in coverage.entries}

        return nc.from_coverage(coverage, decisions, standard_id=STANDARD_ID,
                                revision=REVISION, objective=objective)

    def test_the_packet_carries_coverage_and_applicability(self):
        packet = self.packet("Implement the Rule 5.1 provisions.")
        added = [item for item in packet.constraints if item.origin == cov.CLOSURE]

        self.assertEqual([item.provision for item in packet.constraints],
                         ["Rule 5.1-2", "Rule 5.1-1", "Rule 5.1-3"])
        self.assertTrue(all(item.applicability == cov.APPLICABLE for item in packet.constraints))
        self.assertEqual({item.source_relation for item in added}, {cov.SAME_NORMATIVE_LIST})
        self.assertTrue(packet.coverage_complete)

    def test_a_provision_established_inapplicable_leaves_the_packet(self):
        instance = cov.instance_of(self.record("Rule 5.1-3"))
        packet = self.packet("Implement the Rule 5.1 provisions.", {instance: [
            (cov.NOT_APPLICABLE, "this sender never receives", "projects.json")]})

        self.assertNotIn("Rule 5.1-3", [item.provision for item in packet.constraints])
        self.assertNotIn("Rule 5.1-3", nc.brief(packet))

    def test_unresolved_applicability_or_coverage_blocks_compliance(self):
        resolved = self.packet("Implement the Rule 5.1 provisions.")
        unresolved = self.packet("Make the frames right.")
        satisfied = lambda packet: tuple(nc.ConstraintStatus(item.constraint_id, nc.SATISFIED,
                                                             authority="check:x")
                                         for item in packet.constraints)
        violated = tuple(nc.ConstraintStatus(item.constraint_id, nc.VIOLATED, authority="check:x")
                         for item in unresolved.constraints)

        self.assertEqual(nc.normative_status(resolved, satisfied(resolved)), nc.SATISFIED)
        self.assertEqual(nc.normative_status(unresolved, satisfied(unresolved)),
                         nc.NOT_DEMONSTRATED)
        self.assertEqual(nc.normative_status(unresolved, violated), nc.NOT_DEMONSTRATED)
        self.assertEqual(nc.repairable(unresolved, violated), [])

        from dataclasses import replace

        incomplete = replace(resolved, coverage_complete=False)
        self.assertEqual(nc.normative_status(incomplete, satisfied(incomplete)),
                         nc.NOT_DEMONSTRATED)
        report = nc.coverage_report(unresolved, violated)
        self.assertEqual((report["unresolved_applicability"], report["violated"]), (3, 0))


if __name__ == "__main__":
    unittest.main()
