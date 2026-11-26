"""A field placed where no retrieved description puts it.

Seen in use: the GIC specification says "IRM, bit [40]" and the answer said
bit [4]. The guards checked names and force, never the numbers copied
beside them.
"""

import unittest

from normative import normative_claims as nc
from runtime import answer_repair as ar


def evidence(*texts):
    found = nc.NormativeEvidence()

    for index, text in enumerate(texts):
        found.observe({"text": text, "section": "12.2.22",
                       "source_id": f"std-{index:032d}"})

    return found


SGI1R = evidence("IRM, bit [40] Interrupt Routing Mode.",
                 "INTID, bits [27:24] The INTID of the SGI.",
                 "INTID, bits [23:0] The INTID of the pending interrupt.")


class Positions(unittest.TestCase):
    def kinds(self, answer, found=SGI1R):
        return [(item["identifier"], item["claimed"])
                for item in nc.position_findings(answer, found)]

    def test_a_miscopied_bit_is_found(self):
        self.assertEqual(self.kinds("IRM (Interrupt Routing Mode), bit [4] — routing."),
                         [("IRM", "[4]")])

    def test_the_stated_position_passes(self):
        self.assertEqual(self.kinds("IRM, bit [40] selects the routing mode."), [])

    def test_a_name_reused_across_registers_matches_any_of_its_positions(self):
        self.assertEqual(self.kinds("INTID, bits [27:24] carries the SGI number."), [])
        self.assertEqual(self.kinds("INTID, bits [3:0] carries it."), [("INTID", "[3:0]")])

    def test_table_rows_are_checked(self):
        self.assertEqual(self.kinds("| Field | Bits |\n|---|---|\n| **IRM** | [4] | routing |"),
                         [("IRM", "[4]")])

    def test_a_field_the_evidence_does_not_position_is_left_alone(self):
        self.assertEqual(self.kinds("TargetList, bits [15:0] selects the PEs."), [])
        self.assertEqual(self.kinds("IRM, bit [4]", evidence("No positions here.")), [])

    def test_it_reaches_the_guard_and_is_repairable(self):
        problems = nc.findings("IRM, bit [4] selects the routing mode.", SGI1R)

        self.assertIn(nc.MISPOSITIONED_FIELD, [item["kind"] for item in problems])
        position = [item for item in problems if item["kind"] == nc.MISPOSITIONED_FIELD]
        self.assertTrue(ar.is_repairable(position))
        self.assertIn("[40]", "\n".join(ar._finding_lines(position)))


if __name__ == "__main__":
    unittest.main()
