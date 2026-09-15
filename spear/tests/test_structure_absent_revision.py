"""A revision with no geometry pass answers; it does not raise.

Measured against a second extraction of a bound standard held in its own
store: `standard.get_structure` came back as `StandardStructureError: no
structures for this revision`, an exception the model sees as a broken tool
rather than as a fact about the revision. Never having run the pass is a
state, and the caller has to be able to tell it apart from a store whose
inputs moved under an approval.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import standard_structure_access as access
from standard_store import StandardStore


class AnUnbuiltRevisionIsAState(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = StandardStore(Path(self.tmp.name) / "standards")

    def refusal(self):
        reader = access.StandardStructureAccess(self.store)
        with self.assertRaises(access.StructureAccessError) as caught:
            reader.index("ANSI-VITA-49.2", "never-built")
        return caught.exception

    def test_it_is_named_rather_than_raised_as_a_fault(self):
        self.assertEqual(self.refusal().reason, access.NO_STRUCTURE_STORE)

    def test_it_is_not_reported_as_a_stale_store(self):
        """Nothing has moved under an approval, so calling it stale would
        send an operator looking for a rebuild that would not help."""
        self.assertNotEqual(self.refusal().reason, access.STALE_SEMANTIC_STORE)

    def test_the_reason_survives_into_the_tool_payload(self):
        found = self.refusal().to_dict()

        self.assertEqual(found["error"], access.NO_STRUCTURE_STORE)


if __name__ == "__main__":
    unittest.main()
