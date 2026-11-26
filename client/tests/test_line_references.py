"""File:line references checked against the files a turn touched."""

import tempfile
import unittest
from pathlib import Path

from runtime import line_references as lr

SOURCE = "\n".join([
    "#include <smp.h>",            # 1
    "",                            # 2
    "static void ipi_setup_sgi(int ipi)",  # 3
    "{",                           # 4
    "\terr = request_percpu_irq(irq, ipi_handler);",  # 5
    "}",                           # 6
    "",                            # 7
    "static void other(void)",     # 8
]) + "\n"


class References(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        path = Path(self.root, "arch/kernel/smp.c")
        path.parent.mkdir(parents=True)
        path.write_text(SOURCE)
        self.log = ["terminal {\"command\": \"sed -n '1,8p' arch/kernel/smp.c\"}"]

    def tearDown(self):
        self.temp.cleanup()

    def check(self, answer):
        return lr.check(answer, self.root, self.log)

    def test_a_drifted_line_is_corrected_and_said(self):
        text, fixes = self.check("registered by ipi_setup_sgi() (smp.c:2).")

        self.assertEqual(fixes, [("smp.c", 2, 3, "ipi_setup_sgi")])
        self.assertIn("(smp.c:3)", text)
        self.assertIn("smp.c:2 → 3 (ipi_setup_sgi)", text)

    def test_a_right_reference_and_a_continuation_stand(self):
        answer = "ipi_setup_sgi() (arch/kernel/smp.c:3, request_percpu_irq at :5)."

        self.assertEqual(self.check(answer), (answer, []))

    def test_a_continuation_is_corrected_against_the_file_named_last(self):
        text, fixes = self.check("ipi_setup_sgi (smp.c:3), request_percpu_irq at :6.")

        self.assertEqual(fixes, [("smp.c", 6, 5, "request_percpu_irq")])

    def test_nothing_is_guessed(self):
        for answer in ("smp.c:2 is where it starts.",            # no symbol
                       "ipi_setup_sgi() (smp.c:40)",             # too far
                       "ipi_setup_sgi() (unknown.c:2)",          # untouched file
                       "RS, bits [47:44] and ipi_setup_sgi [7:2]"):  # a bit range
            self.assertEqual(self.check(answer), (answer, []), answer)

    def test_files_the_turn_did_not_touch_are_not_opened(self):
        answer = "ipi_setup_sgi() (smp.c:2)"

        self.assertEqual(lr.check(answer, self.root, []), (answer, []))


if __name__ == "__main__":
    unittest.main()
