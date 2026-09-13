"""The compatibility path that applies a printed code block to a file.

It exists for models that cannot call tools reliably: the model prints a whole
file, the harness writes it.  The failure it must not have is applying a block
to a file that merely got NAMED in the request.
"""
import unittest

import rag_chat


LS_RST = """.. _ls:

ls — List directory contents
############################

The ``ls`` application lists the entries of a directory (or the paths given as
arguments). It supports both short and long format output, colourised display
by entry type, and wildcard pattern matching.

Synopsis
========

::

   ls [-l] [PATTERN] [DIR]

Options
=======

``-l``
   Long format. Each entry is shown with its type, size, time and name.
"""

LS_C = """/*
 * Copyright (C) 2014-2026 REDS Institute from HEIG-VD
 */
#include <stdio.h>
#include <dirent.h>

int main(int argc, char *argv[])
{
    DIR *dir = opendir(argc > 1 ? argv[1] : ".");
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL)
        printf("%s\\n", entry->d_name);

    closedir(dir);
    return 0;
}
"""


class CodeBlockLanguageTests(unittest.TestCase):
    def test_language_is_carried_out_of_the_fence(self):
        block, language = rag_chat.extract_code_block_with_language(
            "here:\n```rst\n" + LS_RST + "```\n")
        self.assertEqual(language, "rst")
        self.assertIn(".. _ls:", block)

    def test_largest_block_wins_and_absence_is_empty(self):
        text = "```\nshort\n```\n```c\n" + LS_C + "```"
        block, language = rag_chat.extract_code_block_with_language(text)
        self.assertEqual(language, "c")
        self.assertIn("#include", block)
        self.assertEqual(rag_chat.extract_code_block_with_language("no fence"),
                         ("", ""))

    def test_old_single_value_helper_still_returns_the_block(self):
        self.assertIn("#include", rag_chat.extract_code_block(
            "```c\n" + LS_C + "```"))


class BlockFitsTargetTests(unittest.TestCase):
    """The regression: 'add a chapter describing the ls.c application'.

    ``ls.c`` is the subject of the sentence, the block is reStructuredText, and
    the fallback was ready to write the chapter into the source file.  Nothing
    but write_file's refusal to overwrite an unread existing file stopped it —
    a target that did not exist would have been created.
    """
    def test_documentation_is_never_written_into_a_source_file(self):
        self.assertFalse(rag_chat.block_fits_target(LS_RST, "rst", "usr/src/ls.c"))
        # ... and not only because the fence said so: an untagged block of the
        # same prose must be refused too, since models often omit the tag.
        self.assertFalse(rag_chat.block_fits_target(LS_RST, "", "usr/src/ls.c"))

    def test_the_chapter_is_accepted_where_it_belongs(self):
        self.assertTrue(rag_chat.block_fits_target(
            LS_RST, "rst", "doc/source/ls.rst"))
        self.assertTrue(rag_chat.block_fits_target(LS_RST, "", "doc/source/ls.rst"))

    def test_real_source_still_applies_to_real_source(self):
        for language in ("c", ""):
            self.assertTrue(rag_chat.block_fits_target(
                LS_C, language, "usr/src/ls.c"),
                f"a C file must remain applicable (language={language!r})")

    def test_a_declared_language_that_contradicts_the_target_refuses(self):
        self.assertFalse(rag_chat.block_fits_target(
            "print('hi')\n", "python", "usr/src/ls.c"))
        self.assertFalse(rag_chat.block_fits_target(
            LS_C, "c", "doc/source/ls.rst"))

    def test_an_unknown_language_is_not_evidence_and_never_refuses_alone(self):
        self.assertTrue(rag_chat.block_fits_target(
            "some: value\n", "someunknownlang", "config.conf"))


if __name__ == "__main__":
    unittest.main()
