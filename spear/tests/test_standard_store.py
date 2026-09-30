import os
import tempfile
import unittest
from pathlib import Path

from standard_store import StandardStore, StandardStoreError


class StandardStoreSecurityTests(unittest.TestCase):
    def test_private_permissions_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "standards"
            store = StandardStore(root)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            for standard, revision in (("../escape", "1"), ("X", "../1"),
                                       ("/abs", "1"), ("X", "a/b")):
                with self.assertRaises(StandardStoreError):
                    store.revision_dir(standard, revision)

    def test_a_store_owned_by_someone_else_opens_as_shipped(self):
        """A container image ships its store owned by root and runs as the
        host user; tightening the mode there is EPERM and made the baked
        standards unreachable."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "standards"
            root.mkdir(mode=0o777)
            os.chmod(root, 0o777)

            with mock.patch("standard_store.os.getuid",
                            return_value=os.getuid() + 1):
                StandardStore(root)

            self.assertEqual(root.stat().st_mode & 0o777, 0o777)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "standards"; root.mkdir()
            outside = Path(directory) / "outside"; outside.mkdir()
            (root / "EVIL").symlink_to(outside, target_is_directory=True)
            store = StandardStore(root)
            with self.assertRaisesRegex(StandardStoreError, "symlink"):
                store.revision_dir("EVIL", "1", create=True)
