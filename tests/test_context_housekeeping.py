"""PR0: active context is archived and compacted before it exceeds the line limit."""

import tempfile
import unittest
from pathlib import Path

from context.housekeeping import ACTIVE_LINE_LIMIT, ActiveContext


class ContextHousekeepingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_default_limit_is_one_thousand(self) -> None:
        store = ActiveContext(self.directory)
        self.assertEqual(ACTIVE_LINE_LIMIT, 1000)
        self.assertEqual(store.limit, 1000)

    def test_append_under_the_limit_does_not_archive(self) -> None:
        store = ActiveContext(self.directory, limit=10)
        for index in range(10):
            result = store.append(f"row {index}")
            self.assertFalse(result.archived)
            self.assertFalse(result.compacted)
        self.assertEqual(store.line_count(), 10)
        self.assertFalse(store.archive_dir.exists())

    def test_append_archives_and_compacts_before_the_file_exceeds_the_limit(self) -> None:
        store = ActiveContext(self.directory, limit=40)
        store.append("# decision keep-me-alpha\nsupporting detail")
        while store.line_count() < 40:
            store.append(f"filler {store.line_count()}")
        self.assertEqual(store.line_count(), 40)
        before = store.active_path.read_text(encoding="utf-8")

        result = store.append("# recent keep-me-beta")

        self.assertTrue(result.archived)
        self.assertTrue(result.compacted)
        self.assertLessEqual(store.line_count(), 40)
        self.assertGreater(store.line_count(), 0)
        active = store.active_path.read_text(encoding="utf-8")
        self.assertIn("keep-me-alpha", active)
        self.assertIn("keep-me-beta", active)
        self.assertLess(store.line_count(), len(before.splitlines()) + 1)
        self.assertEqual(len(result.archive_paths), 1)
        archived = result.archive_paths[0].read_text(encoding="utf-8")
        self.assertEqual(archived, before)
        self.assertNotIn("keep-me-beta", archived)
        self.assertIn("# compacted:", active)
        self.assertIn("Earlier lines compacted:", active)

    def test_repeated_appends_never_pass_the_default_limit(self) -> None:
        store = ActiveContext(self.directory)
        for index in range(1500):
            store.append(f"line {index} decision")
            self.assertLessEqual(store.line_count(), ACTIVE_LINE_LIMIT)
        archives = list(store.archive_dir.glob("active-*.txt"))
        self.assertTrue(archives)
        archived_text = "\n".join(
            path.read_text(encoding="utf-8") for path in archives
        )
        self.assertIn("line 0 decision", archived_text)
        self.assertIn("line 1499 decision", store.active_path.read_text(encoding="utf-8"))

    def test_housekeep_repairs_an_active_file_already_over_the_limit(self) -> None:
        store = ActiveContext(self.directory, limit=20)
        original = "\n".join(f"# item {index}" for index in range(50)) + "\n"
        store.active_path.parent.mkdir(parents=True, exist_ok=True)
        store.active_path.write_text(original, encoding="utf-8")

        result = store.housekeep()

        self.assertTrue(result.archived)
        self.assertTrue(result.compacted)
        self.assertEqual(result.lines_before, 50)
        self.assertLessEqual(result.lines_after, 20)
        self.assertLessEqual(store.line_count(), 20)
        self.assertEqual(result.archive_paths[0].read_text(encoding="utf-8"), original)
        active = store.active_path.read_text(encoding="utf-8")
        self.assertIn("item 0", active)
        self.assertLess(store.line_count(), 50)

    def test_housekeep_leaves_a_file_at_the_limit_unchanged(self) -> None:
        store = ActiveContext(self.directory, limit=5)
        for index in range(5):
            store.append(f"row {index}")
        before = store.active_path.read_text(encoding="utf-8")

        result = store.housekeep()

        self.assertFalse(result.archived)
        self.assertFalse(result.compacted)
        self.assertEqual(store.active_path.read_text(encoding="utf-8"), before)

    def test_full_size_entry_on_nonempty_file_stays_within_limit(self) -> None:
        store = ActiveContext(self.directory, limit=10)
        store.append("seed line")
        before = store.active_path.read_text(encoding="utf-8")
        block = "\n".join(f"wide {index}" for index in range(10))

        result = store.append(block)

        self.assertTrue(result.archived)
        self.assertLessEqual(store.line_count(), 10)
        self.assertEqual(result.archive_paths[0].read_text(encoding="utf-8"), before)
        active = store.active_path.read_text(encoding="utf-8")
        self.assertIn("wide 0", active)
        self.assertIn("wide 9", active)

    def test_oversized_entry_is_archived_and_active_file_stays_within_limit(self) -> None:
        store = ActiveContext(self.directory, limit=10)
        payload = "\n".join(f"big {index}" for index in range(25))

        result = store.append(payload)

        self.assertTrue(result.archived)
        self.assertTrue(result.compacted)
        self.assertLessEqual(store.line_count(), 10)
        archived = "\n".join(
            path.read_text(encoding="utf-8") for path in result.archive_paths
        )
        self.assertIn("big 0", archived)
        self.assertIn("big 24", archived)
        self.assertIn("entry archived:", store.active_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
