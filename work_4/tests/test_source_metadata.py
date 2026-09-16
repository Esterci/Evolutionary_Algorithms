"""Git provenance must remain optional on compute nodes."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.source_metadata import git_revision


class SourceMetadataTests(unittest.TestCase):
    def test_missing_git_preserves_source_hashes(self):
        from runners.train_pinn_de_adam import source_metadata

        with patch("utils.source_metadata.subprocess.run",
                   side_effect=FileNotFoundError(2, "No such file or directory", "git")):
            metadata = source_metadata()
        self.assertIsNone(metadata["framework_commit"])
        self.assertIn("FileNotFoundError", metadata["framework_commit_error"])
        for key in ("framework_source_sha256", "experiment_source_sha256"):
            self.assertTrue(metadata[key])
            for digest in metadata[key].values():
                self.assertEqual(len(digest), 64)
                int(digest, 16)

    def test_non_repository_reports_failure_without_a_commit(self):
        result = subprocess.CompletedProcess([], 128, "", "fatal: not a git repository\n")
        with patch("utils.source_metadata.subprocess.run", return_value=result):
            commit, error = git_revision("/tmp")
        self.assertIsNone(commit)
        self.assertIn("not a git repository", error)

    def test_available_revision_is_preserved(self):
        result = subprocess.CompletedProcess([], 0, "abc123\n", "")
        with patch("utils.source_metadata.subprocess.run", return_value=result):
            self.assertEqual(git_revision("/tmp"), ("abc123", None))


if __name__ == "__main__":
    unittest.main()
