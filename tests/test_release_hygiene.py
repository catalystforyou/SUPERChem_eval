"""Checks which use only the public demo, never local API configuration."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_demo_participants_are_pseudonyms(self):
        with (ROOT/'demo/20251015_baseline_demo.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 13)
        for row in rows:
            self.assertRegex(row['user_id'], r'^participant_\d{3}$')

    def test_cli_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'metrics'
            cmd = [sys.executable, '-m', 'superchem', 'metrics', '--manifest', 'demo/manifest.json', '--output', str(output)]
            first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            with (output/'per_item.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 10)
            second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('exists', second.stderr)


if __name__ == '__main__': unittest.main()
