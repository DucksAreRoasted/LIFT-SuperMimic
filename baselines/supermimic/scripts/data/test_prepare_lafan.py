"""Offline checks for download validation and non-destructive publishing."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_lafan_modelscope import install_snapshot, selected_names, validate_csv


def write_clip(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, 36))
    data[:, 2] = 0.8
    data[:, 6] = 1.0  # xyzw
    np.savetxt(path, data, delimiter=",")


class DownloadValidation(unittest.TestCase):
    def test_fixed_manifest_and_selection(self):
        self.assertEqual(len(set(selected_names("*.csv"))), 40)
        self.assertEqual(selected_names("walk1_subject1.csv"), ["walk1_subject1.csv"])
        with self.assertRaises(ValueError):
            selected_names("not_a_lafan_clip.csv")

    def test_bad_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.csv"
            write_clip(path)
            self.assertEqual(validate_csv(path)["frames"], 4)
            arr = np.loadtxt(path, delimiter=",")
            arr[0, 3:7] = 0
            np.savetxt(path, arr, delimiter=",")
            with self.assertRaisesRegex(ValueError, "quaternion"):
                validate_csv(path)
            write_clip(path)
            arr = np.loadtxt(path, delimiter=",")
            arr[0, 7] = np.nan
            np.savetxt(path, arr, delimiter=",")
            with self.assertRaisesRegex(ValueError, "NaN"):
                validate_csv(path)

    def test_repeat_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, target = Path(tmp) / "snapshot", Path(tmp) / "target"
            write_clip(snapshot / "g1/a.csv")
            rows = install_snapshot(snapshot, target, ["a.csv"])
            before = (target / "a.csv").stat().st_mtime_ns
            self.assertEqual(rows, install_snapshot(snapshot, target, ["a.csv"]))
            self.assertEqual(before, (target / "a.csv").stat().st_mtime_ns)
            self.assertEqual([p.name for p in target.iterdir()], ["a.csv"])

    def test_conflict_checks_all_files_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, target = Path(tmp) / "snapshot", Path(tmp) / "target"
            write_clip(snapshot / "g1/a.csv")
            write_clip(snapshot / "g1/b.csv")
            target.mkdir()
            (target / "b.csv").write_text("user data")
            with self.assertRaises(FileExistsError):
                install_snapshot(snapshot, target, ["a.csv", "b.csv"])
            self.assertFalse((target / "a.csv").exists())
            self.assertEqual((target / "b.csv").read_text(), "user data")

    def test_incomplete_snapshot_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, target = Path(tmp) / "snapshot", Path(tmp) / "target"
            write_clip(snapshot / "g1/a.csv")
            with self.assertRaises(FileNotFoundError):
                install_snapshot(snapshot, target, ["a.csv", "b.csv"])
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
