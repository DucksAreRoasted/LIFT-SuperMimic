"""Desktop regression: the real replay process must exit cleanly after closing.

Run explicitly on a desktop; this opens a short MuJoCo window and needs the
existing K1 reference motion. It is not part of the headless test suite.
"""
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]


class ViewerLifecycle(unittest.TestCase):
    def test_replay_exits_after_window_closes(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/retarget/replay_motion.py"),
             "--motion", str(PROJECT_ROOT / "data/retarget/lafan1_official_v1/k1/dance1_subject2.npz"),
             "--start-time", "20", "--seconds", "0.2"],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
