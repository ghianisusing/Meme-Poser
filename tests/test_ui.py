"""Tests for the meme reveal and snapshot photos (no webcam needed).

Run: python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui as pd  # noqa: E402

MEME = np.full((1080, 1920, 3), (40, 120, 200), np.uint8)


def camera(w=640, h=480):
    return np.full((h, w, 3), (70, 80, 90), np.uint8)


class RevealTest(unittest.TestCase):
    def test_invisible_at_start(self):
        frame = camera()
        pd.draw_reveal(frame, MEME, "hands_behind_head", 0.0)
        np.testing.assert_array_equal(frame, camera())

    def test_fully_shown_after_animation(self):
        frame = camera()
        pd.draw_reveal(frame, MEME, "hands_behind_head", pd.REVEAL_SECONDS)
        # The meme color appears in the top-right area
        self.assertTrue((frame[150, 500] == (40, 120, 200)).all())
        # The left side of the camera view is untouched
        self.assertTrue((frame[150, 50] == (70, 80, 90)).all())

    def test_tiny_and_tall_frames(self):
        for w, h in ((160, 120), (480, 900), (1920, 1080)):
            frame = camera(w, h)
            for t in (0.0, 0.1, 1.0):
                pd.draw_reveal(frame, MEME, "a_very_long_pose_name_that_needs_cutting", t)


class SnapshotTest(unittest.TestCase):
    def test_compose_size(self):
        snap = pd.compose_snapshot(camera(), MEME, "hands_behind_head")
        self.assertEqual(snap.shape[0], 480 + 56)
        self.assertGreater(snap.shape[1], 640 + 12)
        # Camera stays on the left, unchanged
        np.testing.assert_array_equal(snap[:480, :640], camera())

    def test_portrait_meme(self):
        snap = pd.compose_snapshot(camera(), np.zeros((900, 300, 3), np.uint8), "dab")
        self.assertEqual(snap.shape[0], 480 + 56)

    def test_save_writes_png(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pd.save_snapshot(camera(), MEME, "hands_behind_head", folder)
            self.assertTrue(path.endswith(".png"))
            self.assertIsNotNone(cv2.imread(path))


if __name__ == "__main__":
    unittest.main()
