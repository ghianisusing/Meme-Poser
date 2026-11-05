"""Tests for pose scoring and the hold timer, using fake bodies (no webcam).

Run: python -m unittest discover tests
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detection as pd  # noqa: E402
from detection import PL  # noqa: E402

W, H = 640, 480

# A standing body in pixels, shoulders 100 px apart.
BASE = {
    PL.NOSE: (320, 180),
    PL.LEFT_EAR: (295, 175),
    PL.RIGHT_EAR: (345, 175),
    PL.LEFT_SHOULDER: (270, 250),
    PL.RIGHT_SHOULDER: (370, 250),
    PL.LEFT_HIP: (285, 400),
    PL.RIGHT_HIP: (355, 400),
}

HANDS_BEHIND_HEAD = {
    PL.LEFT_ELBOW: (200, 170),
    PL.RIGHT_ELBOW: (440, 170),
    PL.LEFT_WRIST: (290, 150),
    PL.RIGHT_WRIST: (350, 150),
}

HANDS_ON_EARS = {
    PL.LEFT_ELBOW: (250, 320),
    PL.RIGHT_ELBOW: (390, 320),
    PL.LEFT_WRIST: (295, 180),
    PL.RIGHT_WRIST: (345, 180),
}

CHIN_ON_HANDS = {
    PL.LEFT_ELBOW: (280, 330),
    PL.RIGHT_ELBOW: (360, 330),
    PL.LEFT_WRIST: (310, 215),
    PL.RIGHT_WRIST: (330, 215),
}

ARMS_DOWN = {
    PL.LEFT_ELBOW: (260, 330),
    PL.RIGHT_ELBOW: (380, 330),
    PL.LEFT_WRIST: (255, 400),
    PL.RIGHT_WRIST: (385, 400),
}


def make_landmarks(arms, scale=1.0, hidden=(), center=(320, 240)):
    points = {**BASE, **arms}
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for idx, (x, y) in points.items():
        x = center[0] + (x - center[0]) * scale
        y = center[1] + (y - center[1]) * scale
        lm[idx] = SimpleNamespace(x=x / W, y=y / H, visibility=0.0 if idx in hidden else 0.99)
    return lm


def score(arms, **kw):
    return pd.score_poses(make_landmarks(arms, **kw), W, H)["hands_behind_head"]


class HandsBehindHeadTest(unittest.TestCase):
    def test_real_pose_matches(self):
        self.assertGreaterEqual(score(HANDS_BEHIND_HEAD), pd.ENTER_SCORE)

    def test_matches_far_from_camera(self):
        self.assertGreaterEqual(score(HANDS_BEHIND_HEAD, scale=0.4), pd.ENTER_SCORE)

    def test_matches_close_to_camera(self):
        self.assertGreaterEqual(score(HANDS_BEHIND_HEAD, scale=1.8), pd.ENTER_SCORE)

    def test_hands_on_ears_does_not_match(self):
        self.assertLess(score(HANDS_ON_EARS), pd.EXIT_SCORE)

    def test_chin_on_hands_does_not_match(self):
        self.assertLess(score(CHIN_ON_HANDS), pd.EXIT_SCORE)

    def test_arms_down_does_not_match(self):
        self.assertLess(score(ARMS_DOWN), pd.EXIT_SCORE)

    def test_hidden_ears_still_match_using_nose(self):
        hidden = (PL.LEFT_EAR, PL.RIGHT_EAR)
        self.assertGreaterEqual(score(HANDS_BEHIND_HEAD, hidden=hidden), pd.ENTER_SCORE)

    def test_hidden_wrist_does_not_match(self):
        self.assertEqual(score(HANDS_BEHIND_HEAD, hidden=(PL.LEFT_WRIST,)), 0.0)


class PoseTrackerTest(unittest.TestCase):
    def run_frames(self, tracker, score_value, start, seconds, fps=30):
        result = None
        for i in range(int(seconds * fps)):
            result = tracker.update({"p": score_value}, start + i / fps)
        return result

    def test_needs_hold_before_match(self):
        t = pd.PoseTracker()
        self.assertIsNone(self.run_frames(t, 1.0, 0.0, 0.3))
        self.assertEqual(t.candidate, "p")
        self.assertEqual(self.run_frames(t, 1.0, 0.3, 0.5), "p")

    def test_short_flash_does_not_match(self):
        t = pd.PoseTracker()
        self.run_frames(t, 1.0, 0.0, 0.2)
        self.assertIsNone(self.run_frames(t, 0.0, 0.2, 0.5))
        self.assertIsNone(t.candidate)

    def test_stays_matched_through_small_dip(self):
        t = pd.PoseTracker()
        self.run_frames(t, 1.0, 0.0, 1.0)
        self.assertEqual(self.run_frames(t, 0.5, 1.0, 1.0), "p")

    def test_unmatches_when_pose_is_broken(self):
        t = pd.PoseTracker()
        self.run_frames(t, 1.0, 0.0, 1.0)
        self.assertIsNone(self.run_frames(t, 0.0, 1.0, 0.5))

    def test_hold_progress(self):
        t = pd.PoseTracker()
        self.run_frames(t, 1.0, 0.0, 0.3)
        self.assertGreater(t.hold_progress(0.3), 0.4)
        self.assertLess(t.hold_progress(0.3), 1.0)


if __name__ == "__main__":
    unittest.main()
