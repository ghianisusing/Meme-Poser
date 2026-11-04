"""Tests for recorded pose examples (no webcam needed).

Run: python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pose_detection as pd  # noqa: E402
import pose_samples as ps  # noqa: E402
from pose_detection import PL  # noqa: E402
from test_detection import (  # noqa: E402
    ARMS_DOWN,
    CHIN_ON_HANDS,
    H,
    HANDS_BEHIND_HEAD,
    HANDS_ON_EARS,
    W,
    make_landmarks,
)


def vec(arms, **kw):
    return ps.normalize(make_landmarks(arms, **kw), W, H, pd.MIN_VISIBILITY)


def recording(arms, n=40, jitter=0.03, seed=0):
    """Fake a recording: the same pose with small random wobble."""
    rng = np.random.default_rng(seed)
    base = vec(arms)
    return base[None] + rng.normal(0, jitter, (n, *base.shape)).astype(np.float32)


class NormalizeTest(unittest.TestCase):
    def test_same_pose_at_any_distance_and_position(self):
        a = vec(HANDS_BEHIND_HEAD)
        b = vec(HANDS_BEHIND_HEAD, scale=0.5, center=(100, 100))
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_hidden_point_gives_none(self):
        self.assertIsNone(vec(HANDS_BEHIND_HEAD, hidden=(PL.RIGHT_WRIST,)))

    def test_shape(self):
        self.assertEqual(vec(ARMS_DOWN).shape, (len(ps.POINTS), 2))


class ScoreTest(unittest.TestCase):
    def test_recorded_pose_matches_itself(self):
        samples = recording(CHIN_ON_HANDS)
        self.assertGreaterEqual(ps.score(vec(CHIN_ON_HANDS), samples), pd.ENTER_SCORE)

    def test_recorded_pose_matches_far_away(self):
        samples = recording(CHIN_ON_HANDS)
        self.assertGreaterEqual(ps.score(vec(CHIN_ON_HANDS, scale=0.5), samples), pd.ENTER_SCORE)

    def test_other_poses_do_not_match(self):
        samples = recording(CHIN_ON_HANDS)
        for other in (HANDS_BEHIND_HEAD, HANDS_ON_EARS, ARMS_DOWN):
            self.assertLess(ps.score(vec(other), samples), pd.EXIT_SCORE)

    def test_no_samples_scores_zero(self):
        self.assertEqual(ps.score(vec(ARMS_DOWN), np.zeros((0, len(ps.POINTS), 2))), 0.0)


class ScorePosesTest(unittest.TestCase):
    def test_recorded_pose_without_rule(self):
        samples = {"chin_on_hands": recording(CHIN_ON_HANDS)}
        scores = pd.score_poses(make_landmarks(CHIN_ON_HANDS), W, H, samples)
        self.assertGreaterEqual(scores["chin_on_hands"], pd.ENTER_SCORE)
        self.assertLess(scores["hands_behind_head"], pd.EXIT_SCORE)

    def test_best_pose_wins(self):
        samples = {"chin_on_hands": recording(CHIN_ON_HANDS), "arms_down": recording(ARMS_DOWN, seed=1)}
        scores = pd.score_poses(make_landmarks(ARMS_DOWN), W, H, samples)
        self.assertEqual(max(scores, key=scores.get), "arms_down")


class SaveLoadTest(unittest.TestCase):
    def test_save_appends_and_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            ps.save_samples(folder, "dab", recording(ARMS_DOWN, n=20))
            total = ps.save_samples(folder, "dab", recording(ARMS_DOWN, n=25, seed=2))
            self.assertEqual(len(total), 45)
            loaded = ps.load_samples(folder)
            self.assertEqual(list(loaded), ["dab"])
            self.assertEqual(loaded["dab"].shape, (45, len(ps.POINTS), 2))

    def test_bad_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            np.save(os.path.join(folder, "broken.npy"), np.zeros((3, 4)))
            self.assertEqual(ps.load_samples(folder), {})

    def test_missing_folder(self):
        self.assertEqual(ps.load_samples(os.path.join(tempfile.gettempdir(), "no_such_meme_folder")), {})


if __name__ == "__main__":
    unittest.main()
