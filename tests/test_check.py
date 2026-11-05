"""Tests for the mix-up check (check_poses.py).

Run: python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_poses as cp  # noqa: E402
import detection  # noqa: E402
import pose_samples as ps  # noqa: E402
from detection import PL  # noqa: E402
from test_detection import ARMS_DOWN, CHIN_ON_HANDS, HANDS_BEHIND_HEAD  # noqa: E402
from test_samples import recording  # noqa: E402

# Almost the same as CHIN_ON_HANDS: hands a little lower.
CHIN_ON_HANDS_2 = {**CHIN_ON_HANDS, PL.LEFT_WRIST: (310, 225), PL.RIGHT_WRIST: (330, 225)}


def by_name(results):
    return {r.name: r for r in results}


class CheckTest(unittest.TestCase):
    def test_distinct_poses_are_fine(self):
        samples = {"chin": recording(CHIN_ON_HANDS), "down": recording(ARMS_DOWN, seed=1)}
        results = by_name(cp.check(samples, detectors={}))
        for name in ("chin", "down"):
            self.assertGreaterEqual(results[name].recognized, 0.9)
            self.assertEqual(results[name].mixups, {})
        self.assertIn("All recorded poses look fine", cp.format_report(list(results.values()), {}))

    def test_similar_poses_are_reported(self):
        samples = {"chin": recording(CHIN_ON_HANDS), "chin2": recording(CHIN_ON_HANDS_2, seed=1)}
        results = by_name(cp.check(samples, detectors={}))
        self.assertGreater(results["chin"].mixups.get("chin2", 0), 0.5)
        report = cp.format_report(list(results.values()), {})
        self.assertIn("often looks like chin2", report)

    def test_recording_that_triggers_a_rule(self):
        # A new meme recorded with the same pose as the hands_behind_head rule
        samples = {"surprised": recording(HANDS_BEHIND_HEAD)}
        results = by_name(cp.check(samples, detectors=detection.DETECTORS))
        self.assertGreater(results["surprised"].mixups.get("hands_behind_head", 0), 0.9)

    def test_recording_matches_its_own_rule(self):
        # Recording hands_behind_head itself: no mix-up, recognized
        samples = {"hands_behind_head": recording(HANDS_BEHIND_HEAD)}
        r = by_name(cp.check(samples, detectors=detection.DETECTORS))["hands_behind_head"]
        self.assertEqual(r.mixups, {})
        self.assertGreaterEqual(r.recognized, 0.9)

    def test_few_examples_advice(self):
        samples = {"chin": recording(CHIN_ON_HANDS, n=10)}
        r = cp.check(samples, detectors={})[0]
        self.assertIsNone(r.recognized)  # too few to check, not "0%"
        self.assertTrue(any("record it again" in tip for tip in cp.advice(r)))
        self.assertIn("too few", cp.format_report([r], {}))

    def test_rule_lands_on_pose_with_real_scale(self):
        # Converting a recording back to landmarks keeps the rule's measurements
        vec = recording(HANDS_BEHIND_HEAD, n=1, jitter=0)[0]
        body = detection.Body(cp.to_landmarks(vec), 1, 1)
        self.assertEqual(detection.score_hands_behind_head(body), 1.0)


class ReportTest(unittest.TestCase):
    def test_no_recordings(self):
        self.assertIn("No recorded poses yet", cp.format_report([]))

    def test_main_with_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            ps.save_samples(folder, "chin", recording(CHIN_ON_HANDS))
            self.assertEqual(cp.main(["--samples", folder]), 0)

    def test_main_missing_folder(self):
        self.assertEqual(cp.main(["--samples", os.path.join(tempfile.gettempdir(), "no_such_meme_folder")]), 0)


if __name__ == "__main__":
    unittest.main()
