"""Pose recognition: body geometry, hand-written pose rules and the hold timer."""

import math

import pose_samples
from pose_samples import PL, ramp
from settings import ENTER_SCORE, EXIT_SCORE, HOLD_SECONDS, MIN_VISIBILITY, SMOOTHING


# -----------------------------
# Body geometry
# -----------------------------
class Body:
    """Landmarks in pixels, measured in shoulder widths so that
    distance from the camera doesn't change the result."""

    def __init__(self, landmarks, w, h):
        self.lm = landmarks
        self.w, self.h = w, h
        ls, rs = self.px(PL.LEFT_SHOULDER), self.px(PL.RIGHT_SHOULDER)
        self.shoulder_width = max(math.dist(ls, rs), 1e-6)

    def px(self, idx):
        p = self.lm[idx]
        return (p.x * self.w, p.y * self.h)

    def visible(self, *idxs):
        return all(self.lm[i].visibility >= MIN_VISIBILITY for i in idxs)

    def dist(self, a, b):
        """Distance between two points, in shoulder widths."""
        return math.dist(a, b) / self.shoulder_width

    def above(self, a, b):
        """How far point a is above point b, in shoulder widths (negative = below)."""
        return (b[1] - a[1]) / self.shoulder_width

    def head(self):
        """Center of the head from whichever of nose and ears are visible."""
        pts = [self.px(i) for i in (PL.NOSE, PL.LEFT_EAR, PL.RIGHT_EAR) if self.visible(i)]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def angle(a, b, c):
    """Angle at point b, in degrees."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*ab) * math.hypot(*cb)
    if n == 0:
        return 180.0
    cos = (ab[0] * cb[0] + ab[1] * cb[1]) / n
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# -----------------------------
# Pose detectors
# Each returns a score from 0 (not this pose) to 1 (clearly this pose).
# -----------------------------
def score_hands_behind_head(body):
    arms = (PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_ELBOW, PL.RIGHT_ELBOW, PL.LEFT_WRIST, PL.RIGHT_WRIST)
    if not body.visible(*arms):
        return 0.0
    head = body.head()
    if head is None:
        return 0.0

    scores = []
    for s, e, wr in (
        (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
    ):
        shoulder, elbow, wrist = body.px(s), body.px(e), body.px(wr)
        scores += [
            ramp(body.dist(wrist, head), 1.3, 0.7),  # hand close to the head
            ramp(body.above(wrist, shoulder), 0.0, 0.4),  # hand above the shoulder
            ramp(body.above(elbow, shoulder), -0.4, 0.0),  # elbow up at shoulder height
            ramp(angle(shoulder, elbow, wrist), 130, 95),  # elbow bent
        ]

    le, re = body.px(PL.LEFT_ELBOW), body.px(PL.RIGHT_ELBOW)
    scores.append(ramp(abs(le[0] - re[0]) / body.shoulder_width, 1.2, 1.8))  # elbows out wide

    return min(scores)


# Pose name (image file name without extension) -> hand-written scoring function.
# Poses can also be recognized from recorded examples (press r in the game).
DETECTORS = {
    "hands_behind_head": score_hands_behind_head,
}


def score_poses(landmarks, w, h, samples=None):
    """Score every known pose. A pose with both a hand-written rule and
    recorded examples gets whichever score is higher."""
    body = Body(landmarks, w, h)
    scores = {name: detector(body) for name, detector in DETECTORS.items()}
    if samples:
        vec = pose_samples.normalize(landmarks, w, h, MIN_VISIBILITY)
        if vec is not None:
            for name, examples in samples.items():
                scores[name] = max(scores.get(name, 0.0), pose_samples.score(vec, examples))
    return scores


class PoseTracker:
    """Turns per-frame scores into a steady result.

    - Scores are smoothed so one noisy frame doesn't flip the result.
    - A pose has to stay above ENTER_SCORE for HOLD_SECONDS before it matches.
    - Once matched, it stays matched until its score drops below EXIT_SCORE.
    """

    def __init__(self):
        self.smoothed = {}
        self.matched = None
        self.candidate = None
        self.hold_start = None

    def update(self, scores, now):
        for name in set(self.smoothed) | set(scores):
            new = scores.get(name, 0.0)
            old = self.smoothed.get(name, new)
            self.smoothed[name] = SMOOTHING * old + (1 - SMOOTHING) * new

        if self.matched:
            if self.smoothed.get(self.matched, 0.0) >= EXIT_SCORE:
                return self.matched
            self.matched = None

        best = max(self.smoothed, key=self.smoothed.get, default=None)
        if best is None or self.smoothed[best] < ENTER_SCORE:
            self.candidate, self.hold_start = None, None
            return None

        if best != self.candidate:
            self.candidate, self.hold_start = best, now

        if now - self.hold_start >= HOLD_SECONDS:
            self.matched = best
            self.candidate, self.hold_start = None, None
        return self.matched

    def hold_progress(self, now):
        """0..1 progress toward matching the current candidate pose."""
        if not self.candidate or self.hold_start is None:
            return 0.0
        return min(1.0, (now - self.hold_start) / HOLD_SECONDS)
