"""Recorded pose examples and closest-match scoring.

A recorded example is the player's upper body turned into numbers that don't
depend on where they stand: each point is measured from the middle of the
shoulders, in shoulder widths. Examples are saved per pose in
poses/samples/<pose name>.npy.
"""

import os

import mediapipe as mp
import numpy as np

PL = mp.solutions.pose.PoseLandmark

# Body points used for matching recorded poses.
POINTS = (
    PL.NOSE,
    PL.LEFT_SHOULDER,
    PL.RIGHT_SHOULDER,
    PL.LEFT_ELBOW,
    PL.RIGHT_ELBOW,
    PL.LEFT_WRIST,
    PL.RIGHT_WRIST,
)

# Points compared when matching. The shoulders are left out: after normalizing
# they sit in the same place in every pose, so they'd only water down differences.
COMPARE = [i for i, p in enumerate(POINTS) if p not in (PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER)]

# How close you need to be to a recorded example, as the average distance
# per body point in shoulder widths. NEAR or closer scores 1, FAR or more scores 0.
NEAR = 0.15
FAR = 0.35
NEAREST_COUNT = 3  # average over this many closest examples to ignore one-off noise


def ramp(value, bad, good):
    """0 at `bad`, 1 at `good`, linear in between (works in either direction)."""
    if bad == good:
        return 1.0 if value == good else 0.0
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t))


def normalize(landmarks, w, h, min_visibility):
    """Turn landmarks into a (len(POINTS), 2) array, or None if any point
    isn't clearly visible."""
    if any(landmarks[i].visibility < min_visibility for i in POINTS):
        return None
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in POINTS], dtype=np.float32)
    ls, rs = pts[1], pts[2]
    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width < 1e-6:
        return None
    return (pts - (ls + rs) / 2) / shoulder_width


def distances(vec, samples):
    """Average per-point distance from vec to each recorded example."""
    diff = samples[:, COMPARE] - vec[None, COMPARE]
    return np.sqrt((diff**2).sum(axis=2).mean(axis=1))


def score(vec, samples):
    """0..1: how close vec is to the recorded examples of one pose."""
    if samples is None or len(samples) == 0:
        return 0.0
    d = np.sort(distances(vec, samples))[:NEAREST_COUNT]
    return ramp(float(d.mean()), FAR, NEAR)


def samples_path(folder, name):
    return os.path.join(folder, f"{name}.npy")


def load_samples(folder):
    """Load every saved pose. Returns {pose name: array of examples}."""
    result = {}
    if not os.path.isdir(folder):
        return result
    for file in sorted(os.listdir(folder)):
        name, ext = os.path.splitext(file)
        if ext != ".npy":
            continue
        try:
            data = np.load(os.path.join(folder, file))
        except (OSError, ValueError) as e:
            print(f"Skipping recorded pose that could not be read: {file} ({e})")
            continue
        if data.ndim == 3 and data.shape[1:] == (len(POINTS), 2):
            result[name] = data.astype(np.float32)
        else:
            print(f"Skipping recorded pose with an unexpected format: {file}")
    return result


def save_samples(folder, name, new_samples):
    """Add new examples to a pose's saved examples. Returns all examples."""
    os.makedirs(folder, exist_ok=True)
    new = np.asarray(new_samples, dtype=np.float32)
    path = samples_path(folder, name)
    if os.path.exists(path):
        try:
            old = np.load(path).astype(np.float32)
            if old.shape[1:] == new.shape[1:]:
                new = np.concatenate([old, new])
        except (OSError, ValueError):
            pass  # unreadable old file: replace it
    np.save(path, new)
    return new
