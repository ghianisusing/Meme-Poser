import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

import pose_samples
from pose_samples import ramp

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark
mp_drawing = mp.solutions.drawing_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_DIR = os.path.join(SCRIPT_DIR, "poses")
SAMPLES_DIR = os.path.join(POSES_DIR, "samples")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Record mode settings
COUNTDOWN_SECONDS = 3  # time to get into the pose
RECORD_SECONDS = 3  # time spent saving examples
MIN_SAMPLES = 20  # fewer clear frames than this means the recording failed

# Reveal and snapshot settings
REVEAL_SECONDS = 0.25  # length of the meme pop-in animation
SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "snapshots")

# Matching settings
MIN_VISIBILITY = 0.5  # ignore body points MediaPipe isn't sure it can see
ENTER_SCORE = 0.7  # score needed to start holding a pose
EXIT_SCORE = 0.4  # a matched pose stays matched until its score drops below this
HOLD_SECONDS = 0.5  # how long to hold a pose before the meme shows
SMOOTHING = 0.5  # 0 = no smoothing, closer to 1 = smoother but slower

# Colors (BGR)
GREEN = (80, 200, 80)
AMBER = (40, 170, 240)
RED = (60, 60, 230)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


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


# -----------------------------
# Pose library
# -----------------------------
def display_name(name):
    return name.replace("_", " ").replace("-", " ").capitalize()


def load_poses(folder):
    """Load every image in the poses folder. Returns {name: image}."""
    poses = {}
    if not os.path.isdir(folder):
        return poses

    for file in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(os.path.join(folder, file))
        if image is None:
            print(f"Skipping pose image that could not be read: {file}")
            continue
        poses[stem] = image

    return poses


def fit(image, box_w, box_h, background=BLACK):
    """Resize image to fit inside box_w x box_h, keeping its shape."""
    ih, iw = image.shape[:2]
    scale = min(box_w / iw, box_h / ih)
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((box_h, box_w, 3), background, dtype=np.uint8)
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


# -----------------------------
# Drawing helpers
# -----------------------------
def darken(frame, x1, y1, x2, y2, amount=0.6):
    region = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (region * (1 - amount)).astype(np.uint8)


def fit_text(text, max_width, scale, thickness=1):
    """Shorten text with '..' until it fits max_width pixels."""
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    while len(text) > 1:
        text = text[:-1]
        if cv2.getTextSize(text + "..", FONT, scale, thickness)[0][0] <= max_width:
            return text + ".."
    return text


def draw_check(frame, cx, cy, r):
    cv2.circle(frame, (cx, cy), r, GREEN, -1, cv2.LINE_AA)
    pts = np.array(
        [
            (cx - r // 2, cy),
            (cx - r // 8, cy + r * 3 // 8),
            (cx + r // 2, cy - r * 3 // 8),
        ],
        np.int32,
    )
    cv2.polylines(frame, [pts], False, WHITE, max(2, r // 4), cv2.LINE_AA)


def draw_tile(frame, image, name, x, y, tw, th, state, found, label_scale, ready=True, number=None):
    """Draw one meme thumbnail with its name underneath.
    state: "matched", "holding" or None.
    ready: False when the game can't recognize this pose yet.
    number: shown in the corner while choosing a pose to record."""
    text_h = cv2.getTextSize("Ag", FONT, label_scale, 1)[0][1]
    show_label = tw >= 70 and th * 0.22 >= text_h + 4
    label_h = int(th * 0.22) if show_label else 0
    img_h = th - label_h

    frame[y : y + img_h, x : x + tw] = fit(image, tw, img_h)

    if state == "matched":
        color, thickness = GREEN, 3
    elif state == "holding":
        color, thickness = AMBER, 3
    elif found:
        color, thickness = GREEN, 1
    else:
        color, thickness = GREY, 1
    cv2.rectangle(frame, (x, y), (x + tw - 1, y + th - 1), color, thickness)

    if show_label:
        text = fit_text(display_name(name), tw - 8, label_scale)
        text_w = cv2.getTextSize(text, FONT, label_scale, 1)[0][0]
        tx = x + (tw - text_w) // 2
        ty = y + img_h + (label_h + text_h) // 2
        cv2.putText(frame, text, (tx, ty), FONT, label_scale, WHITE if state else GREY, 1, cv2.LINE_AA)

    if found:
        r = max(7, tw // 10)
        draw_check(frame, x + tw - r - 4, y + r + 4, r)

    if not ready and tw >= 70:
        tag = "press r to set up"
        scale = 0.35 if tw < 120 else 0.45
        tag = fit_text(tag, tw - 8, scale)
        tag_w, tag_h = cv2.getTextSize(tag, FONT, scale, 1)[0]
        darken(frame, x + 1, y + 1, x + tw - 1, y + tag_h + 9, 0.7)
        cv2.putText(frame, tag, (x + (tw - tag_w) // 2, y + tag_h + 4), FONT, scale, AMBER, 1, cv2.LINE_AA)

    if number is not None:
        r = max(9, min(16, tw // 7))
        cx, cy = x + r + 4, y + r + 4
        cv2.circle(frame, (cx, cy), r, AMBER, -1, cv2.LINE_AA)
        text = str(number)
        scale = r / 18
        tw_, th_ = cv2.getTextSize(text, FONT, scale, 2)[0]
        cv2.putText(frame, text, (cx - tw_ // 2, cy + th_ // 2), FONT, scale, BLACK, 2, cv2.LINE_AA)


def tile_state(name, matched, holding):
    if name == matched:
        return "matched"
    if name == holding:
        return "holding"
    return None


def draw_gallery_strip(frame, poses, matched, found, holding=None, ready=None, numbered=False):
    """Row of every meme along the bottom of the camera view.
    ready: set of pose names the game can recognize (None = all).
    numbered: show 1, 2, 3... on the tiles for choosing a pose to record."""
    h, w = frame.shape[:2]
    names = list(poses)
    if not names:
        return

    pad = 10
    tile_w = min(140, (w - pad * (len(names) + 1)) // len(names))
    tile_h = int(tile_w * 0.85)
    strip_h = tile_h + pad * 2 + 22
    top = h - strip_h

    darken(frame, 0, top, w, h)
    if numbered:
        header = "Record which pose? Press its number (Esc to cancel)"
    else:
        keys = "g gallery  r record  s skeleton  space photo"
        header = f"Poses you can try: {len(found)}/{len(names)} found   {keys}"
        if cv2.getTextSize(header, FONT, 0.5, 1)[0][0] > w - pad * 2:
            header = f"{len(found)}/{len(names)} found   {keys}"
    cv2.putText(frame, fit_text(header, w - pad * 2, 0.5), (pad, top + 17), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    total_w = len(names) * tile_w + (len(names) - 1) * pad
    x = (w - total_w) // 2
    y = top + 22 + pad
    for i, name in enumerate(names):
        state = tile_state(name, matched, holding)
        is_ready = numbered or ready is None or name in ready  # numbers replace the tag
        number = i + 1 if numbered and i < 9 else None
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, state, name in found, 0.4, is_ready, number)
        x += tile_w + pad


def draw_gallery_grid(frame, poses, matched, found, holding=None, ready=None):
    """Full-screen grid of every meme, opened with 'g'."""
    h, w = frame.shape[:2]
    darken(frame, 0, 0, w, h, 0.8)

    names = list(poses)
    title = f"All poses  -  {len(found)}/{len(names)} found   (g to close)"
    cv2.putText(frame, title, (20, 40), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
    if not names:
        cv2.putText(frame, "No images in the poses folder yet.", (20, 90), FONT, 0.7, GREY, 1, cv2.LINE_AA)
        return

    cols = math.ceil(math.sqrt(len(names) * w / h))
    rows = math.ceil(len(names) / cols)
    pad, top = 20, 60
    tile_w = (w - pad * (cols + 1)) // cols
    tile_h = min(int(tile_w * 0.85), (h - top - pad * (rows + 1)) // rows)
    tile_w = min(tile_w, int(tile_h / 0.85))

    grid_w = cols * tile_w + (cols - 1) * pad
    x0 = (w - grid_w) // 2
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        x = x0 + c * (tile_w + pad)
        y = top + pad + r * (tile_h + pad)
        state = tile_state(name, matched, holding)
        is_ready = ready is None or name in ready
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, state, name in found, 0.6, is_ready)


def draw_status(frame, matched, holding=None, progress=0.0, show_text=True):
    h, w = frame.shape[:2]
    if matched:
        color, text = GREEN, f"MATCH: {display_name(matched).upper()}"
    elif holding:
        color, text = AMBER, f"HOLD IT... {display_name(holding).upper()}"
    else:
        color, text = RED, "STRIKE A POSE"
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    if not show_text:
        return
    cv2.putText(frame, text, (20, 45), FONT, 1, color, 2, cv2.LINE_AA)
    if holding and not matched:
        bar_w = min(300, w - 40)
        cv2.rectangle(frame, (20, 58), (20 + bar_w, 68), GREY, 1)
        cv2.rectangle(frame, (20, 58), (20 + int(bar_w * progress), 68), AMBER, -1)


def draw_center_text(frame, text, y, scale, color, thickness=2):
    w = frame.shape[1]
    text = fit_text(text, w - 40, scale, thickness)
    tw = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    cv2.putText(frame, text, ((w - tw) // 2, y), FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, ((w - tw) // 2, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_record_overlay(frame, image, name, phase, elapsed, count):
    """Countdown and recording screens.
    phase: "countdown" or "recording"; elapsed: seconds since the phase began."""
    h, w = frame.shape[:2]

    # The meme to copy, in the top-right corner
    ref_w = min(260, w // 3)
    ref_h = int(ref_w * 0.6)
    x, y = w - ref_w - 16, 16
    frame[y : y + ref_h, x : x + ref_w] = fit(image, ref_w, ref_h)
    cv2.rectangle(frame, (x, y), (x + ref_w - 1, y + ref_h - 1), WHITE, 1)

    if phase == "countdown":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), AMBER, 8)
        cv2.putText(frame, fit_text(f"GET INTO: {display_name(name).upper()}", x - 30, 0.8, 2), (20, 45), FONT, 0.8, AMBER, 2, cv2.LINE_AA)
        left = max(1, math.ceil(COUNTDOWN_SECONDS - elapsed))
        draw_center_text(frame, str(left), h // 2 + 40, 4, WHITE, 6)
        draw_center_text(frame, "Make sure your face and both arms are in view", h // 2 + 90, 0.6, WHITE, 1)
    else:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 8)
        cv2.circle(frame, (32, 38), 10, RED, -1, cv2.LINE_AA)
        title = "RECORDING - hold the pose" if x - 52 > 380 else "RECORDING"
        cv2.putText(frame, title, (52, 47), FONT, 0.8, RED, 2, cv2.LINE_AA)
        bar_w = min(300, w - 40)
        progress = min(1.0, elapsed / RECORD_SECONDS)
        cv2.rectangle(frame, (20, 60), (20 + bar_w, 70), GREY, 1)
        cv2.rectangle(frame, (20, 60), (20 + int(bar_w * progress), 70), RED, -1)
        cv2.putText(frame, f"{count} examples", (20, 92), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Esc to cancel", (20, h - 20), FONT, 0.5, WHITE, 1, cv2.LINE_AA)


def scale_to_fit(image, max_w, max_h):
    """Resize image to fit inside max_w x max_h, keeping its shape (no padding)."""
    ih, iw = image.shape[:2]
    scale = min(max_w / iw, max_h / ih)
    size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def draw_reveal(frame, image, name, elapsed):
    """Pop the matched meme into the top-right of the camera view.
    elapsed: seconds since the pose matched (drives the pop-in animation)."""
    h, w = frame.shape[:2]
    full = scale_to_fit(image, int(w * 0.42), int(h * 0.42))
    t = min(1.0, max(0.0, elapsed / REVEAL_SECONDS))
    ease = 1 - (1 - t) ** 3  # fast start, gentle stop
    scale = 0.6 + 0.4 * ease

    caption_h = 28
    fh, fw = full.shape[:2]
    cx = w - 16 - fw // 2  # center of the final position
    cy = 70 + fh // 2

    img = cv2.resize(full, (max(1, int(fw * scale)), max(1, int(fh * scale))), interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]
    x, y = cx - iw // 2, cy - ih // 2
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + iw), min(h, y + ih)
    if x2 <= x1 or y2 <= y1:
        return
    crop = img[y1 - y : y2 - y, x1 - x : x2 - x]
    frame[y1:y2, x1:x2] = cv2.addWeighted(crop, ease, frame[y1:y2, x1:x2], 1 - ease, 0)

    if t >= 1.0:
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 1, y2 + 1), GREEN, 3)
        cap_y2 = min(h, y2 + caption_h)
        darken(frame, x1 - 2, y2 + 2, x2 + 2, cap_y2, 0.7)
        text = fit_text(display_name(name), x2 - x1 - 10, 0.6, 2)
        tw = cv2.getTextSize(text, FONT, 0.6, 2)[0][0]
        cv2.putText(frame, text, (x1 + (x2 - x1 - tw) // 2, y2 + 21), FONT, 0.6, WHITE, 2, cv2.LINE_AA)


def compose_snapshot(camera, image, name):
    """You on the left, the meme on the right, the pose name underneath."""
    h = camera.shape[0]
    meme = scale_to_fit(image, int(camera.shape[1] * 1.5), h)
    meme_box = np.zeros((h, max(meme.shape[1], h * 3 // 4), 3), np.uint8)
    mx = (meme_box.shape[1] - meme.shape[1]) // 2
    my = (h - meme.shape[0]) // 2
    meme_box[my : my + meme.shape[0], mx : mx + meme.shape[1]] = meme

    gap = np.zeros((h, 12, 3), np.uint8)
    top = np.hstack([camera, gap, meme_box])
    band = np.zeros((56, top.shape[1], 3), np.uint8)
    text = f"Me as: {display_name(name)}"
    tw = cv2.getTextSize(text, FONT, 0.9, 2)[0][0]
    cv2.putText(band, text, ((top.shape[1] - tw) // 2, 38), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
    return np.vstack([top, band])


def save_snapshot(camera, image, name, folder=SNAPSHOTS_DIR):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(path, compose_snapshot(camera, image, name))
    return path


def draw_message(frame, text, color):
    h, w = frame.shape[:2]
    darken(frame, 0, 80, w, 130, 0.7)
    draw_center_text(frame, text, 113, 0.7, color, 2)


# -----------------------------
# Main loop
# -----------------------------
def main():
    poses = load_poses(POSES_DIR)
    if not poses:
        print(f"No pose images found in {POSES_DIR}")
        sys.exit(1)

    for name in DETECTORS:
        if name not in poses:
            print(f"Warning: detector '{name}' has no image in poses/")

    samples = {name: s for name, s in pose_samples.load_samples(SAMPLES_DIR).items() if name in poses}

    def ready_poses():
        return {name for name in poses if name in DETECTORS or name in samples}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam.")
        sys.exit(1)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 800)
    cv2.moveWindow("Camera", 0, 0)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    names = list(poses)
    tracker = PoseTracker()
    found = set()
    show_grid = False
    show_skeleton = False
    last_matched, match_start = None, 0.0

    # mode: "play", "choose" (picking a pose to record), "countdown" or "recording"
    mode = "play"
    record_pose = None
    phase_start = 0.0
    recorded = []
    message, message_color, message_until = None, WHITE, 0.0

    def start_countdown(name, now):
        nonlocal mode, record_pose, phase_start, show_grid
        mode, record_pose, phase_start, show_grid = "countdown", name, now, False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        camera = frame.copy()  # clean copy for snapshots
        h, w = frame.shape[:2]
        now = time.monotonic()

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        matched, holding, progress = None, None, 0.0
        if mode == "play":
            scores = score_poses(landmarks, w, h, samples) if landmarks else {}
            matched = tracker.update(scores, now)
            if matched not in poses:
                matched = None
            holding = tracker.candidate if tracker.candidate in poses else None
            progress = tracker.hold_progress(now)
            if matched:
                found.add(matched)
        if matched != last_matched:
            last_matched, match_start = matched, now

        if mode == "countdown" and now - phase_start >= COUNTDOWN_SECONDS:
            mode, phase_start, recorded = "recording", now, []

        if mode == "recording":
            if landmarks:
                vec = pose_samples.normalize(landmarks, w, h, MIN_VISIBILITY)
                if vec is not None:
                    recorded.append(vec)
            if now - phase_start >= RECORD_SECONDS:
                if len(recorded) >= MIN_SAMPLES:
                    samples[record_pose] = pose_samples.save_samples(SAMPLES_DIR, record_pose, recorded)
                    message = f"Saved {len(recorded)} examples for {display_name(record_pose)}. Try it!"
                    message_color = GREEN
                else:
                    message = "Couldn't see your face and both arms clearly. Step back and try again."
                    message_color = RED
                message_until = now + 3
                mode = "play"
                tracker = PoseTracker()

        if mode in ("countdown", "recording"):
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_record_overlay(frame, poses[record_pose], record_pose, mode, now - phase_start, len(recorded) if mode == "recording" else 0)
        elif show_grid:
            draw_gallery_grid(frame, poses, matched, found, holding, ready_poses())
            draw_status(frame, matched, holding, progress, show_text=False)
        else:
            if mode == "choose":
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), AMBER, 8)
                cv2.putText(frame, "RECORD A POSE", (20, 45), FONT, 1, AMBER, 2, cv2.LINE_AA)
            else:
                if show_skeleton and results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                draw_status(frame, matched, holding, progress)
            draw_gallery_strip(frame, poses, matched, found, holding, ready_poses(), numbered=mode == "choose")
            if matched:
                draw_reveal(frame, poses[matched], matched, now - match_start)

        if message and now < message_until:
            draw_message(frame, message, message_color)

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if mode == "play":
            if key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            elif key == ord(" "):
                if matched:
                    path = save_snapshot(camera, poses[matched], matched)
                    message = f"Photo saved: snapshots/{os.path.basename(path)}"
                    message_color = GREEN
                else:
                    message, message_color = "Strike a pose first, then press space", AMBER
                message_until = now + 3
            elif key == ord("r"):
                if len(names) == 1:
                    start_countdown(names[0], now)
                else:
                    mode, show_grid = "choose", False
        elif mode == "choose":
            if key in (27, ord("r")):
                mode = "play"
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(names):
                start_countdown(names[key - ord("1")], now)
        elif key == 27:  # cancel countdown or recording
            mode = "play"
            tracker = PoseTracker()

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
