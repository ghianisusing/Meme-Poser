import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_DIR = os.path.join(SCRIPT_DIR, "poses")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

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


def ramp(value, bad, good):
    """0 at `bad`, 1 at `good`, linear in between (works in either direction)."""
    if bad == good:
        return 1.0 if value == good else 0.0
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t))


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


# Pose name (image file name without extension) -> scoring function.
# Images in poses/ without a detector still show in the gallery.
DETECTORS = {
    "hands_behind_head": score_hands_behind_head,
}


def score_poses(landmarks, w, h):
    body = Body(landmarks, w, h)
    return {name: detector(body) for name, detector in DETECTORS.items()}


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


def draw_tile(frame, image, name, x, y, tw, th, state, found, label_scale):
    """Draw one meme thumbnail with its name underneath.
    state: "matched", "holding" or None."""
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


def tile_state(name, matched, holding):
    if name == matched:
        return "matched"
    if name == holding:
        return "holding"
    return None


def draw_gallery_strip(frame, poses, matched, found, holding=None):
    """Row of every meme along the bottom of the camera view."""
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
    cv2.putText(
        frame,
        f"Poses you can try: {len(found)}/{len(names)} found   (g = full gallery)",
        (pad, top + 17),
        FONT,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    total_w = len(names) * tile_w + (len(names) - 1) * pad
    x = (w - total_w) // 2
    y = top + 22 + pad
    for name in names:
        state = tile_state(name, matched, holding)
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, state, name in found, 0.4)
        x += tile_w + pad


def draw_gallery_grid(frame, poses, matched, found, holding=None):
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
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, state, name in found, 0.6)


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

    tracker = PoseTracker()
    found = set()
    show_grid = False
    popup_visible = False
    popup_pose = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scores = score_poses(results.pose_landmarks.landmark, w, h) if results.pose_landmarks else {}
        matched = tracker.update(scores, now)
        if matched not in poses:
            matched = None
        holding = tracker.candidate if tracker.candidate in poses else None
        progress = tracker.hold_progress(now)
        if matched:
            found.add(matched)

        if show_grid:
            draw_gallery_grid(frame, poses, matched, found, holding)
            draw_status(frame, matched, holding, progress, show_text=False)
        else:
            draw_status(frame, matched, holding, progress)
            draw_gallery_strip(frame, poses, matched, found, holding)

        # Meme popup for the matched pose
        if matched:
            if not popup_visible:
                cv2.namedWindow("Your Meme", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Your Meme", 480, 270)
                cv2.moveWindow("Your Meme", 60, 80)
                popup_visible = True
            if matched != popup_pose:
                popup_image = fit(poses[matched], 480, 270)
                popup_pose = matched
            cv2.imshow("Your Meme", popup_image)
        elif popup_visible:
            cv2.destroyWindow("Your Meme")
            popup_visible = False
            popup_pose = None

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_grid = not show_grid

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
