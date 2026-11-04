import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_DIR = os.path.join(SCRIPT_DIR, "poses")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Colors (BGR)
GREEN = (80, 200, 80)
RED = (60, 60, 230)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------
# Pose detectors
# -----------------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def detect_hands_behind_head(landmarks, w, h):
    def px(lm):
        return (int(lm.x * w), int(lm.y * h))

    lw = px(landmarks[PL.LEFT_WRIST])
    rw = px(landmarks[PL.RIGHT_WRIST])
    le = px(landmarks[PL.LEFT_EAR])
    re = px(landmarks[PL.RIGHT_EAR])

    return distance(lw, le) < 120 and distance(rw, re) < 120


# Pose name (image file name without extension) -> detector function.
# Images in poses/ without a detector still show in the gallery.
DETECTORS = {
    "hands_behind_head": detect_hands_behind_head,
}


def detect_pose(landmarks, w, h):
    for name, detector in DETECTORS.items():
        if detector(landmarks, w, h):
            return name
    return None


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


def draw_tile(frame, image, name, x, y, tw, th, active, found, label_scale):
    """Draw one meme thumbnail with its name underneath."""
    text_h = cv2.getTextSize("Ag", FONT, label_scale, 1)[0][1]
    show_label = tw >= 70 and th * 0.22 >= text_h + 4
    label_h = int(th * 0.22) if show_label else 0
    img_h = th - label_h

    frame[y : y + img_h, x : x + tw] = fit(image, tw, img_h)

    if active:
        color, thickness = GREEN, 3
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
        cv2.putText(frame, text, (tx, ty), FONT, label_scale, WHITE if active else GREY, 1, cv2.LINE_AA)

    if found:
        r = max(7, tw // 10)
        draw_check(frame, x + tw - r - 4, y + r + 4, r)


def draw_gallery_strip(frame, poses, active, found):
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
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, name == active, name in found, 0.4)
        x += tile_w + pad


def draw_gallery_grid(frame, poses, active, found):
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
        draw_tile(frame, poses[name], name, x, y, tile_w, tile_h, name == active, name in found, 0.6)


def draw_status(frame, matched, show_text=True):
    h, w = frame.shape[:2]
    if matched:
        color, text = GREEN, f"MATCH: {display_name(matched).upper()}"
    else:
        color, text = RED, "STRIKE A POSE"
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    if show_text:
        cv2.putText(frame, text, (20, 45), FONT, 1, color, 2, cv2.LINE_AA)


# -----------------------------
# Main loop
# -----------------------------
def main():
    poses = load_poses(POSES_DIR)
    if not poses:
        print(f"No pose images found in {POSES_DIR}")
        sys.exit(1)

    missing = [name for name in DETECTORS if name not in poses]
    for name in missing:
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

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        matched = None
        if results.pose_landmarks:
            matched = detect_pose(results.pose_landmarks.landmark, w, h)
        if matched and matched in poses:
            found.add(matched)
        else:
            matched = None

        if show_grid:
            draw_gallery_grid(frame, poses, matched, found)
            draw_status(frame, matched, show_text=False)
        else:
            draw_status(frame, matched)
            draw_gallery_strip(frame, poses, matched, found)

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
