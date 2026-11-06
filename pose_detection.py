"""Meme Poser: strike a pose, get the matching meme.

Run: python pose_detection.py
"""

import os
import sys
import time

import cv2

import pose_samples
from detection import DETECTORS, PoseTracker, score_poses
from library import display_name, load_poses
from pose_model import PoseDetector, ensure_model
from settings import (
    COUNTDOWN_SECONDS,
    MIN_SAMPLES,
    MIN_VISIBILITY,
    POSES_DIR,
    RECORD_SECONDS,
    SAMPLES_DIR,
)
from ui import (
    AMBER,
    GREEN,
    RED,
    WHITE,
    draw_gallery_grid,
    draw_gallery_strip,
    draw_message,
    draw_record_overlay,
    draw_reveal,
    draw_skeleton,
    draw_status,
    put_text,
    save_snapshot,
)

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

    pose = PoseDetector(ensure_model())  # downloads the model on first run

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam.")
        pose.close()
        sys.exit(1)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 800)
    cv2.moveWindow("Camera", 0, 0)

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

        landmarks = pose.detect(frame, now)

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
            if landmarks:
                draw_skeleton(frame, landmarks, MIN_VISIBILITY)
            draw_record_overlay(frame, poses[record_pose], record_pose, mode, now - phase_start, len(recorded) if mode == "recording" else 0)
        elif show_grid:
            draw_gallery_grid(frame, poses, matched, found, holding, ready_poses())
            draw_status(frame, matched, holding, progress, show_text=False)
        else:
            if mode == "choose":
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), AMBER, 8)
                put_text(frame, "RECORD A POSE", (20, 45), 1, AMBER, 2)
            else:
                if show_skeleton and landmarks:
                    draw_skeleton(frame, landmarks, MIN_VISIBILITY)
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
