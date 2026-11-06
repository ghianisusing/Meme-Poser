"""MediaPipe pose model: downloads it on first run and finds body points in frames.

Uses MediaPipe's PoseLandmarker (the "Tasks" API). The model file is fetched
once into models/ and reused after that.
"""

import os
import tempfile
import urllib.request

import cv2
import mediapipe as mp

from settings import MODEL_PATH, MODEL_URL

vision = mp.tasks.vision

# The 33 body points (NOSE, LEFT_SHOULDER, ...) and the lines between them.
PL = vision.PoseLandmark
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    """Download the pose model if it isn't there yet. Returns its path."""
    if os.path.exists(path):
        return path

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    print(f"Downloading the pose model (about 9 MB) to {path} ...")
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"Couldn't download the pose model: {e}\n"
            f"Check your internet connection, or download it yourself from\n  {url}\n"
            f"and save it as {path}"
        ) from e
    print("Done.")
    return path


class PoseDetector:
    """Finds one person's body points in webcam frames."""

    def __init__(self, model_path=MODEL_PATH):
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.last_ms = -1

    def detect(self, frame_bgr, seconds):
        """Body points for this frame (a list of 33 points with x, y and
        visibility), or None if no one is in view.
        seconds: the frame's time; it must keep going up between calls."""
        ms = max(int(seconds * 1000), self.last_ms + 1)
        self.last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def close(self):
        self.landmarker.close()
