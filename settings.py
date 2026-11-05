"""Settings you can tune. Everything that changes how the game feels is here."""

import os

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

# Recorded poses: how close you need to be to a recorded example, as the
# average distance per body point in shoulder widths.
# NEAR or closer scores 1, FAR or more scores 0.
SAMPLE_NEAR = 0.15
SAMPLE_FAR = 0.35
