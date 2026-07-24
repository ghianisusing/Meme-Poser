import cv2
import mediapipe as mp
import math
import os

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -----------------------------
# Load Pose Images
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_DIR = os.path.join(SCRIPT_DIR, "poses")

pose_images = {
    "hands_behind_head": cv2.resize(
        cv2.imread(os.path.join(POSES_DIR, "hands_behind_head.png")), (300, 300)
    )
}

# Check if images loaded
for name, image in pose_images.items():
    if image is None:
        print(f"Failed to load pose image: {name}")
        exit()

current_pose = "hands_behind_head"

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    exit()

# -----------------------------
# Camera Window
# -----------------------------
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1400, 900)
cv2.moveWindow("Camera", 0, 0)

popup_visible = False


# -----------------------------
# Helper Function
# -----------------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    matched = False

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        h, w, _ = frame.shape

        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

        lw = (int(left_wrist.x * w), int(left_wrist.y * h))

        rw = (int(right_wrist.x * w), int(right_wrist.y * h))

        le = (int(left_ear.x * w), int(left_ear.y * h))

        re = (int(right_ear.x * w), int(right_ear.y * h))

        left_distance = distance(lw, le)
        right_distance = distance(rw, re)

        # Hands Behind Head
        if left_distance < 120 and right_distance < 120:
            matched = True

    # -----------------------------
    # Camera UI
    # -----------------------------
    if matched:
        border_color = (0, 255, 0)
        text = "MATCH!"
    else:
        border_color = (0, 0, 255)
        text = "COPY THE POSE"

    cv2.rectangle(
        frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 8
    )

    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)

    # -----------------------------
    # Show Pose Window
    # -----------------------------
    if matched:

        if not popup_visible:

            cv2.namedWindow("Target Pose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Target Pose", 300, 300)
            cv2.moveWindow("Target Pose", 1450, 50)

            popup_visible = True

        cv2.imshow("Target Pose", pose_images[current_pose])

    else:

        if popup_visible:
            cv2.destroyWindow("Target Pose")
            popup_visible = False

    # -----------------------------
    # Show Camera
    # -----------------------------
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()

if popup_visible:
    cv2.destroyWindow("Target Pose")

cv2.destroyAllWindows()
