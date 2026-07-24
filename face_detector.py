import cv2
import mediapipe as mp

print(mp.__version__)

trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Cannot access webcam.")
    exit()

while True:

    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Webcam Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
