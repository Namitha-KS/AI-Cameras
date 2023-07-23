import cv2
import dlib

# Initialize the face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

# Constants for eye aspect ratio (EAR) and eye closed threshold
EAR_THRESHOLD = 0.3
EYE_CLOSED_FRAMES = 3

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = dist(eye_points[1], eye_points[5])
    B = dist(eye_points[2], eye_points[4])
    C = dist(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Euclidean distance between two points
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = landmark_predictor(gray, face)

        # Extract eye points
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate eye aspect ratio (EAR) for left and right eyes
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        # Check if the eyes are closed
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            print("Eyes are closed!")

        # Draw eye landmarks on the frame
        for point in left_eye_points:
            cv2.circle(frame, point, 1, (0, 0, 255), -1)
        for point in right_eye_points:
            cv2.circle(frame, point, 1, (0, 0, 255), -1)

    # Display the frame with eye landmarks
    cv2.imshow("Real-time Face Detection", frame)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
