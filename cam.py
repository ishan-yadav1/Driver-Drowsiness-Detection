# Import the necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- NEW: Function to calculate Mouth Aspect Ratio (MAR) ---
def calculate_mar(mouth):
    # The mouth is represented by 20 (x, y)-coordinates (landmarks 48 to 68)
    # We are interested in the inner mouth region (landmarks 60 to 68)
    # The paper's MAR formula is a simplified one. Let's use the distance between key vertical points.
    A = dist.euclidean(mouth[13], mouth[19]) # 61 vs 67
    B = dist.euclidean(mouth[14], mouth[18]) # 62 vs 66
    C = dist.euclidean(mouth[15], mouth[17]) # 63 vs 65
    # And the horizontal distance
    D = dist.euclidean(mouth[12], mouth[16]) # 60 vs 64
    mar = (A + B + C) / (3.0 * D)
    return mar
# --- End of NEW Function ---


# Define Constants for Drowsiness Detection
EAR_THRESHOLD = 0.22
EAR_CONSECUTIVE_FRAMES = 20

# --- NEW: Define Constants for Yawn Detection ---
MAR_THRESHOLD = 0.4 # This value might need tuning
# --- End of NEW Constants ---


# Initialize dlib's face detector and the facial landmark predictor
print("Loading models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("Models loaded successfully.")

# Define constants for the landmark indices
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)
(M_START, M_END) = (48, 68) # --- NEW: Indices for the mouth ---


def start_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            shape = []
            for i in range(0, 68):
                shape.append((landmarks.part(i).x, landmarks.part(i).y))

            # --- EYE LOGIC (No change) ---
            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]
            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- NEW: MOUTH LOGIC ---
            mouth = shape[M_START:M_END]
            mar = calculate_mar(mouth)

            # Check if the MAR is above the yawn threshold
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # --- End of NEW Logic ---


        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video()