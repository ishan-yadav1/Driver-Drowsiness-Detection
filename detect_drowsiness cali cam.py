# =============================================================================
# IMPLEMENTATION WITH IMPROVEMENT 2: ADAPTIVE THRESHOLD CALIBRATION
# =============================================================================

# 1. Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound
import time


# =============================================================================
# 2. HELPER FUNCTIONS (calculate_ear, calculate_mar) - No Change
# =============================================================================
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar


# =============================================================================
# 3. CONSTANTS - Some will be set by calibration now
# =============================================================================
# This is now a default, but will be overridden by calibration
EAR_THRESHOLD = 0.22
EAR_CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 0.4

# =============================================================================
# 4. INITIALIZE MODELS - No Change
# =============================================================================
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)
(M_START, M_END) = (48, 68)


# =============================================================================
# 5. NEW CALIBRATION FUNCTION
# =============================================================================
def run_calibration():
    """
    Runs a calibration phase to determine a personalized EAR threshold.
    """
    print("[INFO] Starting calibration...")
    print("[INFO] Please look straight at the camera and blink normally for the next 10 seconds.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam for calibration.")
        return None

    ear_values = []
    start_time = time.time()

    while time.time() - start_time < 10:  # Calibrate for 10 seconds
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        # Display instructions on the frame
        cv2.putText(frame, "CALIBRATION: Look straight and blink normally", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(faces) > 0:
            face = faces[0]  # Assume one face for calibration
            landmarks = predictor(gray, face)
            shape = []
            for i in range(0, 68):
                shape.append((landmarks.part(i).x, landmarks.part(i).y))

            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]
            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Only add EAR values for open eyes to get a good average
            # We assume an EAR > 0.2 indicates open eyes during calibration
            if ear > 0.20:
                ear_values.append(ear)

        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if len(ear_values) == 0:
        print("[ERROR] Calibration failed. Could not detect open eyes.")
        return None

    # Calculate the average EAR from the collected values
    average_ear = sum(ear_values) / len(ear_values)
    print(f"[INFO] Your average open-eye EAR is: {average_ear:.2f}")

    # The personalized threshold is a percentage of the average
    # A lower percentage makes it less sensitive. 85% is a good start.
    personalized_threshold = average_ear * 0.85

    print(f"[INFO] Calibration complete. New personalized EAR threshold set to: {personalized_threshold:.2f}")
    return personalized_threshold


# =============================================================================
# 6. MAIN VIDEO PROCESSING FUNCTION (Updated to accept a threshold)
# =============================================================================
def process_video_source(ear_threshold):
    """
    Processes a video source using the provided EAR threshold.
    """
    print("[INFO] Starting main detection...")
    cap = cv2.VideoCapture(0)  # For webcam
    # cap = cv2.VideoCapture("your_video.mp4") # For video file

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    ear_frame_counter = 0
    drowsiness_alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            # The logic for landmarks, EAR, MAR, and alerts is the same
            # But now it uses the new 'ear_threshold' variable
            landmarks = predictor(gray, face)
            shape = []
            for i in range(0, 68):
                shape.append((landmarks.part(i).x, landmarks.part(i).y))

            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]
            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < ear_threshold:  # USE THE PERSONALIZED THRESHOLD
                ear_frame_counter += 1
                if ear_frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    if not drowsiness_alarm_on:
                        drowsiness_alarm_on = True
                        try:
                            playsound("alarm.wav", block=False)
                        except Exception as e:
                            print(f"[ERROR] Could not play sound: {e}")

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ear_frame_counter = 0
                drowsiness_alarm_on = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Threshold: {ear_threshold:.2f}", (480, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mouth = shape[M_START:M_END]
            mar = calculate_mar(mouth)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# 7. SCRIPT ENTRY POINT (Updated Logic)
# =============================================================================

if __name__ == "__main__":
    # First, run the calibration
    new_threshold = run_calibration()

    # If calibration is successful, start the main program with the new threshold
    if new_threshold:
        process_video_source(new_threshold)
    else:
        print("[INFO] Using default EAR threshold.")
        process_video_source(EAR_THRESHOLD)  # Fallback to default if calibration fails