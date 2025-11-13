# =============================================================================
# FINAL INTEGRATED CODE WITH ALL IMPROVEMENTS
# 1. Audible Alarm
# 2. Adaptive Threshold Calibration
# 3. Robust Face Tracking
# =============================================================================

# 1. Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound


# 2. HELPER FUNCTIONS (calculate_ear, calculate_mar) - No Change
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5]);
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3]);
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth):
    A = dist.euclidean(mouth[13], mouth[19]);
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17]);
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar


# 3. CONSTANTS - Some are set by calibration
EAR_CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 0.4
DEFAULT_EAR_THRESHOLD = 0.22  # A fallback default

# 4. MODEL LOADING - No Change
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(L_START, L_END) = (42, 48);
(R_START, R_END) = (36, 42);
(M_START, M_END) = (48, 68)


# =============================================================================
# 5. IMPROVEMENT: CALIBRATION FUNCTION
# =============================================================================
def run_calibration():
    # CHOOSE CALIBRATION SOURCE: WEBCAM (0) or VIDEO FILE
    # calibration_source = 0  # Calibrate with live webcam
    calibration_source = "10-MaleNoGlasses-Normal.avi " # Calibrate with a video file

    print(f"[INFO] Starting calibration from source: {calibration_source}...")
    cap = cv2.VideoCapture(calibration_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open calibration source.")
        return None

    ear_values = []

    # For webcam, calibrate for a fixed duration. For video, process all frames.
    is_video_file = isinstance(calibration_source, str)
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # If using webcam, stop after 10 seconds
        if not is_video_file:
            duration = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if duration > 10.0: break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        cv2.putText(frame, "CALIBRATION: Look straight, blink normally", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            shape = []
            for i in range(0, 68): shape.append((landmarks.part(i).x, landmarks.part(i).y))

            leftEye = shape[L_START:L_END];
            rightEye = shape[R_START:R_END]
            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

            if ear > 0.20: ear_values.append(ear)

        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if len(ear_values) == 0:
        print("[WARNING] Calibration failed. No open eyes detected.")
        return None

    average_ear = sum(ear_values) / len(ear_values)
    personalized_threshold = average_ear * 0.85

    print(f"[INFO] Calibration complete. Personalized EAR threshold: {personalized_threshold:.2f}")
    return personalized_threshold


# =============================================================================
# 6. MAIN VIDEO PROCESSING FUNCTION (WITH ALL IMPROVEMENTS)
# =============================================================================
def start_video_processing(ear_threshold):
    # CHOOSE VIDEO SOURCE: WEBCAM (0) or VIDEO FILE
    # video_source = 0  # Using webcam
    video_source = "10-MaleNoGlasses-Yawning.avi" # Using a video file

    print(f"[INFO] Starting main detection from source: {video_source}...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source.")
        return

    # --- Variables for Tracking ---
    tracker = None
    tracking_face = False
    frame_count = 0

    # --- Variables for Drowsiness Detection ---
    ear_frame_counter = 0
    drowsiness_alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rect = None

        # --- Robust Tracking Logic ---
        if not tracking_face or frame_count % 10 == 0:
            faces = detector(gray, 0)
            if len(faces) > 0:
                face_rect = faces[0]
                tracker = dlib.correlation_tracker();
                tracker.start_track(frame, face_rect)
                tracking_face = True
        else:
            tracker.update(frame);
            pos = tracker.get_position()
            face_rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))

        if face_rect:
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()),
                          (0, 255, 0), 2)

            landmarks = predictor(gray, face_rect)
            shape = []
            for i in range(0, 68): shape.append((landmarks.part(i).x, landmarks.part(i).y))

            # --- EAR/MAR Logic with Audible Alarm ---
            leftEye = shape[L_START:L_END];
            rightEye = shape[R_START:R_END]
            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

            # Use the calibrated threshold passed to the function
            if ear < ear_threshold:
                ear_frame_counter += 1
                if ear_frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    if not drowsiness_alarm_on:
                        drowsiness_alarm_on = True
                        try:
                            playsound("alarm.wav", block=False)
                        except Exception as e:
                            print(f"Error playing sound: {e}")
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ear_frame_counter = 0;
                drowsiness_alarm_on = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Thresh: {ear_threshold:.2f}", (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mouth = shape[M_START:M_END]
            mar = calculate_mar(mouth)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_count += 1
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# 7. SCRIPT ENTRY POINT (Ties Everything Together)
# =============================================================================
if __name__ == "__main__":
    # 1. Run the calibration phase first
    calibrated_threshold = run_calibration()

    # 2. Check if calibration was successful
    if calibrated_threshold is None:
        print("[WARNING] Calibration failed. Using default threshold.")
        final_threshold = DEFAULT_EAR_THRESHOLD
    else:
        final_threshold = calibrated_threshold

    # 3. Start the main detection program with the final threshold
    start_video_processing(final_threshold)


