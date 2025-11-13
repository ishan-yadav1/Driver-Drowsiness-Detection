# =============================================================================
# IMPLEMENTATION WITH IMPROVEMENT: ROBUST FACE TRACKING (FOR VIDEO FILE)
# =============================================================================

# 1. Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound


# 2. HELPER FUNCTIONS (calculate_ear, calculate_mar) - No Change
def calculate_ear(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    A = dist.euclidean(eye[1], eye[5]);
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3]);
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth):
    """Calculates the Mouth Aspect Ratio (MAR) for the mouth."""
    A = dist.euclidean(mouth[13], mouth[19]);
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17]);
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar


# 3. CONSTANTS - No Change
EAR_THRESHOLD = 0.22
EAR_CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 0.4

# 4. MODEL LOADING - No Change
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(L_START, L_END) = (42, 48);
(R_START, R_END) = (36, 42);
(M_START, M_END) = (48, 68)


# =============================================================================
# 5. MAIN VIDEO PROCESSING FUNCTION (Updated for Tracking on a Video File)
# =============================================================================
def start_video_processing():
    # =========================================================================
    #
    #   >>> EDIT THIS SECTION TO CHOOSE YOUR VIDEO SOURCE <<<
    #
    # To use a video file, provide the path. Make sure it's in the same folder.
    video_source = ("10-MaleNoGlasses-Yawning.avi")
    # To use your webcam, change the line above to:
    # video_source = 0
    #
    # =========================================================================

    print(f"[INFO] Starting video stream from: {video_source}...")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
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
        if not ret:
            print("[INFO] Video file has ended. Exiting...")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rect = None

        # --- Detection and Tracking Logic ---
        if not tracking_face or frame_count % 10 == 0:
            faces = detector(gray, 0)
            if len(faces) > 0:
                face_rect = faces[0]
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, face_rect)
                tracking_face = True
        else:
            # Update the tracker and get the new position
            tracker.update(frame)
            pos = tracker.get_position()
            face_rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))

        # If we have a face, process it
        if face_rect:
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()),
                          (0, 255, 0), 2)

            landmarks = predictor(gray, face_rect)
            shape = []
            for i in range(0, 68): shape.append((landmarks.part(i).x, landmarks.part(i).y))

            # --- EAR and MAR Logic (Identical to before) ---
            leftEye = shape[L_START:L_END];
            rightEye = shape[R_START:R_END]
            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
            if ear < EAR_THRESHOLD:
                ear_frame_counter += 1
                if ear_frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    if not drowsiness_alarm_on:
                        drowsiness_alarm_on = True;
                        try:
                            playsound("alarm.wav", block=False)
                        except:
                            pass
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ear_frame_counter = 0;
                drowsiness_alarm_on = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
# 6. SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    start_video_processing()