
# Import the necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound  # NEW: Import the playsound library


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


# Define Constants
EAR_THRESHOLD = 0.22
EAR_CONSECUTIVE_FRAMES = 7
MAR_THRESHOLD = 0.4

# Initialize models
print("Loading models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmqarks.dat")
print("Models loaded successfully.")

# Define landmark indices
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)
(M_START, M_END) = (48, 68)


def start_video_processing():
    # You can switch between your webcam and a video file here
    #cap = cv2.VideoCapture(0) # For webcam
    cap = cv2.VideoCapture("driver3.mp4")  # For video file

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    ear_frame_counter = 0

    # --- NEW: A flag to ensure the alarm plays only once ---
    drowsiness_alarm_on = False
    # --- End of NEW variable ---

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing finished or stream ended.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            shape = []
            for i in range(0, 68):
                shape.append((landmarks.part(i).x, landmarks.part(i).y))

            # --- EAR (Drowsiness) Logic ---
            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]
            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                ear_frame_counter += 1
                if ear_frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    # --- NEW: Alarm Trigger Logic ---
                    # If the alarm is not already on, turn it on
                    if not drowsiness_alarm_on:
                        drowsiness_alarm_on = True
                        # Play the alarm sound
                        # NOTE: This might run in a separate thread.
                        # For a simple implementation, this is fine.
                        try:
                            playsound("alarm.wav")
                        except Exception as e:
                            print(f"Error playing sound: {e}")
                    # --- End of NEW Logic ---

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ear_frame_counter = 0
                # --- NEW: Reset the alarm flag when eyes are open ---
                drowsiness_alarm_on = False
                # --- End of NEW Logic ---

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- MAR (Yawn) Logic (unchanged) ---
            mouth = shape[M_START:M_END]
            mar = calculate_mar(mouth)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_video_processing()