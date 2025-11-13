# =============================================================================
# FINAL CODE FOR AUTOMATED VERIFICATION ACROSS MULTIPLE VIDEOS
# =============================================================================

# 1. Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
import pandas as pd  # To read the CSV label file
import os  # To handle file paths


# =============================================================================
# 2. HELPER FUNCTIONS (No change)
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
# 3. CONSTANTS AND MODEL LOADING (No change)
# =============================================================================
MAR_THRESHOLD = 0.4  # The threshold we are testing
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# =============================================================================
# 4. MAIN VERIFICATION SCRIPT
# =============================================================================
def run_automated_verification():
    """
    Reads a labels.csv file, processes all listed videos, and calculates
    the overall performance metrics for yawn detection.
    """
    # --- Load the Ground Truth Labels ---
    try:
        labels_df = pd.read_csv("labels.csv")
    except FileNotFoundError:
        print("[ERROR] labels.csv not found! Please create it and place it in the project folder.")
        return

    # --- Initialize counters for overall performance ---
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_tn = 0  # True Negatives
    total_fn = 0  # False Negatives

    # --- Get the list of unique videos to process from the CSV file ---
    video_files_to_process = labels_df["video_name"].unique()
    print(f"[INFO] Found {len(video_files_to_process)} videos to verify: {video_files_to_process}")

    # --- Loop through each video file ---
    for video_file in video_files_to_process:
        # Construct the full path to the video file inside the 'videos' folder
        video_path = os.path.join("videos", video_file)

        if not os.path.exists(video_path):
            print(f"[WARNING] Video file not found: {video_path}. Skipping.")
            continue

        print(f"\n[INFO] Processing video: {video_file}")
        cap = cv2.VideoCapture(video_path)

        # Get the specific labels for this video
        video_labels = labels_df[labels_df["video_name"] == video_file]

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            # Determine the ground truth for the current frame
            is_yawn_ground_truth = False
            for _, row in video_labels.iterrows():
                # Normalize the label to string, lowercase, and remove spaces
                label_str = str(row["label"]).strip().lower()

                # Check if the label is any variation of "yawn"
                if label_str in ["yawn", "yawning", "1", "true", "yes"] and row["start_frame"] <= frame_number <= row[
                    "end_frame"]:
                    is_yawn_ground_truth = True
                    break

            # Assume one face per frame for simplicity
            is_yawn_prediction = False  # Default prediction is NO yawn
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                shape = []
                for i in range(0, 68): shape.append((landmarks.part(i).x, landmarks.part(i).y))

                mouth = shape[48:68]
                mar = calculate_mar(mouth)

                if mar > MAR_THRESHOLD:
                    is_yawn_prediction = True

            # Compare prediction to ground truth and update counters
            if is_yawn_prediction and is_yawn_ground_truth:
                total_tp += 1
            elif is_yawn_prediction and not is_yawn_ground_truth:
                total_fp += 1
            elif not is_yawn_prediction and not is_yawn_ground_truth:
                total_tn += 1
            elif not is_yawn_prediction and is_yawn_ground_truth:
                total_fn += 1

        cap.release()
        print(f"[INFO] Finished processing {video_file}.")

    # --- Calculate and Print Final Performance Metrics ---
    print("\n" + "=" * 50)
    print("      OVERALL YAWN DETECTION VERIFICATION RESULTS")
    print("=" * 50)

    total_frames = total_tp + total_fp + total_tn + total_fn
    if total_frames > 0:
        accuracy = (total_tp + total_tn) / total_frames
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        print(f"Total Frames Processed Across All Videos: {total_frames}")
        print(f"True Positives (Correctly detected yawns): {total_tp}")
        print(f"False Positives (Falsely detected yawns): {total_fp}")
        print(f"True Negatives (Correctly ignored non-yawns): {total_tn}")
        print(f"False Negatives (Missed real yawns): {total_fn}")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall (Sensitivity): {recall:.2%}")
    else:
        print("[ERROR] No frames were processed. Please check video paths and labels.")


# =============================================================================
# 5. SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_automated_verification()