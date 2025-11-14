# =============================================================================
# FINAL CODE WITH AUTOMATED VERIFICATION AND GRAPH GENERATION
# =============================================================================

# 1. Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
import pandas as pd  # To read the CSV label file
import os  # To handle file paths
import matplotlib.pyplot as plt  # To generate plots


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
# 4. MAIN SCRIPT
# =============================================================================
def run_verification_and_graphing():
    """
    Processes a single video, calculates performance metrics, and generates
    a graph of the MAR values over time.
    """
    # =========================================================================
    #
    #   >>> EDIT THIS LINE TO CHOOSE THE VIDEO TO ANALYZE <<<
    #
    video_to_analyze = "10-MaleNoGlasses-Yawning.avi "
    #
    # =========================================================================

    # --- Construct the path and check if the video file exists ---
    video_path = os.path.join("videos", video_to_analyze)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}.")
        return

    # --- Load the Ground Truth Labels ---
    try:
        labels_df = pd.read_csv("labels.csv")
    except FileNotFoundError:
        print("[ERROR] labels.csv not found! Please create it.")
        return

    # Get the specific labels for the video we are analyzing
    video_labels = labels_df[labels_df["video_name"] == video_to_analyze]
    if video_labels.empty:
        print(f"[WARNING] No labels found for {video_to_analyze} in labels.csv. Cannot calculate accuracy.")

    print(f"\n[INFO] Processing video: {video_to_analyze}")
    cap = cv2.VideoCapture(video_path)

    # --- Initialize lists and counters ---
    mar_values_over_time = []
    frame_numbers = []
    yawn_tp = 0;
    yawn_fp = 0;
    yawn_tn = 0;
    yawn_fn = 0
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
            if str(row["label"]).strip().lower() in ["yawn", "1"] and row["start_frame"] <= frame_number <= row[
                "end_frame"]:
                is_yawn_ground_truth = True
                break

        current_mar = 0.0  # Default MAR if no face is found
        is_yawn_prediction = False
        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            shape = []
            for i in range(0, 68): shape.append((landmarks.part(i).x, landmarks.part(i).y))
            mouth = shape[48:68]
            current_mar = calculate_mar(mouth)
            if current_mar > MAR_THRESHOLD:
                is_yawn_prediction = True

        # Store MAR value for plotting
        mar_values_over_time.append(current_mar)
        frame_numbers.append(frame_number)

        # Compare prediction to ground truth for verification
        if is_yawn_prediction and is_yawn_ground_truth:
            yawn_tp += 1
        elif is_yawn_prediction and not is_yawn_ground_truth:
            yawn_fp += 1
        elif not is_yawn_prediction and not is_yawn_ground_truth:
            yawn_tn += 1
        elif not is_yawn_prediction and is_yawn_ground_truth:
            yawn_fn += 1

    cap.release()
    print(f"[INFO] Finished processing video. Collected {len(frame_numbers)} data points.")

    # --- Calculate and Print Final Performance Metrics ---
    print("\n" + "=" * 50)
    print(f"      VERIFICATION RESULTS FOR: {video_to_analyze}")
    print("=" * 50)

    total_frames = yawn_tp + yawn_fp + yawn_tn + yawn_fn
    if total_frames > 0:
        accuracy = (yawn_tp + yawn_tn) / total_frames
        precision = yawn_tp / (yawn_tp + yawn_fp) if (yawn_tp + yawn_fp) > 0 else 0
        recall = yawn_tp / (yawn_tp + yawn_fn) if (yawn_tp + yawn_fn) > 0 else 0

        print(f"Total Frames Processed: {total_frames}")
        print(f"True Positives (Correctly detected yawn): {yawn_tp}")
        print(f"False Positives (Falsely detected yawn): {yawn_fp}")
        print(f"True Negatives (Correctly ignored non-yawn): {yawn_tn}")
        print(f"False Negatives (Missed real yawn): {yawn_fn}")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall (Sensitivity): {recall:.2%}")
    else:
        print("[WARNING] No frames were processed for verification.")

    # --- Generate and display the plot ---
    if len(frame_numbers) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(frame_numbers, mar_values_over_time, color='orange')
        plt.title('MAR Graph (MAR V/S Frames)', fontsize=14)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('MAR', fontsize=12)
        plt.ylim(0, max(mar_values_over_time) * 1.1 if max(mar_values_over_time) > 0 else 1)

        output_filename = f"MAR_graph_{video_to_analyze.replace('.avi', '')}.png"
        plt.savefig(output_filename)
        print(f"\n[INFO] Graph saved as: {output_filename}")

        plt.show()


# =============================================================================
# 5. SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_verification_and_graphing()