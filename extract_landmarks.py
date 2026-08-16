import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# --- Download the holistic landmarker model file (one-time) ---
MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading holistic landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)


def landmarks_to_array(landmark_list, num_points, dims=3):
    if landmark_list is None:
        return np.zeros(num_points * dims)
    arr = []
    for lm in landmark_list:
        arr.extend([lm.x, lm.y, lm.z] if dims == 3 else [lm.x, lm.y, lm.z, lm.visibility])
    return np.array(arr)


def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_duration_ms = int(1000 / fps)
    sequence = []

    with HolisticLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_idx * frame_duration_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose = landmarks_to_array(result.pose_landmarks if result.pose_landmarks else None, 33, dims=4)
            lh = landmarks_to_array(result.left_hand_landmarks if result.left_hand_landmarks else None, 21, dims=3)
            rh = landmarks_to_array(result.right_hand_landmarks if result.right_hand_landmarks else None, 21, dims=3)

            frame_landmarks = np.concatenate([pose, lh, rh])
            sequence.append(frame_landmarks)
            frame_idx += 1

    cap.release()
    return np.array(sequence)


def process_all_videos(raw_dir="data/Indian Dataset/ProcessedData_vivit with landmarks", output_dir="data/landmarks"):
    os.makedirs(output_dir, exist_ok=True)
    labels = os.listdir(raw_dir)

    for label in labels:
        label_path = os.path.join(raw_dir, label)
        if not os.path.isdir(label_path):
            continue

        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        for video_file in os.listdir(label_path):
            if not video_file.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            video_path = os.path.join(label_path, video_file)
            output_filename = video_file.rsplit(".", 1)[0] + ".npy"
            output_path = os.path.join(label_output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"Skipping (already done): {output_path}")
                continue

            print(f"Processing: {video_path}")
            sequence = extract_landmarks(video_path)

            output_filename = video_file.rsplit(".", 1)[0] + ".npy"
            output_path = os.path.join(label_output_dir, output_filename)
            np.save(output_path, sequence)
            print(f"Saved: {output_path} — shape: {sequence.shape}")


if __name__ == "__main__":
    process_all_videos()