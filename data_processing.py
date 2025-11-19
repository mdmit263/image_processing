import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return np.array([])

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        frames.append(frame)
        count += 1

        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    return np.array(frames)


if __name__ == "__main__":
    # 👉 Choose ANY single video
    video_path = "D:/downloads/time_cv_action_recognition/data/longjump/v_LongJump_g01_c01.mp4"

    print("Extracting frames...")
    frames = extract_frames(video_path)

    if frames.size == 0:
        print("❌ No frames extracted")
        exit()

    print("✅ Frames extracted:", len(frames))
    print("Frame shape:", frames[0].shape)

    # 👉 Choose which frame to display
    FRAME_NUMBER = 23  # change this to view another frame

    if FRAME_NUMBER < len(frames):
        plt.imshow(frames[FRAME_NUMBER])
        plt.title(f"Frame #{FRAME_NUMBER}")
        plt.axis("off")
        plt.show()    
    else:
        print(f"⚠ Frame {FRAME_NUMBER} does not exist. Video has {len(frames)} frames.")
