import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

def detect_pose_and_activity(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape

    activity_label = "N/A"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose_vector = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()

        activity_label = "exercicio" if pose_vector[25] < pose_vector[27] else "repouso"

    cv2.putText(frame, f"Action: {activity_label}", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)