import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract_pose(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return None

def classify_action(buffer):
    if len(buffer) < 10:
        return "Aguardando"

    recent = np.array(buffer)

    left_wrist_y = recent[:, mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1]
    right_wrist_y = recent[:, mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1]
    nose_y = recent[:, mp_pose.PoseLandmark.NOSE.value * 3 + 1]
    shoulder_var = np.var(recent[:, mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1]) + \
                   np.var(recent[:, mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1])

    if np.mean(right_wrist_y < nose_y) > 0.6 and np.std(right_wrist_y) > 0.05:
        return "Acenando"
    if shoulder_var > 0.01:
        return "Dancando"
    if np.mean(np.abs(recent[:, mp_pose.PoseLandmark.LEFT_WRIST.value * 3] -
                      recent[:, mp_pose.PoseLandmark.RIGHT_WRIST.value * 3])) < 0.05 and \
       np.mean(right_wrist_y > nose_y) > 0.8:
        return "Mexendo no celular"
    if np.var(nose_y) < 0.0005:
        return "Falando"

    return "Parado"