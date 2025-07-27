import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from PIL import ImageFont, ImageDraw, Image

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Buffers para análise temporal
left_wrist_x_buffer = deque(maxlen=60)
right_wrist_x_buffer = deque(maxlen=60)
orientation_buffer = deque(maxlen=60)  # Frente (1) ou costas (-1)

def get_orientation(l_shoulder, r_shoulder):
    """Determina se a pessoa está de frente (1) ou costas (-1) baseado na posição dos ombros"""
    # Quanto maior a diferença entre os ombros, mais de frente está a pessoa
    shoulder_width = abs(r_shoulder.x - l_shoulder.x)
    return 1 if shoulder_width > 0.10 else -1  # limiar ajustável

def detect_activity_from_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape

    activity_labels = []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # LANDMARKS principais
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # ----- BRAÇO LEVANTADO
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            activity_labels.append("Braço levantado")

        # ----- MOVIMENTO DE MÃO
        left_wrist_x_buffer.append(left_wrist.x)
        right_wrist_x_buffer.append(right_wrist.x)
        if len(left_wrist_x_buffer) >= 30:
            amp_esq = max(left_wrist_x_buffer) - min(left_wrist_x_buffer)
            amp_dir = max(right_wrist_x_buffer) - min(right_wrist_x_buffer)
            if amp_esq > 0.08 or amp_dir > 0.08:
                activity_labels.append("Movimento de mão")

        # ----- ORIENTAÇÃO (Frente/Costas) para DANÇAR
        orientation = get_orientation(left_shoulder, right_shoulder)
        orientation_buffer.append(orientation)

        # Detecta alternância de frente ↔ costas dentro de 2s
        if len(orientation_buffer) >= 30:
            if 1 in orientation_buffer and -1 in orientation_buffer:
                activity_labels.append("Dançando")

    else:
        # Zerar buffers se não detecta corpo
        left_wrist_x_buffer.clear()
        right_wrist_x_buffer.clear()
        orientation_buffer.clear()

    if not activity_labels:
        activity_labels.append("Nenhuma atividade detectada")

    texto = "Atividades: " + ", ".join(activity_labels)
    frame = draw_text_with_accent(frame, texto, position=(10, h - 60))
    return frame

def draw_text_with_accent(frame, text, position=(10, 30)):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", 24)
    draw.text(position, text, font=font, fill=(255, 255, 0))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
