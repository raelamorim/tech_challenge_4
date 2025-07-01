import mediapipe as mp
import cv2
import numpy as np
from deepface import DeepFace
from collections import deque, Counter

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def draw_face_mesh(frame, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

# Global (ou em cache por rosto)
emotion_history = deque(maxlen=10)  # mantém as últimas 10 emoções

def draw_face_mesh_and_emotion(frame, face_mesh):
    """
    Processa o frame com FaceMesh e desenha bounding box e emoção no rosto.
    Retorna o frame anotado.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            cropped_face = frame[y_min:y_max, x_min:x_max]
            emotion_label = "N/A"

            try:
                if cropped_face.size > 0:
                    result = DeepFace.analyze(cropped_face, actions=["emotion"], enforce_detection=False)
                    predicted_emotion = result[0]['dominant_emotion']
                    emotion_history.append(predicted_emotion)

                    # usa a emoção mais comum nas últimas 10
                    emotion_label = Counter(emotion_history).most_common(1)[0][0]
            except:
                emotion_label = "Erro"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(frame, emotion_label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame