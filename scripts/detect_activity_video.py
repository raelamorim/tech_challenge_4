import sys
import os
# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import mediapipe as mp
import joblib
import os
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from app.face_mesh_service import get_face_mesh, draw_face_mesh, draw_face_mesh_and_emotion



def aplicar_modelo(video_path, output_video_path, output_csv_path, modelo_path, modelo_encoder_path, scaler_path):  # NOVO
    # Carregar modelo, codificador e SCALER
    model = joblib.load(modelo_path)
    le = joblib.load(modelo_encoder_path)
    scaler = joblib.load(scaler_path)  # NOVO

    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Criar vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Criar CSV de saída
    csv_file = open(output_csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['frame', 'action'])
    last_action = 'unknown'
    last_action_frame = 0

    face_mesh = get_face_mesh()

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        for frame_index in tqdm(range(total_frames), desc="Analisando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            frame, emotion_label = draw_face_mesh_and_emotion(frame, face_mesh)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            action_label = 'unknown'

            if results.pose_landmarks:
                features = []
                for lm in results.pose_landmarks.landmark:
                    features += [lm.x, lm.y, lm.z, lm.visibility]

                if len(features) == 132:
                    try:
                        # Aplicar normalização com o SCALER usado no treinamento
                        features_scaled = scaler.transform([features])  # NOVO

                        probs = model.predict_proba(features_scaled)[0]
                        confidence = max(probs)
                        pred_idx = probs.argmax()
                        predicted_label = le.inverse_transform([pred_idx])[0]

                        if confidence >= 0.3:
                            action_label = predicted_label
                        else:
                            action_label = 'unknown'

                        if action_label == 'unknown':
                            if last_action_frame < 100:
                                action_label = last_action
                                last_action_frame += 1
                            else:
                                last_action = 'unknown'
                        else:
                            last_action = action_label
                            last_action_frame = 0

                        text_action = f'Atividade: {action_label}' if action_label == "unknown" else f'Atividade: {action_label} ({confidence:.2f})'

                        cv2.putText(frame, text_action, (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except:
                        action_label = "unknown"

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            writer.writerow([frame_index, action_label, emotion_label])
            out.write(frame)

    cap.release()
    out.release()
    csv_file.close()
    print(f"Processamento finalizado. Vídeo salvo em: {output_video_path}")
    print(f"CSV salvo em: {output_csv_path}")


# Caminhos
base_path = Path(__file__).parent.parent
input_video =  base_path / 'assets\\andando.mp4'
output_video = base_path / 'assets\\andando_output_video_activity.mp4'
output_csv = base_path / 'landmarks\\acoes_detectadas.csv'
modelo_path = base_path / 'modelo_classificador\\activity_recognition_model.pkl'
modelo_encoder_path = base_path / 'modelo_classificador\\label_encoder.pkl'
scaler_path = base_path / 'modelo_classificador\\scaler.pkl'  # NOVO

# Chamar função
aplicar_modelo(input_video, output_video, output_csv, modelo_path, modelo_encoder_path, scaler_path)

