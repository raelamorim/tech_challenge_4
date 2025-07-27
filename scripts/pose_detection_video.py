import cv2
import mediapipe as mp
import os
import csv
from tqdm import tqdm

# Lista com as ações (início, fim, rótulo)
acoes = [
    (0, 139, 'reading'),
    (182, 225, 'waving'),
    (542, 717, 'dancing'),
    (902, 1078, 'hands_on_face'),
    (1080, 1259, 'resting'),
    (1260, 1438, 'grimacing'),
    (1440, 1555, 'laughing'),
    (1570, 1616, 'suffering'),
    (1835, 2008, 'removing_blindfold'),
    (2011, 2185, 'examining'),
    (2190, 2300, 'walking'),
    (2301, 2368, 'raising_arm'),
    (2402, 2580, 'typing'),
    (2583, 2624, 'typing'),
    (2632, 2756, 'typing'),
    (2940, 2977, 'using_cellphone'),
    (3025, 3118, 'handshaking'),
    (3130, 3298, 'talking'),
    (3300, 3325, 'typing')
]

def obter_label_por_frame(frame_index):
    for inicio, fim, label in acoes:
        if inicio <= frame_index <= fim:
            return label
    return 'unknown'  # Caso o frame não esteja em nenhuma faixa

def detect_pose(video_path, output_video_path, output_csv_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['label']
        for i in range(33):
            header += [f'x{i}', f'y{i}', f'z{i}', f'visibility{i}']
        writer.writerow(header)

        with mp_pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:

            for frame_index in tqdm(range(total_frames), desc="Processando vídeo"):
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                # Obter rótulo da ação baseado no frame atual
                label_acao = obter_label_por_frame(frame_index)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Desenhar número do frame no vídeo
                    cv2.putText(frame, f'Frame: {frame_index}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'Action: {label_acao}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    row = [label_acao]
                    for lm in results.pose_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z, lm.visibility]

                    writer.writerow(row)

                out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Processamento concluído. Dados salvos em {output_csv_path}.')

# Caminhos
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video_pose.mp4')
output_csv_path = os.path.join(script_dir, 'landmarks.csv')

# Executar
detect_pose(input_video_path, output_video_path, output_csv_path)
