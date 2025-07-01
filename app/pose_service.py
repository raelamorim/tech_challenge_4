from collections import deque
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Por pessoa (idealmente), mas começamos com 1 global
feature_sequence = deque(maxlen=30)  # ~1s a 30fps

   
def classify_activity_by_pose(rgb_frame):
    """
    Classifica atividade com base nos landmarks (simplificado).
    """
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    activity_label = "Desconhecida"
    
    if not pose_results.pose_landmarks:
        return activity_label
    
    features = extract_pose_hand_features(pose_results, hands_results)
    
    # Garante que todos os vetores tenham o mesmo tamanho
    if features.shape[0] == (33 + 42) * 3:  # 75 pontos * 3 coords
        feature_sequence.append(features)
        
    return classify_action_from_sequence()
    
def extract_pose_hand_features(pose_results, hands_results):
    pose_vector = [[0, 0, 0]] * 33  # 33 pontos de pose
    hand_vector = [[0, 0, 0]] * 42  # 2 mãos x 21 pontos

    if pose_results and pose_results.pose_landmarks:
        pose_vector = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]

    if hands_results and hands_results.multi_hand_landmarks:
        hand_vector = []
        for hand_landmarks in hands_results.multi_hand_landmarks:
            hand_vector.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        # Se só detectou uma mão, preencha a segunda com zeros
        while len(hand_vector) < 42:
            hand_vector.extend([[0, 0, 0]])

    feature_vector = np.array(pose_vector + hand_vector).flatten()
    return feature_vector


def classify_action_from_sequence():
    if len(feature_sequence) < feature_sequence.maxlen:
        return "Carregando..."

    recent = np.array(feature_sequence)

    # === Cálculo de métricas básicas ===
    left_wrist = recent[:, mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1]
    right_wrist = recent[:, mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1]
    nose = recent[:, mp_pose.PoseLandmark.NOSE.value * 3 + 1]
    left_shoulder = recent[:, mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1]
    right_shoulder = recent[:, mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1]

    # ============================
    # Lógica Heurística das Ações
    # ============================

    # Acenando: mão sobe e desce alternadamente
    right_wrist_std = np.std(recent[:, mp_pose.PoseLandmark.RIGHT_WRIST.value * 3])
    if np.mean(right_wrist < nose) > 0.6 and right_wrist_std > 0.05:
        return "Acenando"

    # Dançando: muita movimentação de ombros
    shoulder_var = np.var(left_shoulder) + np.var(right_shoulder)
    if shoulder_var > 0.01:
        return "Dançando"

    # Mexendo no celular: mãos perto e abaixo do rosto
    left_wrist_x = recent[:, mp_pose.PoseLandmark.LEFT_WRIST.value * 3]
    right_wrist_x = recent[:, mp_pose.PoseLandmark.RIGHT_WRIST.value * 3]
    dist_wrist_x = np.abs(left_wrist_x - right_wrist_x)

    if np.mean(dist_wrist_x) < 0.05 and np.mean(right_wrist > nose) > 0.8:
        return "Mexendo no celular"

    # Falando: pequenas variações na posição da boca (usaria face landmarks com mais precisão)
    # Pode usar variação do nariz como proxy de cabeça estável
    nose_var = np.var(nose)
    if nose_var < 0.0005:
        return "Falando"

    # Caso nenhuma ação clara seja detectada
    return "Parado"