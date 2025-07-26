from collections import Counter
import cv2
from app.detector import detect_people
from app.tracker import track_people
from app.person_buffers import pose_history, emotion_history
from app.pose_service import extract_pose, classify_action
from app.emotion_service import detect_emotion


def process_video(video_path=0):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detections = detect_people(frame)
        tracks = track_people(frame, detections)

        height, width, _ = frame.shape
        
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, w, h = map(int, track.to_tlwh())
            x2, y2 = x1 + w, y1 + h

            # Ajustar para garantir que não saia dos limites da imagem
            height, width, _ = frame.shape
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            w = max(0, min(w, width - x1))
            h = max(0, min(h, height - y1))

            # box invalida
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Pega só a parte superior para o rosto (40% da altura)
            face_y2 = y1 + int(h * 0.4)
            face_y2 = min(face_y2, height)

            if face_y2 <= y1 or x2 <= x1:
                print(f"[WARNING] Faixa de rosto inválida para track_id {track_id}: x1={x1}, x2={x2}, y1={y1}, face_y2={face_y2}")
                continue

            person_img = frame[y1:face_y2, x1:x2]
            if person_img is None or person_img.size == 0:
                print(f"[ERROR] Imagem vazia para track_id {track_id}, coordenadas inválidas?")
                continue
            
            pose_vec = extract_pose(person_img)
            
            if pose_vec is not None:
                pose_history[track_id].append(pose_vec)
                action = classify_action(pose_history[track_id])
            else:
                action = "Desconhecido"

            emotion = detect_emotion(person_img)
            if emotion:
                emotion_history[track_id].append(emotion)
                final_emotion = Counter(emotion_history[track_id]).most_common(1)[0][0]
            else:
                final_emotion = "?"

            cv2.putText(frame, f"ID:{track_id} - {action} - {final_emotion}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imshow("Rastreamento com Emoções e Ações", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()