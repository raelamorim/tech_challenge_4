import cv2
from infra.video_capture import get_video_capture
from app.face_mesh_service import get_face_mesh, draw_face_mesh, draw_face_mesh_and_emotion

def process_video(video_source="0"):
    cap = get_video_capture(video_source)
    face_mesh = get_face_mesh()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
 
        processed_frame = draw_face_mesh_and_emotion(frame, face_mesh)
        
        cv2.imshow("Rostos + Emoções", processed_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
