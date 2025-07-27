import cv2
from app.pose_service import detect_pose_and_activity
from app.detect_activity import detect_activity_from_pose
from infra.video_capture import get_video_capture
from app.face_mesh_service import get_face_mesh, draw_face_mesh, draw_face_mesh_and_emotion

def process_video(video_source="0"):
    cap = get_video_capture(video_source)
    face_mesh = get_face_mesh()

    frame_count = 0
    frame_inicio = 30  * 1 # equivale a 1 seg a cada 30 frames
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count < frame_inicio:
            continue 
        
        if frame is None:
            print("Frame é None, pulando frame.")
            continue
        
        processed_frame = draw_face_mesh_and_emotion(frame, face_mesh)
        processed_frame = detect_activity_from_pose(processed_frame)
        
        cv2.imshow("Smart People Detector", processed_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
