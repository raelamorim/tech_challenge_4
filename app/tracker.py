from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=10, n_init=3)  # menor tolerância a rastreamento perdido


def track_people(frame, detections):
    formatted_detections = []
    for det in detections:
        x, y, w, h, conf, class_id = det
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        formatted_detections.append([[x1, y1, x2, y2], conf, class_id])
    
    tracks = tracker.update_tracks(formatted_detections, frame=frame)
    return tracks
