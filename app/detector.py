from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # baixa automaticamente

def detect_people(frame):
    results = model.predict(frame, classes=[0], conf=0.5, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x_center, y_center, w, h = box.xywh[0].tolist()
        conf = box.conf[0].item()

        # Converter de centro para canto superior esquerdo
        x1 = x_center - w / 2
        y1 = y_center - h / 2

        detections.append([x1, y1, w, h, conf, 0])
    return detections
