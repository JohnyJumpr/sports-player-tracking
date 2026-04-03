from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, classes=[0])[0] # Class 0 = person
        detections = []

        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())

            # Detect persons
            if results.names[cls] == "person" and conf > 0.4:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        return detections