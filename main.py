import cv2
import numpy as np
from ultralytics import YOLO

from baseline_cv2 import MultiColorTracker

class YOLOMarkerDetector:
    def __init__(self, model_path='best.pt'):

        self.model = YOLO(model_path)


    def detect(self, frame):
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()

            if conf > 0.5:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                detections.append({'center': (center_x, center_y), 'conf': conf})

        return detections

tracker = MultiColorTracker()

detector = YOLOMarkerDetector(model_path='best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    detections = detector.detect(frame)

    tracker.update(detections)

    output = tracker.draw(frame.copy())
    cv2.imshow("YOLO Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()