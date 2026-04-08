import cv2
from ultralytics import YOLO
import numpy as np
import random


class MultiPersonTracker:
    def __init__(self):
        self.model = YOLO('yolo26n-pose.pt')
        self.id_colors = {}
        self.people_data = {}
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def get_color(self, track_id):
        if track_id not in self.id_colors:
            hue = random.randint(0, 179)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            self.id_colors[track_id] = tuple(map(int, color))
        return self.id_colors[track_id]

    def process_frame(self, frame):
        output = frame.copy()
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes.id is None:
            return output

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confidences = results[0].keypoints.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            track_id = int(track_ids[i])
            color = self.get_color(track_id)

            if track_id not in self.people_data:
                self.people_data[track_id] = {
                    'history': {kp: [] for kp in range(17)}
                }

            kpts = keypoints[i]
            confs = confidences[i]

            for bone in self.skeleton:
                pt1_idx, pt2_idx = bone
                if confs[pt1_idx] > 0.5 and confs[pt2_idx] > 0.5:
                    pt1 = tuple(map(int, kpts[pt1_idx]))
                    pt2 = tuple(map(int, kpts[pt2_idx]))
                    cv2.line(output, pt1, pt2, color, 2)

            for kp_idx in range(17):
                if confs[kp_idx] > 0.5:
                    x, y = map(int, kpts[kp_idx])

                    history = self.people_data[track_id]['history'][kp_idx]
                    history.append((x, y))
                    if len(history) > 30:
                        history.pop(0)

                    cv2.circle(output, (x, y), 5, color, -1)

                    for j in range(1, len(history)):
                        pt_prev = history[j - 1]
                        pt_curr = history[j]
                        dist = np.hypot(pt_curr[0] - pt_prev[0], pt_curr[1] - pt_prev[1])
                        if dist < 100:
                            alpha = j / len(history)
                            cv2.line(output, pt_prev, pt_curr, color, 1)

            x1, y1, x2, y2 = map(int, box)
            cv2.putText(output, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output


cap = cv2.VideoCapture(0)
tracker = MultiPersonTracker()


while True:
    ret, frame = cap.read()
    if not ret: break

    result = tracker.process_frame(frame)
    cv2.putText(result, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Multi-Person Skeleton Tracking', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()