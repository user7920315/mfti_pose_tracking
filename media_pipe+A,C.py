import os
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import urllib.request


class TrajectoryPredictor:
    def __init__(self, history_length=10, prediction_horizon=5):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon

    def predict_marker(self, trajectory, n_steps=5):
        if len(trajectory) < 3:
            return []

        points = np.array(trajectory[-self.history_length:])
        if len(points) < 2:
            return []

        velocities = np.diff(points, axis=0)
        avg_velocity = np.mean(velocities, axis=0)

        predictions = []
        last_point = points[-1]

        for i in range(n_steps):
            next_point = last_point + avg_velocity * (i + 1)
            predictions.append(next_point)

        return predictions


class MotionClassifier:
    def __init__(self):
        self.window_size = 50

    def classify_simple(self, trajectories):
        if 25 not in trajectories or 26 not in trajectories:
            return 'static', 0.0

        knee_left = list(trajectories[25])[-20:]
        knee_right = list(trajectories[26])[-20:]

        if len(knee_left) < 10 or len(knee_right) < 10:
            return 'static', 0.0

        knee_left = np.array(knee_left)
        knee_right = np.array(knee_right)

        vertical_left = np.max(knee_left[:, 1]) - np.min(knee_left[:, 1])
        vertical_right = np.max(knee_right[:, 1]) - np.min(knee_right[:, 1])
        vertical_movement = max(vertical_left, vertical_right)

        all_knees = np.vstack([knee_left, knee_right])
        if len(all_knees) > 1:
            velocities = np.diff(all_knees, axis=0)
            avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
        else:
            avg_speed = 0

        avg_y = np.mean(all_knees[:, 1])

        if vertical_movement > 80 and avg_speed > 8:
            return 'jump', 0.85
        elif vertical_movement > 40 and avg_y > 250:
            return 'squat', 0.80
        elif avg_speed > 5 and vertical_movement < 40:
            return 'step', 0.75
        elif avg_speed > 3:
            return 'wave_hand', 0.70
        else:
            return 'static', 0.90


class FullBodyHandTracker:
    def __init__(self):
        self.pose_path = self.download_model("pose_landmarker_lite.task",
                                             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        self.hand_path = self.download_model("hand_landmarker.task",
                                             "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.pose_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE
            )
        )

        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2
            )
        )

        self.smooth_coords = {i: [0.0, 0.0] for i in range(142)}
        self.trajectories = {i: deque(maxlen=50) for i in range(142)}
        self.smooth_alpha = 0.2

        self.classifier = MotionClassifier()
        self.predictor = TrajectoryPredictor(history_length=10, prediction_horizon=5)

        self.body_connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (12, 14), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]


        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]


    def download_model(self, name, url):
        if not os.path.exists(name):
            urllib.request.urlretrieve(url, name)
        return name

    def get_smooth_point(self, landmark, w, h, idx):
        raw_x, raw_y = landmark.x * w, landmark.y * h
        if self.smooth_coords[idx] == [0.0, 0.0]:
            self.smooth_coords[idx] = [raw_x, raw_y]
        sm_x = self.smooth_coords[idx][0] * (1 - self.smooth_alpha) + raw_x * self.smooth_alpha
        sm_y = self.smooth_coords[idx][1] * (1 - self.smooth_alpha) + raw_y * self.smooth_alpha
        self.smooth_coords[idx] = [sm_x, sm_y]
        return int(sm_x), int(sm_y)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        output = frame.copy()
        h, w, _ = frame.shape

        pose_result = self.pose_landmarker.detect(mp_image)
        if pose_result.pose_landmarks:
            for pose in pose_result.pose_landmarks:
                for conn in self.body_connections:
                    if all(idx < len(pose) for idx in conn):
                        p1, p2 = pose[conn[0]], pose[conn[1]]
                        if p1.visibility > 0.4 and p2.visibility > 0.4:
                            cv2.line(output,
                                     self.get_smooth_point(p1, w, h, conn[0]),
                                     self.get_smooth_point(p2, w, h, conn[1]),
                                     (200, 200, 200), 2)

                for i in range(33):
                    if i in [15, 16]: continue
                    if pose[i].visibility > 0.4:
                        x, y = self.get_smooth_point(pose[i], w, h, i)
                        cv2.circle(output, (x, y), 6, (0, 0, 255), -1)
                        self.trajectories[i].append((x, y))

        hand_result = self.hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks:
            for idx, hand in enumerate(hand_result.hand_landmarks):
                color = (0, 0, 255)

                for conn in self.hand_connections:
                    pt1 = (int(hand[conn[0]].x * w), int(hand[conn[0]].y * h))
                    pt2 = (int(hand[conn[1]].x * w), int(hand[conn[1]].y * h))
                    cv2.line(output, pt1, pt2, (255, 255, 255), 2)

                for i, lm in enumerate(hand):
                    hand_idx = 100 + idx * 21 + i
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(output, (x, y), 5, color, -1)
                    self.trajectories[hand_idx].append((x, y))


        motion_class, confidence = self.classifier.classify_simple(self.trajectories)
        cv2.putText(output, f"Motion: {motion_class}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, f"Conf: {confidence:.2f}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        if 25 in self.trajectories and len(self.trajectories[25]) > 10:
            predictions = self.predictor.predict_marker(list(self.trajectories[25]), n_steps=5)

            if predictions:
                last_actual = self.trajectories[25][-1]
                for i, pred in enumerate(predictions):
                    pred_point = (int(pred[0]), int(pred[1]))
                    cv2.circle(output, pred_point, 4, (0, 255, 255), -1)
                    if i == 0:
                        cv2.line(output, last_actual, pred_point, (0, 255, 255), 2)

        return output



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = FullBodyHandTracker()


    while True:
        ret, frame = cap.read()
        if not ret: break

        result = tracker.process_frame(frame)
        cv2.putText(result, "Press 'q' to quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('MoCap + A + C', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
