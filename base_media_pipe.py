import os
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import urllib.request


def get_model_path(task_name, url):
    if not os.path.exists(task_name):
        urllib.request.urlretrieve(url, task_name)
    return task_name


class FullBodyHandTracker:
    def __init__(self):
        pose_path = get_model_path("pose_landmarker_lite.task",
                                   "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

        hand_path = get_model_path("hand_landmarker.task",
                                   "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=pose_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=2
            )
        )

        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=hand_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2
            )
        )

        self.smooth_coords = {i: [0.0, 0.0] for i in range(142)}
        self.trajectories = {i: deque(maxlen=25) for i in range(142)}
        self.smooth_alpha = 0.2

        self.body_connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (0, 4), (5, 4),(8,6), (7,3), (9, 10),
            (5, 6),  (11, 12),
            (11, 13), (12, 14),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def get_smooth_point(self, landmark, w, h, idx):
        raw_x = landmark.x * w
        raw_y = landmark.y * h

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
                    idx1, idx2 = conn
                    if idx1 < len(pose) and idx2 < len(pose):
                        p1, p2 = pose[idx1], pose[idx2]
                        if p1.visibility > 0.4 and p2.visibility > 0.4:
                            pt1 = self.get_smooth_point(p1, w, h, idx1)
                            pt2 = self.get_smooth_point(p2, w, h, idx2)
                            cv2.line(output, pt1, pt2, (200, 200, 200), 2)


                for i in range(33):
                    if i in [15, 16]: continue
                    if pose[i].visibility > 0.4:
                        x, y = self.get_smooth_point(pose[i], w, h, i)
                        cv2.circle(output, (x, y), 6, (0, 0, 255), -1)
                        self.trajectories[i].append((x, y))
                        self.draw_trail(output, i)


        hand_result = self.hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks:
            for idx, hand in enumerate(hand_result.hand_landmarks):
                color = (0, 0, 255)

                for conn in self.hand_connections:
                    i1, i2 = conn
                    p1, p2 = hand[i1], hand[i2]
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(output, pt1, pt2, (255, 255, 255), 2)

                for i, lm in enumerate(hand):
                    hand_idx = 100 + idx * 21 + i

                    x, y = self.get_smooth_point(lm, w, h, hand_idx)

                    cv2.circle(output, (x, y), 5, color, -1)

                    self.trajectories[hand_idx].append((x, y))
                    self.draw_trail(output, hand_idx)

        return output

    def draw_trail(self, frame, idx):
        hist = list(self.trajectories[idx])
        if len(hist) > 1:
            for j in range(1, len(hist)):
                if np.hypot(hist[j][0] - hist[j - 1][0], hist[j][1] - hist[j - 1][1]) < 50:
                    cv2.line(frame, hist[j - 1], hist[j], (0, 255, 0), 1)


cap = cv2.VideoCapture(0)
tracker = FullBodyHandTracker()

while True:
    ret, frame = cap.read()
    if not ret: break

    result = tracker.process_frame(frame)
    cv2.putText(result, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Professional MoCap', result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()