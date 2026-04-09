import cv2
import numpy as np


class RedMarkerTracker:
    def __init__(self, min_area=100, max_dist=50):
        self.min_area = min_area
        self.max_dist = max_dist
        self.prev_centers = {}
        self.next_id = 0

    def get_red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 120, 120])
        upper1 = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        lower2 = np.array([160, 120, 120])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def process_frame(self, frame):
        mask = self.get_red_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            (x, y), r = cv2.minEnclosingCircle(cnt)
            current_markers.append({
                'center': (int(x), int(y)),
                'radius': int(r),
                'area': area
            })


        markers = self._assign_ids(current_markers)
        return markers, mask

    def _assign_ids(self, markers):
        curr_map = {m['center']: m for m in markers}
        used = set()

        for pid, prev_pos in list(self.prev_centers.items()):
            best_c = None
            min_d = float('inf')
            for c, m in curr_map.items():
                if c in used: continue
                d = np.hypot(c[0] - prev_pos[0], c[1] - prev_pos[1])
                if d < min_d and d < self.max_dist:
                    min_d, best_c = d, c

            if best_c:
                curr_map[best_c]['id'] = pid
                used.add(best_c)
                self.prev_centers[pid] = best_c

        for c, m in curr_map.items():
            if c not in used:
                m['id'] = self.next_id
                self.prev_centers[self.next_id] = c
                self.next_id += 1

        return list(curr_map.values())


tracker = RedMarkerTracker(min_area=150, max_dist=60)
history = {}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    markers, mask = tracker.process_frame(frame)

    for m in markers:
        mid, (x, y), r = m['id'], m['center'], m['radius']

        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{mid}", (x - 20, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        history.setdefault(mid, []).append((x, y))
        if len(history[mid]) > 150:
            history[mid].pop(0)

        pts = np.array(history[mid], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 165, 255), 2)

    cv2.imshow('Red Markers', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()




















'''import cv2
import numpy as np
from collections import deque
import time



class StableMarker:
    def __init__(self, marker_id, center, color_range_name=None):
        self.id = marker_id
        self.center = center
        self.raw_center = center
        self.history = deque(maxlen=150)
        self.history.append((center[0], center[1], time.time()))

        self.missed_frames = 0
        self.max_missed_frames = 15

        self.is_active = True
        self.color_name = color_range_name

        self.smoothing_alpha = 0.3

    def update(self, new_center):
        self.raw_center = new_center
        self.missed_frames = 0
        self.is_active = True

        smoothed_x = self.center[0] * (1 - self.smoothing_alpha) + new_center[0] * self.smoothing_alpha
        smoothed_y = self.center[1] * (1 - self.smoothing_alpha) + new_center[1] * self.smoothing_alpha

        self.center = (smoothed_x, smoothed_y)
        self.history.append((self.center[0], self.center[1], time.time()))

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > self.max_missed_frames:
            self.is_active = False


class MultiColorTracker:
    def __init__(self):
        self.markers = {}
        self.next_id = 0

        self.color_ranges = {
            'red': ([0, 100, 100], [15, 255, 255]),
            'blue': ([100, 150, 150], [130, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255]),
        }

        self.color_bgr = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
        }

    def detect_blobs(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []

        for name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 10000:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        detections.append({'center': (cX, cY), 'color': name, 'area': area})

        return detections

    def update(self, detections):
        for marker in self.markers.values():
            marker.mark_missed()

        used_detections_indices = set()
        used_marker_ids = set()

        active_markers = [m for m in self.markers.values() if m.is_active or m.missed_frames < m.max_missed_frames]

        matches = []
        for i, det in enumerate(detections):
            best_marker = None
            min_dist = float('inf')

            for marker in self.markers.values():
                if not marker.is_active and marker.missed_frames >= marker.max_missed_frames:
                    continue

                if marker.id in used_marker_ids:
                    continue

                dist = np.hypot(det['center'][0] - marker.center[0],
                                det['center'][1] - marker.center[1])

                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    best_marker = marker

            if best_marker:
                matches.append({'det_idx': i, 'marker': best_marker})
                used_marker_ids.add(best_marker.id)
                used_detections_indices.add(i)

        for match in matches:
            match['marker'].update(detections[match['det_idx']]['center'])
            match['marker'].is_active = True

        for i, det in enumerate(detections):
            if i not in used_detections_indices:
                new_id = self.next_id
                self.next_id += 1
                self.markers[new_id] = StableMarker(new_id, det['center'], det['color'])

        dead_ids = [mid for mid, m in self.markers.items() if
                    not m.is_active and m.missed_frames >= m.max_missed_frames]
        for mid in dead_ids:
            del self.markers[mid]

    def draw(self, frame):
        for marker in self.markers.values():
            x, y = int(marker.center[0]), int(marker.center[1])

            if marker.is_active:
                color = self.color_bgr.get(marker.color_name, (255, 255, 255))
                thickness = -1
            else:
                color = (100, 100, 100)
                thickness = 1

            history_points = [(int(p[0]), int(p[1])) for p in marker.history]
            for i in range(1, len(history_points)):
                cv2.line(frame, history_points[i - 1], history_points[i], color, 2)

            cv2.circle(frame, (x, y), 8, color, thickness)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)

            status = "Active" if marker.is_active else "Ghost"
            cv2.putText(frame, f"ID:{marker.id}", (x - 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


cap = cv2.VideoCapture(0)
tracker = MultiColorTracker()

while True:
    ret, frame = cap.read()
    if not ret: break

    detections = tracker.detect_blobs(frame)

    tracker.update(detections)

    output = tracker.draw(frame.copy())

    cv2.imshow("Stable Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''