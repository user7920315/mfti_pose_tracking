import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ColorMarkerTracker:

    def __init__(self, min_area=100, max_dist=50):
        self.min_area = min_area
        self.max_dist = max_dist
        self.prev_centers = {}
        self.next_id = 0

    def get_color_mask(self, frame, h_min, h_max, s_min, s_max, v_min, v_max):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if h_min <= h_max:
            mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        else:
            mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max]))
            mask2 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([180, s_max, v_max]))
            mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def process_frame(self, frame, h_min, h_max, s_min, s_max, v_min, v_max):
        mask = self.get_color_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_markers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area: continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue

            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            (x, y), r = cv2.minEnclosingCircle(cnt)
            current_markers.append({'center': (int(x), int(y)), 'radius': int(r), 'area': area})

        markers = self._assign_ids(current_markers)
        return markers, mask

    def _assign_ids(self, markers):
        curr_map = {m['center']: m for m in markers}
        used = set()
        for pid, prev_pos in list(self.prev_centers.items()):
            best_c, min_d = None, float('inf')
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


class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Универсальный Трекер Цветов")
        self.root.geometry("1050x650")
        self.root.minsize(900, 500)

        self.tracker = ColorMarkerTracker(min_area=150, max_dist=60)
        self.cap = cv2.VideoCapture(0)
        self.history = {}
        self.running = True

        self._build_ui()
        self._update_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=320)

        self.video_frame = ttk.LabelFrame(main, text="Видеопоток", padding=5)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.LabelFrame(main, text="Настройки", padding=10)
        ctrl.grid(row=0, column=1, sticky="ns", padx=5, pady=5)

        ttk.Label(ctrl, text="Быстрые пресеты:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        presets_frame = ttk.Frame(ctrl)
        presets_frame.pack(fill=tk.X, pady=5)

        presets = {
            "Красный": (0, 15, 120, 255, 120, 255),
            "Зеленый": (40, 70, 100, 255, 100, 255),
            "Синий": (100, 130, 150, 255, 150, 255),
            "Желтый": (20, 40, 150, 255, 150, 255),
            "Оранжевый": (10, 25, 150, 255, 150, 255),
            "Фиолетовый": (130, 160, 100, 255, 100, 255)
        }
        for name, val in presets.items():
            ttk.Button(presets_frame, text=name, command=lambda v=val: self._set_preset(v)).pack(side=tk.LEFT, padx=2,
                                                                                                 pady=2, fill=tk.X,
                                                                                                 expand=True)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)


        sliders_frame = ttk.Frame(ctrl)
        sliders_frame.pack(fill=tk.X, pady=5)
        sliders_frame.columnconfigure(1, weight=1)

        self.vars = {}
        for row, (name, default, mx) in enumerate([
            ("H Min", 0, 179), ("H Max", 15, 179),
            ("S Min", 120, 255), ("S Max", 255, 255),
            ("V Min", 120, 255), ("V Max", 255, 255)
        ]):
            ttk.Label(sliders_frame, text=name).grid(row=row, column=0, sticky=tk.W, pady=3)
            v = tk.IntVar(value=default)
            ttk.Scale(sliders_frame, from_=0, to=mx, variable=v, orient=tk.HORIZONTAL).grid(row=row, column=1,
                                                                                            sticky=tk.EW, padx=5)
            ttk.Label(sliders_frame, textvariable=v, width=3).grid(row=row, column=2)
            self.vars[name] = v



        ttk.Label(sliders_frame, text="Min Area").grid(row=6, column=0, sticky=tk.W, pady=3)
        self.vars["Area"] = tk.IntVar(value=150)
        ttk.Scale(sliders_frame, from_=50, to=1000, variable=self.vars["Area"], orient=tk.HORIZONTAL).grid(row=6,
                                                                                                           column=1,
                                                                                                           sticky=tk.EW,
                                                                                                           padx=5)
        ttk.Label(sliders_frame, textvariable=self.vars["Area"], width=3).grid(row=6, column=2)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.color_preview = tk.Label(ctrl, width=20, height=1, relief=tk.SUNKEN, background="#000000")
        self.color_preview.pack(fill=tk.X, pady=5)


        self.info = ttk.Label(ctrl, text="Камера инициализируется...", wraplength=300, justify=tk.LEFT)
        self.info.pack(pady=5)



        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        ttk.Button(btn_frame, text="Сбросить ID", command=self._reset_ids).pack(side=tk.LEFT, padx=2, fill=tk.X,
                                                                                  expand=True)
        ttk.Button(btn_frame, text="Сохранить PNG", command=self._save_frame).pack(side=tk.LEFT, padx=2, fill=tk.X,
                                                                                     expand=True)

    def _set_preset(self, vals):
        keys = ["H Min", "H Max", "S Min", "S Max", "V Min", "V Max"]
        for k, v in zip(keys, vals):
            self.vars[k].set(v)

    def _get_params(self):
        return {k: self.vars[k].get() for k in self.vars}

    def _update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.info.config(text="Камера недоступна")
        else:
            p = self._get_params()
            self.tracker.min_area = p["Area"]
            markers, mask = self.tracker.process_frame(frame, p["H Min"], p["H Max"], p["S Min"], p["S Max"],
                                                       p["V Min"], p["V Max"])

            for m in markers:
                x, y, r, mid = m['center'][0], m['center'][1], m['radius'], m['id']
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{mid}", (x - 20, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self.history.setdefault(mid, []).append((x, y))
                if len(self.history[mid]) > 150: self.history[mid].pop(0)
                pts = np.array(self.history[mid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 165, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            h, s, v = (p["H Min"] + p["H Max"]) // 2, (p["S Min"] + p["S Max"]) // 2, 200
            bgr_color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
            hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
            self.color_preview.config(background=hex_color)

            self.info.config(
                text=f"Маркеров: {len(markers)}\nHSV: {p['H Min']}-{p['H Max']} | S: {p['S Min']}-{p['S Max']}")

        if self.running:
            self.root.after(30, self._update_loop)

    def _reset_ids(self):
        self.tracker.prev_centers = {}
        self.tracker.next_id = 0
        self.history = {}
        self.info.config(text="История и ID сброшены")

    def _save_frame(self):
        ret, frame = self.cap.read()
        if ret:
            fname = f"capture_{int(cv2.getTickCount())}.png"
            cv2.imwrite(fname, frame)
            self.info.config(text=f"Сохранено: {fname}")

    def _on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()