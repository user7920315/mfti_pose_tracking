import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq


class Point:
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class LandmarkList:
    def __init__(self, points):
        self.landmark = points


class UniversalPoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        rgb.flags.writeable = True
        if not res.pose_landmarks:
            return None
        return LandmarkList([
            Point(lm.x, lm.y, lm.z, lm.visibility)
            for lm in res.pose_landmarks.landmark])

    def close(self):
        try:
            self.pose.close()
        except:
            pass


class MP:
    NOSE = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_ELBOW = 13
    R_ELBOW = 14
    L_WRIST = 15
    R_WRIST = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANKLE = 27
    R_ANKLE = 28
    L_HEEL = 29
    R_HEEL = 30
    L_FOOT = 31
    R_FOOT = 32

    UPPER_BODY = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    LOWER_BODY = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    HANDS = [L_WRIST, R_WRIST]
    TORSO = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


class PoseNormalizer:
    def __init__(self):
        self.scale_buf = deque(maxlen=30)
        self.cx_buf = deque(maxlen=30)
        self.cy_buf = deque(maxlen=30)

    def get_torso_params(self, lm):
        def pt(idx):
            p = lm[idx]
            return (p.x, p.y) if p.visibility > 0.15 else None

        ls = pt(MP.L_SHOULDER)
        rs = pt(MP.R_SHOULDER)
        lh = pt(MP.L_HIP)
        rh = pt(MP.R_HIP)

        if ls and rs and lh and rh:
            sho_cx, sho_cy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
            hip_cx, hip_cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            scale = float(np.hypot(sho_cx - hip_cx, sho_cy - hip_cy))
            cx, cy = (sho_cx + hip_cx) / 2, (sho_cy + hip_cy) / 2
        elif lh and rh:
            cx, cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            scale = float(abs(lh[0] - rh[0])) * 1.5
        elif ls and rs:
            cx, cy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
            scale = float(abs(ls[0] - rs[0])) * 1.5
        else:
            return 0.5, 0.5, 0.25

        if scale > 0.01:
            self.scale_buf.append(scale)
            self.cx_buf.append(cx)
            self.cy_buf.append(cy)

        s = float(np.median(list(self.scale_buf))) if self.scale_buf else 0.25
        cx = float(np.median(list(self.cx_buf))) if self.cx_buf else 0.5
        cy = float(np.median(list(self.cy_buf))) if self.cy_buf else 0.5
        return cx, cy, s

    def normalize(self, lm_list):
        lm = lm_list.landmark
        cx, cy, scale = self.get_torso_params(lm)
        norm_dict = {
            idx: (
                float((pt.x - cx) / scale),
                float((pt.y - cy) / scale),
                float(pt.visibility)
            ) for idx, pt in enumerate(lm)
        }
        return norm_dict, scale

    def reset(self):
        self.scale_buf.clear()
        self.cx_buf.clear()
        self.cy_buf.clear()


class TrajectoryBuffer:
    def __init__(self, window=60, fps=30.0):
        self.window = window
        self.fps = fps
        self.dt = 1.0 / fps
        self.data = {}
        self.n_frames = 0
        self.scale_history = deque(maxlen=window)

    def add_frame(self, norm_dict, scale=0.0):
        self.n_frames += 1
        self.scale_history.append(scale)
        for mid, (rx, ry, vis) in norm_dict.items():
            if mid not in self.data:
                self.data[mid] = deque(maxlen=self.window)
            self.data[mid].append((float(rx), float(ry), float(vis)))

    def vis(self, mid):
        if mid not in self.data or not self.data[mid]:
            return 0.0
        return float(np.mean([d[2] for d in list(self.data[mid])[-10:]]))

    def visible(self, mid, thr=0.3):
        return self.vis(mid) >= thr

    def get_xy(self, mid, min_vis=0.2, last_n=0):
        if mid not in self.data:
            return None
        raw = list(self.data[mid])[-last_n:] if last_n > 0 else list(self.data[mid])
        pts = [(d[0], d[1]) for d in raw if d[2] >= min_vis]
        if len(pts) < 5:
            return None
        return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))

    def get_xy_aligned(self, mids, min_vis=0.2, last_n=0):
        results = {mid: self.get_xy(mid, min_vis, last_n=last_n) for mid in mids}
        if any(v is None for v in results.values()):
            return None
        min_len = min(len(v[0]) for v in results.values())
        if min_len < 5:
            return None
        return {mid: (x[-min_len:], y[-min_len:]) for mid, (x, y) in results.items()}

    def get_raw(self, mid, last_n=0):
        if mid not in self.data:
            return None
        raw = list(self.data[mid])[-last_n:] if last_n > 0 else list(self.data[mid])
        if len(raw) < 5:
            return None
        return (np.array([d[0] for d in raw]), np.array([d[1] for d in raw]), np.array([d[2] for d in raw]))

    def group_visible_ratio(self, group, thr=0.3):
        return sum(1 for m in group if self.visible(m, thr)) / max(len(group), 1)

    def filled(self, ratio=0.5):
        return self.n_frames >= self.window * ratio


    def get_scale_change(self):
        sc = list(self.scale_history)
        if len(sc) < 5:
            return 0.0
        arr = np.array(sc, dtype=float)
        mn = arr.mean()
        return float(arr.std() / (mn + 1e-9))

    def reset(self):
        self.data.clear()
        self.n_frames = 0
        self.scale_history.clear()


class Kin:
    def __init__(self, fps=30.0):
        self.dt = 1.0 / fps

    @staticmethod
    def smooth(a, w=9):
        n = len(a)
        if n < 5:
            return a.copy()
        w = min(w, n - (n % 2 == 0))
        if w < 3:
            return a.copy()
        try:
            return savgol_filter(a, w, min(3, w - 1))
        except:
            return a.copy()

    def speed(self, x, y):
        return np.sqrt(np.gradient(x, self.dt) ** 2 + np.gradient(y, self.dt) ** 2)

    def band_power(self, sig, fmin, fmax):
        n = len(sig)
        if n < 12:
            return 0.0
        s = sig - sig.mean()
        if s.std() < 1e-9:
            return 0.0
        pwr = np.abs(rfft(s)) ** 2
        fr = rfftfreq(n, d=self.dt)
        if (total_pwr := pwr.sum()) < 1e-12:
            return 0.0
        mask = (fr >= fmin) & (fr <= fmax)
        return float(pwr[mask].sum() / total_pwr)

    def periodicity(self, sig):
        if len(sig) < 16:
            return 0.0
        s = sig - sig.mean()
        if s.std() < 1e-9:
            return 0.0
        ac = np.correlate(s, s, 'full')[len(s) - 1:]
        ac /= (ac[0] + 1e-9)
        for i in range(2, len(ac) - 1):
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.08:
                return float(ac[i])
        return 0.0

    def corr(self, a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def angle3(self, ax, ay, bx, by, cx, cy):
        vax, vay = ax - bx, ay - by
        vcx, vcy = cx - bx, cy - by
        d = vax * vcx + vay * vcy
        na, nc = np.sqrt(vax ** 2 + vay ** 2) + 1e-9, np.sqrt(vcx ** 2 + vcy ** 2) + 1e-9
        return float(np.degrees(np.arccos(np.clip(d / (na * nc), -1, 1))))

    def angle_history(self, buf, idx_a, idx_b, idx_c, min_vis=0.3, last_n=0):
        ra, rb, rc = buf.get_raw(idx_a, last_n), buf.get_raw(idx_b, last_n), buf.get_raw(idx_c, last_n)
        if ra is None or rb is None or rc is None:
            return None

        n = min(len(ra[0]), len(rb[0]), len(rc[0]))
        ok = (ra[2][:n] >= min_vis) & (rb[2][:n] >= min_vis) & (rc[2][:n] >= min_vis)

        if ok.sum() < max(4, n // 3):
            return None

        angles = []
        for i in range(n):
            if ok[i]:
                ang = self.angle3(ra[0][i], ra[1][i], rb[0][i], rb[1][i], rc[0][i], rc[1][i])
                angles.append(ang)

        if len(angles) < 5:
            return None

        idx_valid = np.where(ok)[0]
        angles_interp = np.interp(np.arange(n), idx_valid, angles)

        return self.smooth(angles_interp)

    def interp_vis(self, raw_tuple, min_vis=0.2):
        x_r, y_r, vis_r = raw_tuple
        mask = vis_r >= min_vis
        if mask.sum() < 4:
            return None, None
        idx = np.where(mask)[0]
        xi = np.interp(np.arange(len(x_r)), idx, x_r[mask])
        yi = np.interp(np.arange(len(y_r)), idx, y_r[mask])
        return xi, yi

    def rel_diff_x(self, buf, mid_a, mid_b, min_vis=0.20, last_n=0, w_smooth=9):
        ra, rb = buf.get_raw(mid_a, last_n), buf.get_raw(mid_b, last_n)
        if ra is None or rb is None:
            return None
        n = min(len(ra[0]), len(rb[0]))
        ok = (ra[2][:n] >= min_vis) & (rb[2][:n] >= min_vis)
        if ok.sum() < max(4, n // 3):
            return None
        idx = np.where(ok)[0]
        dv = ra[0][:n][ok] - rb[0][:n][ok]
        d = np.interp(np.arange(n), idx, dv)
        return self.smooth(d, w=w_smooth)


@dataclass
class Features:
    vis_upper: float = 0.0
    vis_lower: float = 0.0
    vis_hands: float = 0.0
    scale_change: float = 0.0
    torso_scale: float = 0.25
    body_speed: float = 0.0

    ankle_dx_range: float = 0.0
    ankle_dx_bp: float = 0.0
    ankle_dx_period: float = 0.0
    ankle_dx_freq: float = 0.0
    ankle_speed_l: float = 0.0
    ankle_speed_r: float = 0.0
    ankle_corr: float = 0.0
    rel_x_l_range: float = 0.0
    rel_x_r_range: float = 0.0
    rel_x_l_bp: float = 0.0
    rel_x_r_bp: float = 0.0

    torso_y_velocity_max: float = 0.0
    torso_y_acceleration_max: float = 0.0
    knee_y_velocity_max: float = 0.0
    ankle_y_velocity_max: float = 0.0
    vertical_impulse: float = 0.0
    has_takeoff_landing: bool = False
    knee_angle_velocity_max: float = 0.0
    knee_extension_range: float = 0.0

    knee_angle_current: float = 180.0
    knee_angle_min: float = 180.0
    knee_angle_range: float = 0.0
    knee_bend_duration: float = 0.0
    is_knees_bent: bool = False
    hip_y_std: float = 0.0
    torso_y_range: float = 0.0

    lr_symmetry: float = 0.0

    wave_dist_l: float = 0.0
    wave_dist_r: float = 0.0
    wave_range_l: float = 0.0
    wave_range_r: float = 0.0
    wave_bp_l: float = 0.0
    wave_bp_r: float = 0.0
    wave_speed_l: float = 0.0
    wave_speed_r: float = 0.0
    shoulder_speed: float = 0.0


class FeatureExtractor:
    def __init__(self, fps=30.0):
        self.kin = Kin(fps)
        self.jw = max(20, int(fps * 0.8))
        self.sw = max(30, int(fps * 1.5))

    def spd(self, buf, mid):
        r = buf.get_xy(mid, min_vis=0.15)
        if r is None:
            return 0.0
        return float(self.kin.speed(self.kin.smooth(r[0]), self.kin.smooth(r[1])).mean())

    def compute(self, buf):
        ft = Features()
        ft.vis_upper = buf.group_visible_ratio(MP.UPPER_BODY)
        ft.vis_lower = buf.group_visible_ratio(MP.LOWER_BODY)
        ft.vis_hands = buf.group_visible_ratio(MP.HANDS)
        ft.scale_change = buf.get_scale_change()
        if buf.scale_history:
            ft.torso_scale = float(np.median(list(buf.scale_history)))

        spds = [self.spd(buf, m) for m in MP.UPPER_BODY + MP.LOWER_BODY]
        if any(s > 0 for s in spds):
            ft.body_speed = float(np.mean([s for s in spds if s > 0]))

        self.step_features(buf, ft)
        self.jump_features(buf, ft)
        self.squat_features(buf, ft)
        self.wave_features(buf, ft)
        return ft

    def step_features(self, buf, ft):
        kin = self.kin
        adx = kin.rel_diff_x(buf, MP.L_ANKLE, MP.R_ANKLE, min_vis=0.15)
        if adx is not None and len(adx) >= 10:
            ft.ankle_dx_range = float(adx.max() - adx.min())
            ft.ankle_dx_bp = kin.band_power(adx, 0.4, 3.0)
            ft.ankle_dx_period = kin.periodicity(adx)
        rl, rr = buf.get_raw(MP.L_ANKLE), buf.get_raw(MP.R_ANKLE)
        if rl is not None and rr is not None:
            n = min(len(rl[0]), len(rr[0]))
            if ((rl[2][:n] >= 0.15) & (rr[2][:n] >= 0.15)).sum() >= 8:
                ft.ankle_corr = kin.corr(kin.smooth(rl[0][:n]), kin.smooth(rr[0][:n]))
        ft.ankle_speed_l = self.spd(buf, MP.L_ANKLE)
        ft.ankle_speed_r = self.spd(buf, MP.R_ANKLE)
        for a, h, r_a, bp_a in [(MP.L_ANKLE, MP.L_HIP, 'rel_x_l_range', 'rel_x_l_bp'),
                                (MP.R_ANKLE, MP.R_HIP, 'rel_x_r_range', 'rel_x_r_bp')]:
            d = kin.rel_diff_x(buf, a, h, min_vis=0.15)
            if d is not None:
                setattr(ft, r_a, float(d.max() - d.min()))
                setattr(ft, bp_a, kin.band_power(d, 0.4, 3.0))

    def jump_features(self, buf, ft):
        kin, jw = self.kin, self.jw

        aligned = buf.get_xy_aligned(MP.TORSO, min_vis=0.25, last_n=jw)
        if aligned and len(next(iter(aligned.values()))[0]) >= 10:
            torso_y = kin.smooth(np.mean([aligned[m][1] for m in MP.TORSO], axis=0))
            vy = -np.gradient(torso_y, kin.dt)
            ft.torso_y_velocity_max = float(max(abs(vy.max()), abs(vy.min())))
            ay = np.gradient(vy, kin.dt)
            ft.torso_y_acceleration_max = float(max(abs(ay.max()), abs(ay.min())))
            ft.vertical_impulse = float(np.abs(vy).sum() * kin.dt)
            idx_up = np.argmax(vy)
            idx_down = np.argmin(vy)
            if idx_up < idx_down and (idx_down - idx_up) > 3:
                ft.has_takeoff_landing = True

        knee_ys = []
        for knee in [MP.L_KNEE, MP.R_KNEE]:
            rk = buf.get_raw(knee, last_n=jw)
            if rk is not None and (rk[2] >= 0.25).sum() > 5:
                ky = kin.smooth(rk[1])
                vy_k = -np.gradient(ky, kin.dt)
                knee_ys.append(float(max(abs(vy_k.max()), abs(vy_k.min()))))
        if knee_ys:
            ft.knee_y_velocity_max = float(np.mean(knee_ys))

        ankle_ys = []
        for ankle in [MP.L_ANKLE, MP.R_ANKLE]:
            ra = buf.get_raw(ankle, last_n=jw)
            if ra is not None and (ra[2] >= 0.20).sum() > 5:
                ay = kin.smooth(ra[1])
                vy_a = -np.gradient(ay, kin.dt)
                ankle_ys.append(float(max(abs(vy_a.max()), abs(vy_a.min()))))
        if ankle_ys:
            ft.ankle_y_velocity_max = float(np.mean(ankle_ys))

        angle_vels = []
        extension_ranges = []

        for hip, knee, ankle in [(MP.L_HIP, MP.L_KNEE, MP.L_ANKLE),
                                 (MP.R_HIP, MP.R_KNEE, MP.R_ANKLE)]:
            angles = kin.angle_history(buf, hip, knee, ankle, min_vis=0.25, last_n=jw)
            if angles is not None and len(angles) >= 10:
                angle_vel = np.abs(np.gradient(angles, kin.dt))
                angle_vels.append(float(angle_vel.max()))
                extension_ranges.append(float(angles.max() - angles.min()))

        if angle_vels:
            ft.knee_angle_velocity_max = float(np.mean(angle_vels))
        if extension_ranges:
            ft.knee_extension_range = float(np.mean(extension_ranges))

        rl = buf.get_raw(MP.L_KNEE, last_n=jw)
        rr = buf.get_raw(MP.R_KNEE, last_n=jw)
        if rl is not None and rr is not None:
            n = min(len(rl[1]), len(rr[1]))
            if ((rl[2][:n] >= 0.20) & (rr[2][:n] >= 0.20)).sum() >= 5:
                ft.lr_symmetry = abs(kin.corr(rl[1][:n], rr[1][:n]))

    def squat_features(self, buf, ft):
        kin, sw = self.kin, self.sw

        angles_all = []
        angles_recent = []

        for hip, knee, ankle in [(MP.L_HIP, MP.L_KNEE, MP.L_ANKLE),
                                 (MP.R_HIP, MP.R_KNEE, MP.R_ANKLE)]:
            angles = kin.angle_history(buf, hip, knee, ankle, min_vis=0.30, last_n=0)
            if angles is not None and len(angles) >= 10:
                angles_all.append(angles)
                ft.knee_angle_min = min(ft.knee_angle_min, float(angles.min()))
                ft.knee_angle_range = max(ft.knee_angle_range, float(angles.max() - angles.min()))

            angles_curr = kin.angle_history(buf, hip, knee, ankle, min_vis=0.30, last_n=10)
            if angles_curr is not None:
                angles_recent.append(float(angles_curr[-1]))

        if angles_recent:
            ft.knee_angle_current = float(np.mean(angles_recent))
            ft.is_knees_bent = ft.knee_angle_current < 140.0

        if len(angles_all) > 0:
            ranges = [float(a.max() - a.min()) for a in angles_all]
            ft.knee_angle_range = float(np.mean(ranges))

        if angles_all:
            bent_frames = sum((a < 140.0).sum() for a in angles_all) / len(angles_all)
            ft.knee_bend_duration = float(bent_frames * kin.dt)

        hip_ys = []
        for hip in [MP.L_HIP, MP.R_HIP]:
            rh = buf.get_raw(hip, last_n=sw)
            if rh is not None and (rh[2] >= 0.25).sum() > 10:
                hy = kin.smooth(rh[1])
                hip_ys.append(hy)

        if hip_ys:
            avg_hip_y = np.mean(hip_ys, axis=0)
            ft.hip_y_std = float(avg_hip_y.std())

        aligned = buf.get_xy_aligned(MP.TORSO, min_vis=0.25, last_n=sw)
        if aligned and len(next(iter(aligned.values()))[0]) >= 10:
            torso_y = kin.smooth(np.mean([aligned[m][1] for m in MP.TORSO], axis=0))
            ft.torso_y_range = float(torso_y.max() - torso_y.min())

        rl = buf.get_raw(MP.L_KNEE, last_n=sw)
        rr = buf.get_raw(MP.R_KNEE, last_n=sw)
        if rl is not None and rr is not None:
            n = min(len(rl[1]), len(rr[1]))
            if ((rl[2][:n] >= 0.25) & (rr[2][:n] >= 0.25)).sum() >= 10:
                ft.lr_symmetry = max(ft.lr_symmetry, abs(kin.corr(rl[1][:n], rr[1][:n])))

    def wave_features(self, buf, ft):
        kin = self.kin
        for w_mid, s_mid, d_a, r_a, bp_a, sp_a in [
            (MP.L_WRIST, MP.L_SHOULDER, 'wave_dist_l', 'wave_range_l', 'wave_bp_l', 'wave_speed_l'),
            (MP.R_WRIST, MP.R_SHOULDER, 'wave_dist_r', 'wave_range_r', 'wave_bp_r', 'wave_speed_r')]:
            rw, rs = buf.get_raw(w_mid), buf.get_raw(s_mid)
            if rw is None or rs is None:
                continue
            xw, yw = kin.interp_vis(rw, 0.12)
            xs, ys = kin.interp_vis(rs, 0.12)
            if xw is None or xs is None:
                continue
            xw, yw, xs, ys = kin.smooth(xw), kin.smooth(yw), kin.smooth(xs), kin.smooth(ys)
            n = min(len(xw), len(xs))
            ds = kin.smooth(np.sqrt((xw[:n] - xs[:n]) ** 2 + (yw[:n] - ys[:n]) ** 2))
            setattr(ft, d_a, float(ds.mean()))
            setattr(ft, r_a, float(ds.max() - ds.min()))
            setattr(ft, bp_a, kin.band_power(ds, 0.5, 4.0))
            setattr(ft, sp_a, float(kin.speed(xw[:n], yw[:n]).mean()))
        sps = [float(kin.speed(kin.smooth(r[0]), kin.smooth(r[1])).mean()) for mid in [MP.L_SHOULDER, MP.R_SHOULDER]
               if (r := buf.get_xy(mid, 0.15))]
        if sps:
            ft.shoulder_speed = float(np.mean(sps))


@dataclass
class ClassResult:
    name: str
    score: float
    confidence: float
    evidence: list = field(default_factory=list)


class MathClassifier:
    CLASSES = ['static', 'step', 'jump', 'squat', 'wave']
    THR = {
        'st_body_spd': 0.10,
        'step_dx_range': 0.20, 'step_dx_bp': 0.12, 'step_dx_period': 0.08, 'step_spd': 0.10,
        'jump_torso_vy': 1.5, 'jump_torso_ay': 8.0, 'jump_knee_vy': 1.0, 'jump_impulse': 0.8,
        'jump_angle_vel': 150.0, 'jump_min_vis': 0.40,
        'squat_angle_max': 140.0, 'squat_angle_min': 70.0, 'squat_angle_range': 25.0,
        'squat_torso_range_max': 0.15, 'squat_hip_std_max': 0.08, 'squat_min_vis': 0.40,
        'wave_dist': 0.80, 'wave_range': 0.20, 'wave_bp': 0.08, 'wave_min_vis': 0.25,
        'cam_thr': 0.05,
    }

    def __init__(self, thr=None):
        self.THR = self.__class__.THR.copy()
        if thr:
            self.THR.update(thr)

    @staticmethod
    def r(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)) if hi > lo else float(x >= lo)

    def cam_pen(self, ft):
        return float(np.clip(1.0 - self.r(ft.scale_change, self.THR['cam_thr'], self.THR['cam_thr'] * 5), 0.15, 1.0))

    def score_static(self, ft):
        ev, T = [], self.THR
        s1 = 1.0 - self.r(ft.body_speed, T['st_body_spd'] * 0.3, T['st_body_spd'] * 3.0)
        s2 = 1.0 - self.r(ft.ankle_dx_bp, 0.03, 0.25)
        s3 = 1.0 - self.r(ft.ankle_dx_range, 0.05, 0.40)
        s4 = 1.0 - self.r(ft.torso_y_velocity_max, 0.1, 0.8)
        sc = float(np.average([s1, s2, s3, s4], weights=[2.5, 3.5, 3.0, 2.0]))
        return float(np.clip(sc, 0, 1)), ev

    def score_step(self, ft):
        ev, T = [], self.THR
        if ft.ankle_dx_range < T['step_dx_range'] * 0.3 or ft.ankle_dx_bp < T['step_dx_bp'] * 0.25:
            return 0.02, ev
        s1 = self.r(ft.ankle_dx_range, T['step_dx_range'], T['step_dx_range'] * 4)
        s2 = self.r(ft.ankle_dx_bp, T['step_dx_bp'], T['step_dx_bp'] * 4)
        s3 = self.r(ft.ankle_dx_period, T['step_dx_period'] * 0.5, T['step_dx_period'] * 5)
        s4 = self.r(max(ft.ankle_speed_l, ft.ankle_speed_r), T['step_spd'] * 0.5, T['step_spd'] * 5)
        s5 = 1.0 - self.r(ft.torso_y_velocity_max, 0.5, 2.0)
        s6 = 1.0 - self.r(ft.knee_angle_current, 90, 140)
        sc = float(np.average([s1, s2, s3, s4, s5, s6], weights=[5, 5, 3, 3, 2, 2]))
        return float(np.clip(sc, 0, 1)), ev

    def score_jump(self, ft):
        ev, T, cam = [], self.THR, self.cam_pen(ft)

        if ft.vis_lower < T['jump_min_vis']:
            return 0.0, ev

        if not ft.has_takeoff_landing:
            return 0.01 * cam, ev

        if ft.torso_y_velocity_max < T['jump_torso_vy'] * 0.4:
            return 0.02 * cam, ev

        s_torso_vy = self.r(ft.torso_y_velocity_max, T['jump_torso_vy'], T['jump_torso_vy'] * 3)
        s_torso_ay = self.r(ft.torso_y_acceleration_max, T['jump_torso_ay'], T['jump_torso_ay'] * 3)
        s_knee_vy = self.r(ft.knee_y_velocity_max, T['jump_knee_vy'], T['jump_knee_vy'] * 3)
        s_impulse = self.r(ft.vertical_impulse, T['jump_impulse'], T['jump_impulse'] * 3)
        s_angle_vel = self.r(ft.knee_angle_velocity_max, T['jump_angle_vel'], T['jump_angle_vel'] * 3)
        s_extension = self.r(ft.knee_extension_range, 30.0, 80.0)
        s_sym = self.r(ft.lr_symmetry, 0.60, 0.95)
        s_no_step = 1.0 - self.r(ft.ankle_dx_bp, 0.1, 0.4)

        raw = float(np.average(
            [s_torso_vy, s_torso_ay, s_knee_vy, s_impulse, s_angle_vel, s_extension, s_sym, s_no_step],
            weights=[6.0, 5.0, 4.0, 4.0, 3.5, 2.5, 2.0, 2.0]
        ))

        sc = raw * cam
        return float(np.clip(sc, 0, 1)), ev

    def score_squat(self, ft):
        ev, T, cam = [], self.THR, self.cam_pen(ft)

        if ft.vis_lower < T['squat_min_vis']:
            return 0.0, ev

        if ft.knee_angle_current > T['squat_angle_max']:
            return 0.01 * cam, ev

        knee_bend = 180.0 - ft.knee_angle_min
        s_depth = self.r(knee_bend, 180.0 - T['squat_angle_max'], 180.0 - T['squat_angle_min'])
        s_current = self.r(180.0 - ft.knee_angle_current, 40.0, 110.0)
        s_range = self.r(ft.knee_angle_range, T['squat_angle_range'], T['squat_angle_range'] * 3)
        s_stable_torso = 1.0 - self.r(ft.torso_y_range, T['squat_torso_range_max'] * 0.5,
                                      T['squat_torso_range_max'] * 2)
        s_stable_hip = 1.0 - self.r(ft.hip_y_std, T['squat_hip_std_max'] * 0.5, T['squat_hip_std_max'] * 2)
        s_no_jump = 1.0 - self.r(ft.torso_y_velocity_max, 0.3, 1.5)
        s_no_step = 1.0 - self.r(ft.ankle_dx_range, 0.15, 0.5)
        s_sym = self.r(ft.lr_symmetry, 0.60, 0.95)

        raw = float(np.average(
            [s_depth, s_current, s_range, s_stable_torso, s_stable_hip, s_no_jump, s_no_step, s_sym],
            weights=[5.0, 5.0, 3.0, 4.0, 3.0, 4.0, 2.0, 2.0]
        ))

        sc = raw * cam
        return float(np.clip(sc, 0, 1)), ev

    def score_wave(self, ft):
        ev, T = [], self.THR
        if ft.vis_hands < T['wave_min_vis']:
            return 0.0, ev
        dist_max = max(ft.wave_dist_l, ft.wave_dist_r)
        range_max = max(ft.wave_range_l, ft.wave_range_r)
        bp_max = max(ft.wave_bp_l, ft.wave_bp_r)
        if dist_max < T['wave_dist'] * 0.5 or (range_max < T['wave_range'] * 0.3 and bp_max < T['wave_bp'] * 0.3):
            return 0.02, ev
        s1 = self.r(dist_max, T['wave_dist'] * 0.7, T['wave_dist'] * 2.5)
        s2 = self.r(range_max, T['wave_range'] * 0.5, T['wave_range'] * 4)
        s3 = self.r(bp_max, T['wave_bp'] * 0.4, T['wave_bp'] * 4)
        s4 = float(np.clip(1.0 - self.r(ft.shoulder_speed, 0.05, 0.30), 0, 1))
        sc = float(np.average([s1, s2, s3, s4], weights=[4.0, 5.5, 3.5, 2.0]))
        return float(np.clip(sc, 0, 1)), ev

    def classify(self, ft):
        raw = {c: getattr(self, f'score_{c}')(ft) for c in self.CLASSES}
        vals = np.array([raw[c][0] for c in self.CLASSES], dtype=float)
        ex = np.exp(np.clip(vals / 0.15, -15, 15))
        prob = ex / (ex.sum() + 1e-9)
        res = [ClassResult(c, raw[c][0], float(prob[i]), raw[c][1]) for i, c in enumerate(self.CLASSES)]
        return sorted(res, key=lambda x: x.confidence, reverse=True)


class PredictionSmoother:
    def __init__(self, n, history=5):
        self.n = n
        self.buf = deque(maxlen=history)

    def update(self, probs):
        self.buf.append(probs.copy())

    def get(self):
        if not self.buf:
            return np.ones(self.n) / self.n
        w = np.exp(np.linspace(-1.2, 0, len(self.buf)))
        return np.average(list(self.buf), axis=0, weights=w / w.sum())


CLASS_COLORS = {
    'static': (180, 180, 180),
    'step': (50, 205, 50),
    'jump': (0, 60, 255),
    'squat': (0, 165, 255),
    'wave': (255, 0, 200),
    '?': (100, 100, 100)
}


SKEL = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
        (26, 28)]


def draw_frame(frame, lm_list, results):
    h, w = frame.shape[:2]
    top = results[0]
    col = CLASS_COLORS.get(top.name, (255, 255, 0))

    if lm_list:
        lm = lm_list.landmark
        for a, b in SKEL:
            if lm[a].visibility > 0.3 and lm[b].visibility > 0.3:
                cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)), col, 2, cv2.LINE_AA)
        for p in lm:
            if p.visibility > 0.3:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, col, -1, cv2.LINE_AA)

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 160), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, top.name.upper(), (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.8, col, 3, cv2.LINE_AA)

    bx, by, bw = 20, 70, 300
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 18), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(top.confidence * bw), by + 18), col, -1)

    classes = MathClassifier.CLASSES
    cw = min(55, (w - 40) // max(len(classes), 1))
    for i, cls in enumerate(classes):
        cf = next((r.confidence for r in results if r.name == cls), 0.0)
        bx2, bh = 20 + i * cw, int(cf * 60)
        cc = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(frame, (bx2, 100), (bx2 + cw - 4, 160), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx2, 160 - bh), (bx2 + cw - 4, 160), cc, -1)
        cv2.putText(frame, cls[:5], (bx2, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


class MathPoseClassifier:
    def __init__(self, config=None):
        cfg = {
            'camera_id': 0,
            'frame_w': 1280,
            'frame_h': 720,
            'fps': 30.0,
            'window': 60,
            'mp_det': 0.5,
            'mp_track': 0.5,
            'mp_complex': 1,
            'thresholds': {}
        }
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.normalizer = PoseNormalizer()
        self.buf = TrajectoryBuffer(cfg['window'], cfg['fps'])
        self.extractor = FeatureExtractor(cfg['fps'])
        self.classifier = MathClassifier(cfg['thresholds'])
        self.smoother = PredictionSmoother(len(MathClassifier.CLASSES), history=5)
        self.detector = UniversalPoseDetector(cfg['mp_det'], cfg['mp_track'], cfg['mp_complex'])
        self.last_res = [ClassResult('?', 0.0, 0.0)]
        self.last_lm = None

    def process(self, frame):
        lm = self.detector.detect(frame)
        self.last_lm = lm
        if lm:
            nd, sc = self.normalizer.normalize(lm)
            self.buf.add_frame(nd, scale=sc)
            if self.buf.filled(0.4):
                ft = self.extractor.compute(self.buf)
                res = self.classifier.classify(ft)
                cls_s = sorted(MathClassifier.CLASSES)
                probs = np.array([next((r.confidence for r in res if r.name == c), 0.0) for c in cls_s])
                self.smoother.update(probs)
                sm = self.smoother.get()
                for r in res:
                    r.confidence = float(sm[cls_s.index(r.name)])
                res.sort(key=lambda x: x.confidence, reverse=True)
                self.last_res = res

        return draw_frame(frame, self.last_lm, self.last_res)

    def run(self):
        cap = cv2.VideoCapture(self.cfg['camera_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg['frame_w'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg['frame_h'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow('Pose Classifier', cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self.process(cv2.flip(frame, 1))
                cv2.imshow('Pose Classifier', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.buf.reset()
                    self.normalizer.reset()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


if __name__ == '__main__':
    MathPoseClassifier().run()