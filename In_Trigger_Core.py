# title: "In-Trigger Core"
# date: "09/19/2025"
# author: "Babur Erdem"
# update date: "09/19/2025"

import os, time, datetime, collections
import cv2, numpy as np

# Quiet OpenCV + prefer stable backends
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1000")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "500")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# ---------------- Serial utilities ----------------
SERIAL_PORT = "COM6"
BAUD = 115200
_ser = None

def serial_ports():
    try:
        import serial.tools.list_ports as lp
        return [p.device for p in lp.comports()]
    except Exception:
        return []

def _open_serial():
    global _ser
    try:
        import serial, serial.tools.list_ports
        if SERIAL_PORT:
            try:
                _ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
            except Exception:
                _ser = None
        if not (_ser and _ser.is_open):
            for p in serial.tools.list_ports.comports():
                if any(t in (p.description or "").lower() for t in
                       ["arduino", "wch", "ch340", "usb serial", "ttyacm", "ttyusb"]):
                    try:
                        _ser = serial.Serial(p.device, BAUD, timeout=0)
                        break
                    except Exception:
                        pass
        if _ser and _ser.is_open:
            time.sleep(2)  # allow Arduino reset
        else:
            _ser = None
    except Exception:
        _ser = None

def set_serial_port(port: str) -> bool:
    global SERIAL_PORT, _ser
    if port:
        SERIAL_PORT = port
    if _ser:
        try: _ser.close()
        except: pass
        _ser = None
    _open_serial()
    return bool(_ser and _ser.is_open)

def serial_status():
    return (bool(_ser and _ser.is_open), (getattr(_ser, "port", None) or SERIAL_PORT or ""))

def _swrite(b: bytes):
    if not (_ser and _ser.is_open):
        _open_serial()
    if _ser and _ser.is_open:
        try: _ser.write(b)
        except: pass

# ---------------- Helpers ----------------
def _unique_base_in(dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    base = time.strftime("%y%m%d_%H%M%S")
    if not (os.path.exists(os.path.join(dirpath, f"{base}.mp4")) or
            os.path.exists(os.path.join(dirpath, f"{base}.txt"))):
        return base
    i = 1
    while True:
        cand = f"{base}_{i:02d}"
        if not (os.path.exists(os.path.join(dirpath, f"{cand}.mp4")) or
                os.path.exists(os.path.join(dirpath, f"{cand}.txt"))):
            return cand
        i += 1

def _unique_mp4_in(dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    base = time.strftime("%y%m%d_%H%M%S")
    p = os.path.join(dirpath, f"{base}.mp4")
    if not os.path.exists(p): return p
    i = 1
    while True:
        cand = os.path.join(dirpath, f"{base}_{i:02d}.mp4")
        if not os.path.exists(cand): return cand
        i += 1

def _norm_rect(r):
    x1, y1, x2, y2 = r
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def _pt_in_rect(x, y, r):
    if not r: return False
    x1, y1, x2, y2 = r
    return x1 <= x <= x2 and y1 <= y <= y2

def _pt_in_circle(x, y, c):
    if not c: return False
    cx, cy, rr = c
    return (x - cx) ** 2 + (y - cy) ** 2 <= rr * rr

# ---------------- Engine ----------------
class Engine:
    # Tunables
    CAM_INDEX = 0
    FLIP_HORIZONTAL = False
    MIN_AREA = 1000
    MAX_AREA = 100000
    MATCH_DIST = 50
    CONFIRM_FRAMES = 3
    MAX_CHANGE_RATIO = 0.20
    REFIRE_COOLDOWN = 0.50
    SHOCK_LAG_SEC = 1.00
    SHOCK_ON_SEC = 1.00

    def __init__(self):
        # camera and state
        self.cap = None
        self.exp_locked = True
        self.backsub = None
        self.prev_tracks = []
        self.rois = []
        self.last_fire_ts_by_roi = {}
        self.current_on_rois = set()
        self.led_state_by_pin = {}
        self.led_events = []
        self.ts_hist = collections.deque(maxlen=120)
        self.decided_fps = None
        self.cam_fps = None
        self.recording = False
        self.writer = None
        self.last_out_path = ""
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.buffer_frames = []
        self.MAX_BUFFER_FRAMES = 600
        self.record_grace_deadline = None
        self.autofocus = False
        self.focus_raw = 128
        self.save_dir = os.getcwd()
        self.session_base = None
        self.log_fp = None
        self.frame_counter = 0
        self.bg_relearn_bursts = 0
        self._pending_def = (None, None, None)
        self._last_ts = None
        self._dt_window = collections.deque(maxlen=90)
        self._need_reopen = False   # UI will handle reopen via worker
        self.busy = False           # UI sets while a worker operates

    # ---------- lifecycle ----------
    def start(self):
        self.set_save_dir(self.save_dir)
        self._log("SESSION_START", self.session_base)
        self._open_camera(self.CAM_INDEX)

    def stop(self):
        if self.writer:
            self.writer.release(); self.writer = None
            self._log("RECORD_STOP", self.last_out_path)
        if self.cap:
            self.cap.release(); self.cap = None
        if self.log_fp:
            self._log("SESSION_END"); self.log_fp.close(); self.log_fp = None
        if _ser:
            try: _ser.close()
            except: pass

    # ---------- camera ops (blocking; call from worker) ----------
    def _open_camera(self, index: int):
        self.CAM_INDEX = int(index)
        def _try_open(idx, backend):
            try:
                c = cv2.VideoCapture(idx, backend)
                return c if c.isOpened() else None
            except Exception:
                return None

        if self.cap:  # clean prior
            try: self.cap.release()
            except: pass
            self.cap = None

        self.cap = (_try_open(self.CAM_INDEX, cv2.CAP_MSMF) or
                    _try_open(self.CAM_INDEX, cv2.CAP_DSHOW) or
                    _try_open(self.CAM_INDEX, cv2.CAP_ANY))
        if not (self.cap and self.cap.isOpened()):
            raise RuntimeError("Camera open failed")

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

        pw, ph, pfps = self._pending_def
        try:
            if pw:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(pw))
            if ph:  self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(ph))
            pass  # FPS setting removed by request
        except Exception:
            pass

        # prime device
        for _ in range(6):
            try: self.cap.read()
            except Exception: pass
            time.sleep(0.02)

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.cam_fps = fps if 0.1 < fps < 240.0 else None

        self._set_exposure_lock(self.exp_locked)
        self.set_autofocus(self.autofocus)
        if not self.autofocus:
            self.set_focus_raw(self.focus_raw)

        # processing state
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=25, detectShadows=True)
        self.prev_tracks = []
        self.ts_hist.clear(); self.decided_fps = None
        self.buffer_frames.clear()
        self._last_ts = None; self._dt_window.clear()
        self.frame_counter = 0
        self._need_reopen = False

    def switch_camera(self, index: int):
        self._open_camera(index)
        self._log("CAM_SWITCH", str(index))

    def set_camera_definition(self, width: int | None = None, height: int | None = None, fps: float | None = None):
        self._pending_def = (width, height, fps)
        # apply by reopening with same index
        self._open_camera(self.CAM_INDEX)

    def current_size(self):
        if not self.cap: return (0, 0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def detect_supported_presets(self, res_candidates: list[tuple[int, int]]) -> dict:
        out = {}
        if not self.cap: return out

        cw, ch = self.current_size()
        rfps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        saved_def = (cw, ch, rfps if rfps > 0 else None)

        for (w, h) in res_candidates:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                for _ in range(4): self.cap.read()
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if abs(aw - w) > 8 or abs(ah - h) > 8:
                    continue
                ok_fps = []
                for f in fps_candidates:
                    try:
                        if f and f > 0: self.cap.set(cv2.CAP_PROP_FPS, float(f))
                    except Exception:
                        pass
                    times = []
                    last = None
                    frames = 0
                    t0 = time.time()
                    # shorter timing window to speed up probing
                    while frames < 12 and (time.time() - t0) < 0.5:
                        ok, _fr = self.cap.read()
                        if not ok: continue
                        t = time.time()
                        if last is not None:
                            dt = t - last
                            if 0.002 < dt < 1.0: times.append(dt)
                        last = t
                        frames += 1
                    if len(times) >= 6:
                        med = sorted(times)[len(times) // 2]
                        m_fps = 1.0 / med
                        if abs(m_fps - f) <= max(2.0, 0.2 * f):
                            ok_fps.append(int(round(f)))
                if ok_fps:
                    out[(aw, ah)] = sorted(set(ok_fps))
            except Exception:
                pass

        # restore
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(saved_def[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(saved_def[1]))
            pass  # FPS restore removed
            for _ in range(3): self.cap.read()
        except Exception:
            pass

        return out

    # ---------- logging ----------
    def _open_log(self):
        path = os.path.join(self.save_dir, f"{self.session_base}.txt")
        new_file = not os.path.exists(path)
        self.log_fp = open(path, "a", buffering=1)
        if new_file:
            header = "Date\tTime\t"
            for i in range(len(self.rois)):
                header += f"ROI_{i + 1}_Objects\t"
            header += "Trigger_Status\tVideo_Name\tFrame_Number\n"
            self.log_fp.write(header)

    def _log(self, event, info=""):
        if not self.log_fp: return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ms = int((time.time() % 1) * 1000)
        self.log_fp.write(f"{ts}.{ms:03d}\t{event}\t{info}\n")

    def _log_entry(self, now, counts_by_roi, shock_status, frame_num):
        if not self.log_fp: return
        dt_object = datetime.datetime.fromtimestamp(now)
        date_str = dt_object.strftime("%Y-%m-%d")
        time_str = dt_object.strftime("%H:%M:%S.%f")[:-3]
        row = f"{date_str}\t{time_str}\t"
        for i in range(len(self.rois)):
            row += f"{counts_by_roi.get(i, 0)}\t"
        video_name = os.path.basename(self.last_out_path) if self.last_out_path else "NA"
        row += f"{shock_status}\t{video_name}\t{frame_num}\n"
        self.log_fp.write(row)

    # ---------- config ----------
    def set_save_dir(self, path: str) -> str:
        path = os.path.abspath(path or os.getcwd())
        os.makedirs(path, exist_ok=True)
        self.save_dir = path
        if self.log_fp:
            self._log("DIR_CHANGE", path)
            self.log_fp.close(); self.log_fp = None
        self.session_base = _unique_base_in(self.save_dir)
        self._open_log()
        return self.save_dir

    def set_areas(self, min_area: int, max_area: int):
        self.MIN_AREA = max(0, int(min_area))
        self.MAX_AREA = max(self.MIN_AREA + 1, int(max_area))

    def set_flip(self, flip: bool):
        self.FLIP_HORIZONTAL = bool(flip)

    def set_shock_timing(self, lag_s: float, on_s: float):
        self.SHOCK_LAG_SEC = max(0.0, float(lag_s))
        self.SHOCK_ON_SEC = max(0.0, float(on_s))

    def set_rois(self, rois: list):
        self.rois = []
        for r in rois:
            if r.get("shape") == "rect" and "rect" in r:
                x1, y1, x2, y2 = _norm_rect(r["rect"])
                if x2 - x1 > 2 and y2 - y1 > 2:
                    self.rois.append({"shape": "rect", "rect": (int(x1), int(y1), int(x2), int(y2))})
            elif r.get("shape") == "circle" and "circle" in r:
                cx, cy, rr = r["circle"]
                rr = max(2, int(rr))
                self.rois.append({"shape": "circle", "circle": (int(cx), int(cy), rr)})
        self.last_fire_ts_by_roi = {i: self.last_fire_ts_by_roi.get(i, 0.0) for i in range(len(self.rois))}
        self.current_on_rois = {i for i in self.current_on_rois if i < len(self.rois)}

    def clear_rois(self):
        self.rois = []; self.current_on_rois.clear(); self.last_fire_ts_by_roi.clear()

    def arm_recording(self, enable: bool):
        self.recording = bool(enable)
        if enable:
            self.record_grace_deadline = time.time() + 2.0
            self.frame_counter = 0
        else:
            self.record_grace_deadline = None
            if self.writer:
                self.writer.release(); self.writer = None
                self._log("RECORD_STOP", self.last_out_path)
            self.buffer_frames.clear()

    def fast_relearn(self, bursts: int = 5):
        self.bg_relearn_bursts = max(self.bg_relearn_bursts, int(bursts))
        self._log("BG_RELEARN", str(bursts))

    def exposure_lock(self, enable: bool):
        self._set_exposure_lock(bool(enable))
        self._log("EXPOSURE_LOCK", str(int(self.exp_locked)))

    def toggle_exposure(self):
        self._set_exposure_lock(not self.exp_locked)
        self._log("EXPOSURE_LOCK", str(int(self.exp_locked)))

    def set_autofocus(self, enable: bool):
        self.autofocus = bool(enable)
        try:
            if self.cap: self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
        except Exception:
            pass

    def set_focus_raw(self, value: int):
        self.focus_raw = int(max(0, min(255, value)))
        try:
            if self.cap: self.cap.set(cv2.CAP_PROP_FOCUS, float(self.focus_raw))
        except Exception:
            pass

    def _set_exposure_lock(self, lock=True):
        self.exp_locked = lock
        try:
            if self.cap:
                if lock:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        except Exception:
            pass

    # ---------- I/O helpers ----------
    def _pin_for_roi(self, roi_idx: int):
        return int(roi_idx + 2)

    def _shock_on_pin(self, pin: int):
        _swrite(f"N{int(pin)}\n".encode())
        roi_idx = pin - 2
        self.led_state_by_pin[pin] = True
        self.current_on_rois.add(roi_idx)
        self._log_entry(time.time(), {}, "Shock ON", self.frame_counter)

    def _shock_off_pin(self, pin: int):
        _swrite(f"F{int(pin)}\n".encode())
        roi_idx = pin - 2
        self.led_state_by_pin[pin] = False
        self.current_on_rois.discard(roi_idx)
        self._log_entry(time.time(), {}, "Shock OFF", self.frame_counter)

    def _measured_fps(self):
        if len(self._dt_window) < 20:
            return None
        dts = sorted(self._dt_window)
        med = dts[len(dts) // 2]
        if med <= 0:
            return None
        fps = 1.0 / med
        if fps < 5.0 or fps > 240.0:
            return None
        return fps

    # ---------- main step ----------
    def step(self):
        # if a previous read failed, request reopen via UI worker
        if self._need_reopen:
            return None, None

        self.frame_counter += 1
        raw = None
        for _ in range(2):
            try:
                ok, raw = self.cap.read() if self.cap else (False, None)
                if ok and raw is not None and raw.size > 0:
                    break
            except cv2.error:
                ok, raw = False, None
            time.sleep(0.01)

        if raw is None or (isinstance(raw, np.ndarray) and raw.size == 0):
            # do not reopen here; flag and let UI schedule worker
            self._need_reopen = True
            return None, None

        now = time.time()
        if self._last_ts is not None:
            dt = now - self._last_ts
            if 0.001 < dt < 1.0:
                self._dt_window.append(dt)
        self._last_ts = now

        base = cv2.flip(raw, 1) if self.FLIP_HORIZONTAL else raw
        h, w = base.shape[:2]

        self.ts_hist.append(now)
        if self.decided_fps is None and len(self.ts_hist) >= 20:
            dt = self.ts_hist[-1] - self.ts_hist[0]
            if dt > 0:
                self.decided_fps = float(max(5.0, min(240.0, (len(self.ts_hist) - 1) / dt)))

        proc = base.copy()

        # background learning
        if self.backsub is None:
            self.backsub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=25, detectShadows=True)

        if self.bg_relearn_bursts > 0:
            for _ in range(self.bg_relearn_bursts):
                self.backsub.apply(proc, learningRate=0.5)
            self.bg_relearn_bursts = 0

        fg = self.backsub.apply(proc, learningRate=0.005)
        moving = (fg == 255).astype(np.uint8) * 255

        change_ratio = float(np.count_nonzero(fg)) / fg.size
        lighting_change = change_ratio > self.MAX_CHANGE_RATIO
        if lighting_change:
            self.backsub.apply(proc, learningRate=0.5)
            moving[:] = 0

        # ROI mask
        mask = np.zeros(proc.shape[:2], dtype=np.uint8)
        if self.rois:
            for r in self.rois:
                if r["shape"] == "rect":
                    x1, y1, x2, y2 = r["rect"]
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                else:
                    cx, cy, rr = r["circle"]
                    cv2.circle(mask, (cx, cy), rr, 255, -1)
        else:
            mask[:] = 255

        masked = cv2.bitwise_and(moving, mask)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, k_open, 1)
        masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, k_close, 1)

        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = proc.copy()
        red = (0, 0, 255); green = (0, 255, 0)

        # draw ROIs with ID and center dot only
        for i, r in enumerate(self.rois):
            idx = i + 1
            if r["shape"] == "rect":
                x1, y1, x2, y2 = r["rect"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), red, 2)
                cv2.putText(vis, str(idx), (x1 + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2, cv2.LINE_AA)
            else:
                cx, cy, rr = r["circle"]
                cv2.circle(vis, (cx, cy), rr, red, 2)
                cv2.putText(vis, str(idx), (cx - 10, cy - rr - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 5, red, -1)

        curr_pts = []
        counts_by_roi = collections.defaultdict(int)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.MIN_AREA or area > self.MAX_AREA: continue
            x, y, wc, hc = cv2.boundingRect(c)
            ar = max(wc, hc) / max(1, min(wc, hc))
            if ar > 3.0: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            roi_idx = self._which_roi(cx, cy)
            if roi_idx is not None:
                counts_by_roi[roi_idx] += 1
            cv2.rectangle(vis, (x, y), (x + wc, y + hc), green, 2)
            cv2.circle(vis, (cx, cy), 3, green, -1)
            curr_pts.append((cx, cy, roi_idx))

        matched = self._match_tracks(curr_pts)
        next_tracks = []
        for cx, cy, roi_idx, prev_i in matched:
            counts = {}
            if prev_i is not None:
                _, _, prev_counts = self.prev_tracks[prev_i]
                counts.update(prev_counts)
            if roi_idx is not None:
                prev = counts.get(roi_idx, 0)
                counts[roi_idx] = prev + 1
                if counts[roi_idx] == self.CONFIRM_FRAMES and not lighting_change:
                    self._try_fire_roi(roi_idx, now, counts_by_roi)
            for k in list(counts.keys()):
                if k != roi_idx:
                    counts[k] = 0
            next_tracks.append((cx, cy, counts))
        self.prev_tracks = next_tracks

        # process queued LED events
        if self.led_events:
            self.led_events.sort(key=lambda t: t[0])
            while self.led_events and self.led_events[0][0] <= now:
                _, act, r_idx = self.led_events.pop(0)
                pin = self._pin_for_roi(r_idx)
                if act == 'on':
                    self._shock_on_pin(pin)
                else:
                    self._shock_off_pin(pin)

        frame_out = base.copy()
        if self.current_on_rois:
            label = "ON " + ",".join(str(i + 1) for i in sorted(self.current_on_rois))
            self._draw_text_br(frame_out, label, color=red)
            self._draw_text_br(vis, label, color=red)

        fps_use = self._measured_fps() or self.cam_fps or self.decided_fps
        if self.recording and self.writer is None:
            if not fps_use and self.record_grace_deadline and time.time() > self.record_grace_deadline:
                fps_use = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if fps_use:
                fps_use = float(max(5.0, min(120.0, fps_use)))
                out_path = _unique_mp4_in(self.save_dir)
                self.writer = cv2.VideoWriter(out_path, self.fourcc, fps_use, (w, h))
                if self.writer.isOpened():
                    self.last_out_path = out_path
                    image_path = os.path.splitext(out_path)[0] + ".jpg"
                    cv2.imwrite(image_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    self._log("IMAGE_SAVE", f"Path={image_path}")
                    for f in self.buffer_frames: self.writer.write(f)
                    self.buffer_frames.clear()
                    self._log("RECORD_START", f"{out_path} fps={fps_use:.2f}")
                else:
                    self.writer = None
                    self.recording = False
                    self.buffer_frames.clear()

        if self.recording:
            if self.writer is not None:
                self.writer.write(frame_out)
            else:
                self.buffer_frames.append(frame_out.copy())
                if len(self.buffer_frames) > self.MAX_BUFFER_FRAMES:
                    self.buffer_frames.pop(0)

        return vis, masked

    # ---------- helpers ----------
    def _which_roi(self, x: int, y: int):
        for i, r in enumerate(self.rois):
            if r["shape"] == "rect":
                if _pt_in_rect(x, y, r["rect"]): return i
            else:
                if _pt_in_circle(x, y, r["circle"]): return i
        return None

    def _match_tracks(self, curr_pts):
        matched = []
        used = set()
        for cx, cy, roi_idx in curr_pts:
            best_i, best_d = None, 1e18
            for i, (px, py, _) in enumerate(self.prev_tracks):
                if i in used: continue
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d <= self.MATCH_DIST ** 2:
                matched.append((cx, cy, roi_idx, best_i)); used.add(best_i)
            else:
                matched.append((cx, cy, roi_idx, None))
        return matched

    def _try_fire_roi(self, roi_idx: int, now_ts: float, counts_by_roi):
        last = self.last_fire_ts_by_roi.get(roi_idx, 0.0)
        if now_ts - last < self.REFIRE_COOLDOWN: return
        self.last_fire_ts_by_roi[roi_idx] = now_ts
        on_at = now_ts + max(0.0, self.SHOCK_LAG_SEC)
        off_at = on_at + max(0.0, self.SHOCK_ON_SEC)
        self.led_events.append((on_at, 'on', roi_idx))
        self.led_events.append((off_at, 'off', roi_idx))
        self._log_entry(now_ts, counts_by_roi, "Object Detected", self.frame_counter)

    def _draw_text_br(self, img, text, color=(0, 0, 255), scale=1.0, thickness=2, margin=12):
        h, w = img.shape[:2]
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = (w - tw - margin, h - bl - margin)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
