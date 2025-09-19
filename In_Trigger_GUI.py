# title: "In-Trigger GUI"
# date: "09/19/2025"
# author: "Babur Erdem"
# update date: "09/19/2025"

import sys, os, subprocess, importlib, time

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "1000"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "500"

REQUIRED = [("PySide6", "PySide6"), ("cv2", "opencv-python"), ("numpy", "numpy"), ("serial", "pyserial")]


def _ensure(mod_name, pip_name):
    try:
        importlib.import_module(mod_name);
        return True
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", pip_name], check=False)
            importlib.import_module(mod_name);
            return True
        except Exception:
            print(f"Missing package '{pip_name}'. Install manually: pip install {pip_name}")
            return False


_missing = [p for (m, p) in REQUIRED if not _ensure(m, p)]
if _missing: sys.exit(1)

import cv2, numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

from PySide6 import QtCore, QtGui, QtWidgets
from In_Trigger_Core import (
    Engine, set_serial_port, serial_ports, serial_status
)

DEFAULT_RES_CANDIDATES = [(960, 540), (1280, 720), (1920, 1080), (3840, 2160)]

HELP_TEXT = (
    "Preview window:\n"
    "• Add ROI: Left-drag. 'Add Region Of Interest' to determine trigger area\n"
    "with defined shape as Rect or Circle.\n"
    "• Move ROI: Left-drag inside.\n"
    "• Resize Circle: Mouse wheel when selected.\n\n"
    "Camera:\n"
    "• Camera: Select the camera.\n"
    "• Resolution: Select resolution of recorded video.\n"
    "• Record: Begin video recording.\n"
    "• Flip Horizontal: Flip the video horizontally.\n"
    "• Autofocus: Choose the focus in automatic mode or make focus manually.\n"
    "• Focus: If autofocus off, adjust the focus manually.\n\n"
    "Detection and ROIs:\n"
    "• Min Object Size: Define minimum object size (in pixel) to catch in ROIs.\n"
    "• Max Object Size: Define maximum object size (in pixel) to catch in ROIs.\n"
    "• Relearn Background: Fastly learn background in the changes as moving cam\n"
    "or changing illumination.\n"
    "• Add ROI: Define shape of the ROIs, Rectangle or Circle.\n"
    "• Delete Selected ROI: Select a ROI with left click inside,\n"
    "then press this button.\n"
    "• Clear All ROIs: Delete all defined ROIs.\n\n"
    "Trigger and Serial:\n"
    "• Trigger Lag (s): Lag between the object detection and triggering in sec.\n"
    "• Trigger On (s): Duration of triggering on in sec.\n"
    "• Serial Port: USB port that Arduino connected.\n\n"
    "Output:\n"
    "• Directory: Choose directory to save recorded video, log file and snapshot.\n\n"
    "• Quit: Click two times to quit.\n\n"
    "For further inquiries or assistance:\n"
    "contact with Babur Erdem, e-mail: ebabur@metu.edu.tr\n"

)

ALLOW_CLOSE = False


def _norm(r): x1, y1, x2, y2 = r; return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _backend_order(): return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]


def list_cameras(max_index=6):
    out = []
    for i in range(max_index + 1):
        cap = None
        try:
            for be in _backend_order():
                cap = cv2.VideoCapture(i, be)
                if cap.isOpened(): out.append(i); break
                cap.release()
        except Exception:
            pass
        if cap: cap.release()
    return out


class FuncWorker(QtCore.QThread):
    done = QtCore.Signal(object, object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn;
        self.args = args;
        self.kwargs = kwargs
        self._result = None;
        self._error = None

    def run(self):
        try:
            self._result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self._error = e
        self.done.emit(self._result, self._error)


class VideoView(QtWidgets.QLabel):
    roisChanged = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet("QWidget { font-size: 10px; }")
        self.setScaledContents(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.frame_size = (0, 0);
        self.rois = [];
        self.sel_idx = -1
        self.add_shape = "rect";
        self.drawing = False;
        self.dragging = False
        self.anchor = (0, 0);
        self.roi_anchor_geom = None;
        self.WHEEL_STEP = 5

    def setFrameSize(self, w, h):
        if w <= 0 or h <= 0: return
        old = self.frame_size
        if old != (0, 0) and (old != (w, h)) and self.rois:
            sx = w / old[0];
            sy = h / old[1];
            s = min(sx, sy)
            new = []
            for r in self.rois:
                if r["shape"] == "rect":
                    x1, y1, x2, y2 = _norm(r["rect"])
                    new.append({"shape": "rect", "rect": (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)),
                                                          int(round(y2 * sy)))})
                else:
                    cx, cy, rr = r["circle"]
                    new.append({"shape": "circle",
                                "circle": (int(round(cx * sx)), int(round(cy * sy)), max(2, int(round(rr * s))))})
            self.rois = new;
            self.roisChanged.emit()
        self.frame_size = (w, h)

    def setAddShape(self, shape: str):
        self.add_shape = "circle" if shape == "circle" else "rect"

    def clearAll(self):
        self.rois.clear(); self.sel_idx = -1; self.roisChanged.emit()

    def deleteSelected(self):
        if 0 <= self.sel_idx < len(self.rois):
            del self.rois[self.sel_idx];
            self.sel_idx = min(self.sel_idx, len(self.rois) - 1);
            self.roisChanged.emit()

    def _map(self, qp: QtCore.QPoint):
        w, h = self.frame_size
        if w == 0 or h == 0: return (0, 0)
        sx = w / max(1, self.width());
        sy = h / max(1, self.height())
        x = int(qp.x() * sx);
        y = int(qp.y() * sy)
        x = max(0, min(w - 1, x));
        y = max(0, min(h - 1, y));
        return (x, y)

    def _inside(self, i, x, y):
        r = self.rois[i]
        if r["shape"] == "rect":
            x1, y1, x2, y2 = _norm(r["rect"]);
            return x1 <= x <= x2 and y1 <= y <= y2
        else:
            cx, cy, rr = r["circle"];
            return (x - cx) ** 2 + (y - cy) ** 2 <= rr * rr

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton: return
        qp = ev.position().toPoint();
        x, y = self._map(qp);
        self.anchor = (x, y)
        hit = -1
        for i in range(len(self.rois) - 1, -1, -1):
            if self._inside(i, x, y): hit = i; break
        if hit >= 0:
            self.sel_idx = hit;
            self.dragging = True
            r = self.rois[self.sel_idx]
            self.roi_anchor_geom = r["rect"] if r["shape"] == "rect" else r["circle"]
        else:
            self.drawing = True
            if self.add_shape == "rect":
                self.rois.append({"shape": "rect", "rect": (x, y, x, y)})
            else:
                self.rois.append({"shape": "circle", "circle": (x, y, 0)})
            self.sel_idx = len(self.rois) - 1;
            self.roisChanged.emit()

    def mouseMoveEvent(self, ev):
        qp = ev.position().toPoint();
        x, y = self._map(qp)
        if self.dragging and 0 <= self.sel_idx < len(self.rois):
            dx, dy = x - self.anchor[0], y - self.anchor[1];
            r = self.rois[self.sel_idx]
            if r["shape"] == "rect":
                x1, y1, x2, y2 = self.roi_anchor_geom
                self.rois[self.sel_idx]["rect"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            else:
                cx, cy, rr = self.roi_anchor_geom
                self.rois[self.sel_idx]["circle"] = (cx + dx, cy + dy, rr)
            self.roisChanged.emit()
        elif self.drawing and 0 <= self.sel_idx < len(self.rois):
            r = self.rois[self.sel_idx]
            if r["shape"] == "rect":
                x1, y1, _, _ = r["rect"];
                self.rois[self.sel_idx]["rect"] = (x1, y1, x, y)
            else:
                cx, cy, _ = r["circle"];
                rr = int(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
                self.rois[self.sel_idx]["circle"] = (cx, cy, rr)
            self.roisChanged.emit()

    def mouseReleaseEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton: return
        if self.dragging:
            self.dragging = False; self.roi_anchor_geom = None
        elif self.drawing:
            self.drawing = False
            if 0 <= self.sel_idx < len(self.rois):
                r = self.rois[self.sel_idx]
                if r["shape"] == "rect":
                    x1, y1, x2, y2 = _norm(r["rect"])
                    if x2 - x1 <= 2 or y2 - y1 <= 2: self.deleteSelected(); return
                    self.rois[self.sel_idx]["rect"] = (x1, y1, x2, y2)
                else:
                    cx, cy, rr = r["circle"]
                    if rr < 2: self.deleteSelected(); return
                    self.rois[self.sel_idx]["circle"] = (cx, cy, max(2, rr))
                self.roisChanged.emit()

    def wheelEvent(self, ev):
        if 0 <= self.sel_idx < len(self.rois):
            r = self.rois[self.sel_idx]
            if r["shape"] == "circle":
                delta = 1 if ev.angleDelta().y() > 0 else -1
                cx, cy, rr = r["circle"];
                rr = max(2, rr + delta * 5)
                self.rois[self.sel_idx]["circle"] = (cx, cy, rr);
                self.roisChanged.emit()


class PreviewWindow(QtWidgets.QMainWindow):
    def __init__(self, eng: Engine):
        super().__init__();
        self.setWindowTitle("Preview");
        self.eng = eng
        self.view = VideoView();
        left = QtWidgets.QVBoxLayout();
        left.addWidget(self.view, 1)
        hw = QtWidgets.QWidget();
        hw.setLayout(left);
        self.setCentralWidget(hw)
        self.view.roisChanged.connect(self._push_rois)

    def closeEvent(self, e: QtGui.QCloseEvent):
        global ALLOW_CLOSE
        if not ALLOW_CLOSE:
            e.ignore()
        else:
            super().closeEvent(e)

    def _push_rois(self):
        rois = []
        for r in self.view.rois:
            if r["shape"] == "rect":
                x1, y1, x2, y2 = _norm(r["rect"])
                rois.append({"shape": "rect", "rect": (int(x1), int(y1), int(x2), int(y2))})
            else:
                cx, cy, rr = r["circle"]
                rois.append({"shape": "circle", "circle": (int(cx), int(cy), int(rr))})
        self.eng.set_rois(rois)

    def show_bgr(self, img_bgr):
        if img_bgr is None: return
        h, w = img_bgr.shape[:2];
        self.view.setFrameSize(w, h)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg))


class ControlWindow(QtWidgets.QMainWindow):
    def __init__(self, eng: Engine, preview: PreviewWindow):
        super().__init__();
        self.setWindowTitle("Controls")
        self.eng = eng;
        self.preview = preview

        # worker, debounce, timer MUST exist before any _run_worker
        self._worker = None
        self._preset_timer = QtCore.QTimer(self);
        self._preset_timer.setSingleShot(True)
        self._preset_timer.timeout.connect(self._apply_preset_now)
        self.timer = QtCore.QTimer(self);
        self.timer.timeout.connect(self._tick);
        self.timer.start(1)

        # ---- Filtered preview
        preview_box = QtWidgets.QGroupBox("Filtered (motion mask)")
        self.view_fg = QtWidgets.QLabel();
        self.view_fg.setMinimumSize(300, 170);
        self.view_fg.setScaledContents(True)
        vb = QtWidgets.QVBoxLayout();
        vb.addWidget(self.view_fg);
        preview_box.setLayout(vb)

        # ---- Camera
        cam_box = QtWidgets.QGroupBox("Camera")
        self.combo_cams = QtWidgets.QComboBox()
        self.btn_cams_refresh = QtWidgets.QPushButton("Refresh")
        self.combo_res = QtWidgets.QComboBox()
        self.btn_detect = QtWidgets.QPushButton("Detect Supported Presets")
        self.btn_rec = QtWidgets.QPushButton("Record");
        self.btn_rec.setCheckable(True)
        self.btn_flip = QtWidgets.QPushButton("Flip Horizontal");
        self.btn_flip.setCheckable(True);
        self.btn_flip.setChecked(self.eng.FLIP_HORIZONTAL)
        self.btn_exp = QtWidgets.QPushButton("Exposure Lock Off");
        self.btn_exp.setCheckable(True)
        self.btn_af = QtWidgets.QPushButton("Autofocus Off");
        self.btn_af.setCheckable(True)
        self.sl_focus = QtWidgets.QSlider(QtCore.Qt.Horizontal);
        self.sl_focus.setRange(0, 255);
        self.sl_focus.setValue(128)
        g_cam = QtWidgets.QGridLayout()
        g_cam.addWidget(QtWidgets.QLabel("Camera"), 0, 0);
        g_cam.addWidget(self.combo_cams, 0, 1);
        g_cam.addWidget(self.btn_cams_refresh, 0, 2)
        g_cam.addWidget(QtWidgets.QLabel("Resolution"), 1, 0);
        g_cam.addWidget(self.combo_res, 1, 1)
        g_cam.addWidget(self.btn_detect, 2, 0, 1, 2);
        g_cam.addWidget(self.btn_rec, 2, 2);
        g_cam.addWidget(self.btn_flip, 2, 3)
        g_cam.addWidget(self.btn_exp, 3, 0);
        g_cam.addWidget(QtWidgets.QLabel("Focus"), 3, 1);
        g_cam.addWidget(self.sl_focus, 3, 2);
        g_cam.addWidget(self.btn_af, 3, 3)
        cam_w = QtWidgets.QWidget();
        cam_w.setLayout(g_cam);
        cam_box.setLayout(g_cam)

        # ---- Detection & ROI

        det_box = QtWidgets.QGroupBox("Detection and ROIs")
        self.sp_min = QtWidgets.QSpinBox();
        self.sp_min.setRange(0, 300000);
        self.sp_min.setValue(self.eng.MIN_AREA)
        self.sp_max = QtWidgets.QSpinBox();
        self.sp_max.setRange(1, 400000);
        self.sp_max.setValue(self.eng.MAX_AREA)
        self.btn_bg = QtWidgets.QPushButton("Relearn Background")
        self.btn_add_shape = QtWidgets.QPushButton("Add ROI: Rect")
        self.btn_delete = QtWidgets.QPushButton("Delete Selected ROI")
        self.btn_clear = QtWidgets.QPushButton("Clear All ROIs")
        g_det = QtWidgets.QGridLayout()
        g_det.addWidget(QtWidgets.QLabel("Min Object Size"), 0, 0);
        g_det.addWidget(self.sp_min, 0, 1)
        g_det.addWidget(QtWidgets.QLabel("Max Object Size"), 0, 2);
        g_det.addWidget(self.sp_max, 0, 3)
        g_det.addWidget(self.btn_bg, 1, 0);
        g_det.addWidget(self.btn_add_shape, 1, 1);
        g_det.addWidget(self.btn_delete, 1, 2);
        g_det.addWidget(self.btn_clear, 1, 3)
        det_box.setLayout(g_det)

        # ---- Trigger & Serial
        sh_box = QtWidgets.QGroupBox("Trigger and Serial")
        self.sp_lag = QtWidgets.QDoubleSpinBox();
        self.sp_lag.setRange(0.0, 10.0);
        self.sp_lag.setSingleStep(0.05);
        self.sp_lag.setValue(self.eng.SHOCK_LAG_SEC)
        self.sp_on = QtWidgets.QDoubleSpinBox();
        self.sp_on.setRange(0.0, 10.0);
        self.sp_on.setSingleStep(0.05);
        self.sp_on.setValue(self.eng.SHOCK_ON_SEC)
        self.combo_ports = QtWidgets.QComboBox();
        self.btn_ports_refresh = QtWidgets.QPushButton("Refresh")
        self.lbl_port_status = QtWidgets.QLabel("Serial: not connected")
        g_sh = QtWidgets.QGridLayout()
        g_sh.addWidget(QtWidgets.QLabel("Trigger Lag (s)"), 0, 0);
        g_sh.addWidget(self.sp_lag, 0, 1)
        g_sh.addWidget(QtWidgets.QLabel("Trigger On (s)"), 0, 2);
        g_sh.addWidget(self.sp_on, 0, 3)
        g_sh.addWidget(QtWidgets.QLabel("Serial Port"), 1, 0);
        g_sh.addWidget(self.combo_ports, 1, 1);
        g_sh.addWidget(self.btn_ports_refresh, 1, 2)
        g_sh.addWidget(self.lbl_port_status, 1, 3);
        sh_box.setLayout(g_sh)

        # ---- Output
        out_box = QtWidgets.QGroupBox("Output")
        self.edit_dir = QtWidgets.QLineEdit(os.getcwd());
        self.edit_dir.setMinimumWidth(260)
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        g_out = QtWidgets.QGridLayout()
        g_out.addWidget(QtWidgets.QLabel("Directory"), 0, 0);
        g_out.addWidget(self.edit_dir, 0, 1, 1, 2);
        g_out.addWidget(self.btn_browse, 0, 3)
        out_box.setLayout(g_out)

        # Busy banner
        self.busyBar = QtWidgets.QLabel("")
        self.busyBar.setAlignment(QtCore.Qt.AlignCenter)
        self.busyBar.setStyleSheet("QLabel {color:#222; background:#f0d98a; padding:6px; border:1px solid #e0c765;}")
        self.busyBar.setVisible(False)

        # ---- Root layout with fixed bottom bar
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.setContentsMargins(6, 6, 6, 6);
        content_layout.setSpacing(8)
        content_layout.addWidget(preview_box);
        content_layout.addWidget(cam_box)
        content_layout.addWidget(det_box);
        content_layout.addWidget(sh_box);
        content_layout.addWidget(out_box);
        content_layout.addStretch(1)

        content_w = QtWidgets.QWidget();
        content_w.setLayout(content_layout)
        scroll = QtWidgets.QScrollArea();
        scroll.setWidgetResizable(True);
        scroll.setWidget(content_w)

        bottombar = QtWidgets.QHBoxLayout()
        self.btn_help = QtWidgets.QPushButton("Help");
        self.btn_quit = QtWidgets.QPushButton("Quit")
        bottombar.addWidget(self.btn_help);
        bottombar.addStretch(1);
        bottombar.addWidget(self.btn_quit)
        bottombar_w = QtWidgets.QWidget();
        bottombar_w.setLayout(bottombar)

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self.busyBar, 0);
        root.addWidget(scroll, 1);
        root.addWidget(bottombar_w, 0)
        root_w = QtWidgets.QWidget();
        root_w.setLayout(root);
        self.setCentralWidget(root_w)

        # quit
        self._quit_armed = False
        self._quit_timer = QtCore.QTimer(self);
        self._quit_timer.setSingleShot(True);
        self._quit_timer.timeout.connect(self._reset_quit)

        # signals
        self.btn_help.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Help", HELP_TEXT))
        self.btn_quit.clicked.connect(self._quit_clicked)
        self.btn_add_shape.clicked.connect(self._shape_toggle)
        self.btn_delete.clicked.connect(self.preview.view.deleteSelected)
        self.btn_clear.clicked.connect(self.preview.view.clearAll)
        self.btn_bg.clicked.connect(lambda: self.eng.fast_relearn(5))
        self.btn_exp.clicked.connect(self._exp_click)
        self.btn_rec.toggled.connect(self._rec_toggled)
        self.btn_flip.toggled.connect(self._flip_toggled)
        self.btn_af.toggled.connect(self._af_toggled)
        self.sl_focus.valueChanged.connect(self._focus_changed)
        self.sp_min.valueChanged.connect(lambda v: self.eng.set_areas(v, self.sp_max.value()))
        self.sp_max.valueChanged.connect(lambda v: self.eng.set_areas(self.sp_min.value(), v))
        self.sp_lag.valueChanged.connect(lambda v: self.eng.set_shock_timing(v, self.sp_on.value()))
        self.sp_on.valueChanged.connect(lambda v: self.eng.set_shock_timing(self.sp_lag.value(), v))
        self.combo_ports.currentTextChanged.connect(self._port_changed)
        self.btn_ports_refresh.clicked.connect(self._refresh_ports)
        self.combo_cams.currentTextChanged.connect(self._cam_changed)
        self.btn_cams_refresh.clicked.connect(self._refresh_cams)
        self.combo_res.currentTextChanged.connect(self._apply_preset_debounced)
        self.btn_detect.clicked.connect(self._detect_presets_clicked)
        self.btn_browse.clicked.connect(self._browse_dir)
        self.edit_dir.editingFinished.connect(self._dir_changed)

        # init
        self._refresh_ports();
        self._refresh_cams()
        self._cam_changed(self.combo_cams.currentText())
        self._update_rec_button();
        self._update_flip_button();
        self._update_af_button();
        self._update_exp_button()
        self.edit_dir.setText(self.eng.save_dir)
        self.setMinimumWidth(400);
        self.setMaximumWidth(480)
        self.setStyleSheet("QWidget { font-size: 10px; }")

    def closeEvent(self, e: QtGui.QCloseEvent):
        global ALLOW_CLOSE
        if not ALLOW_CLOSE:
            e.ignore()
        else:
            super().closeEvent(e)

    def _quit_clicked(self):
        global ALLOW_CLOSE
        if not self._quit_armed:
            self._quit_armed = True;
            self.btn_quit.setText("Quit (click again)");
            self._quit_timer.start(2000)
        else:
            ALLOW_CLOSE = True;
            QtWidgets.QApplication.instance().quit()

    def _reset_quit(self):
        self._quit_armed = False;
        self.btn_quit.setText("Quit")

    def _setBusy(self, text: str | None):
        busy = bool(text)
        self.busyBar.setVisible(busy);
        self.busyBar.setText(text or "")
        self.setCursor(QtCore.Qt.BusyCursor if busy else QtCore.Qt.ArrowCursor)
        for w in [self.combo_cams, self.combo_res, self.btn_detect,
                  self.btn_cams_refresh, self.btn_rec, self.btn_flip, self.btn_exp,
                  self.btn_af, self.sl_focus, self.btn_bg, self.btn_add_shape,
                  self.btn_delete, self.btn_clear, self.sp_min, self.sp_max,
                  self.sp_lag, self.sp_on, self.combo_ports, self.btn_ports_refresh,
                  self.btn_browse]:
            w.setEnabled(not busy)
        self.eng.busy = busy
        if hasattr(self, "timer"):
            if busy:
                self.timer.stop()
            else:
                self.timer.start(1)

    def _update_flip_button(self):
        self.btn_flip.setText(f"Flip Horizontal {'On' if self.btn_flip.isChecked() else 'Off'}")

    def _update_af_button(self):
        checked = self.btn_af.isChecked()
        self.btn_af.setText(f"Autofocus {'On' if checked else 'Off'}")
        self.sl_focus.setEnabled(not checked)

    def _shape_toggle(self):
        new_shape = "circle" if self.preview.view.add_shape == "rect" else "rect"
        self.preview.view.setAddShape(new_shape)
        self.btn_add_shape.setText(f"Add ROI: {'Circle' if new_shape == 'circle' else 'Rect'}")

    def _exp_click(self):
        self.eng.toggle_exposure();
        self._update_exp_button()

    def _update_exp_button(self):
        self.btn_exp.setText(f"Exposure Lock {'On' if self.eng.exp_locked else 'Off'}");
        self.btn_exp.setChecked(self.eng.exp_locked)

    def _rec_toggled(self, checked):
        self.eng.arm_recording(bool(checked));
        self._update_rec_button()

    def _update_rec_button(self):
        if self.btn_rec.isChecked():
            self.btn_rec.setStyleSheet("QPushButton { background-color:#d00000; color:white; padding:5px 8px; }")
        else:
            self.btn_rec.setStyleSheet("QPushButton { padding:5px 8px; }")

    def _flip_toggled(self, checked):
        self.eng.set_flip(bool(checked));
        self._update_flip_button()

    def _af_toggled(self, checked):
        self.eng.set_autofocus(bool(checked));
        self._update_af_button()

    def _focus_changed(self, val):
        if not self.btn_af.isChecked(): self.eng.set_focus_raw(int(val))

    def _refresh_ports(self):
        cur = self.combo_ports.currentText()
        self.combo_ports.blockSignals(True);
        self.combo_ports.clear()
        ports = serial_ports();
        self.combo_ports.addItems(ports if ports else [""])
        self.combo_ports.blockSignals(False)
        if cur and cur in ports:
            self.combo_ports.setCurrentText(cur)
        elif ports:
            self.combo_ports.setCurrentIndex(0)
        self._port_changed(self.combo_ports.currentText())

    def _port_changed(self, text):
        set_serial_port(text)
        ok, port = serial_status()
        self.lbl_port_status.setText(f"Serial: {'connected' if ok else 'not connected'} ({port})")
        pal = self.lbl_port_status.palette();
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("green" if ok else "red"));
        self.lbl_port_status.setPalette(pal)

    def _refresh_cams(self):
        cur = self.combo_cams.currentText()
        cams = list_cameras(6);
        labels = [str(i) for i in cams] if cams else ["0"]
        self.combo_cams.blockSignals(True);
        self.combo_cams.clear();
        self.combo_cams.addItems(labels);
        self.combo_cams.blockSignals(False)
        if cur and cur in labels:
            self.combo_cams.setCurrentText(cur)
        elif labels:
            self.combo_cams.setCurrentIndex(0)

    def _cam_changed(self, text):
        try:
            idx = int(text)
        except Exception:
            idx = 0
        self._run_worker(self.eng.switch_camera, "Switching camera…", idx, on_done=self._after_cam_switched)

    def _after_cam_switched(self, _result, err):
        if err:
            QtWidgets.QMessageBox.critical(self, "Camera Error", f"Failed to open camera.\n{err}")
            return
        self.combo_res.blockSignals(True)
        self.combo_res.clear()
        self.combo_res.addItems([f"{w}×{h}" for (w, h) in DEFAULT_RES_CANDIDATES])
        self.combo_res.blockSignals(False)

    def _detect_presets_clicked(self):
        self._run_worker(self.eng.detect_supported_presets, "Detecting presets…",
                         DEFAULT_RES_CANDIDATES,
                         on_done=self._after_detect_presets)

    def _after_detect_presets(self, result, err):
        if err:
            QtWidgets.QMessageBox.warning(self, "Detection Error", f"Preset detection failed.\n{err}")
            return
        modes = result or {}
        self.combo_res.blockSignals(True)
        self.combo_res.clear()
        if modes:
            res_list = [f"{w}×{h}" for (w, h) in sorted(modes.keys())]
            self.combo_res.addItems(res_list)
        else:
            self.combo_res.addItems([f"{w}×{h}" for (w, h) in DEFAULT_RES_CANDIDATES])
        self.combo_res.blockSignals(False)

    def _apply_preset_debounced(self, _=None):
        self._preset_timer.start(600)

    def _apply_preset_now(self):
        res = self.combo_res.currentText()
        if "×" in res:
            try:
                w, h = [int(t) for t in res.split("×")]
            except Exception:
                w, h = (0, 0)
        else:
            w, h = (0, 0)
        fps = 0.0  # FPS ignored
        self._run_worker(self.eng.set_camera_definition, "Applying camera preset…", w, h, on_done=lambda *_: None)

    def _browse_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory",
                                                          self.edit_dir.text() or os.getcwd())
        if path: self.edit_dir.setText(path); self._dir_changed()

    def _dir_changed(self):
        path = self.edit_dir.text().strip()
        if not path: return
        real = self.eng.set_save_dir(path);
        self.edit_dir.setText(real)

    def _tick(self):
        if getattr(self.eng, "_need_reopen", False) and not self.eng.busy and self._worker is None:
            self._run_worker(self.eng._open_camera, "Reopening camera…", self.eng.CAM_INDEX, on_done=lambda *_: None)
            return
        vis, fg = self.eng.step()
        if vis is None: return
        self.preview.show_bgr(vis)
        if fg is not None:
            qg = QtGui.QImage(fg.data, fg.shape[1], fg.shape[0], fg.strides[0], QtGui.QImage.Format_Grayscale8)
            self.view_fg.setPixmap(QtGui.QPixmap.fromImage(qg))

    def _run_worker(self, fn, busy_text, *args, on_done=None, **kwargs):
        if self._worker is not None: return
        self._setBusy(busy_text)
        self._worker = FuncWorker(fn, *args, **kwargs)
        self._worker.done.connect(lambda res, err: self._finish_worker(res, err, on_done))
        self._worker.start()

    def _finish_worker(self, result, err, cb):
        self._worker = None
        self._setBusy(None)
        if cb: cb(result, err)


def main():
    app = QtWidgets.QApplication(sys.argv)
    eng = Engine();
    eng.start()
    preview = PreviewWindow(eng)
    controls = ControlWindow(eng, preview)
    screen = app.primaryScreen().availableGeometry()
    controls.resize(420, min(640, screen.height() - 80))
    controls.move(screen.x() + screen.width() - controls.width() - 30, screen.y() + 30)
    preview.resize(int(screen.width() * 0.66), int(screen.height() * 0.66))
    preview.move(screen.x() + 20, screen.y() + 20)
    preview.show();
    controls.show()
    ret = app.exec()
    try:
        eng.stop()
    finally:
        sys.exit(ret)


if __name__ == "__main__":
    main()
