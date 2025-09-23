import time
import math
import threading
import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

"""
Single‑camera, multi‑visualization GUI (Tkinter + OpenCV), non‑blocking URL connect
----------------------------------------------------------------------------------
• One camera source (webcam index or URL).
• The grid shows selected visualizations (Normal, Gray, Difference, Binary).
• Selection lives in the ALWAYS‑visible top bar (checkboxes). No single‑view mode.
• Visualizations that need parameters (Difference/Binary) show a live T slider.
• Bottom‑right stats box per tile: FPS and avg fg pixels.
• URL connects on a background thread to avoid UI freezes; auto‑retries if it drops.

Run
  pip install opencv-python pillow numpy
  python multicam_gui.py

Edit CAMERA_SOURCE below if desired.
"""

# ---------------- Configuration ----------------
CAMERA_SOURCE = 0               # int index (0) or URL string
UI_FPS = 30                     # target UI refresh rate (Hz)
RETRY_DELAY_S = 2.0             # wait before retrying a failed connection

# ------------------------------------------------


def bgr_to_tk(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(image=im)


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def resize_to_fit(img, target_w, target_h):
    if img is None or target_w <= 0 or target_h <= 0:
        return img
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_stats(img, lines, margin=8):
    if img is None:
        return img
    overlay = img.copy()
    h, w = img.shape[:2]

    # text metrics
    line_h = 18
    pad = 6
    box_h = pad * 2 + line_h * len(lines)
    box_w = pad * 2 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines)

    x1 = max(0, w - box_w - margin)
    y1 = max(0, h - box_h - margin)
    x2 = w - margin
    y2 = h - margin

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)

    y = y1 + pad + line_h - 6
    for t in lines:
        cv2.putText(img, t, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return img


class CameraStream:
    """Background, non‑blocking camera reader with auto‑reconnect.

    - set_source() is instant (doesn't block UI).
    - A worker thread tries to open/read frames, stores the latest frame and FPS.
    - If a backend blocks, it does so in the worker thread, not the UI thread.
    """

    def __init__(self, source, name="Camera"):
        self.name = name
        self._lock = threading.Lock()
        self._frame = None
        self._fps_times = deque(maxlen=30)
        self._fps = 0.0
        self._status = 'disconnected'   # 'connecting' | 'open' | 'disconnected'

        # thread control
        self._stop = False
        self._reopen = True
        self._source = source
        self._cap = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # cached derivatives for GUI processing
        self._prev_gray = None
        self.gray = None
        self.diff = None

    # ----- public API -----
    @property
    def fps(self):
        return self._fps

    @property
    def status(self):
        return self._status

    def set_source(self, source):
        # Accept int or str
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        with self._lock:
            self._source = source
            self._reopen = True
            self._frame = None
            self._prev_gray = None
            self.gray = None
            self.diff = None

    def snapshot_and_prepare(self):
        """Copy latest frame and compute gray/diff for visualizations."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            fps = self._fps

        if frame is None:
            return None, fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        self.gray = gray
        self.diff = diff
        return frame, fps

    def stop(self):
        self._stop = True
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

    # ----- worker thread -----
    def _open_cap(self):
        # Close previous
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        # Try to open; don't block UI thread
        self._status = 'connecting'
        src = self._source
        cap = cv2.VideoCapture(src)
        # A tiny buffer keeps things responsive on some backends
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            self._status = 'disconnected'
            return None
        self._status = 'open'
        return cap

    def _run(self):
        last_tick = None
        while not self._stop:
            if self._reopen or self._cap is None:
                self._cap = self._open_cap()
                self._reopen = False
                if self._cap is None:
                    time.sleep(RETRY_DELAY_S)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._status = 'disconnected'
                time.sleep(RETRY_DELAY_S)
                self._reopen = True
                continue

            # FPS calc
            now = time.perf_counter()
            if last_tick is not None:
                dt = max(1e-6, now - last_tick)
                self._fps_times.append(1.0 / dt)
                self._fps = sum(self._fps_times) / max(1, len(self._fps_times))
            last_tick = now

            with self._lock:
                self._frame = frame


class VizTile(ttk.Frame):
    """One visualization tile with optional slider."""
    def __init__(self, parent, label, needs_threshold, enabled_var: tk.BooleanVar):
        super().__init__(parent)
        self.label_text = label
        self.needs_threshold = needs_threshold
        self.enabled = enabled_var
        self.t_val = tk.IntVar(value=25)

        # Header (no View button; selection handled in top bar)
        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text=label).pack(side=tk.LEFT, padx=(10, 8))

        # Image area
        self.image_label = ttk.Label(self)
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Slider if needed
        self.slider_frame = ttk.Frame(self)
        self.slider = ttk.Scale(self.slider_frame, from_=0, to=255, variable=self.t_val, command=self._slider_changed)
        self.slider_label = ttk.Label(self.slider_frame, text=f"T = {self.t_val.get()}")
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.slider_label.pack(side=tk.RIGHT, padx=8)
        if needs_threshold:
            self.slider_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._fg_vals = deque(maxlen=60)
        self._tk_img = None

    def _slider_changed(self, *args):
        self.slider_label.config(text=f"T = {self.t_val.get()}")

    def wants_in_grid(self):
        return self.enabled.get()

    def render(self, stream: CameraStream, base_frame, fps, target_w, target_h):
        # base_frame is BGR or None
        out = None
        avg_fg = 0
        if base_frame is None:
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            msg = f"{stream.name}: {stream.status}" if stream.status != 'open' else f"{stream.name}: no frame"
            cv2.putText(placeholder, msg, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            out = placeholder
            fps = 0
        else:
            label = self.label_text
            T = self.t_val.get()
            if label == 'Normal':
                out = base_frame.copy()
                fg = 0
            elif label == 'Gray':
                out = to_bgr(stream.gray)
                fg = 0
            elif label == 'Difference':
                out = to_bgr(stream.diff)
                fg = int(np.count_nonzero(stream.diff > T))
            elif label == 'Binary':
                _, bin_img = cv2.threshold(stream.diff, T, 255, cv2.THRESH_BINARY)
                out = to_bgr(bin_img)
                fg = int(np.count_nonzero(bin_img))
            else:
                out = base_frame.copy()
                fg = 0
            self._fg_vals.append(fg)
            avg_fg = int(sum(self._fg_vals) / max(1, len(self._fg_vals)))

        out = overlay_stats(out, [f"fps: {int(fps)}", f"avg_fg: {avg_fg}"])
        out = resize_to_fit(out, target_w, target_h)
        self._tk_img = bgr_to_tk(out)
        self.image_label.configure(image=self._tk_img)


class App(tk.Tk):
    def __init__(self, source):
        super().__init__()
        self.title("Single‑Cam Multi‑View (Non‑blocking)")
        self.geometry("1200x820")

        # ---- Top: source & selection ----
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        src_box = ttk.Frame(top)
        src_box.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(src_box, text="Source:").pack(side=tk.LEFT, padx=(8, 4))
        self.src_var = tk.StringVar(value=str(source))
        self.src_entry = ttk.Entry(src_box, textvariable=self.src_var, width=50)
        self.src_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(src_box, text="Connect / Reconnect", command=self.reconnect).pack(side=tk.LEFT, padx=6)
        self.status_lbl = ttk.Label(src_box, text="status: …")
        self.status_lbl.pack(side=tk.LEFT, padx=(10, 0))

        sel_box = ttk.Frame(top)
        sel_box.pack(side=tk.TOP, fill=tk.X, pady=(6, 2))
        ttk.Label(sel_box, text="Views:").pack(side=tk.LEFT, padx=(8, 4))

        viz_defs = [
            ('Normal', False),
            ('Gray', False),
            ('Difference', True),
            ('Binary', True),
        ]

        # Shared BooleanVars so top bar controls which tiles are shown
        self.view_vars = {}
        default_on = {'Normal', 'Binary'}
        for name, _ in viz_defs:
            v = tk.BooleanVar(value=(name in default_on))
            self.view_vars[name] = v
            ttk.Checkbutton(sel_box, text=name, variable=v, command=self._request_layout).pack(side=tk.LEFT, padx=4)
        ttk.Button(sel_box, text="All", command=self._select_all).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(sel_box, text="None", command=self._select_none).pack(side=tk.LEFT, padx=2)

        # ---- Container for tiles ----
        self.container = ttk.Frame(self)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.container.bind('<Configure>', lambda e: self._request_layout())

        # ---- Camera stream (async) ----
        src = CAMERA_SOURCE
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.stream = CameraStream(src)

        # ---- Build tiles ----
        self.tiles = []
        for name, needs_T in viz_defs:
            tile = VizTile(self.container, name, needs_T, enabled_var=self.view_vars[name])
            self.tiles.append(tile)

        self._layout_dirty = True
        self._layout_id = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(0, self._update_loop)
        self._do_layout()

    # ---- Selection helpers ----
    def _select_all(self):
        for v in self.view_vars.values():
            v.set(True)
        self._request_layout()

    def _select_none(self):
        for v in self.view_vars.values():
            v.set(False)
        self._request_layout()

    # ---- Source mgmt ----
    def reconnect(self):
        val = self.src_var.get().strip()
        src = int(val) if val.isdigit() else val
        self.stream.set_source(src)

    # ---- Layout ----
    def _request_layout(self):
        self._layout_dirty = True
        if self._layout_id is None:
            self._layout_id = self.after(50, self._do_layout)

    def _do_layout(self):
        self._layout_id = None
        if not self._layout_dirty:
            return
        self._layout_dirty = False

        for w in self.container.winfo_children():
            w.grid_forget()
            w.pack_forget()

        visible = [t for t in self.tiles if t.wants_in_grid()]
        visible = visible or self.tiles  # if none selected, show all

        n = len(visible)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / float(cols)))

        for r in range(rows):
            self.container.rowconfigure(r, weight=1)
        for c in range(cols):
            self.container.columnconfigure(c, weight=1)

        i = 0
        for t in visible:
            r = i // cols
            c = i % cols
            t.grid(row=r, column=c, sticky='nsew', padx=4, pady=4)
            i += 1

    # ---- Update loop ----
    def _update_loop(self):
        try:
            # Snapshot once per tick and reuse for all tiles
            frame, fps = self.stream.snapshot_and_prepare()
            self.status_lbl.config(text=f"status: {self.stream.status}")

            visible = [t for t in self.tiles if t.wants_in_grid()]
            visible = visible or self.tiles
            n = len(visible)
            cols = max(1, int(math.ceil(math.sqrt(n))))
            rows = int(math.ceil(n / float(cols)))
            cw = max(1, self.container.winfo_width())
            ch = max(1, self.container.winfo_height())
            tile_w = cw // cols - 8
            tile_h = ch // rows - 8
            for t in visible:
                t.render(self.stream, frame, fps, tile_w, tile_h)
        except Exception:
            pass
        finally:
            self.after(int(1000 / UI_FPS), self._update_loop)

    def on_close(self):
        self.stream.stop()
        self.destroy()


if __name__ == '__main__':
    app = App(CAMERA_SOURCE)
    app.mainloop()
