import kivy
kivy.require("2.0.0")

APP_NAME = "Will's Volume Toolkit"
APP_VERSION = "v1.0"

import json
import ssl
import threading
import time
from datetime import datetime
from io import BytesIO
import http.client

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, BooleanProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.scatter import Scatter
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput

from PIL import Image as PILImage

# QR (OpenCV)
import cv2
import numpy as np


def pil_rotate_90s(img, ang):
    """Rotate image by 0/90/180/270 degrees; compatible with old+new Pillow."""
    ang = int(ang) % 360
    if ang == 0:
        return img
    try:
        T = PILImage.Transpose
        if ang == 90:
            return img.transpose(T.ROTATE_90)
        if ang == 180:
            return img.transpose(T.ROTATE_180)
        if ang == 270:
            return img.transpose(T.ROTATE_270)
        return img
    except Exception:
        if ang == 90:
            return img.transpose(PILImage.ROTATE_90)
        if ang == 180:
            return img.transpose(PILImage.ROTATE_180)
        if ang == 270:
            return img.transpose(PILImage.ROTATE_270)
        return img


class PreviewOverlay(FloatLayout):
    """
    No-stretch preview with correct overlays:
      - Image is keep_ratio/contain.
      - Overlays are drawn in Image.canvas.after and use Image.norm_image_size.
    """
    show_border = BooleanProperty(True)
    show_grid = BooleanProperty(True)
    show_57 = BooleanProperty(True)
    show_810 = BooleanProperty(True)
    show_oval = BooleanProperty(True)

    # QR
    show_qr = BooleanProperty(True)

    grid_n = NumericProperty(3)

    oval_cx = NumericProperty(0.5)
    oval_cy = NumericProperty(0.5)
    oval_w = NumericProperty(0.55)
    oval_h = NumericProperty(0.75)

    preview_rotation = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.img = Image(allow_stretch=True, keep_ratio=True)
        try:
            self.img.fit_mode = "contain"
        except Exception:
            pass
        self.add_widget(self.img)

        lw = 2  # physical pixels

        with self.img.canvas.after:
            # Border (blue)
            self._c_border = Color(0.2, 0.6, 1.0, 1.0)
            self._ln_border = Line(width=lw)

            # Grid (orange) - vertical/horizontal only
            self._c_grid = Color(1.0, 0.6, 0.0, 0.85)
            self._ln_grid = Line(width=lw)

            # Crops
            self._c_57 = Color(1.0, 0.2, 0.2, 0.95)
            self._ln_57 = Line(width=lw)

            self._c_810 = Color(1.0, 0.9, 0.2, 0.95)
            self._ln_810 = Line(width=lw)

            # Oval
            self._c_oval = Color(0.7, 0.2, 1.0, 0.95)
            self._ln_oval = Line(width=lw)

            # QR overlay (green)
            self._c_qr = Color(0.2, 1.0, 0.2, 0.95)
            self._ln_qr = Line(width=lw, close=True)
            # (Text label is handled by app-level Label overlay for simplicity)

        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(
            show_border=self._redraw, show_grid=self._redraw, show_57=self._redraw,
            show_810=self._redraw, show_oval=self._redraw, show_qr=self._redraw,
            grid_n=self._redraw,
            oval_cx=self._redraw, oval_cy=self._redraw, oval_w=self._redraw, oval_h=self._redraw
        )
        self.img.bind(pos=self._redraw, size=self._redraw, texture=self._redraw, texture_size=self._redraw)

        # Latest QR points in image pixel coords (rotated image coords)
        self._qr_points_px = None  # list[(x,y)] length 4

        self._redraw()

    def set_texture(self, texture):
        self.img.texture = texture
        self._redraw()

    def set_qr(self, points_px):
        """points_px: list of 4 (x,y) in rotated image pixel coordinates, or None"""
        self._qr_points_px = points_px
        self._redraw()

    def _drawn_rect(self):
        wx, wy = self.img.pos
        ww, wh = self.img.size
        try:
            iw, ih = self.img.norm_image_size
        except Exception:
            return (wx, wy, ww, wh)
        dx = wx + (ww - iw) / 2.0
        dy = wy + (wh - ih) / 2.0
        return (dx, dy, iw, ih)

    @staticmethod
    def _center_crop_rect(frame_x, frame_y, frame_w, frame_h, aspect):
        frame_aspect = frame_w / frame_h
        if frame_aspect >= aspect:
            h = frame_h
            w = h * aspect
        else:
            w = frame_w
            h = w / aspect
        x = frame_x + (frame_w - w) / 2.0
        y = frame_y + (frame_h - h) / 2.0
        return (x, y, w, h)

    def _crop_aspect(self, a_w, a_h, fw, fh):
        # Landscape => swap so crop is oriented correctly
        if fw >= fh:
            return float(a_h) / float(a_w)
        return float(a_w) / float(a_h)

    def _clear_line_modes(self, ln: Line):
        # Clearing all modes prevents stale geometry artifacts.
        try:
            ln.points = []
        except Exception:
            pass
        try:
            ln.rectangle = (0, 0, 0, 0)
        except Exception:
            pass

    def _redraw(self, *args):
        fx, fy, fw, fh = self._drawn_rect()

        # Border
        self._ln_border.rectangle = (fx, fy, fw, fh) if self.show_border else (0, 0, 0, 0)

        # Crops
        if self.show_57:
            asp57 = self._crop_aspect(5.0, 7.0, fw, fh)
            self._ln_57.rectangle = self._center_crop_rect(fx, fy, fw, fh, asp57)
        else:
            self._ln_57.rectangle = (0, 0, 0, 0)

        if self.show_810:
            asp810 = self._crop_aspect(4.0, 5.0, fw, fh)
            self._ln_810.rectangle = self._center_crop_rect(fx, fy, fw, fh, asp810)
        else:
            self._ln_810.rectangle = (0, 0, 0, 0)

        # Grid (NO diagonals)
        n = int(self.grid_n)
        if self.show_grid and n >= 2:
            pts = []
            for i in range(1, n):
                x = fx + fw * (i / n)
                pts += [x, fy, x, fy + fh]
            for i in range(1, n):
                y = fy + fh * (i / n)
                pts += [fx, y, fx + fw, y]
            self._ln_grid.points = pts
        else:
            self._ln_grid.points = []

        # Oval: explicitly clear other modes to prevent stale Line geometry artifacts.
        if self.show_oval:
            cx = fx + fw * float(self.oval_cx)
            cy = fy + fh * float(self.oval_cy)
            ow = fw * float(self.oval_w)
            oh = fh * float(self.oval_h)

            ow = max(0.05 * fw, min(ow, fw))
            oh = max(0.05 * fh, min(oh, fh))

            left = max(fx, min(cx - ow / 2.0, fx + fw - ow))
            bottom = max(fy, min(cy - oh / 2.0, fy + fh - oh))

            self._clear_line_modes(self._ln_oval)
            self._ln_oval.ellipse = (left, bottom, ow, oh)
        else:
            self._clear_line_modes(self._ln_oval)
            self._ln_oval.ellipse = (0, 0, 0, 0)

        # QR polygon overlay
        if self.show_qr and self._qr_points_px and self.img.texture and self.img.texture.size[0] > 0:
            iw, ih = self.img.texture.size  # rotated image size
            dx, dy, dw, dh = fx, fy, fw, fh

            pts = []
            for (x, y) in self._qr_points_px:
                u = float(x) / float(iw)
                v = float(y) / float(ih)
                sx = dx + u * dw
                sy = dy + v * dh
                pts += [sx, sy]

            self._ln_qr.points = pts
        else:
            self._ln_qr.points = []


class CanonLiveViewApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.connected = False
        self.camera_ip = "192.168.34.29"

        self.live_running = False
        self.session_started = False

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_jpeg_ts = 0.0
        self._last_decoded_ts = 0.0

        self._fetch_thread = None
        self._display_event = None

        self._flip_conn = None
        self._flip_ssl_ctx = ssl._create_unverified_context()
        self._flip_timeout = 3.0

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._log_lines = []
        self._max_log_lines = 300

        self._frame_texture = None
        self._frame_size = None

        self.dropdown = None
        self.show_log = True

        # QR state
        self.qr_enabled = True
        self.qr_interval_s = 0.15
        self.qr_new_gate_s = 0.70
        self._qr_detector = cv2.QRCodeDetector()
        self._qr_thread = None
        self._latest_qr_text = None
        self._latest_qr_points = None  # list[(x,y)] in rotated image px coords
        self._qr_seen = set()
        self._qr_last_add_time = 0.0

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        header = BoxLayout(size_hint=(1, None), height=dp(40), spacing=dp(6))
        header.add_widget(Label(text=f"{APP_NAME}  {APP_VERSION}", font_size=sp(18)))
        self.menu_btn = Button(text="Menu", size_hint=(None, 1), width=dp(90), font_size=sp(16))
        header.add_widget(self.menu_btn)
        root.add_widget(header)

        row1 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.ip_input = TextInput(text=self.camera_ip, multiline=False, font_size=sp(16), padding=[dp(10)] * 4)
        row1.add_widget(self.ip_input)
        root.add_widget(row1)

        row2 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(44))
        self.connect_btn = Button(text="Connect", font_size=sp(16), background_normal="", background_down="",
                                  background_color=(0.35, 0.75, 1.0, 1.0), color=(0, 0, 0, 1))
        self.start_btn = Button(text="Start", disabled=True, font_size=sp(16), background_normal="", background_down="",
                                background_color=(0.25, 0.85, 0.35, 1.0), color=(0, 0, 0, 1))
        self.stop_btn = Button(text="Stop", disabled=True, font_size=sp(16), background_normal="", background_down="",
                               background_color=(1.0, 0.35, 0.35, 1.0), color=(0, 0, 0, 1))
        row2.add_widget(self.connect_btn)
        row2.add_widget(self.start_btn)
        row2.add_widget(self.stop_btn)
        root.add_widget(row2)

        row3 = BoxLayout(spacing=dp(6), size_hint=(1, None), height=dp(40))
        row3.add_widget(Label(text="Display FPS", size_hint=(None, 1), width=dp(110), font_size=sp(14)))
        self.fps_slider = Slider(min=5, max=30, value=12, step=1)
        self.fps_label = Label(text="12", size_hint=(None, 1), width=dp(50), font_size=sp(14))
        row3.add_widget(self.fps_slider)
        row3.add_widget(self.fps_label)
        root.add_widget(row3)

        self.status = Label(text="Status: not connected", size_hint=(1, None), height=dp(22), font_size=sp(13))
        self.metrics = Label(text="Delay: -- ms | Fetch: 0 | Decode: 0 | Display: 0",
                             size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.status)
        root.add_widget(self.metrics)

        # QR status line
        self.qr_status = Label(text="QR: none", size_hint=(1, None), height=dp(22), font_size=sp(13))
        root.add_widget(self.qr_status)

        preview_holder = AnchorLayout(anchor_x="center", anchor_y="center", size_hint=(1, 0.60))
        self.preview_scatter = Scatter(do_translation=True, do_scale=True, do_rotation=False, scale_min=0.5, scale_max=2.5)
        self.preview_scatter.size_hint = (None, None)

        self.preview = PreviewOverlay(size_hint=(None, None))
        self.preview_scatter.add_widget(self.preview)
        preview_holder.add_widget(self.preview_scatter)
        root.add_widget(preview_holder)

        def fit_preview_to_holder(*_):
            w = max(dp(220), preview_holder.width * 0.92)
            h = max(dp(220), preview_holder.height * 0.92)
            self.preview_scatter.size = (w, h)
            self.preview.size = (w, h)
            self.preview_scatter.scale = 1.0
            self.preview_scatter.pos = (
                preview_holder.x + (preview_holder.width - w) / 2.0,
                preview_holder.y + (preview_holder.height - h) / 2.0
            )

        preview_holder.bind(pos=fit_preview_to_holder, size=fit_preview_to_holder)

        # Log section (hide/show)
        self.log_holder = BoxLayout(orientation="vertical", size_hint=(1, None), height=dp(150))
        log_sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.log_label = Label(text="", size_hint_y=None, halign="left", valign="top", font_size=sp(11))
        self.log_label.bind(width=lambda *_: setattr(self.log_label, "text_size", (self.log_label.width, None)))
        self.log_label.bind(texture_size=lambda *_: setattr(self.log_label, "height", self.log_label.texture_size[1]))
        log_sv.add_widget(self.log_label)
        self.log_holder.add_widget(log_sv)
        root.add_widget(self.log_holder)

        self.dropdown = self._build_dropdown(fit_preview_to_holder)
        self.menu_btn.bind(on_release=lambda *_: self.dropdown.open(self.menu_btn))

        self.connect_btn.bind(on_press=lambda *_: self.connect_camera())
        self.start_btn.bind(on_press=lambda *_: self.start_liveview())
        self.stop_btn.bind(on_press=lambda *_: self.stop_liveview())
        self.fps_slider.bind(value=self._on_fps_change)

        Window.bind(on_resize=lambda *_: self._update_orientation_label())

        self._reschedule_display_loop(int(self.fps_slider.value))
        self._set_controls_idle()
        self._update_orientation_label()
        self.log("Ready")
        return root

    def _orientation(self):
        w, h = Window.size
        return "Landscape" if w >= h else "Portrait"

    def _update_orientation_label(self):
        if hasattr(self, "orientation_label"):
            self.orientation_label.text = f"Orientation: {self._orientation()}"

    def _set_log_visible(self, visible: bool):
        self.show_log = bool(visible)
        self.log_holder.height = dp(150) if self.show_log else 0
        self.log_holder.opacity = 1 if self.show_log else 0
        self.log_holder.disabled = not self.show_log

    def _style_menu_button(self, b):
        b.background_normal = ""
        b.background_down = ""
        b.background_color = (0.10, 0.10, 0.10, 0.80)
        b.color = (1, 1, 1, 1)
        return b

    def _build_dropdown(self, reset_callback):
        dd = DropDown(auto_dismiss=True)
        dd.auto_width = False
        dd.width = dp(330)
        dd.max_height = dp(560)

        with dd.canvas.before:
            Color(0.0, 0.0, 0.0, 0.80)
            panel = Rectangle(pos=dd.pos, size=dd.size)
        dd.bind(pos=lambda *_: setattr(panel, "pos", dd.pos), size=lambda *_: setattr(panel, "size", dd.size))

        def add_header(text):
            dd.add_widget(Label(text=text, size_hint_y=None, height=dp(28), font_size=sp(15), color=(1, 1, 1, 1)))

        def add_button(text, on_press):
            b = Button(text=text, size_hint_y=None, height=dp(44), font_size=sp(14))
            self._style_menu_button(b)
            b.bind(on_release=lambda *_: on_press())
            dd.add_widget(b)

        def add_row_bg(row):
            with row.canvas.before:
                Color(0.10, 0.10, 0.10, 0.80)
                r = Rectangle(pos=row.pos, size=row.size)
            row.bind(pos=lambda *_: setattr(r, "pos", row.pos), size=lambda *_: setattr(r, "size", row.size))

        def add_toggle(text, initial, on_change):
            row = BoxLayout(size_hint_y=None, height=dp(40), padding=[dp(6), 0, dp(6), 0])
            add_row_bg(row)
            row.add_widget(Label(text=text, font_size=sp(14), color=(1, 1, 1, 1)))
            cb = CheckBox(active=initial, size_hint=(None, 1), width=dp(44))
            cb.bind(active=lambda inst, val: on_change(val))
            row.add_widget(cb)
            dd.add_widget(row)

        def add_slider(text, minv, maxv, val, step, on_change, label_w=dp(110)):
            row = BoxLayout(size_hint_y=None, height=dp(42), padding=[dp(6), 0, dp(6), 0], spacing=dp(6))
            add_row_bg(row)
            row.add_widget(Label(text=text, size_hint=(None, 1), width=label_w, font_size=sp(13), color=(1, 1, 1, 1)))
            s = Slider(min=minv, max=maxv, value=val, step=step)
            vlab = Label(text=str(int(val) if step == 1 else f"{val:.2f}"),
                         size_hint=(None, 1), width=dp(55), font_size=sp(13), color=(1, 1, 1, 1))

            def _upd(_, v):
                vlab.text = str(int(v)) if step == 1 else f"{v:.2f}"
                on_change(v)

            s.bind(value=_upd)
            row.add_widget(s)
            row.add_widget(vlab)
            dd.add_widget(row)

        add_header("Device")
        self.orientation_label = Label(text=f"Orientation: {self._orientation()}",
                                       size_hint_y=None, height=dp(26), font_size=sp(13), color=(1, 1, 1, 1))
        dd.add_widget(self.orientation_label)

        add_header("Framing")
        add_button("Reset framing", reset_callback)

        rot_row = BoxLayout(size_hint_y=None, height=dp(44), padding=[dp(6), 0, dp(6), 0], spacing=dp(6))
        add_row_bg(rot_row)
        rot_row.add_widget(Label(text="Rotate", size_hint=(None, 1), width=dp(110), font_size=sp(13), color=(1, 1, 1, 1)))
        for a in (0, 90, 180, 270):
            b = Button(text=str(a), size_hint=(None, 1), width=dp(52), font_size=sp(13))
            self._style_menu_button(b)
            b.bind(on_release=lambda btn, ang=a: setattr(self.preview, "preview_rotation", ang))
            rot_row.add_widget(b)
        dd.add_widget(rot_row)

        add_header("Overlays")
        add_toggle("Border (blue)", True, lambda v: setattr(self.preview, "show_border", v))
        add_toggle("Grid (orange)", True, lambda v: setattr(self.preview, "show_grid", v))
        add_toggle("Crop 5:7 (red)", True, lambda v: setattr(self.preview, "show_57", v))
        add_toggle("Crop 8:10 (yellow)", True, lambda v: setattr(self.preview, "show_810", v))
        add_toggle("Oval (purple)", True, lambda v: setattr(self.preview, "show_oval", v))

        add_header("QR")
        add_toggle("QR overlay", True, lambda v: setattr(self.preview, "show_qr", v))
        add_toggle("QR detect (OpenCV)", True, lambda v: self._set_qr_enabled(v))
        add_slider("QR interval (s)", 0.05, 1.0, self.qr_interval_s, 0.01, lambda v: self._set_qr_interval(v), label_w=dp(140))

        add_header("Grid")
        add_slider("Divisions", 0, 10, 3, 1, lambda v: setattr(self.preview, "grid_n", int(v)))

        add_header("Oval")
        add_slider("Center X", 0.0, 1.0, 0.5, 0.01, lambda v: setattr(self.preview, "oval_cx", float(v)))
        add_slider("Center Y", 0.0, 1.0, 0.5, 0.01, lambda v: setattr(self.preview, "oval_cy", float(v)))
        add_slider("Width", 0.1, 1.0, 0.55, 0.01, lambda v: setattr(self.preview, "oval_w", float(v)))
        add_slider("Height", 0.1, 1.0, 0.75, 0.01, lambda v: setattr(self.preview, "oval_h", float(v)))

        add_header("UI")
        add_toggle("Show log", True, lambda v: self._set_log_visible(v))
        add_toggle("Allow drag/scale", True, lambda v: self._set_adjust_enabled(v))

        return dd

    def _set_qr_enabled(self, enabled: bool):
        self.qr_enabled = bool(enabled)
        if not self.qr_enabled:
            self._set_qr_ui(None, None, note="QR: off")

    def _set_qr_interval(self, interval_s: float):
        try:
            self.qr_interval_s = float(interval_s)
        except Exception:
            self.qr_interval_s = 0.15

    def _set_adjust_enabled(self, enabled: bool):
        self.preview_scatter.do_translation = bool(enabled)
        self.preview_scatter.do_scale = bool(enabled)

    def _on_fps_change(self, *_):
        fps = int(self.fps_slider.value)
        self.fps_label.text = str(fps)
        self._reschedule_display_loop(fps)

    def _reschedule_display_loop(self, fps):
        if self._display_event is not None:
            self._display_event.cancel()
        fps = max(1, int(fps))
        self._display_event = Clock.schedule_interval(self._ui_decode_and_display, 1.0 / fps)

    def _set_controls_idle(self):
        self.ip_input.disabled = False
        self.connect_btn.disabled = False
        self.start_btn.disabled = not self.connected
        self.stop_btn.disabled = True

    def _set_controls_running(self):
        self.ip_input.disabled = True
        self.connect_btn.disabled = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        if hasattr(self, "log_label"):
            self.log_label.text = "\n".join(self._log_lines)

    def _json_call(self, method, path, payload=None, timeout=8.0):
        ctx = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(self.camera_ip, 443, timeout=timeout, context=ctx)
        headers = {"Host": self.camera_ip}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        status = f"{resp.status} {resp.reason}"
        j = None
        try:
            if data and b"{" in data:
                j = json.loads(data[data.find(b"{"):].decode("utf-8", errors="ignore"))
        except Exception:
            j = None
        conn.close()
        return status, j

    def connect_camera(self):
        if self.live_running:
            self.log("Connect disabled while live view is running. Stop first.")
            return

        self.camera_ip = self.ip_input.text.strip()
        if not self.camera_ip:
            self.status.text = "Status: enter an IP"
            return

        self.status.text = f"Status: connecting to {self.camera_ip}:443..."
        self.log(f"Connecting to {self.camera_ip}:443")

        try:
            status, data = self._json_call("GET", "/ccapi/ver100/deviceinformation", None, timeout=8.0)
            if status.startswith("200") and data:
                self.connected = True
                self.status.text = f"Status: connected ({data.get('productname', 'camera')})"
                self.log("Connected OK")
            else:
                raise Exception(status)
        except Exception as e:
            self.connected = False
            self.status.text = f"Status: connect failed ({e})"
            self.log(f"Connect failed: {e}")

        self._set_controls_idle()

    def start_liveview(self):
        if not self.connected or self.live_running:
            return

        payload = {"liveviewsize": "small", "cameradisplay": "on"}
        self.log("Starting live view size=small, cameradisplay=on")

        status, _ = self._json_call("POST", "/ccapi/ver100/shooting/liveview", payload, timeout=10.0)
        if not status.startswith("200"):
            self.status.text = f"Status: live view start failed ({status})"
            self.log(f"Live view start failed: {status}")
            return

        self.session_started = True
        self.live_running = True
        self._set_controls_running()
        self.status.text = "Status: live"

        with self._lock:
            self._latest_jpeg = None
            self._latest_jpeg_ts = 0.0
        self._last_decoded_ts = 0.0
        self._frame_texture = None
        self._frame_size = None

        # reset QR state
        self._latest_qr_text = None
        self._latest_qr_points = None
        self._qr_seen = set()
        self._qr_last_add_time = 0.0
        self._set_qr_ui(None, None, note="QR: none")

        self._fetch_count = 0
        self._decode_count = 0
        self._display_count = 0
        self._stat_t0 = time.time()

        self._fetch_thread = threading.Thread(target=self._flip_fetch_keepalive_loop, daemon=True)
        self._fetch_thread.start()

        # QR worker
        self._qr_thread = threading.Thread(target=self._qr_loop, daemon=True)
        self._qr_thread.start()

    def stop_liveview(self):
        if not self.live_running:
            self._set_controls_idle()
            return

        self.live_running = False
        self._close_flip_conn()

        if self.session_started:
            try:
                self._json_call("DELETE", "/ccapi/ver100/shooting/liveview", None, timeout=6.0)
            except Exception:
                pass
            self.session_started = False

        self._set_qr_ui(None, None, note="QR: none")
        self.status.text = "Status: connected (live stopped)" if self.connected else "Status: not connected"
        self.log("Live view stopped")
        self._set_controls_idle()

    def _close_flip_conn(self):
        try:
            if self._flip_conn is not None:
                self._flip_conn.close()
        except Exception:
            pass
        self._flip_conn = None

    def _flip_conn_get(self):
        if self._flip_conn is None:
            self._flip_conn = http.client.HTTPSConnection(
                self.camera_ip, 443, timeout=self._flip_timeout, context=self._flip_ssl_ctx
            )
        return self._flip_conn

    def _flip_fetch_keepalive_loop(self):
        while self.live_running:
            try:
                conn = self._flip_conn_get()
                conn.request("GET", "/ccapi/ver100/shooting/liveview/flip",
                             headers={"Host": self.camera_ip, "Connection": "keep-alive"})
                resp = conn.getresponse()
                body = resp.read()
                if resp.status == 200 and body:
                    with self._lock:
                        self._latest_jpeg = body
                        self._latest_jpeg_ts = time.time()
                    self._fetch_count += 1
                else:
                    time.sleep(0.03)
            except Exception as e:
                self.log(f"flip reconnect: {e}")
                self._close_flip_conn()
                time.sleep(0.10)

    def _qr_loop(self):
        while self.live_running:
            if not self.qr_enabled:
                time.sleep(0.10)
                continue

            # grab the newest JPEG snapshot
            with self._lock:
                jpeg = self._latest_jpeg
                jpeg_ts = self._latest_jpeg_ts

            if not jpeg:
                time.sleep(0.05)
                continue

            try:
                pil = PILImage.open(BytesIO(jpeg)).convert("RGB")
                pil = pil_rotate_90s(pil, self.preview.preview_rotation)

                rgb = np.array(pil)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                data, points, _ = self._qr_detector.detectAndDecode(bgr)

                qr_text = data.strip() if isinstance(data, str) else ""
                qr_points = None
                if points is not None:
                    try:
                        pts = points.astype(int).reshape(-1, 2)
                        if len(pts) >= 4:
                            qr_points = [(int(pts[i][0]), int(pts[i][1])) for i in range(4)]
                    except Exception:
                        qr_points = None

                # Update overlay even if undecoded, if corners are present
                if qr_text or qr_points:
                    self._publish_qr(qr_text if qr_text else None, qr_points)
                else:
                    self._publish_qr(None, None)

            except Exception:
                # keep quiet; QR is best-effort
                pass

            time.sleep(max(0.05, float(self.qr_interval_s)))

    def _publish_qr(self, text, points):
        # Gate "new QR" events (log/list behaviors), but still allow overlay refresh.
        now = time.time()
        if text:
            if (text not in self._qr_seen) and (now - self._qr_last_add_time >= self.qr_new_gate_s):
                self._qr_seen.add(text)
                self._qr_last_add_time = now
                self.log(f"QR: {text}")

        note = None
        if not self.qr_enabled:
            note = "QR: off"
        elif text:
            note = f"QR: {text[:80]}"
        elif points:
            note = "QR: detected (undecoded)"
        else:
            note = "QR: none"

        Clock.schedule_once(lambda *_: self._set_qr_ui(text, points, note=note), 0)

    def _set_qr_ui(self, text, points, note="QR: none"):
        self._latest_qr_text = text
        self._latest_qr_points = points
        self.qr_status.text = note
        self.preview.set_qr(points)

    def _ui_decode_and_display(self, dt):
        if not self.live_running:
            return

        with self._lock:
            jpeg = self._latest_jpeg
            jpeg_ts = self._latest_jpeg_ts

        self._display_count += 1
        if not jpeg or jpeg_ts <= self._last_decoded_ts:
            self._update_metrics(jpeg_ts)
            return

        try:
            pil = PILImage.open(BytesIO(jpeg)).convert("RGB")
            pil = pil_rotate_90s(pil, self.preview.preview_rotation)

            w, h = pil.size
            rgb = pil.tobytes()

            if self._frame_texture is None or self._frame_size != (w, h):
                tex = Texture.create(size=(w, h), colorfmt="rgb")
                tex.flip_vertical()
                self._frame_texture = tex
                self._frame_size = (w, h)
                self.log(f"texture init size={w}x{h}")

            self._frame_texture.blit_buffer(rgb, colorfmt="rgb", bufferfmt="ubyte")
            self.preview.set_texture(self._frame_texture)

            self._decode_count += 1
            self._last_decoded_ts = jpeg_ts

        except Exception as e:
            self.log(f"ui decode err: {e}")

        self._update_metrics(jpeg_ts)

    def _update_metrics(self, frame_ts):
        now = time.time()
        if now - self._stat_t0 >= 1.0:
            dt_s = now - self._stat_t0
            fetch_fps = self._fetch_count / dt_s
            dec_fps = self._decode_count / dt_s
            disp_fps = self._display_count / dt_s
            delay_ms = int((now - frame_ts) * 1000) if frame_ts else -1
            self.metrics.text = (
                f"Delay: {delay_ms if delay_ms >= 0 else '--'} ms | "
                f"Fetch: {fetch_fps:.1f} | Decode: {dec_fps:.1f} | Display: {disp_fps:.1f}"
            )
            self._fetch_count = 0
            self._decode_count = 0
            self._display_count = 0
            self._stat_t0 = now

    def on_stop(self):
        try:
            self.stop_liveview()
        except Exception:
            pass


if __name__ == "__main__":
    CanonLiveViewApp().run()
