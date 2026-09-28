"""Module 3 - low-latency multi-person and YOLO pose camera manager.

This module intentionally does not import or change detection_manager_M2.py.
The pipeline is: capture thread -> latest-frame buffer -> AI worker -> GUI/display.
Only the newest unprocessed frame is retained, preventing latency backlogs.
"""
import os
import queue
import shutil
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import messagebox, ttk

import cv2


# ----------------------------- Configuration -----------------------------
DEFAULT_CAMERA_URL = "http://192.168.1.10:8080/video"
VIDEO_FOLDER = "videos"
SCREENSHOT_FOLDER = "screenshots"
MODEL_FOLDER = "models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "yolo11n.pt")
POSE_MODEL_PATH = os.path.join(MODEL_FOLDER, "yolo11n-pose.pt")
CONFIDENCE_THRESHOLD = 0.50
POSE_KEYPOINT_CONFIDENCE = 0.40
INFERENCE_WIDTH, INFERENCE_HEIGHT = 640, 640
DETECTION_INTERVAL = 1  # 1 means process every newest available frame.
PERSON_CLASS_ID = 0
CAPTURE_BUFFER_SIZE = 1
PREVIEW_UPDATE_MS = 15
GUI_UPDATE_MS = 250
RECONNECT_DELAY = 3
AUTO_RECONNECT = True
RECORDING_FPS = 20
RECORDING_QUEUE_SIZE = 30
SHOW_TIMESTAMP = True
SHOW_FPS = True
RESIZE_FRAME = False
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

CAMERAS = [
    {"name": "IP Webcam", "type": "IP Webcam", "source": DEFAULT_CAMERA_URL},
    {"name": "USB Camera", "type": "USB Camera", "source": 0},
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]
# COCO-17 topology.  A segment is painted only if both joint confidences pass.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


class RollingFPS:
    """A short time-window FPS meter that does not use inference duration."""

    def __init__(self, window_seconds=2.0):
        self.window_seconds = window_seconds
        self.timestamps = deque()
        self.lock = threading.Lock()

    def tick(self):
        now = time.perf_counter()
        with self.lock:
            self.timestamps.append(now)
            cutoff = now - self.window_seconds
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()

    def value(self):
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
            duration = self.timestamps[-1] - self.timestamps[0]
            return (len(self.timestamps) - 1) / duration if duration else 0.0


class CameraManager:
    """Tkinter camera manager with independent capture, AI, and recording workers."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera Manager - Module 3")
        self.root.geometry("455x1010")
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        self.camera = None
        self.camera_source = None
        self.camera_name = None
        self.preview_window_name = None
        self.preview_running = False
        self.reconnecting = False
        self.shutdown_event = threading.Event()
        self.capture_stop_event = threading.Event()
        self.detection_stop_event = threading.Event()
        self.recording_stop_event = threading.Event()
        self.camera_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        self.current_frame = None              # latest original capture frame
        self.current_frame_id = 0
        self.latest_result = self.empty_result()
        self.capture_thread = None
        self.detection_thread = None
        self.recording_thread = None
        self.recording_queue = queue.Queue(maxsize=RECORDING_QUEUE_SIZE)
        self.video_writer = None
        self.is_recording = False
        self.current_video_path = None
        self.screenshot_counter = 0
        self.selected_camera_index = None
        self.capture_fps = RollingFPS()
        self.detection_fps = RollingFPS()
        self.display_fps = RollingFPS()
        self.device = self.detect_device()
        self.person_model = None
        self.pose_model = None
        self.model_load_lock = threading.Lock()
        self.person_model_error = None
        self.pose_model_error = None

        # Tk variables are written only in the GUI thread.
        self.status_variable = tk.StringVar(value="Status: No camera connected")
        self.person_detection_enabled = tk.BooleanVar(value=False)
        self.pose_detection_enabled = tk.BooleanVar(value=False)
        self.detection_status_variable = tk.StringVar(value="OFF")
        self.pose_status_variable = tk.StringVar(value="OFF")
        self.person_count_variable = tk.StringVar(value="0")
        self.camera_fps_variable = tk.StringVar(value="0.0")
        self.detection_fps_variable = tk.StringVar(value="0.0")
        self.display_fps_variable = tk.StringVar(value="0.0")
        self.inference_time_variable = tk.StringVar(value="0.0 ms")
        self.person_model_status_variable = tk.StringVar(value="Not loaded")
        self.pose_model_status_variable = tk.StringVar(value="Not loaded")
        self.device_variable = tk.StringVar(value=self.device.upper())
        self.person_detection_active = False
        self.pose_detection_active = False

        self.create_output_folders()
        self.load_initial_models()
        self.create_gui()

    def empty_result(self):
        return {"timestamp": 0.0, "frame_id": -1, "persons": [],
                "person_count": 0, "inference_time_ms": 0.0,
                "mode": "off", "error": None}

    def detect_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def create_output_folders(self):
        for folder in (VIDEO_FOLDER, SCREENSHOT_FOLDER, MODEL_FOLDER):
            os.makedirs(folder, exist_ok=True)

    def _load_model(self, path, label):
        """Load once; ask Ultralytics to download a missing standard model once."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics package is not installed. Run: pip install ultralytics")
            return None, "Not loaded"
        try:
            if not os.path.exists(path):
                filename = os.path.basename(path)
                print(f"{label} model missing; requesting one-time download: {filename}")
                YOLO(filename)
                if os.path.exists(filename):
                    shutil.move(filename, path)
            model = YOLO(path)
            print(f"{label} model loaded from {path} on {self.device.upper()}")
            return model, "Loaded"
        except Exception as error:
            print(f"ERROR: {label} model loading failed: {error}")
            return None, "Not loaded"

    def load_initial_models(self):
        # Loading is once per application lifetime. Failures leave camera features usable.
        self.person_model, person_status = self._load_model(MODEL_PATH, "Person")
        self.pose_model, pose_status = self._load_model(POSE_MODEL_PATH, "Pose")
        self.person_model_initial_status = person_status
        self.pose_model_initial_status = pose_status

    def create_gui(self):
        tk.Label(self.root, text="Camera Manager - Module 3", font=("Arial", 16, "bold")).pack(pady=(12, 6))
        tk.Label(self.root, textvariable=self.status_variable, fg="#0f766e").pack(pady=(0, 10))
        cameras = tk.LabelFrame(self.root, text="Cameras", padx=8, pady=8)
        cameras.pack(fill="x", padx=12, pady=6)
        self.camera_listbox = tk.Listbox(cameras, height=5)
        self.camera_listbox.pack(fill="x", pady=(0, 8))
        self.camera_listbox.bind("<<ListboxSelect>>", self.select_camera)
        row = tk.Frame(cameras); row.pack(fill="x")
        tk.Button(row, text="Connect", command=self.connect_selected_camera).pack(side=tk.LEFT, fill="x", expand=True, padx=2)
        tk.Button(row, text="Disconnect", command=self.disconnect_camera).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        details = tk.LabelFrame(self.root, text="Camera Details", padx=8, pady=8)
        details.pack(fill="x", padx=12, pady=6)
        tk.Label(details, text="Camera Name").pack(anchor="w")
        self.name_entry = tk.Entry(details); self.name_entry.pack(fill="x", pady=(0, 6))
        tk.Label(details, text="Camera Type").pack(anchor="w")
        self.type_combobox = ttk.Combobox(details, values=["IP Webcam", "USB Camera", "RTSP Camera"], state="normal")
        self.type_combobox.pack(fill="x", pady=(0, 6))
        tk.Label(details, text="Camera Source").pack(anchor="w")
        self.source_entry = tk.Entry(details); self.source_entry.pack(fill="x", pady=(0, 8))
        row = tk.Frame(details); row.pack(fill="x")
        for text, callback in (("Add Camera", self.add_camera), ("Edit Camera", self.edit_camera), ("Remove Camera", self.remove_camera)):
            tk.Button(row, text=text, command=callback).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        controls = tk.LabelFrame(self.root, text="Controls", padx=8, pady=8)
        controls.pack(fill="x", padx=12, pady=6)
        self.record_button = tk.Button(controls, text="Start Recording", command=self.start_recording)
        self.record_button.pack(fill="x", pady=3)
        tk.Button(controls, text="Capture Screenshot", command=self.capture_screenshot).pack(fill="x", pady=3)
        tk.Button(controls, text="Exit Application", command=self.close_application).pack(fill="x", pady=(10, 3))

        detection = tk.LabelFrame(self.root, text="Detection Controls", padx=8, pady=8)
        detection.pack(fill="x", padx=12, pady=6)
        tk.Checkbutton(detection, text="Enable Person Detection", variable=self.person_detection_enabled, command=self.handle_detection_toggle).pack(anchor="w")
        tk.Checkbutton(detection, text="Enable Pose Detection", variable=self.pose_detection_enabled, command=self.handle_detection_toggle).pack(anchor="w", pady=(0, 8))
        for label, value in (("Detection Status:", self.detection_status_variable), ("Pose Status:", self.pose_status_variable),
                             ("Person Count:", self.person_count_variable), ("Camera FPS:", self.camera_fps_variable),
                             ("Detection FPS:", self.detection_fps_variable), ("Display FPS:", self.display_fps_variable),
                             ("Inference Time:", self.inference_time_variable), ("Person Model:", self.person_model_status_variable),
                             ("Pose Model:", self.pose_model_status_variable), ("Device:", self.device_variable)):
            self.create_status_row(detection, label, value)
        self.person_model_status_variable.set(self.person_model_initial_status)
        self.pose_model_status_variable.set(self.pose_model_initial_status)
        self.refresh_camera_list(); self.select_first_camera()

    def create_status_row(self, parent, label, variable):
        row = tk.Frame(parent); row.pack(fill="x", pady=1)
        tk.Label(row, text=label).pack(side=tk.LEFT)
        tk.Label(row, textvariable=variable, font=("Arial", 9, "bold")).pack(side=tk.RIGHT)

    # ------------------------------ Camera GUI ------------------------------
    def refresh_camera_list(self):
        self.camera_listbox.delete(0, tk.END)
        for camera in CAMERAS:
            self.camera_listbox.insert(tk.END, f"{camera['name']}  |  {camera['type']}  |  {camera['source']}")

    def select_first_camera(self):
        if CAMERAS:
            self.camera_listbox.selection_set(0); self.select_camera()

    def get_selected_camera_index(self):
        selected = self.camera_listbox.curselection()
        return int(selected[0]) if selected else None

    def select_camera(self, event=None):
        index = self.get_selected_camera_index()
        if index is None: return
        self.selected_camera_index = index; camera = CAMERAS[index]
        for widget, value in ((self.name_entry, camera["name"]), (self.source_entry, camera["source"])):
            widget.delete(0, tk.END); widget.insert(0, str(value))
        self.type_combobox.set(str(camera["type"]))

    @staticmethod
    def normalize_camera_source(source):
        source = str(source).strip()
        return int(source) if source.isdigit() else source

    def read_camera_form(self):
        name, kind = self.name_entry.get().strip(), self.type_combobox.get().strip() or "IP Webcam"
        source = self.normalize_camera_source(self.source_entry.get())
        if not name or source == "":
            messagebox.showwarning("Missing Camera Details", "Enter a camera name and source."); return None
        return {"name": name, "type": kind, "source": source}

    def add_camera(self):
        camera = self.read_camera_form()
        if camera: CAMERAS.append(camera); self.refresh_camera_list()

    def edit_camera(self):
        index, camera = self.get_selected_camera_index(), self.read_camera_form()
        if index is None: messagebox.showwarning("No Camera Selected", "Select a camera to edit.")
        elif camera: CAMERAS[index] = camera; self.refresh_camera_list(); self.camera_listbox.selection_set(index)

    def remove_camera(self):
        index = self.get_selected_camera_index()
        if index is None: messagebox.showwarning("No Camera Selected", "Select a camera to remove.")
        else: print(f"Removed camera: {CAMERAS.pop(index)['name']}"); self.refresh_camera_list()

    def open_camera(self, source):
        print(f"Connecting to camera: {source}")
        camera = cv2.VideoCapture(source); camera.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not camera.isOpened(): camera.release(); print("ERROR: Camera could not be opened."); return None
        return camera

    def connect_selected_camera(self):
        index = self.get_selected_camera_index()
        if index is None: messagebox.showwarning("No Camera Selected", "Select a camera first."); return
        self.disconnect_camera(keep_status=True)
        selected = CAMERAS[index]; self.camera_source, self.camera_name = selected["source"], selected["name"]
        camera = self.open_camera(self.camera_source)
        if camera is None:
            self.status_variable.set("Status: Connection failed"); messagebox.showerror("Connection Failed", "The selected camera stream could not be opened."); return
        with self.camera_lock: self.camera = camera
        self.preview_running = True; self.capture_stop_event.clear(); self.detection_stop_event.clear()
        self.preview_window_name = str(self.camera_name)
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        self.status_variable.set(f"Status: Connected to {self.camera_name}")
        self.capture_thread = threading.Thread(target=self.capture_frames, name="camera-capture", daemon=False)
        self.detection_thread = threading.Thread(target=self.detection_worker, name="ai-detection", daemon=False)
        self.capture_thread.start()
        # Keep inference completely stopped until at least one AI control is on.
        with self.settings_lock:
            ai_enabled = self.person_detection_active or self.pose_detection_active
        if ai_enabled:
            self.detection_thread.start()

    # --------------------------- Worker pipeline ---------------------------
    def capture_frames(self):
        """Capture only: publishes one newest original frame; no YOLO work here."""
        while not self.capture_stop_event.is_set() and not self.shutdown_event.is_set():
            with self.camera_lock: camera = self.camera
            if camera is None:
                time.sleep(.05); continue
            success, frame = camera.read()
            if success and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame; self.current_frame_id += 1
                    frame_id = self.current_frame_id
                self.capture_fps.tick()
                self.enqueue_recording_frame(frame)
                continue
            print("ERROR: Camera read failed.")
            if AUTO_RECONNECT: self.attempt_reconnect()
            else: self.root.after(0, self.disconnect_camera); return

    def attempt_reconnect(self):
        if self.reconnecting or self.capture_stop_event.is_set(): return
        self.reconnecting = True
        self.root.after(0, lambda: self.status_variable.set(f"Status: Reconnecting in {RECONNECT_DELAY} seconds..."))
        with self.camera_lock:
            if self.camera is not None: self.camera.release(); self.camera = None
        if self.capture_stop_event.wait(RECONNECT_DELAY): self.reconnecting = False; return
        camera = self.open_camera(self.camera_source)
        if camera is not None:
            with self.camera_lock: self.camera = camera
            self.root.after(0, lambda: self.status_variable.set("Status: Reconnected"))
        else:
            print("ERROR: Reconnect failed; retrying after next read cycle.")
        self.reconnecting = False

    def handle_detection_toggle(self):
        """Runs on Tk's main thread; worker reads only plain protected booleans."""
        with self.settings_lock:
            self.person_detection_active = bool(self.person_detection_enabled.get() and self.person_model is not None)
            self.pose_detection_active = bool(self.pose_detection_enabled.get() and self.pose_model is not None)
            person_on, pose_on = self.person_detection_active, self.pose_detection_active
        if self.person_detection_enabled.get() and not person_on:
            print("ERROR: Person detection cannot start because its model is unavailable."); self.person_detection_enabled.set(False)
        if self.pose_detection_enabled.get() and not pose_on:
            print("ERROR: Pose detection cannot start because its model is unavailable."); self.pose_detection_enabled.set(False)
        self.detection_status_variable.set("ON" if (person_on or pose_on) else "OFF")
        self.pose_status_variable.set("ON" if pose_on else "OFF")
        if not person_on and not pose_on:
            with self.result_lock: self.latest_result = self.empty_result()
        elif self.preview_running and (self.detection_thread is None or not self.detection_thread.is_alive()):
            self.detection_stop_event.clear()
            self.detection_thread = threading.Thread(target=self.detection_worker, name="ai-detection", daemon=False)
            self.detection_thread.start()

    def detection_worker(self):
        """AI-only worker. It skips stale frames and uses pose boxes to avoid duplicate inference."""
        last_id, examined = -1, 0
        while not self.detection_stop_event.is_set() and not self.shutdown_event.is_set():
            with self.settings_lock: person_on, pose_on = self.person_detection_active, self.pose_detection_active
            if not person_on and not pose_on:
                time.sleep(.08); continue
            with self.frame_lock:
                frame, frame_id = self.current_frame, self.current_frame_id
            if frame is None or frame_id == last_id:
                time.sleep(.004); continue
            last_id = frame_id; examined += 1
            if DETECTION_INTERVAL > 1 and examined % DETECTION_INTERVAL:
                continue
            # This resize is the only intentional AI copy. Display/recording keep original size.
            height, width = frame.shape[:2]
            ai_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            try:
                started = time.perf_counter()
                if pose_on:
                    # Pose model emits person boxes and keypoints, satisfying both modes efficiently.
                    persons = self.run_pose_detection(ai_frame, width, height)
                    mode = "pose"
                else:
                    persons = self.run_person_detection(ai_frame, width, height)
                    mode = "person"
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = {"timestamp": time.time(), "frame_id": frame_id, "persons": persons,
                          "person_count": len(persons), "inference_time_ms": elapsed_ms,
                          "mode": mode, "error": None}
                with self.result_lock: self.latest_result = result
                self.detection_fps.tick()
            except Exception as error:
                print(f"ERROR: {'Pose' if pose_on else 'Person'} detection failed: {error}")
                with self.result_lock:
                    self.latest_result = {**self.empty_result(), "frame_id": frame_id, "error": str(error)}

    def predict(self, model, frame):
        return model.predict(source=frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                             imgsz=(INFERENCE_HEIGHT, INFERENCE_WIDTH), device=self.device, verbose=False)

    def run_person_detection(self, ai_frame, original_width, original_height):
        results = self.predict(self.person_model, ai_frame)
        if not results or results[0].boxes is None: return []
        sx, sy = original_width / INFERENCE_WIDTH, original_height / INFERENCE_HEIGHT
        people = []
        for index, box in enumerate(results[0].boxes):
            if int(box.cls[0]) != PERSON_CLASS_ID: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            people.append({"person_id": None, "bbox": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)),
                           "confidence": float(box.conf[0]), "class_id": PERSON_CLASS_ID, "keypoints": []})
        return people

    def run_pose_detection(self, ai_frame, original_width, original_height):
        results = self.predict(self.pose_model, ai_frame)
        if not results or results[0].boxes is None: return []
        result = results[0]; sx, sy = original_width / INFERENCE_WIDTH, original_height / INFERENCE_HEIGHT
        xy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
        confidence = result.keypoints.conf.cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else []
        people = []
        for index, box in enumerate(result.boxes):
            if int(box.cls[0]) != PERSON_CLASS_ID: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist(); points = []
            if index < len(xy):
                for joint, name in enumerate(KEYPOINT_NAMES):
                    if joint >= len(xy[index]): break
                    x, y = xy[index][joint]; conf = float(confidence[index][joint]) if index < len(confidence) else 0.0
                    points.append({"name": name, "x": int(x*sx), "y": int(y*sy), "confidence": conf})
            people.append({"person_id": None, "bbox": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)),
                           "confidence": float(box.conf[0]), "class_id": PERSON_CLASS_ID, "keypoints": points})
        return people

    # ------------------------------ Presentation -----------------------------
    def draw_detections(self, frame, result):
        for ordinal, person in enumerate(result["persons"], 1):
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_label = f"Person {person['confidence'] * 100:.0f}%"
            cv2.putText(frame, person_label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 0), 2, cv2.LINE_AA)
            self.draw_pose_skeleton(frame, person["keypoints"])
        cv2.putText(frame, f"Persons: {result['person_count']}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_pose_skeleton(self, frame, keypoints):
        visible = {i: point for i, point in enumerate(keypoints) if point["confidence"] >= POSE_KEYPOINT_CONFIDENCE}
        for start, end in SKELETON:
            if start in visible and end in visible:
                a, b = visible[start], visible[end]
                cv2.line(frame, (a["x"], a["y"]), (b["x"], b["y"]), (255, 180, 0), 2, cv2.LINE_AA)
        for point in visible.values():
            cv2.circle(frame, (point["x"], point["y"]), 4, (0, 0, 255), -1, cv2.LINE_AA)

    def add_overlay_text(self, frame):
        if SHOW_FPS:
            cv2.putText(frame, f"Display FPS: {self.display_fps.value():.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2, cv2.LINE_AA)
        if SHOW_TIMESTAMP:
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 58), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def update_preview(self):
        if self.preview_running and self.preview_window_name:
            with self.frame_lock: frame = self.current_frame
            if frame is not None and frame.size:
                with self.result_lock: result = dict(self.latest_result); result["persons"] = list(self.latest_result["persons"])
                display = frame.copy()  # needed exclusively for visual overlays
                if result["mode"] != "off": self.draw_detections(display, result)
                if RESIZE_FRAME: display = cv2.resize(display, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                self.display_fps.tick(); cv2.imshow(self.preview_window_name, self.add_overlay_text(display))
            try:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")): self.disconnect_camera()
            except cv2.error as error: print(f"ERROR: Preview operation failed: {error}")
        if not self.shutdown_event.is_set(): self.root.after(PREVIEW_UPDATE_MS, self.update_preview)

    def update_detection_gui(self):
        with self.result_lock: result = self.latest_result
        self.person_count_variable.set(str(result["person_count"]))
        self.camera_fps_variable.set(f"{self.capture_fps.value():.1f}")
        self.detection_fps_variable.set(f"{self.detection_fps.value():.1f}")
        self.display_fps_variable.set(f"{self.display_fps.value():.1f}")
        self.inference_time_variable.set(f"{result['inference_time_ms']:.1f} ms")
        if not self.shutdown_event.is_set(): self.root.after(GUI_UPDATE_MS, self.update_detection_gui)

    # ------------------------------ Recording ------------------------------
    def enqueue_recording_frame(self, frame):
        with self.recording_lock: recording = self.is_recording
        if not recording: return
        try: self.recording_queue.put_nowait(frame)
        except queue.Full:
            try: self.recording_queue.get_nowait()
            except queue.Empty: pass
            try: self.recording_queue.put_nowait(frame)
            except queue.Full: pass

    def start_recording(self):
        with self.frame_lock: frame = self.current_frame
        if frame is None: messagebox.showwarning("No Frame Available", "Connect to a camera first."); return
        with self.recording_lock:
            if self.is_recording: return
            path = os.path.join(VIDEO_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.avi"))
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), RECORDING_FPS, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened(): writer.release(); messagebox.showerror("Recording Failed", "Could not create the video file."); return
            self.video_writer, self.current_video_path, self.is_recording = writer, path, True
        self.recording_stop_event.clear()
        self.recording_thread = threading.Thread(target=self.recording_worker, name="video-recording", daemon=False)
        self.recording_thread.start(); self.record_button.config(text="Stop Recording", command=self.stop_recording)
        print(f"Recording started: {path}")

    def recording_worker(self):
        while not self.recording_stop_event.is_set() or not self.recording_queue.empty():
            try: frame = self.recording_queue.get(timeout=.15)
            except queue.Empty: continue
            with self.recording_lock: writer = self.video_writer
            if writer is not None:
                try: writer.write(frame)
                except Exception as error: print(f"ERROR: VideoWriter failed: {error}")

    def stop_recording(self):
        with self.recording_lock:
            if not self.is_recording: return
            self.is_recording = False; path = self.current_video_path
        self.recording_stop_event.set()
        if self.recording_thread and self.recording_thread is not threading.current_thread(): self.recording_thread.join(timeout=2)
        with self.recording_lock:
            if self.video_writer is not None: self.video_writer.release(); self.video_writer = None
            self.current_video_path = None
        while not self.recording_queue.empty():
            try: self.recording_queue.get_nowait()
            except queue.Empty: break
        self.record_button.config(text="Start Recording", command=self.start_recording); print(f"Recording stopped: {path}")

    def capture_screenshot(self):
        with self.frame_lock: frame = None if self.current_frame is None else self.current_frame.copy()
        if frame is None: messagebox.showwarning("No Frame Available", "Connect to a camera first."); return
        self.screenshot_counter += 1
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.screenshot_counter:06d}.jpg"
        if cv2.imwrite(os.path.join(SCREENSHOT_FOLDER, filename), frame): print(f"Screenshot saved: {filename}")
        else: print("ERROR: Screenshot could not be saved.")

    # ------------------------------- Shutdown -------------------------------
    def disconnect_camera(self, keep_status=False):
        self.preview_running = False; self.capture_stop_event.set(); self.detection_stop_event.set(); self.stop_recording()
        with self.camera_lock:
            if self.camera is not None: self.camera.release(); self.camera = None
        for worker in (self.capture_thread, self.detection_thread):
            # A Thread object may exist without having been started when all
            # detection controls remain off; joining such a thread raises.
            if worker and worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=2)
        with self.frame_lock: self.current_frame = None
        with self.result_lock: self.latest_result = self.empty_result()
        self.person_count_variable.set("0")
        if self.preview_window_name:
            try: cv2.destroyWindow(self.preview_window_name)
            except cv2.error: pass
            self.preview_window_name = None
        if not keep_status: self.status_variable.set("Status: Camera disconnected")

    def close_application(self):
        self.shutdown_event.set(); self.disconnect_camera()
        try: cv2.destroyAllWindows()
        except cv2.error: pass
        self.root.destroy()

    def run(self):
        self.root.after(PREVIEW_UPDATE_MS, self.update_preview)
        self.root.after(GUI_UPDATE_MS, self.update_detection_gui)
        self.root.mainloop()


if __name__ == "__main__":
    CameraManager().run()
