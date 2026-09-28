"""
camera_manager.py

Camera Manager module for previewing and managing USB, IP Webcam,
and RTSP-compatible camera streams.

Features:
- Camera management GUI
- Live preview in a separate window named after the camera
- Auto reconnect
- Screenshot capture
- Video recording
- Optional YOLO11 Nano person detection
- In-memory camera list
"""

import os
import time
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

import cv2


# ============================================================
# Configuration
# ============================================================

DEFAULT_CAMERA_URL = "http://192.168.1.10:8080/video"

SHOW_FPS = True
SHOW_TIMESTAMP = True
RESIZE_FRAME = False
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

AUTO_RECONNECT = True
RECONNECT_DELAY = 3

ENABLE_RECORDING = True
ENABLE_SCREENSHOTS = True

WINDOW_NAME = "Camera Preview"

VIDEO_FOLDER = "videos"
SCREENSHOT_FOLDER = "screenshots"
RECORDING_FPS = 20

# Best-effort low-latency buffer setting; support depends on OpenCV backend.
CAPTURE_BUFFER_SIZE = 1

# Delay between preview refreshes, in milliseconds.
PREVIEW_UPDATE_MS = 1


# ============================================================
# Module 2: Person Detection Configuration
# ============================================================

# Detection remains disabled by default, preserving existing behavior.
ENABLE_PERSON_DETECTION = False

# YOLO11 Nano is stored locally after its first download.
MODEL_PATH = "models/yolo11n.pt"

# Minimum confidence required to show a person detection.
CONFIDENCE_THRESHOLD = 0.50

# Detection overlay display options.
SHOW_BOXES = True
SHOW_CONFIDENCE = True
SHOW_PERSON_COUNT = True
SHOW_DETECTION_FPS = True

# YOLO's COCO class ID for "person".
PERSON_CLASS_ID = 0


# Cameras are stored in memory as simple dictionaries.
# This structure can later be saved to or loaded from a JSON file.
CAMERAS = [
    {
        "name": "IP Webcam",
        "type": "IP Webcam",
        "source": DEFAULT_CAMERA_URL,
    },
    {
        "name": "USB Camera",
        "type": "USB Camera",
        "source": 0,
    },
]


class CameraManager:
    """
    Manage camera connections, live preview, recording, screenshots,
    and optional person detection.
    """

    def __init__(self):
        """
        Initialize application state, output folders, GUI, and YOLO model state.
        """
        self.root = tk.Tk()
        self.root.title("Camera Manager")

        self.camera = None
        self.camera_source = None
        self.camera_name = None
        self.preview_window_name = None
        self.preview_running = False
        self.reconnecting = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None

        self.video_writer = None
        self.is_recording = False
        self.current_video_path = None

        self.screenshot_counter = 0
        self.selected_camera_index = None
        self.status_variable = tk.StringVar(value="Status: No camera connected")

        # Preview FPS values.
        self.frame_count = 0
        self.fps_start_time = time.time()

        # ----------------------------------------------------
        # Module 2: Detection state
        # ----------------------------------------------------

        # Tkinter variable controlled by the detection checkbox.
        self.person_detection_enabled = tk.BooleanVar(
            value=ENABLE_PERSON_DETECTION
        )

        # The model is loaded once and reused for every inference call.
        self.yolo_model = None

        # These values appear in the Detection Controls GUI section.
        self.model_status_variable = tk.StringVar(value="Not Loaded")
        self.detection_status_variable = tk.StringVar(value="OFF")
        self.person_count_variable = tk.StringVar(value="0")
        self.detection_fps_variable = tk.StringVar(value="0.0")
        self.inference_time_variable = tk.StringVar(value="0.0 ms")

        # Stores simple detection data for a later tracking module.
        # Each item contains: bbox, confidence, and class_id.
        self.latest_person_detections = []

        self.create_output_folders()
        self.load_person_detection_model()
        self.create_gui()

    def create_output_folders(self):
        """
        Create screenshot, video, and model folders when they do not exist.
        """
        if ENABLE_RECORDING:
            os.makedirs(VIDEO_FOLDER, exist_ok=True)

        if ENABLE_SCREENSHOTS:
            os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

        # The models folder is always created for YOLO model storage.
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    def load_person_detection_model(self):
        """
        Load YOLO11 Nano once during application startup.

        If the local model file is missing, Ultralytics downloads the
        requested YOLO11 Nano file once and stores it in models/.

        Returns:
            None
        """
        try:
            # Import here so missing ultralytics does not crash the program.
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics package is not installed.")
            print("pip install ultralytics")
            self.model_status_variable.set("Not Loaded")
            return

        try:
            model_directory = os.path.dirname(MODEL_PATH)
            model_filename = os.path.basename(MODEL_PATH)

            # If missing, request the standard filename first so Ultralytics
            # downloads it. Then move it into the configured models folder.
            if not os.path.exists(MODEL_PATH):
                print("YOLO model not found locally. Downloading once...")

                downloaded_model = YOLO(model_filename)

                # Ultralytics normally downloads to the current directory.
                # Save the downloaded file in the requested models folder.
                if os.path.exists(model_filename):
                    os.replace(model_filename, MODEL_PATH)

                self.yolo_model = YOLO(MODEL_PATH)

            else:
                # Always prefer the existing local model file.
                print(f"Loading local YOLO model: {MODEL_PATH}")
                self.yolo_model = YOLO(MODEL_PATH)

            self.model_status_variable.set("Loaded")
            print("YOLO11 Nano model loaded successfully.")

        except Exception as error:
            self.yolo_model = None
            self.model_status_variable.set("Not Loaded")
            print(f"ERROR: Could not load YOLO model: {error}")

    def create_gui(self):
        """
        Create the existing Camera Manager GUI and Detection Controls section.
        """
        self.root.geometry("430x900")

        tk.Label(
            self.root,
            text="Camera Manager",
            font=("Arial", 16, "bold"),
        ).pack(pady=(12, 6))

        tk.Label(
            self.root,
            textvariable=self.status_variable,
            fg="#0f766e",
            font=("Arial", 10),
        ).pack(pady=(0, 10))

        camera_frame = tk.LabelFrame(self.root, text="Cameras", padx=8, pady=8)
        camera_frame.pack(fill="x", padx=12, pady=6)

        self.camera_listbox = tk.Listbox(camera_frame, height=5)
        self.camera_listbox.pack(fill="x", pady=(0, 8))
        self.camera_listbox.bind("<<ListboxSelect>>", self.select_camera)

        button_frame = tk.Frame(camera_frame)
        button_frame.pack(fill="x")

        tk.Button(
            button_frame,
            text="Connect",
            command=self.connect_selected_camera,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        tk.Button(
            button_frame,
            text="Disconnect",
            command=self.disconnect_camera,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        edit_frame = tk.LabelFrame(self.root, text="Camera Details", padx=8, pady=8)
        edit_frame.pack(fill="x", padx=12, pady=6)

        tk.Label(edit_frame, text="Camera Name").pack(anchor="w")
        self.name_entry = tk.Entry(edit_frame)
        self.name_entry.pack(fill="x", pady=(0, 6))

        tk.Label(edit_frame, text="Camera Type").pack(anchor="w")
        self.type_combobox = ttk.Combobox(
            edit_frame,
            values=["IP Webcam", "USB Camera", "RTSP Camera"],
            state="normal",
        )
        self.type_combobox.pack(fill="x", pady=(0, 6))

        tk.Label(edit_frame, text="Camera Source").pack(anchor="w")
        self.source_entry = tk.Entry(edit_frame)
        self.source_entry.pack(fill="x", pady=(0, 8))

        camera_edit_buttons = tk.Frame(edit_frame)
        camera_edit_buttons.pack(fill="x")

        tk.Button(
            camera_edit_buttons,
            text="Add Camera",
            command=self.add_camera,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        tk.Button(
            camera_edit_buttons,
            text="Edit Camera",
            command=self.edit_camera,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        tk.Button(
            camera_edit_buttons,
            text="Remove Camera",
            command=self.remove_camera,
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        controls = tk.LabelFrame(self.root, text="Controls", padx=8, pady=8)
        controls.pack(fill="x", padx=12, pady=6)

        self.record_button = tk.Button(
            controls,
            text="Start Recording",
            command=self.start_recording,
        )
        self.record_button.pack(fill="x", pady=3)

        self.screenshot_button = tk.Button(
            controls,
            text="Capture Screenshot",
            command=self.capture_screenshot,
        )
        self.screenshot_button.pack(fill="x", pady=3)

        tk.Button(
            controls,
            text="Exit Application",
            command=self.close_application,
        ).pack(fill="x", pady=(10, 3))

        # ----------------------------------------------------
        # Module 2: Detection Controls
        # This adds controls below the existing GUI without
        # changing camera management controls above it.
        # ----------------------------------------------------
        detection_frame = tk.LabelFrame(
            self.root,
            text="Detection Controls",
            padx=8,
            pady=8,
        )
        detection_frame.pack(fill="x", padx=12, pady=6)

        tk.Checkbutton(
            detection_frame,
            text="Enable Person Detection",
            variable=self.person_detection_enabled,
            command=self.handle_detection_toggle,
        ).pack(anchor="w", pady=(0, 8))

        self.create_detection_status_row(
            detection_frame,
            "Detection Status:",
            self.detection_status_variable,
        )

        self.create_detection_status_row(
            detection_frame,
            "Person Count:",
            self.person_count_variable,
        )

        self.create_detection_status_row(
            detection_frame,
            "Model Status:",
            self.model_status_variable,
        )

        self.create_detection_status_row(
            detection_frame,
            "Detection FPS:",
            self.detection_fps_variable,
        )

        self.create_detection_status_row(
            detection_frame,
            "Inference Time:",
            self.inference_time_variable,
        )

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        self.refresh_camera_list()
        self.select_first_camera()

    def create_detection_status_row(self, parent, label_text, value_variable):
        """
        Create one label-value row in the Detection Controls section.

        Args:
            parent: Tkinter parent widget for the row.
            label_text (str): Fixed label text shown on the left.
            value_variable (tk.StringVar): Dynamic value shown on the right.

        Returns:
            None
        """
        row = tk.Frame(parent)
        row.pack(fill="x", pady=2)

        tk.Label(row, text=label_text).pack(side=tk.LEFT)

        tk.Label(
            row,
            textvariable=value_variable,
            font=("Arial", 9, "bold"),
        ).pack(side=tk.RIGHT)

    def handle_detection_toggle(self):
        """
        Update visible detection status when the checkbox is changed.

        Detection inference is still performed only inside update_preview(),
        so checkbox interaction never blocks the GUI.
        """
        if self.person_detection_enabled.get() and self.yolo_model is not None:
            self.detection_status_variable.set("ON")
            return

        if self.person_detection_enabled.get() and self.yolo_model is None:
            self.person_detection_enabled.set(False)
            self.detection_status_variable.set("OFF")
            print("ERROR: Person detection cannot start because the model is unavailable.")
            return

        self.detection_status_variable.set("OFF")
        self.person_count_variable.set("0")
        self.detection_fps_variable.set("0.0")
        self.inference_time_variable.set("0.0 ms")
        self.latest_person_detections = []

    def refresh_camera_list(self):
        """
        Refresh the GUI camera list from the in-memory CAMERAS list.
        """
        self.camera_listbox.delete(0, tk.END)

        for camera in CAMERAS:
            self.camera_listbox.insert(
                tk.END,
                f"{camera['name']}  |  {camera['type']}  |  {camera['source']}",
            )

    def select_first_camera(self):
        """
        Select the first available camera when the application starts.
        """
        if CAMERAS:
            self.camera_listbox.selection_set(0)
            self.select_camera()

    def get_selected_camera_index(self):
        """
        Get the currently selected camera index from the GUI list.

        Returns:
            int or None: Selected camera index, or None if unavailable.
        """
        selection = self.camera_listbox.curselection()

        if not selection:
            return None

        return int(selection[0])

    def select_camera(self, event=None):
        """
        Load the selected camera's details into the editable fields.

        Args:
            event: Optional Tkinter selection event.

        Returns:
            None
        """
        index = self.get_selected_camera_index()

        if index is None:
            return

        camera = CAMERAS[index]
        self.selected_camera_index = index

        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, str(camera["name"]))

        self.type_combobox.set(str(camera["type"]))

        self.source_entry.delete(0, tk.END)
        self.source_entry.insert(0, str(camera["source"]))

    def normalize_camera_source(self, source):
        """
        Convert numeric text sources to integer USB camera indexes.

        Args:
            source: Camera source entered by the user.

        Returns:
            int or str: USB camera index or network camera URL.
        """
        source_text = str(source).strip()

        if source_text.isdigit():
            return int(source_text)

        return source_text

    def read_camera_form(self):
        """
        Read and validate the camera form fields.

        Returns:
            dict or None: Camera details, or None when validation fails.
        """
        camera_name = self.name_entry.get().strip()
        camera_type = self.type_combobox.get().strip() or "IP Webcam"
        camera_source = self.normalize_camera_source(self.source_entry.get())

        if not camera_name:
            messagebox.showwarning("Missing Camera Name", "Enter a camera name.")
            return None

        if camera_source == "":
            messagebox.showwarning("Missing Camera Source", "Enter a camera source.")
            return None

        return {
            "name": camera_name,
            "type": camera_type,
            "source": camera_source,
        }

    def add_camera(self):
        """
        Add the entered camera to the in-memory camera list.

        Returns:
            None
        """
        camera = self.read_camera_form()

        if camera is None:
            return

        CAMERAS.append(camera)
        self.refresh_camera_list()

    def edit_camera(self):
        """
        Replace the selected camera with the details from the form.

        Returns:
            None
        """
        index = self.get_selected_camera_index()

        if index is None:
            messagebox.showwarning("No Camera Selected", "Select a camera to edit.")
            return

        camera = self.read_camera_form()

        if camera is None:
            return

        CAMERAS[index] = camera
        self.refresh_camera_list()
        self.camera_listbox.selection_set(index)

    def remove_camera(self):
        """
        Remove the selected camera from the in-memory camera list.

        Returns:
            None
        """
        index = self.get_selected_camera_index()

        if index is None:
            messagebox.showwarning("No Camera Selected", "Select a camera to remove.")
            return

        removed_camera = CAMERAS.pop(index)
        self.refresh_camera_list()

        print(f"Removed camera: {removed_camera['name']}")

    def open_camera(self, source):
        """
        Open a camera source using OpenCV VideoCapture.

        Args:
            source (int or str): USB camera index, IP Webcam URL, or RTSP URL.

        Returns:
            cv2.VideoCapture or None: Opened camera object or None on failure.
        """
        print(f"Connecting to camera: {source}")

        camera = cv2.VideoCapture(source)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)

        if not camera.isOpened():
            camera.release()
            print("ERROR: Camera could not be opened.")
            return None

        print("Camera connected successfully.")
        return camera

    def connect_selected_camera(self):
        """
        Connect to the currently selected camera from the camera list.

        Returns:
            None
        """
        index = self.get_selected_camera_index()

        if index is None:
            messagebox.showwarning("No Camera Selected", "Select a camera first.")
            return

        self.disconnect_camera(keep_status=True)

        selected_camera = CAMERAS[index]
        self.camera_source = selected_camera["source"]
        self.camera_name = selected_camera["name"]

        camera = self.open_camera(self.camera_source)

        if camera is None:
            self.status_variable.set("Status: Connection failed")
            messagebox.showerror(
                "Connection Failed",
                "The selected camera stream could not be opened.",
            )
            return

        self.camera = camera
        self.preview_running = True
        self.reconnecting = False

        self.frame_count = 0
        self.fps_start_time = time.time()

        self.preview_window_name = str(self.camera_name)
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)

        if RESIZE_FRAME:
            cv2.resizeWindow(
                self.preview_window_name,
                FRAME_WIDTH,
                FRAME_HEIGHT,
            )

        self.status_variable.set(f"Status: Connected to {self.camera_name}")

        self.capture_thread = threading.Thread(
            target=self.capture_frames,
            daemon=True,
        )
        self.capture_thread.start()

    def capture_frames(self):
        """
        Continuously read frames and handle automatic reconnection.

        Returns:
            None
        """
        while self.preview_running:
            if self.camera is None:
                time.sleep(0.1)
                continue

            success, frame = self.camera.read()

            if success and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                continue

            print("ERROR: Could not read a frame from the camera stream.")

            if not AUTO_RECONNECT:
                self.root.after(0, self.disconnect_camera)
                break

            self.attempt_reconnect()

    def attempt_reconnect(self):
        """
        Release the failed camera and reconnect after the configured delay.

        Returns:
            None
        """
        if self.reconnecting or not self.preview_running:
            return

        self.reconnecting = True
        self.root.after(
            0,
            lambda: self.status_variable.set(
                f"Status: Reconnecting in {RECONNECT_DELAY} seconds..."
            ),
        )

        print(f"Reconnecting in {RECONNECT_DELAY} seconds...")

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        time.sleep(RECONNECT_DELAY)

        if not self.preview_running:
            self.reconnecting = False
            return

        camera = self.open_camera(self.camera_source)

        if camera is None:
            print("ERROR: Reconnect failed. Trying again later.")
            self.reconnecting = False
            time.sleep(RECONNECT_DELAY)
            return

        self.camera = camera
        self.reconnecting = False

        self.root.after(
            0,
            lambda: self.status_variable.set("Status: Reconnected"),
        )

    def resize_frame_if_needed(self, frame):
        """
        Resize a frame when resizing is enabled.

        Args:
            frame: OpenCV image frame.

        Returns:
            OpenCV frame: Original or resized frame.
        """
        if not RESIZE_FRAME:
            return frame

        return cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def add_overlay_text(self, frame, fps):
        """
        Add optional preview FPS and timestamp overlays to a frame.

        Args:
            frame: OpenCV image frame.
            fps (float): Estimated preview FPS.

        Returns:
            OpenCV frame: Frame with overlay text.
        """
        text_y = 30

        if SHOW_FPS:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            text_y += 30

        if SHOW_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(
                frame,
                timestamp,
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    def run_person_detection(self, frame):
        """
        Run YOLO inference on one frame and return person detections.

        This method only runs when the detection checkbox is enabled.
        It deliberately returns simple dictionaries so Module 3 can use
        the same detections later for multi-person tracking.

        Args:
            frame: Original OpenCV BGR camera frame.

        Returns:
            tuple: A list of person detections and inference time in milliseconds.
        """
        if self.yolo_model is None:
            return [], 0.0

        if frame is None or frame.size == 0:
            return [], 0.0

        try:
            start_time = time.perf_counter()

            # Filter at inference time so only person class ID 0 is returned.
            results = self.yolo_model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            inference_time_ms = (time.perf_counter() - start_time) * 1000
            detections = []

            if not results:
                return detections, inference_time_ms

            result = results[0]

            if result.boxes is None:
                return detections, inference_time_ms

            # Convert YOLO output into tracking-ready detection dictionaries.
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "bbox": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )

            return detections, inference_time_ms

        except Exception as error:
            print(f"ERROR: Person detection inference failed: {error}")
            return [], 0.0

    def draw_person_detections(self, frame, detections):
        """
        Draw person boxes, confidence labels, and person count on a frame.

        Args:
            frame: OpenCV BGR frame used for preview display.
            detections (list): Detection dictionaries from run_person_detection().

        Returns:
            OpenCV frame: Preview frame with detection overlays.
        """
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            if SHOW_BOXES:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            if SHOW_CONFIDENCE:
                label = f"Person {confidence * 100:.0f}%"

                label_y = max(25, y1 - 10)

                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if SHOW_PERSON_COUNT:
            cv2.putText(
                frame,
                f"Persons: {len(detections)}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    def update_detection_gui(self, detections, inference_time_ms):
        """
        Update Detection Controls values after a successful inference call.

        Args:
            detections (list): Person detection dictionaries.
            inference_time_ms (float): YOLO inference duration in milliseconds.

        Returns:
            None
        """
        person_count = len(detections)

        if inference_time_ms > 0:
            detection_fps = 1000 / inference_time_ms
        else:
            detection_fps = 0.0

        self.detection_status_variable.set("ON")
        self.person_count_variable.set(str(person_count))
        self.detection_fps_variable.set(f"{detection_fps:.1f}")
        self.inference_time_variable.set(f"{inference_time_ms:.1f} ms")

    def start_recording(self):
        """
        Start recording original camera frames into the videos folder.

        Returns:
            None
        """
        if not ENABLE_RECORDING:
            print("Recording is disabled.")
            return

        if self.is_recording:
            print("Recording is already active.")
            return

        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame

        if frame is None:
            messagebox.showwarning("No Frame Available", "Connect to a camera first.")
            return

        frame_height, frame_width = frame.shape[:2]
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.avi")
        filepath = os.path.join(VIDEO_FOLDER, filename)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(
            filepath,
            fourcc,
            RECORDING_FPS,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            video_writer.release()
            messagebox.showerror("Recording Failed", "Could not create the video file.")
            return

        self.video_writer = video_writer
        self.current_video_path = filepath
        self.is_recording = True
        self.record_button.config(text="Stop Recording", command=self.stop_recording)

        print(f"Recording started: {filepath}")

    def stop_recording(self):
        """
        Stop the active recording and release the VideoWriter.

        Returns:
            None
        """
        if not self.is_recording:
            return

        self.is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        print(f"Recording stopped: {self.current_video_path}")
        self.current_video_path = None

        self.record_button.config(
            text="Start Recording",
            command=self.start_recording,
        )

    def capture_screenshot(self):
        """
        Save the latest original camera frame as a JPG screenshot.

        Returns:
            None
        """
        if not ENABLE_SCREENSHOTS:
            print("Screenshots are disabled.")
            return

        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()

        if frame is None:
            messagebox.showwarning("No Frame Available", "Connect to a camera first.")
            return

        self.screenshot_counter += 1

        filename = (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{self.screenshot_counter:06d}.jpg"
        )
        filepath = os.path.join(SCREENSHOT_FOLDER, filename)

        success = cv2.imwrite(filepath, frame)

        if success:
            print(f"Screenshot saved: {filename}")
        else:
            print("ERROR: Screenshot could not be saved.")

    def write_frame_to_video(self, frame):
        """
        Write an original non-resized frame to the active recording file.

        Args:
            frame: OpenCV BGR camera frame.

        Returns:
            None
        """
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def update_preview(self):
        """
        Refresh the separate OpenCV preview and optionally apply detection.

        The original frame remains unchanged for screenshots and recordings.
        Detection drawings are applied only to the preview frame.
        """
        if self.preview_running and self.preview_window_name is not None:
            with self.frame_lock:
                frame = self.current_frame

            if frame is not None and frame.size > 0:
                self.frame_count += 1

                elapsed_time = time.time() - self.fps_start_time
                preview_fps = (
                    self.frame_count / elapsed_time
                    if elapsed_time > 0
                    else 0.0
                )

                # Recording continues to receive the original camera frame.
                self.write_frame_to_video(frame)

                # A copy is required only for drawing preview overlays.
                display_frame = frame.copy()

                # Run inference only while detection is explicitly enabled.
                if (
                    self.person_detection_enabled.get()
                    and self.yolo_model is not None
                ):
                    detections, inference_time_ms = self.run_person_detection(frame)

                    # Save basic detection data for future tracking integration.
                    self.latest_person_detections = detections

                    self.update_detection_gui(detections, inference_time_ms)

                    display_frame = self.draw_person_detections(
                        display_frame,
                        detections,
                    )

                display_frame = self.resize_frame_if_needed(display_frame)
                display_frame = self.add_overlay_text(display_frame, preview_fps)

                cv2.imshow(self.preview_window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                self.disconnect_camera()

        self.root.after(PREVIEW_UPDATE_MS, self.update_preview)

    def disconnect_camera(self, keep_status=False):
        """
        Stop recording, stop preview, close the preview window,
        and release the active camera.

        Args:
            keep_status (bool): Keep the current GUI status unchanged.

        Returns:
            None
        """
        self.stop_recording()
        self.preview_running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        with self.frame_lock:
            self.current_frame = None

        self.latest_person_detections = []
        self.person_count_variable.set("0")

        if self.preview_window_name is not None:
            try:
                cv2.destroyWindow(self.preview_window_name)
            except cv2.error:
                pass

            self.preview_window_name = None

        if not keep_status:
            self.status_variable.set("Status: Camera disconnected")

    def close_application(self):
        """
        Release camera, video, detection-preview resources, and close the GUI.

        Returns:
            None
        """
        self.disconnect_camera()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        """
        Start the Tkinter GUI and main-thread OpenCV preview updates.

        Returns:
            None
        """
        self.root.after(PREVIEW_UPDATE_MS, self.update_preview)
        self.root.mainloop()


def main():
    """
    Launch the Camera Manager application.
    """
    app = CameraManager()
    app.run()


if __name__ == "__main__":
    main()