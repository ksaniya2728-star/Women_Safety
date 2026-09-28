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

# Delay between preview window refreshes, in milliseconds.
PREVIEW_UPDATE_MS = 1


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
    Manage camera connections, live preview, recording, and screenshots.
    """

    def __init__(self):
        """
        Initialize camera state, GUI state, and output folders.
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

        # FPS measurement for the preview overlay.
        self.frame_count = 0
        self.fps_start_time = time.time()

        self.create_output_folders()
        self.create_gui()

    def create_output_folders(self):
        """
        Create the screenshot and video output folders when missing.
        """
        if ENABLE_RECORDING:
            os.makedirs(VIDEO_FOLDER, exist_ok=True)

        if ENABLE_SCREENSHOTS:
            os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

    def create_gui(self):
        """
        Create the lightweight Camera Manager and control-panel GUI.
        """
        self.root.geometry("430x720")

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

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        self.refresh_camera_list()
        self.select_first_camera()

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
        """
        camera = self.read_camera_form()

        if camera is None:
            return

        CAMERAS.append(camera)
        self.refresh_camera_list()

    def edit_camera(self):
        """
        Replace the selected camera with the details from the form.
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

        # Reset FPS measurement for the new session.
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Create the separate preview window named after the camera.
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
        Add optional FPS and timestamp overlays to a frame.

        Args:
            frame: OpenCV image frame.
            fps (float): Estimated average FPS.

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

    def start_recording(self):
        """
        Start recording original camera frames into the videos folder.
        """
        if not ENABLE_RECORDING:
            print("Recording is disabled.")
            return

        if self.is_recording:
            print("Recording is already active.")
            return

        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()

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
        Save the latest camera frame as a JPG screenshot.
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
        Write an original, non-resized frame to the active recording file.

        Args:
            frame: OpenCV image frame.
        """
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def update_preview(self):
        """
        Refresh the preview window from the main GUI thread.

        Runs through root.after() so the OpenCV window stays on the main
        thread, which is required for imshow() to work reliably.
        """
        if self.preview_running and self.preview_window_name is not None:
            with self.frame_lock:
                frame = (
                    None
                    if self.current_frame is None
                    else self.current_frame.copy()
                )

            if frame is not None:
                self.frame_count += 1
                elapsed_time = time.time() - self.fps_start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

                self.write_frame_to_video(frame)

                display_frame = self.resize_frame_if_needed(frame.copy())
                display_frame = self.add_overlay_text(display_frame, fps)

                # Show the frame in the separate camera-named window.
                cv2.imshow(self.preview_window_name, display_frame)

            # Keep the preview window responsive and allow Q to disconnect.
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
        """
        self.stop_recording()
        self.preview_running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        with self.frame_lock:
            self.current_frame = None

        # Close only this camera's preview window if it exists.
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
        Release every camera and video resource, then close the GUI.
        """
        self.disconnect_camera()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        """
        Start the Tkinter GUI and the main-thread preview updater.
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