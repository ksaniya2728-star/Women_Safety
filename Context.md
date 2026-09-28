# Project Context: AI Sexual Detection Model

## 1. Purpose and Scope

This project is a modular computer-vision prototype for detecting and analysing interactions between people in camera streams. Its current design combines:

1. Camera acquisition and recording.
2. YOLO-based person and pose detection.
3. Multi-person tracking-style identifiers and pose feature extraction.
4. Geometric and temporal interaction analysis.
5. Explainable rule-based safety-risk scoring.
6. Evidence capture and database integration.

The project is **not an autonomous crime, harassment, intent, guilt, or sexual-assault classifier**. The rules engine explicitly treats its output as a review signal for a human operator. The source files describe image-space features and potentially unsafe interaction patterns; they do not establish what actually happened.

The supplied files are prototype modules rather than one completely integrated package. Several modules deliberately state that they should not import or modify each other. The primary intended data flow is:

```text
Camera stream
    -> frame capture
    -> latest-frame buffer
    -> YOLO person/pose inference
    -> tracked persons and keypoints
    -> interaction features
    -> temporal rule engine
    -> event/evidence recording
    -> human review and optional database persistence
```

## 2. Supplied Modules

| File | Logical module | Main responsibility |
|---|---|---|
| `camera_manager_M1.py` | Module 1 | Camera management, preview, screenshots, recording |
| `detection_manager_M2.py` | Module 2 | Module 1 plus optional YOLO person detection and detection controls |
| `multi_person_detection_pose_M3.py` | Module 3 | Low-latency multi-person detection, pose detection, display, and recording |
| `interactive_analyser_M4.py` | Module 4 | Converts detected/tracked people and pose keypoints into interaction features |
| `pose_detection_M4.py` | M4 outline | Text-only architecture outline for the interaction analyser |
| `rules_engine_M5.py` | Module 5 | Explainable temporal safety-risk assessment |
| `rules_engine_tuning_M5.py` | M5 tuning variant | Expanded/configurable variant of the rules engine |
| `database_evidence_M6.py` | Module 6 | Evidence buffering, event finalization, metadata, and database integration |

`detection_manager_M2.py` is an enhanced continuation of Module 1. `multi_person_detection_pose_M3.py` is a more complete and concurrency-oriented camera pipeline. `interactive_analyser_M4.py` is the main feature-analysis layer. Module 5 consumes M4 feature dictionaries, while Module 6 consumes M4-compatible features and connects risk events to evidence and storage.

## 3. Architecture at a Glance

### 3.1 Acquisition layer

The acquisition layer uses OpenCV `VideoCapture` to open:

- USB camera indexes such as `0`.
- IP Webcam HTTP streams such as `http://192.168.1.10:8080/video`.
- RTSP-compatible sources, depending on the local OpenCV backend and URL.

The camera modules maintain an in-memory list of camera dictionaries:

```python
{
    "name": "USB Camera",
    "type": "USB Camera",
    "source": 0,
}
```

### 3.2 Processing layer

Module 3 separates capture, AI processing, display, and recording. It intentionally keeps only the newest frame for AI processing. This prevents an inference backlog from causing the display to become increasingly delayed.

### 3.3 Feature layer

Module 4 does not run YOLO. It expects person records containing IDs, bounding boxes, centers, movement information, and optional pose keypoints. It calculates normalized proximity, movement, approach direction, following, facing, hand proximity, and temporal interaction features.

### 3.4 Decision layer

Module 5 receives feature dictionaries. It stores per-pair history, smooths risk scores, applies persistence windows, applies rule weights and combination bonuses, and emits a risk state plus explainable reason codes.

### 3.5 Evidence layer

Module 6 is intended to preserve context around an event. It contains configuration for pre-event and post-event evidence windows, segment finalization, queues, event history, manual metadata, and MySQL settings. It is designed to avoid treating the analytical score as proof.

## 4. Module 1: `camera_manager_M1.py`

### 4.1 Responsibility

Module 1 is the baseline Tkinter camera manager. It provides a GUI for selecting and editing cameras, opening a live stream, capturing screenshots, recording video, and reconnecting after failures.

### 4.2 Configuration

Important settings include:

- `DEFAULT_CAMERA_URL`: default IP Webcam URL.
- `SHOW_FPS`: enables FPS overlay.
- `SHOW_TIMESTAMP`: enables timestamps on preview frames.
- `RESIZE_FRAME`: controls optional frame resizing.
- `FRAME_WIDTH` and `FRAME_HEIGHT`: target dimensions when resizing is enabled.
- `AUTO_RECONNECT`: enables reconnect logic.
- `RECONNECT_DELAY`: delay before reconnect attempts.
- `ENABLE_RECORDING`: creates and uses the `videos` folder.
- `ENABLE_SCREENSHOTS`: creates and uses the `screenshots` folder.
- `RECORDING_FPS`: FPS used by the video writer.
- `CAPTURE_BUFFER_SIZE`: requested OpenCV capture buffer size, normally `1` for low latency.
- `PREVIEW_UPDATE_MS`: GUI preview refresh delay.

### 4.3 `CameraManager` state

The class maintains:

- `camera`: current `cv2.VideoCapture` instance.
- `camera_source`: active source URL or USB index.
- `camera_name`: active camera name.
- `preview_running`: whether the preview loop is active.
- `reconnecting`: prevents duplicate reconnect operations.
- `current_frame`: most recent frame.
- `frame_lock`: protects access to the current frame from worker and GUI contexts.
- `capture_thread`: background capture worker.
- `video_writer`: current OpenCV writer.
- `is_recording`: recording state.
- `current_video_path`: output video path.
- `selected_camera_index`: selected entry in `CAMERAS`.
- `status_variable`: Tkinter status text.

### 4.4 Camera CRUD

The GUI supports:

1. Selecting a camera from a list.
2. Loading its name, type, and source into form fields.
3. Adding a camera to the in-memory `CAMERAS` list.
4. Editing the selected camera.
5. Removing the selected camera.

Camera sources consisting only of digits are converted to integers. Therefore, `0` becomes a USB camera index, while URLs remain strings.

### 4.5 Connection lifecycle

The expected lifecycle is:

```text
select camera
    -> disconnect previous camera
    -> cv2.VideoCapture(source)
    -> set capture buffer size
    -> verify isOpened()
    -> start capture/preview loop
```

If the source cannot be opened, the module releases the capture object, updates the status, and displays an error message.

### 4.6 Recording and screenshots

The module creates `videos/` and `screenshots/` directories when enabled. Screenshots are generated from the current frame. Video recording uses an OpenCV `VideoWriter`, a configured codec, the current frame dimensions, and `RECORDING_FPS`.

### 4.7 Limitations

- Camera configuration is stored only in memory.
- No authentication or access control is implemented for camera URLs.
- Network URLs may contain sensitive credentials if credentials are embedded in the URL.
- OpenCV backend support varies by operating system and stream type.
- The Tkinter UI and camera pipeline are tightly coupled in the same file.

## 5. Module 2: `detection_manager_M2.py`

### 5.1 Responsibility

Module 2 extends the camera manager with optional YOLO11 Nano person detection. Detection is disabled by default through `ENABLE_PERSON_DETECTION = False`, preserving the basic Module 1 behaviour until the operator enables it.

### 5.2 Model loading

The model path is:

```text
models/yolo11n.pt
```

At startup:

1. The module creates the model directory.
2. It imports `YOLO` lazily from `ultralytics`.
3. If the local model is missing, it requests the standard model file.
4. It moves the downloaded file into `models/` when possible.
5. It loads the local model once and reuses it for inference.

Lazy importing prevents a missing Ultralytics installation from immediately crashing the application, although detection will remain unavailable.

### 5.3 Detection settings

- `CONFIDENCE_THRESHOLD = 0.50`: minimum detection confidence.
- `PERSON_CLASS_ID = 0`: COCO person class.
- `SHOW_BOXES`: draw person bounding boxes.
- `SHOW_CONFIDENCE`: show confidence values.
- `SHOW_PERSON_COUNT`: show the number of detected people.
- `SHOW_DETECTION_FPS`: show detection FPS.

### 5.4 Detection state

The GUI stores:

- Model status.
- Detection ON/OFF state.
- Current person count.
- Detection FPS.
- Inference time.
- Latest person detections.

Each latest detection is intended to contain a bounding box, confidence, and class ID. This is a useful handoff format for later tracking or feature modules.

### 5.5 Relationship to Module 1

Module 2 retains the camera management features from Module 1 and adds a detection controls section. It should be treated as an incremental prototype, not as a clean library inheritance hierarchy.

## 6. Module 3: `multi_person_detection_pose_M3.py`

### 6.1 Responsibility

Module 3 is the main real-time vision pipeline. It supports multiple people, YOLO person detection, YOLO pose detection, low-latency frame handling, recording, preview, and model/device status.

The key design statement is:

```text
capture thread -> latest-frame buffer -> AI worker -> GUI/display
```

Only the newest unprocessed frame is retained for inference. This is preferable for live monitoring because stale frames are discarded instead of building an unbounded queue.

### 6.2 Configuration

Important settings include:

- `MODEL_PATH = models/yolo11n.pt`.
- `POSE_MODEL_PATH = models/yolo11n-pose.pt`.
- `CONFIDENCE_THRESHOLD = 0.50`.
- `POSE_KEYPOINT_CONFIDENCE = 0.40`.
- Inference size of `640 x 640`.
- `DETECTION_INTERVAL = 1`, meaning each newest available frame can be processed.
- `PERSON_CLASS_ID = 0`.
- `CAPTURE_BUFFER_SIZE = 1`.
- `PREVIEW_UPDATE_MS = 15`.
- `GUI_UPDATE_MS = 250`.
- `RECONNECT_DELAY = 3`.
- `RECORDING_FPS = 20`.
- `RECORDING_QUEUE_SIZE = 30`.
- Display resolution defaults of `1280 x 720`.

### 6.3 Model types

The module supports two local model paths:

- Person model: `yolo11n.pt`.
- Pose model: `yolo11n-pose.pt`.

Models are loaded once. If a model is absent, the module requests the standard filename and attempts to move it into `models/`. It detects whether PyTorch CUDA is available and reports `CUDA` or `CPU` in the GUI.

### 6.4 `RollingFPS`

`RollingFPS` stores recent timestamps in a deque. Its FPS is calculated from the number of events divided by the duration between the oldest and newest retained timestamps. It deliberately measures the selected pipeline activity rather than incorrectly using inference duration as camera FPS.

Separate counters are maintained for capture, detection, and display.

### 6.5 Threading model

The `CameraManager` uses separate concepts for:

- Capture stop event.
- Detection stop event.
- Recording stop event.
- Camera lock.
- Frame lock.
- Result lock.
- Settings lock.
- Recording lock.

The frame state includes:

- `current_frame`: latest original frame.
- `current_frame_id`: monotonically increasing frame identifier.
- `latest_result`: latest AI output.

This arrangement reduces the chance that the GUI blocks while waiting for camera or inference operations.

### 6.6 Person and pose output

The module is expected to produce person records with bounding boxes, IDs or detection order, confidence, and optional keypoints. Pose uses COCO-17 keypoints:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

The skeleton topology is rendered only when the necessary keypoints meet the confidence threshold.

### 6.7 Recording architecture

Recording uses a bounded queue of size `30`. A bounded queue prevents recording from consuming unlimited memory if disk writing temporarily falls behind. The output is written by a recording worker rather than by the GUI thread.

### 6.8 Strengths

- Low-latency design.
- Explicit shutdown events.
- Separate model loading and inference state.
- CPU/GPU detection.
- Pose support.
- Capture, detection, display, and recording FPS metrics.

### 6.9 Risks and limitations

- YOLO model download at runtime requires network access during first startup.
- Models are trusted local binary artifacts and should be obtained from a controlled source.
- Camera streams may fail intermittently.
- Person IDs need a real tracker if persistent identity across frames is required; detection order alone is not sufficient.
- GUI operations must remain in the Tkinter main thread.
- The system has no privacy masking or retention policy in the supplied code.

## 7. Module 4: `interactive_analyser_M4.py`

### 7.1 Responsibility

Module 4 consumes tracking and pose output. It does not run YOLO and does not make criminal, behavioural, or safety conclusions. Its outputs are geometric and temporal feature dictionaries for a later consumer.

The architecture outline in `pose_detection_M4.py` lists:

```text
InteractionManager
├── PairDetector
├── ProximityAnalyzer
├── DistanceAnalyzer
├── MovementAnalyzer
├── ApproachAnalyzer
├── FollowingAnalyzer
├── FacingAnalyzer
├── HandInteractionAnalyzer
├── TemporalAnalyzer
└── InteractionStateManager
```

### 7.2 Core configuration

Important thresholds include:

- Keypoint confidence threshold: `0.40`.
- Minimum person height: `20` pixels.
- Very close, close, and moderate normalized-distance thresholds: `0.85`, `1.65`, and `3.0`.
- Movement speed thresholds: `8.0` and `45.0` image-space units per second.
- Approach history length: `10` observations.
- Minimum approach duration: `0.60` seconds.
- Approach slope threshold: `0.08`.
- Following history length: `12` observations.
- Following score threshold: `0.65`.
- Minimum following duration: `1.0` second.
- Facing score threshold: `0.60`.
- Hand proximity threshold: `0.80`.
- Pair timeout: `3.0` seconds.
- Minimum state duration: `0.35` seconds.
- Feature history window: `20` observations.

### 7.3 Person input contract

A person record is expected to have at least:

```python
{
    "person_id": 1,
    "bbox": [x1, y1, x2, y2],
    "center": [x, y],              # optional; derived from bbox otherwise
    "keypoints": {
        "nose": {"x": 0, "y": 0, "confidence": 0.9},
        # other COCO-17 joints...
    }
}
```

Missing or low-confidence keypoints are not fabricated. They remain unavailable for calculations that depend on them.

### 7.4 `PairDetector`

`PairDetector` filters out people without IDs or bounding boxes, sorts valid people by ID, and returns unique combinations of two people. Pair keys are normalized so `(1, 2)` and `(2, 1)` represent the same pair.

### 7.5 `ProximityAnalyzer`

The analyzer calculates pixel distance between person centers and normalizes it by the average person height:

```text
normalized_distance = center_distance / average_person_height
```

It categorizes the result as:

- `VERY_CLOSE` when normalized distance is at most `0.85`.
- `CLOSE` when it is at most `1.65`.
- `MODERATE` when it is at most `3.0`.
- `FAR` otherwise.

This is relative image-space proximity, not a physical measurement in metres.

### 7.6 `DistanceAnalyzer`

This component adds:

- Previous normalized distance.
- Current normalized distance.
- Distance delta.

A negative delta means the normalized distance decreased between observations; a positive delta means it increased.

### 7.7 `MovementAnalyzer`

For each person ID, the analyzer stores timestamped center positions in a bounded deque. Velocity is calculated from the newest and previous positions:

```text
velocity = position_delta / time_delta
speed = magnitude(velocity)
```

Movement states are:

- `STATIONARY` below `8.0`.
- `MOVING` from `8.0` up to `45.0`.
- `FAST_MOVING` at or above `45.0`.

### 7.8 `ApproachAnalyzer`

Approach detection is intentionally temporal rather than single-frame. It requires enough distance history and at least `0.60` seconds of history. It fits a trend to normalized distance and evaluates consistency between consecutive distance values.

It can report:

- Approaching.
- Separating.
- Unknown.
- Approach score.
- Whether the target is moving away.
- Approacher and target IDs.

A single close frame is therefore not enough to create an approach signal.

### 7.9 Following, facing, hand, and temporal analysis

The module contains analyzers for:

- Following: compares movement direction, velocity, trajectory, distance, and duration.
- Facing: uses pose orientation cues when visible.
- Hand interaction: uses wrists and body geometry when keypoints are sufficiently confident.
- Temporal analysis: maintains histories and prevents rapid state oscillation.
- Interaction state management: organizes current pair-level feature state.

The weighted interaction score uses the following configured components:

- Proximity: `0.24`.
- Approach: `0.18`.
- Following: `0.18`.
- Facing: `0.14`.
- Hand interaction: `0.16`.
- Temporal evidence: `0.10`.

These are feature-combination weights, not probabilities and not legal conclusions.

### 7.10 Output contract to Module 5

The rules engine expects M4-compatible dictionaries containing fields such as:

```python
{
    "person_a": 1,
    "person_b": 2,
    "normalized_distance": 1.2,
    "proximity_level": "CLOSE",
    "close_duration": 3.5,
    "approach_score": 0.8,
    "approaching": True,
    "following_score": 0.7,
    "following": True,
    "a_speed": 30.0,
    "b_speed": 25.0,
    "facing_score": 0.6,
    "hand_near_body_score": 0.5,
    "interaction_score": 0.65,
    "target_moving_away": True,
}
```

Only fields actually provided by the producer should be evaluated. Module 5 explicitly avoids fabricating unavailable features.

## 8. Module 5: `rules_engine_M5.py` and `rules_engine_tuning_M5.py`

### 8.1 Responsibility

Module 5 provides an explainable, temporal, CPU-light risk-assessment engine. It consumes M4 feature dictionaries and evaluates each person pair over time.

It does not determine guilt, intent, harassment, or confirmed criminal behaviour. Its intended output is a review-oriented signal with reason codes.

### 8.2 `M5Config`

The configuration controls:

- History size: `90` observations.
- Evaluation interval.
- Default FPS, explicitly set to `0.0` so the system does not silently assume 30 FPS.
- Maximum frame delta: `1.5` seconds.
- Short tracking gap: `2.0` seconds.
- Stale pair timeout: `45.0` seconds.
- Episode gap: `4.0` seconds.
- Event history size: `100`.
- EMA smoothing alpha: `0.28`.
- Median window: `5`.
- Persistence durations for closeness, following, movement, hands, and overlap.
- Escalation window: `3.0` seconds.
- Rapid approach threshold: `0.65`.
- High-speed thresholds: `70.0` and `180.0`.
- Meaningful overlap threshold: `0.18`.
- Repeated approach/retreat window: `18.0` seconds.
- Full repeated-cycle signal: `3` cycles.
- Trend deadband: `0.08`.
- Group signal minimum: `2` nearby people.

### 8.3 Rule weights

Default weights are:

| Rule | Weight |
|---|---:|
| Persistent proximity | 18 |
| Interaction duration | 9 |
| Following | 17 |
| Rapid approach | 10 |
| Proximity escalation | 9 |
| Movement context | 8 |
| Body overlap | 10 |
| Hand proximity | 7 |
| Repeated approach | 8 |
| Group surrounding | 8 |

A score contribution is approximately:

```text
rule contribution = rule weight * normalized signal
```

The engine also applies combination bonuses, including:

- Following plus duration: `8`.
- Approach, close proximity, and movement: `7`.
- Overlap, hand proximity, and movement: `7`.
- Group proximity persistence: `6`.
- Facing context: `2`.

The total combination bonus is capped at `18`, and the raw score is capped at `100`.

### 8.4 Risk states and hysteresis

Risk states include:

- `NORMAL`.
- `LOW`.
- `MONITORING`.
- `SUSPICIOUS`.
- `HIGH_RISK`.
- `RESOLVED`.

Enter thresholds are:

| State | Entry score |
|---|---:|
| LOW | 12 |
| MONITORING | 26 |
| SUSPICIOUS | 50 |
| HIGH_RISK | 75 |

Exit thresholds are lower:

| State | Exit score |
|---|---:|
| LOW | 6 |
| MONITORING | 18 |
| SUSPICIOUS | 39 |
| HIGH_RISK | 63 |

This hysteresis prevents the state from repeatedly switching when the score is close to a threshold.

### 8.5 `PairMemory`

Each canonical pair has independent memory containing:

- First and last timestamps.
- Last evaluated timestamp.
- Active duration.
- Current risk state.
- Smoothed EMA risk.
- Last alert timestamp.
- Current episode.
- Distance history.
- Approach/retreat trends.
- Per-rule signal histories.
- Risk history.

The pair key is always normalized, preventing duplicate state for `(1, 2)` and `(2, 1)`.

### 8.6 `Episode`

An episode records a continuous interaction period. It stores:

- Unique episode ID.
- Pair.
- Start, latest, and end timestamps.
- Peak raw risk.
- Peak smoothed risk.
- Average risk.
- Observation count.
- Final state.
- Triggered rules.
- Reason codes.
- Alerted states.

The `snapshot()` method creates a serializable dictionary suitable for event logs or persistence.

### 8.7 `FeatureAdapter`

The adapter reads M4-compatible mappings without inventing unavailable fields. Its declared available rules are:

- Persistent proximity.
- Interaction duration.
- Following.
- Rapid approach.
- Proximity escalation.
- Movement context.
- Hand proximity.
- Repeated approach.

Overlap may be accepted if a future producer supplies `body_overlap` or `bbox_overlap`, but the current M4 code does not expose it by default. Group geometry, acceleration, and direction changes are likewise unavailable unless a producer adds them.

The adapter also extracts a confidence value from supported confidence fields and clamps it to the range `[0, 1]`.

### 8.8 Temporal safety controls

The engine reduces single-frame false positives using:

- Sliding histories.
- Median or EMA smoothing.
- Persistence requirements.
- Frame timestamp validation.
- Stale-pair expiry.
- Episode gaps.
- Alert cooldowns.
- Hysteresis thresholds.
- Multi-signal combination requirements.

Configured alert cooldown is `20` seconds. High-risk persistence must last at least `1.5` seconds. At least two strong combinations are required for some elevated outcomes.

### 8.9 Tuning variant

`rules_engine_tuning_M5.py` duplicates the main rules-engine design and includes the same central configuration structure. It should be treated as an experimental tuning branch or alternate copy. Before production use, select one canonical implementation to avoid divergent fixes and behaviours.

## 9. Module 6: `database_evidence_M6.py`

### 9.1 Responsibility

Module 6 is the evidence and database integration layer. It consumes feature dictionaries from M4 and event/risk information from the M5-style engine. It is designed to preserve the surrounding evidence needed for a human review.

The file states that it never imports or modifies M4. Therefore, integration should occur through stable dictionaries, callbacks, queues, or a dedicated adapter rather than direct modification of the feature analyser.

### 9.2 Configuration

`M6Config` includes:

- `evidence_directory = "evidence"`.
- `buffer_directory = ".evidence_buffer"`.
- Pre-event evidence duration: `300` seconds.
- Post-event evidence duration: `300` seconds.
- Segment duration: `20` seconds.
- Recorder queue size: `3`.
- Finalizer queue size: `8`.
- Event history size: `100`.
- Maximum event duration: `1800` seconds.
- Database queue size: `100`.

### 9.3 MySQL settings

The module reads these environment variables:

| Variable | Default |
|---|---|
| `MYSQL_HOST` | `127.0.0.1` |
| `MYSQL_PORT` | `3306` |
| `MYSQL_DATABASE` | `women_safety` |
| `MYSQL_USER` | `saniya` |
| `MYSQL_PASSWORD` | empty string |

Passwords should be supplied through environment variables or a secret manager, not committed to source control.

### 9.4 Evidence window

The intended evidence model is:

```text
rolling pre-event buffer
    + event-triggered recording
    + post-event continuation
    -> segment finalization
    -> evidence metadata
    -> optional database persistence
```

A five-minute pre-event and five-minute post-event window can create large storage requirements. Retention, deletion, encryption, access control, and consent must be defined before deployment.

### 9.5 Metadata

`PersonMetadataConfig` is described as manual, non-sensitive metadata. It includes:

- Enabled flag.
- Cache timeout of `3` seconds.
- Maximum note length of `120` characters.

The source explicitly indicates that this metadata is not generated through video classification.

## 10. End-to-End Data Contracts

### 10.1 Camera-to-vision contract

```text
OpenCV frame: numpy.ndarray, normally BGR format
```

The frame must remain valid while the selected worker reads it. Copies should be used when a worker can outlive the capture iteration.

### 10.2 Vision-to-feature contract

```python
{
    "person_id": int,
    "bbox": [x1, y1, x2, y2],
    "center": [x, y],
    "confidence": float,
    "keypoints": {
        "joint_name": {
            "x": float,
            "y": float,
            "confidence": float,
        }
    },
}
```

### 10.3 Feature-to-rules contract

```python
{
    "person_a": int,
    "person_b": int,
    "normalized_distance": float,
    "proximity_level": str,
    "close_duration": float,
    "approach_score": float,
    "approaching": bool,
    "following_score": float,
    "following": bool,
    "a_speed": float,
    "b_speed": float,
    "facing_score": float,
    "hand_near_body_score": float,
    "interaction_score": float,
    "target_moving_away": bool,
}
```

Optional fields must remain optional. Missing fields should disable a rule rather than be replaced with guessed values.

### 10.4 Rules-to-evidence contract

A risk event should include at minimum:

```python
{
    "event_id": str,
    "episode_id": str,
    "pair": [int, int],
    "timestamp": float,
    "state": "MONITORING | SUSPICIOUS | HIGH_RISK",
    "raw_risk": float,
    "smoothed_risk": float,
    "triggered_rules": list[str],
    "reason_codes": list[str],
    "feature_snapshot": dict,
}
```

Module 6 can use this event to start or finalize evidence handling. It should never rewrite the original evidence in a way that hides uncertainty or removes the feature snapshot.

## 11. Startup and Runtime Sequence

### Basic camera operation

1. Install Python dependencies.
2. Ensure Tkinter is available for the operating system.
3. Start a camera manager module.
4. Select or add a camera.
5. Connect to the source.
6. Confirm that frames are arriving.
7. Capture screenshots or start recording.
8. Disconnect before closing.

### Vision operation

1. Create `models/`, `videos/`, and `screenshots/` directories.
2. Load the person model.
3. Optionally load the pose model.
4. Open the camera.
5. Start the capture worker.
6. Publish the latest frame.
7. Run person and pose inference.
8. Render detections and pose skeletons.
9. Create stable tracked person records.
10. Send person records to M4.

### Interaction and risk operation

1. M4 validates person records.
2. M4 creates unique pairs.
3. It calculates geometry and movement.
4. It accumulates temporal histories.
5. It emits feature dictionaries.
6. M5 canonicalizes the pair.
7. M5 updates pair memory.
8. M5 computes rule contributions and combination bonuses.
9. M5 smooths the risk score.
10. M5 updates the hysteresis state.
11. M5 emits a review event when alert conditions are met.
12. M6 preserves pre-event, event, and post-event evidence.

## 12. Installation Baseline

The source directly references these Python packages or standard-library components:

```text
opencv-python
ultralytics
torch
tkinter
mysql connector package, if the MySQL persistence path is enabled
```

Most imported modules are from Python’s standard library: `os`, `time`, `threading`, `queue`, `logging`, `math`, `json`, `csv`, `statistics`, `uuid`, `pathlib`, `dataclasses`, `enum`, `collections`, `itertools`, and `typing`.

A baseline setup is:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install opencv-python ultralytics torch mysql-connector-python
```

Tkinter is commonly installed through the operating-system package manager rather than pip. On Debian/Ubuntu systems, it is commonly provided by the `python3-tk` package.

The exact Torch build should be selected for the target CPU/GPU environment. A CPU-only installation is simpler; CUDA requires a compatible NVIDIA driver, CUDA runtime, and matching Torch build.

## 13. Suggested Project Layout

```text
project/
├── camera_manager_M1.py
├── detection_manager_M2.py
├── multi_person_detection_pose_M3.py
├── interactive_analyser_M4.py
├── pose_detection_M4.py
├── rules_engine_M5.py
├── rules_engine_tuning_M5.py
├── database_evidence_M6.py
├── models/
│   ├── yolo11n.pt
│   └── yolo11n-pose.pt
├── videos/
├── screenshots/
├── evidence/
├── .evidence_buffer/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

Recommended `.env.example` fields:

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=women_safety
MYSQL_USER=application_user
MYSQL_PASSWORD=change_me
```

Do not commit `.env`, model secrets, camera credentials, recorded video, screenshots, or evidence files.

## 14. Important Integration Gaps

### 14.1 No complete import pipeline

The modules are intentionally standalone. A production application needs an orchestrator that connects M3 output to M4, M4 output to M5, and M5 events to M6.

### 14.2 Tracking is not fully defined

Detection IDs must remain stable over time. If YOLO detections are assigned fresh IDs on every frame, movement, approach, following, and pair histories will be unreliable. Use a tested multi-object tracker and explicitly define ID loss and re-identification behaviour.

### 14.3 M4 and M5 feature names must be normalized

M4 may expose fields with slightly different names than M5 expects. Create one adapter layer and test it with fixtures. Do not silently map semantically different values.

### 14.4 M5 is not a sexual-content classifier

The current code analyses person interaction, proximity, movement, pose cues, and temporal combinations. It does not identify explicit sexual content. Any product claim must match the actual supported signals.

### 14.5 Evidence storage is sensitive

Video evidence may contain faces, bodies, location information, and bystanders. Implement encryption at rest, access logging, role-based access, retention/deletion policies, consent notices, and secure transport before deployment.

### 14.6 Database schema is not shown in the supplied material

The M6 file includes MySQL configuration and integration structures, but a complete schema, migration process, indexes, and transaction policy should be documented separately.

### 14.7 Duplicate M5 files

Keep one canonical rules-engine implementation. Treat the other as a tuning branch, or refactor shared logic into a package with configuration files.

## 15. Testing Strategy

### Unit tests

Test:

- Camera-source normalization.
- Pair canonicalization.
- Bounding-box center and height calculation.
- Proximity thresholds.
- Velocity and movement state.
- Approach trend detection.
- Following score calculation.
- Missing-keypoint handling.
- Risk threshold hysteresis.
- Episode creation and closure.
- Alert cooldown behaviour.
- Evidence filename and metadata generation.

### Synthetic temporal fixtures

Create deterministic sequences for:

- Two stationary people remaining far apart.
- Two people gradually approaching.
- A close interaction that immediately ends.
- Sustained proximity.
- Repeated approach and retreat.
- A person moving while another follows.
- Missing frames and tracking gaps.
- Person ID changes.
- Timestamp rewind.
- Confidence dropping below the threshold.

### Integration tests

Verify:

1. M3 output can be consumed by M4.
2. M4 output satisfies the M5 adapter contract.
3. M5 events enter M6 without blocking capture.
4. Evidence queues remain bounded.
5. MySQL failures do not crash the camera pipeline.
6. Application shutdown releases cameras, writers, worker threads, and windows.

### Performance tests

Measure:

- Capture FPS.
- Inference FPS.
- Display FPS.
- Average and percentile inference latency.
- Queue depth.
- CPU usage.
- GPU memory usage.
- Evidence disk throughput.
- Database write latency.

## 16. Production Hardening Checklist

- Replace global mutable camera lists with a configuration service or validated repository.
- Add structured logging instead of relying only on `print`.
- Add configuration validation at startup.
- Pin dependency versions after compatibility testing.
- Add model checksum verification.
- Use a real multi-object tracker.
- Keep all UI updates on the Tkinter main thread.
- Add graceful worker shutdown and join timeouts.
- Add bounded queues everywhere a producer can outrun a consumer.
- Handle camera reconnects without duplicate threads.
- Prevent concurrent video-writer access.
- Encrypt evidence storage.
- Add authentication and authorization for evidence access.
- Add audit logs for evidence viewing and export.
- Define evidence retention and deletion rules.
- Separate development/test data from real camera data.
- Do not expose raw camera streams or MySQL ports publicly.
- Store credentials in a secret manager.
- Calibrate thresholds using reviewed, representative data.
- Report uncertainty and missing inputs in every event.
- Maintain a human-review workflow.

## 17. Final System Interpretation

The project is best understood as a modular, real-time interaction-analysis and evidence-assistance system:

```text
M1/M2: camera acquisition and basic detection
M3: low-latency person and pose pipeline
M4: geometric and temporal interaction features
M5: explainable temporal risk scoring
M6: evidence and database integration
```

Its strongest design choices are the separation of acquisition from inference, bounded/latest-frame processing, temporal rather than single-frame analysis, canonical person-pair handling, explainable rule contributions, and explicit disclaimers against interpreting scores as proof.

Its most important next engineering task is to build and test a single orchestrator with stable data contracts, reliable tracking IDs, clear evidence lifecycle controls, and a privacy-preserving human-review process.
