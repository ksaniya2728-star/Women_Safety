"""
safety_manager_m3.py

Module 3 - Safety Manager
AI-Driven Surveillance System for Women Safety

Responsibilities:
- Receive multiple-person detection results from Module 2
- Track proximity between detected people
- Detect group formation
- Detect restricted-zone entry
- Generate safety events
- Prevent repeated alerts
- Maintain event history
"""

import cv2
import math
import time
from datetime import datetime


class SafetyManagerM3:

    def __init__(self):
        # -----------------------------
        # Configuration
        # -----------------------------

        # Distance threshold in pixels
        self.proximity_threshold = 120

        # Number of frames an event must persist
        self.required_frames = 20

        # Alert cooldown in seconds
        self.alert_cooldown = 10

        # Restricted zone
        # Format: x1, y1, x2, y2
        self.restricted_zone = (50, 50, 300, 300)

        # Event history
        self.event_history = []

        # Counters for persistent events
        self.proximity_counter = {}
        self.zone_counter = {}

        # Last alert time
        self.last_alert_time = {}

        print("[M3] Safety Manager initialized")

    # =========================================================
    # Calculate center of bounding box
    # =========================================================

    def get_center(self, bbox):

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    # =========================================================
    # Calculate distance between two people
    # =========================================================

    def calculate_distance(self, bbox1, bbox2):

        center1 = self.get_center(bbox1)
        center2 = self.get_center(bbox2)

        distance = math.sqrt(
            (center1[0] - center2[0]) ** 2 +
            (center1[1] - center2[1]) ** 2
        )

        return distance

    # =========================================================
    # Check whether a point is inside restricted zone
    # =========================================================

    def point_inside_zone(self, point):

        x, y = point

        x1, y1, x2, y2 = self.restricted_zone

        return (
            x1 <= x <= x2 and
            y1 <= y <= y2
        )

    # =========================================================
    # Check restricted zone
    # =========================================================

    def check_restricted_zone(self, person):

        person_id = person["id"]
        bbox = person["bbox"]

        center = self.get_center(bbox)

        inside = self.point_inside_zone(center)

        if inside:

            self.zone_counter[person_id] = \
                self.zone_counter.get(person_id, 0) + 1

        else:

            self.zone_counter[person_id] = 0

        # Generate event after persistence
        if self.zone_counter[person_id] >= self.required_frames:

            self.generate_event(
                event_type="RESTRICTED_ZONE_ENTRY",
                description=f"Person {person_id} entered restricted zone",
                person_ids=[person_id]
            )

            self.zone_counter[person_id] = 0

    # =========================================================
    # Check proximity between people
    # =========================================================

    def check_proximity(self, people):

        for i in range(len(people)):

            for j in range(i + 1, len(people)):

                person1 = people[i]
                person2 = people[j]

                id1 = person1["id"]
                id2 = person2["id"]

                distance = self.calculate_distance(
                    person1["bbox"],
                    person2["bbox"]
                )

                pair_id = tuple(sorted((id1, id2)))

                if distance < self.proximity_threshold:

                    self.proximity_counter[pair_id] = \
                        self.proximity_counter.get(pair_id, 0) + 1

                else:

                    self.proximity_counter[pair_id] = 0

                # Persistent proximity
                if self.proximity_counter[pair_id] >= self.required_frames:

                    self.generate_event(
                        event_type="CLOSE_PROXIMITY",
                        description=(
                            f"Persons {id1} and {id2} "
                            f"remain in close proximity"
                        ),
                        person_ids=[id1, id2]
                    )

                    self.proximity_counter[pair_id] = 0

    # =========================================================
    # Generate safety event
    # =========================================================

    def generate_event(
        self,
        event_type,
        description,
        person_ids
    ):

        current_time = time.time()

        # Prevent repeated alerts
        last_time = self.last_alert_time.get(event_type, 0)

        if current_time - last_time < self.alert_cooldown:
            return

        self.last_alert_time[event_type] = current_time

        event = {

            "timestamp":
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "event_type":
                event_type,

            "description":
                description,

            "person_ids":
                person_ids
        }

        self.event_history.append(event)

        print("\n================ SAFETY ALERT ================")

        print("Time       :", event["timestamp"])
        print("Event      :", event["event_type"])
        print("Description:", event["description"])
        print("Persons    :", event["person_ids"])

        print("================================================\n")

    # =========================================================
    # Process detection results from Module 2
    # =========================================================

    def process_detections(self, people):

        """
        Expected input:

        people = [
            {
                "id": 1,
                "bbox": (100, 100, 200, 300)
            },

            {
                "id": 2,
                "bbox": (250, 120, 350, 320)
            }
        ]
        """

        if people is None:
            return

        if len(people) == 0:
            return

        # Check restricted zones
        for person in people:

            self.check_restricted_zone(person)

        # Check person-to-person proximity
        if len(people) > 1:

            self.check_proximity(people)

    # =========================================================
    # Draw safety information on frame
    # =========================================================

    def draw_safety_overlay(self, frame, people):

        # Draw restricted zone

        x1, y1, x2, y2 = self.restricted_zone

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Restricted Zone",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Draw proximity lines

        for i in range(len(people)):

            for j in range(i + 1, len(people)):

                person1 = people[i]
                person2 = people[j]

                distance = self.calculate_distance(
                    person1["bbox"],
                    person2["bbox"]
                )

                if distance < self.proximity_threshold:

                    center1 = self.get_center(
                        person1["bbox"]
                    )

                    center2 = self.get_center(
                        person2["bbox"]
                    )

                    cv2.line(
                        frame,
                        center1,
                        center2,
                        (0, 0, 255),
                        2
                    )

                    midpoint = (
                        int((center1[0] + center2[0]) / 2),
                        int((center1[1] + center2[1]) / 2)
                    )

                    cv2.putText(
                        frame,
                        f"{int(distance)} px",
                        midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        # Display recent alert

        if self.event_history:

            latest_event = self.event_history[-1]

            cv2.putText(
                frame,
                f"ALERT: {latest_event['event_type']}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame

    # =========================================================
    # Get event history
    # =========================================================

    def get_event_history(self):

        return self.event_history

    # =========================================================
    # Clear event history
    # =========================================================

    def clear_event_history(self):

        self.event_history.clear()

        print("[M3] Event history cleared")