import cv2
import time
import math
import numpy as np
import pickle
from pose_estimation.angle_calculation import calculate_angle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PushUp:
    def __init__(self):
        self.counter = 0
        self.stage = "Initial"
        self.angle_threshold_up = 150
        self.angle_threshold_down = 70
        self.last_counter_update = time.time()

        # ---------------------------------------------
        # LOAD YOUR TRAINED PUSH-UP MODEL
        # ---------------------------------------------
        try:
            with open("models/pushup_model.pkl", "rb") as f:
                self.model = pickle.load(f)
        except:
            self.model = None
            print("⚠ Could NOT load pushup_model.pkl — running only rule-based")

    # ---------------------------------------------
    # Compute ground-relative angle between a limb
    # ---------------------------------------------
    def angle_with_ground(self, p1, p2):
        # p2-p1 vector
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        return abs(np.degrees(np.arctan2(dy, dx)))

    # ---------------------------------------------------
    # FEATURE EXTRACTION — EXACT MATCH TO YOUR TRAINING
    # ---------------------------------------------------
    def prepare_features(self, lm):
        # LEFT SIDE ONLY (your dataset appears single-side)
        shoulder = np.array([lm[11].x, lm[11].y])
        elbow = np.array([lm[13].x, lm[13].y])
        wrist = np.array([lm[15].x, lm[15].y])
        hip = np.array([lm[23].x, lm[23].y])
        knee = np.array([lm[25].x, lm[25].y])
        ankle = np.array([lm[27].x, lm[27].y])

        # Virtual ground point (slightly below ankle)
        ground_pt = np.array([ankle[0], ankle[1] + 0.1])

        # -----------------------------
        # Your 10 training angles:
        # -----------------------------
        Shoulder_Angle = calculate_angle(hip, shoulder, elbow)
        Elbow_Angle = calculate_angle(shoulder, elbow, wrist)
        Hip_Angle = calculate_angle(shoulder, hip, knee)
        Knee_Angle = calculate_angle(hip, knee, ankle)
        Ankle_Angle = calculate_angle(knee, ankle, ground_pt)

        Shoulder_Ground_Angle = self.angle_with_ground(shoulder, elbow)
        Elbow_Ground_Angle = self.angle_with_ground(elbow, wrist)
        Hip_Ground_Angle = self.angle_with_ground(hip, shoulder)
        Knee_Ground_Angle = self.angle_with_ground(knee, hip)
        Ankle_Ground_Angle = self.angle_with_ground(ankle, knee)

        return [
            Shoulder_Angle, Elbow_Angle, Hip_Angle, Knee_Angle, Ankle_Angle,
            Shoulder_Ground_Angle, Elbow_Ground_Angle, Hip_Ground_Angle,
            Knee_Ground_Angle, Ankle_Ground_Angle
        ]

    # ---------------------------------------------------
    # RULE-BASED TRACKING + ML CLASSIFICATION
    # ---------------------------------------------------
    def track_push_up(self, landmarks, frame):
        # BASIC LANDMARKS (from your original code)
        shoulder_left = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_left =    [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        wrist_left =    [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]

        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_right =    [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        wrist_right =    [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]

        # Left side used for rule-based counting
        angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)

        # Draw visuals (unchanged)
        self.draw_line_with_style(frame, shoulder_left, elbow_left, (0, 0, 255), 2)
        self.draw_line_with_style(frame, elbow_left, wrist_left, (0, 0, 255), 2)
        self.draw_line_with_style(frame, shoulder_right, elbow_right, (102, 0, 0), 2)
        self.draw_line_with_style(frame, elbow_right, wrist_right, (102, 0, 0), 2)

        self.draw_circle(frame, shoulder_left, (0, 0, 255), 8)
        self.draw_circle(frame, elbow_left, (0, 0, 255), 8)
        self.draw_circle(frame, wrist_left, (0, 0, 255), 8)
        self.draw_circle(frame, shoulder_right, (102, 0, 0), 8)
        self.draw_circle(frame, elbow_right, (102, 0, 0), 8)
        self.draw_circle(frame, wrist_right, (102, 0, 0), 8)

        # Display left elbow angle
        cv2.putText(frame, f'Angle: {int(angle_left)}',
                    (elbow_left[0] + 10, elbow_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # REP COUNT LOGIC (unchanged)
        current_time = time.time()

        if angle_left > self.angle_threshold_up:
            self.stage = "Starting position"
        elif self.angle_threshold_down < angle_left < self.angle_threshold_up and self.stage == "Starting position":
            self.stage = "Descent"
        elif angle_left < self.angle_threshold_down and self.stage == "Descent":
            self.stage = "Ascent"
            if current_time - self.last_counter_update > 1:
                self.counter += 1
                self.last_counter_update = current_time

        # -----------------------------------------
        # ML MODEL PREDICTION
        # -----------------------------------------
        ml_label = None
        ml_confidence = None

        if self.model is not None:
            try:
                features = self.prepare_features(landmarks)
                ml_label = int(self.model.predict(features)[0])

                if hasattr(self.model, "predict_proba"):
                    ml_confidence = float(self.model.predict_proba(features)[0].max())

            except Exception as e:
                print("Pushup ML prediction error:", e)

        return self.counter, angle_left, self.stage, ml_label, ml_confidence

    def draw_line_with_style(self, frame, start_point, end_point, color, thickness):
        cv2.line(frame, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)

    def draw_circle(self, frame, center, color, radius):
        cv2.circle(frame, center, radius, color, -1)
