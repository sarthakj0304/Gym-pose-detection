import cv2
import pickle
from pose_estimation.angle_calculation import calculate_angle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Squat:
    def __init__(self):
        self.counter = 0
        self.stage = None

        # ----------------------------
        # LOAD SQUAT TRAINED ML MODEL
        # ----------------------------
        try:
            with open("models/squat_model.pkl", "rb") as f:
                self.model = pickle.load(f)
        except:
            self.model = None
            print("⚠ Could NOT load squat_model.pkl — running rule-based only")
        try:
            with open("models/squat_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except:
            self.scaler=None
            print('Could not load the standard scaler')

    # ---------------------------------------------------
    # FEATURE EXTRACTION — EXACT ORDER OF YOUR DATASET
    # ---------------------------------------------------
    def prepare_features(self, lm):
        def L(i):
            return [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]

        # EXACT TRAINING ORDER
        nose = L(0)

        left_shoulder = L(11)
        right_shoulder = L(12)

        left_hip = L(23)
        right_hip = L(24)

        left_knee = L(25)
        right_knee = L(26)

        left_ankle = L(27)
        right_ankle = L(28)

        # Combine in same order as CSV
        features = (
            nose +
            left_shoulder + right_shoulder +
            left_hip + right_hip +
            left_knee + right_knee +
            left_ankle + right_ankle
        )

        features = np.array(features).reshape(1, -1)

        #  APPLY SCALING
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except Exception as e:
                print("Scaler transform failed:", e)

        return features


    # ---------------------------------------------------
    # RULE-BASED TRACKING + ML CLASSIFICATION
    # ---------------------------------------------------
    def track_squat(self, landmarks, frame):
        # RULE BASED ANGLES (your original code)
        hip = [int(landmarks[23].x * frame.shape[1]), int(landmarks[23].y * frame.shape[0])]
        knee = [int(landmarks[25].x * frame.shape[1]), int(landmarks[25].y * frame.shape[0])]
        shoulder = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]

        hip_right = [int(landmarks[24].x * frame.shape[1]), int(landmarks[24].y * frame.shape[0])]
        knee_right = [int(landmarks[26].x * frame.shape[1]), int(landmarks[26].y * frame.shape[0])]
        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]

        # Calculate angles
        angle = calculate_angle(shoulder, hip, knee)
        angle_right = calculate_angle(shoulder_right, hip_right, knee_right)

        # Drawing (unchanged)
        self.draw_line_with_style(frame, shoulder, hip, (178, 102, 255), 2)
        self.draw_line_with_style(frame, hip, knee, (178, 102, 255), 2)
        self.draw_line_with_style(frame, shoulder_right, hip_right, (51, 153, 255), 2)
        self.draw_line_with_style(frame, hip_right, knee_right, (51, 153, 255), 2)

        self.draw_circle(frame, shoulder, (178, 102, 255), 8)
        self.draw_circle(frame, hip, (178, 102, 255), 8)
        self.draw_circle(frame, knee, (178, 102, 255), 8)
        self.draw_circle(frame, shoulder_right, (51, 153, 255), 8)
        self.draw_circle(frame, hip_right, (51, 153, 255), 8)
        self.draw_circle(frame, knee_right, (51, 153, 255), 8)

        cv2.putText(frame, f'Angle L: {int(angle)}', (knee[0] + 10, knee[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Angle R: {int(angle_right)}', (knee_right[0] + 10, knee_right[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # REP COUNTING (unchanged)
        if angle > 170:
            self.stage = "Starting Position"
        elif 90 < angle < 170 and self.stage == "Starting Position":
            self.stage = "Descent"
        elif angle < 90 and self.stage == "Descent":
            self.stage = "Ascent"
            self.counter += 1

        # -----------------------------------------
        # ML PREDICTION
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
                print("Squat ML prediction error:", e)

        return self.counter, angle, self.stage, ml_label, ml_confidence

    # Drawing helpers
    def draw_line_with_style(self, frame, start_point, end_point, color, thickness):
        cv2.line(frame, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)

    def draw_circle(self, frame, center, color, radius):
        cv2.circle(frame, center, radius, color, -1)
