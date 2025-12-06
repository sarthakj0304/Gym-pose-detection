import cv2
import numpy as np
import pickle
from pose_estimation.angle_calculation import calculate_angle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class HammerCurl:
    def __init__(self):
        self.counter_right = 0
        self.counter_left = 0
        self.stage_right = None
        self.stage_left = None

        # Thresholds (same as your original)
        self.angle_threshold = 40
        self.flexion_angle_up = 155
        self.flexion_angle_down = 35
        self.angle_threshold_up = 155
        self.angle_threshold_down = 47

        # ----------------------
        # LOAD YOUR TRAINED MODEL
        # ----------------------
        try:
            with open("models/hammer_curl_model.pkl", "rb") as f:
                self.model = pickle.load(f)
        except:
            self.model = None
            print(" Could NOT load hammer_curl_model.pkl — using rule-based only")
        try:
            with open("models/hammer_scaler.pkl", "rb") as f:
                
                self.scaler = pickle.load(f)
        except:
            self.scaler=None
            print('Could not load the standard scaler')

    # ------------------------------------------
    # FEATURE PREPARATION FOR ML MODEL
    # ------------------------------------------
    def prepare_features(self, lm):
        # Helper to extract (x, y, z, v)
        def L(i):
            return [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]

        # FOLLOW TRAINING ORDER EXACTLY
        nose = L(0)

        left_shoulder = L(11)
        right_shoulder = L(12)

        right_elbow = L(14)
        left_elbow = L(13)

        right_wrist = L(16)
        left_wrist = L(15)

        left_hip = L(23)
        right_hip = L(24)

        # EXACT SAME ORDER AS TRAINING CSV
        features = (
            nose +
            left_shoulder + right_shoulder +
            right_elbow + left_elbow +        # <-- FIXED ORDER
            right_wrist + left_wrist +        # <-- FIXED ORDER
            left_hip + right_hip
        )

        features = np.array(features).reshape(1, -1)

        #  APPLY SCALING
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except Exception as e:
                print("Scaler transform failed:", e)

        return features


    # ----------------------------------------------------------
    # MAIN TRACKING FUNCTION (RULE BASED + ML PREDICTION ADDED)
    # ----------------------------------------------------------
    def track_hammer_curl(self, landmarks, frame):

        # (1) Extract points for rule-based tracking (unchanged)
        shoulder_right = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_right =    [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        hip_right =      [int(landmarks[23].x * frame.shape[1]), int(landmarks[23].y * frame.shape[0])]
        wrist_right =    [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]

        shoulder_left = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_left =    [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        hip_left =      [int(landmarks[24].x * frame.shape[1]), int(landmarks[24].y * frame.shape[0])]
        wrist_left =    [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]

        # ANGLE CALCULATIONS (unchanged)
        angle_right_counter = calculate_angle(shoulder_right, elbow_right, wrist_right)
        angle_left_counter =  calculate_angle(shoulder_left, elbow_left, wrist_left)

        angle_right = calculate_angle(elbow_right, shoulder_right, hip_right)
        angle_left =  calculate_angle(elbow_left, shoulder_left, hip_left)

        # REP COUNTING (unchanged)
        if angle_right_counter > self.angle_threshold_up:
            self.stage_right = "Flex"
        elif self.angle_threshold_down < angle_right_counter < self.angle_threshold_up and self.stage_right == "Flex":
            self.stage_right = "Up"
        elif angle_right_counter < self.angle_threshold_down and self.stage_right=="Up":
            self.stage_right = "Down"
            self.counter_right +=1

        if angle_left_counter > self.angle_threshold_up:
            self.stage_left = "Flex"
        elif self.angle_threshold_down < angle_left_counter < self.angle_threshold_up and self.stage_left == "Flex":
            self.stage_left = "Up"
        elif angle_left_counter < self.angle_threshold_down and self.stage_left == "Up":
            self.stage_left = "Down"
            self.counter_left +=1

        # PROGRESS
        progress_right = 1 if self.stage_right == "up" else 0
        progress_left = 1 if self.stage_left == "up" else 0

        # ------------------------------------------------------
        # (2) ML MODEL PREDICTION
        # ------------------------------------------------------
        ml_label = None
        ml_confidence = None

        if self.model is not None:
            features = self.prepare_features(landmarks)

            try:
                ml_label = int(self.model.predict(features)[0])

                if hasattr(self.model, "predict_proba"):
                    ml_confidence = float(self.model.predict_proba(features)[0].max())
                else:
                    ml_confidence = None
            except Exception as e:
                print("ML prediction error:", e)

        # ------------------------------------------------------
        # RETURN EVERYTHING (ADDING ML OUTPUT)
        # ------------------------------------------------------
        return (
            self.counter_right,
            angle_right_counter,

            self.counter_left,
            angle_left_counter,

            None,  # warning right (you can add later)
            None,  # warning left

            progress_right,
            progress_left,

            self.stage_right,
            self.stage_left,

            ml_label,
            ml_confidence
        )
