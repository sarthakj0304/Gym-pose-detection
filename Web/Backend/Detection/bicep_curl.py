import cv2
import numpy as np
import mediapipe as mp
import pickle
from utils import calculate_angle  # You already have this function
from math import degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class BicepCurlAnalyzer:
    def __init__(self):
        # ---- Load model and scaler ----
        with open("Models/bicep_curl.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("Models/bicep_curl/input_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # ---- Thresholds for counting reps ----
        self.stage_down_threshold = 120
        self.stage_up_threshold = 60
        self.stage = "down"
        self.counter = 0

        # ---- Visualization config ----
        self.has_error = False

        # ---- Important landmarks ----
        self.important_landmarks = [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
        ]

    # -----------------------------------------------------------
    def extract_angles(self, landmarks):
        """
        Extract key angles for the bicep curl
        """

        # Get coordinates of left arm joints
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Get coordinates of right arm joints
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # ---- Compute angles ----
        left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        return [left_angle, right_angle]

    # -----------------------------------------------------------
    def predict_accuracy(self, angles):
        """
        Scale the angles and predict accuracy score from model
        """
        X = np.array(angles).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        accuracy = self.model.predict(X_scaled)[0]
        return accuracy

    # -----------------------------------------------------------
    def update_counter(self, angle):
        """
        Simple counter logic based on elbow angle
        """
        if angle > self.stage_down_threshold:
            self.stage = "down"
        elif angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

    # -----------------------------------------------------------
    def analyze_pose(self, landmarks, frame, results, timestamp):
        """
        Analyze the pose for one frame:
        - extract angles
        - predict accuracy
        - update counter
        - draw results
        """

        if not landmarks:
            return frame, None

        # ---- Extract angles ----
        left_angle, right_angle = self.extract_angles(landmarks)

        # ---- Predict accuracy ----
        accuracy = self.predict_accuracy([left_angle, right_angle])

        # ---- Update counter ----
        self.update_counter(left_angle)

        # ---- Draw landmarks ----
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 117, 16), thickness=2, circle_radius=1),
        )

        # ---- Display accuracy and reps ----
        cv2.rectangle(frame, (0, 0), (300, 80), (245, 117, 16), -1)
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Reps: {self.counter}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame, accuracy
