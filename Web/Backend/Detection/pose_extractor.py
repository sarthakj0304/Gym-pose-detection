import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseExtractor:
    def __init__(self, visibility_threshold=0.6):
        self.pose = mp_pose.Pose()
        self.visibility_threshold = visibility_threshold

    def get_landmarks(self, frame):
        results = self.pose.process(frame)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark
