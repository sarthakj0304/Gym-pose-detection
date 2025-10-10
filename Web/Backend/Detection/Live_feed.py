from pose_extractor import PoseExtractor
from bicep_curl import BicepCurlAnalyzer
import cv2

pose_extractor = PoseExtractor()
bicep_analyzer = BicepCurlAnalyzer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_extractor.pose.process(frame)
    landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

    frame, acc = bicep_analyzer.analyze_pose(landmarks, frame, results, 0)

    cv2.imshow("Bicep Curl Analysis", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
