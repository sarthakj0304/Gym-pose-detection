# feedback/layout.py

from feedback.indicators import (
    draw_squat_indicators,
    draw_pushup_indicators,
    draw_hammercurl_indicators
)

from feedback.form_confidence import (
    squat_confidence,
    pushup_confidence,
    curl_confidence
)

from utils.draw_text_with_background import draw_text_with_background
import cv2


def layout_indicators(frame, exercise_type, exercise_data):

    if exercise_type == "squat":
        counter, angle, stage = exercise_data
        draw_squat_indicators(frame, counter, angle, stage)

        # ✅ RULE-BASED CONFIDENCE
        confidence = squat_confidence(angle)

        draw_text_with_background(
            frame,
            f"Accuracy: {confidence}%",
            (40, 200),
            cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255),
            (0, 140, 90),
            1
        )

    elif exercise_type == "push_up":
        counter, angle, stage = exercise_data
        draw_pushup_indicators(frame, counter, angle, stage)

        # ✅ RULE-BASED CONFIDENCE
        confidence = pushup_confidence(angle)

        draw_text_with_background(
            frame,
            f"Accuracy: {confidence}%",
            (40, 200),
            cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255),
            (0, 140, 90),
            1
        )

    elif exercise_type == "hammer_curl":
        (counter_right, angle_right,
         counter_left, angle_left,
         warning_message_right, warning_message_left,
         progress_right, progress_left,
         stage_right, stage_left) = exercise_data

        draw_hammercurl_indicators(
            frame,
            counter_right, angle_right,
            counter_left, angle_left,
            stage_right, stage_left
        )

        # ✅ USE BEST ARM FOR CONFIDENCE
        best_angle = min(angle_right, angle_left)
        confidence = curl_confidence(best_angle)

        draw_text_with_background(
            frame,
            f"Accuracy: {confidence}%",
            (40, 200),
            cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255),
            (0, 140, 90),
            1
        )
