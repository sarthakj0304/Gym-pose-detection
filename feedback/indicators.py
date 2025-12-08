# feedback/indicators.py
from utils.drawing_utils import (
    draw_gauge_meter,
    draw_progress_bar,
    display_stage,
    display_counter
)
import cv2

display_counter_poisiton = (40, 240)
display_stage_poisiton = (40, 270)


# ----------------------------------------
# ✅ SQUAT FEEDBACK
# ----------------------------------------
def draw_squat_indicators(frame, counter, angle, stage):
    display_counter(frame, counter, position=display_counter_poisiton,
                    color=(0, 0, 0), background_color=(192, 192, 192))

    display_stage(frame, stage, "Stage", position=display_stage_poisiton,
                  color=(0, 0, 0), background_color=(192, 192, 192))

    draw_progress_bar(frame, exercise="squat", value=counter,
                      position=(40, 170), size=(200, 20),
                      color=(163, 245, 184, 1), background_color=(255, 255, 255))

    # -------- REAL-TIME COACHING --------
    if angle > 140:
        feedback = "Go deeper"
        color = (0, 0, 255)
    elif 90 < angle <= 140:
        feedback = "Good depth"
        color = (0, 165, 255)
    else:
        feedback = "Perfect squat!"
        color = (0, 200, 0)

    cv2.putText(frame, f"Feedback: {feedback}", (40, 320),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)


# ----------------------------------------
# ✅ PUSH-UP FEEDBACK
# ----------------------------------------
def draw_pushup_indicators(frame, counter, angle, stage):
    display_counter(frame, counter, position=display_counter_poisiton,
                    color=(0, 0, 0), background_color=(192, 192, 192))

    display_stage(frame, stage, "Stage", position=display_stage_poisiton,
                  color=(0, 0, 0), background_color=(192, 192, 192))

    draw_progress_bar(frame, exercise="push_up", value=counter,
                      position=(40, 170), size=(200, 20),
                      color=(163, 245, 184, 1), background_color=(255, 255, 255))

    # -------- REAL-TIME COACHING --------
    if angle > 160:
        feedback = "Lower your chest"
        color = (0, 0, 255)
    elif 90 < angle <= 160:
        feedback = "Good depth"
        color = (0, 165, 255)
    else:
        feedback = "Push back up!"
        color = (0, 200, 0)

    cv2.putText(frame, f"Feedback: {feedback}", (40, 320),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)


# ----------------------------------------
# ✅ HAMMER CURL FEEDBACK
# ----------------------------------------
def draw_hammercurl_indicators(frame,
                               counter_right, angle_right,
                               counter_left, angle_left,
                               stage_right, stage_left):

    display_counter_poisiton_left_arm = (40, 300)

    display_counter(frame, counter_right, position=display_counter_poisiton,
                    color=(0, 0, 0), background_color=(192, 192, 192))

    display_stage(frame, stage_right, "Right Stage",
                  position=display_stage_poisiton,
                  color=(0, 0, 0), background_color=(192, 192, 192))

    display_stage(frame, stage_left, "Left Stage",
                  position=display_counter_poisiton_left_arm,
                  color=(0, 0, 0), background_color=(192, 192, 192))

    draw_progress_bar(frame, exercise="hammer_curl",
                      value=(counter_right + counter_left) / 2,
                      position=(40, 170), size=(200, 20),
                      color=(163, 245, 184, 1), background_color=(255, 255, 255))

    # -------- RIGHT ARM COACHING --------
    if angle_right > 150:
        feedback_r = "Extend right arm"
        color_r = (0, 0, 255)
    elif 80 < angle_right <= 150:
        feedback_r = "Curl up"
        color_r = (0, 165, 255)
    else:
        feedback_r = "Good! Go down slowly"
        color_r = (0, 200, 0)

    # -------- LEFT ARM COACHING --------
    if angle_left > 150:
        feedback_l = "Extend left arm"
        color_l = (0, 0, 255)
    elif 80 < angle_left <= 150:
        feedback_l = "Curl up"
        color_l = (0, 165, 255)
    else:
        feedback_l = "Good! Go down slowly"
        color_l = (0, 200, 0)

    cv2.putText(frame, f"Right: {feedback_r}", (40, 350),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, color_r, 2)

    cv2.putText(frame, f"Left: {feedback_l}", (40, 380),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, color_l, 2)
