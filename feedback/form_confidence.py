import numpy as np

def normalize(value, min_v, max_v):
    return max(0, min(1, (value - min_v) / (max_v - min_v)))

def squat_confidence(angle):
    score = 1 - normalize(abs(angle - 85), 0, 40)
    return int(score * 100)

def pushup_confidence(arm_angle):
    score = 1 - normalize(abs(arm_angle - 85), 0, 60)
    return int(score * 100)

def curl_confidence(curl_angle):
    score = 1 - normalize(abs(curl_angle - 60), 0, 70)
    return int(score * 100)
