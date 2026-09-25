import cv2
import numpy as np

def draw_colored_overlay(frame, bbox, color, alpha=0.35):
    x1, y1, x2, y2 = map(int, bbox)

    overlay = frame.copy()

    # Filled transparent rectangle
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay onto frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def draw_object(frame, bbox, class_name, track_id):
    # Pick class color or default red
    color = CLASS_COLORS.get(class_name, (0, 0, 255))

    # Transparent overlay
    frame = draw_colored_overlay(frame, bbox, color)

    # Bounding box outline
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label text (class + id)
    label_text = f"{class_name} #{track_id}"

    cv2.putText(frame, label_text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

    return frame


# BGR, one per class the model predicts; anything unknown falls back to red
CLASS_COLORS = {
    "player": (255, 144, 30),                # Blue
    "player-in-possession": (255, 0, 255),   # Magenta
    "player-jump-shot": (255, 255, 0),       # Cyan
    "player-layup-dunk": (180, 105, 255),    # Pink
    "player-shot-block": (130, 0, 75),       # Indigo
    "referee": (200, 200, 200),              # Light grey
    "number": (0, 255, 255),                 # Yellow
    "ball": (0, 165, 255),                   # Orange
    "ball-in-basket": (0, 255, 0),           # Green
    "rim": (0, 100, 0),                      # Dark green
}
