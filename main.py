import cv2
import numpy as np

from vision.hand_tracking import HandTracker
from vision.gesture_classifier import GestureClassifier
from state.gesture_fsm import GestureFSM
from state.swipe_detector import SwipeDetector

from drawing.canvas import Canvas
from drawing.stroke_engine import StrokeEngine
from drawing.shape_corrector import ShapeCorrector


# ================= COLOR PALETTE =================
COLORS = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 165, 255),   # Orange
    (0, 0, 255),     # Red
    (0, 0, 0)        # Black (eraser)
]

PALETTE_X = 10
PALETTE_Y = 150
BOX_SIZE = 40
COLOR_SELECT_THRESHOLD = 15


# ================= SHAPE BAR =================
SHAPES = ["FREE", "LINE", "RECTANGLE", "CIRCLE", "TRIANGLE"]
SHAPE_BOX = 40
SHAPE_SELECT_THRESHOLD = 15


def normalize_points(points):
    pts = np.array(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    return pts


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        return

    # ---------------- INIT MODULES ----------------
    tracker = HandTracker()
    classifier = GestureClassifier()
    fsm = GestureFSM()
    swipe = SwipeDetector()

    canvas = Canvas(frame.shape)
    stroke = StrokeEngine()
    shape_corrector = ShapeCorrector()

    current_color = COLORS[0]
    current_shape = "FREE"
    color_hold = 0
    shape_hold = 0
    prev_state = None

    SHAPE_BAR_X = frame.shape[1] - 60
    SHAPE_BAR_Y = 150

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ================= HAND TRACKING =================
        landmarks = tracker.process(frame)
        gesture = classifier.classify(landmarks)
        state = fsm.update(gesture)

        # ================= PALM ERASER OVERRIDE =================
        if classifier.is_palm_open(landmarks):
            state = "ERASE"

        # ================= SWIPE UNDO / REDO =================
        finger_count = classifier.count_fingers(landmarks)
        if landmarks:
            swipe.update(landmarks[8])

        direction = swipe.detect()
        if direction and finger_count == 2:
            if direction == "LEFT":
                canvas.undo()
            elif direction == "RIGHT":
                canvas.redo()
            swipe.reset()

        # ================= COLOR & SHAPE SELECTION =================
        if state == "SELECT" and landmarks:
            x, y = landmarks[8]

            # ---- COLOR PALETTE ----
            if x < frame.shape[1] // 2:
                hovering = False
                for i, color in enumerate(COLORS):
                    box_y = PALETTE_Y + i * (BOX_SIZE + 10)
                    if PALETTE_X < x < PALETTE_X + BOX_SIZE and box_y < y < box_y + BOX_SIZE:
                        hovering = True
                        color_hold += 1
                        if color_hold >= COLOR_SELECT_THRESHOLD:
                            current_color = color
                            color_hold = 0
                        break
                if not hovering:
                    color_hold = 0
                shape_hold = 0

            # ---- SHAPE BAR ----
            else:
                hovering = False
                for i, shape in enumerate(SHAPES):
                    box_y = SHAPE_BAR_Y + i * (SHAPE_BOX + 10)
                    if SHAPE_BAR_X < x < SHAPE_BAR_X + SHAPE_BOX and box_y < y < box_y + SHAPE_BOX:
                        hovering = True
                        shape_hold += 1
                        if shape_hold >= SHAPE_SELECT_THRESHOLD:
                            current_shape = shape
                            shape_hold = 0
                        break
                if not hovering:
                    shape_hold = 0
                color_hold = 0

        # ================= STROKE END → SHAPE =================
        if prev_state == "DRAW" and state != "DRAW" and stroke.current_stroke:
            if current_shape != "FREE":
                canvas.undo()

            if current_shape != "FREE":
                pts = normalize_points(stroke.current_stroke)
                if pts is not None:
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

                    if current_shape == "LINE":
                        cv2.line(canvas.canvas, tuple(pts[0]), tuple(pts[-1]), current_color, 4)

                    elif current_shape == "RECTANGLE":
                        cv2.rectangle(canvas.canvas, (x_min, y_min), (x_max, y_max), current_color, 4)

                    elif current_shape == "CIRCLE":
                        r = max(x_max - x_min, y_max - y_min) // 2
                        cv2.circle(canvas.canvas, (cx, cy), r, current_color, 4)

                    elif current_shape == "TRIANGLE":
                        triangle = np.array([
                            (cx, y_min),
                            (x_min, y_max),
                            (x_max, y_max)
                        ])
                        cv2.polylines(canvas.canvas, [triangle], True, current_color, 4)

            stroke.reset()

        # ================= DRAWING =================
        if state == "DRAW" and landmarks:
            if stroke.prev is None:
                canvas.save()
            stroke.draw(canvas.canvas, landmarks[8], current_color)

        # ================= PALM ERASER =================
        if state == "ERASE" and landmarks:
            if prev_state != "ERASE":
                canvas.save()
            hand_pts = np.array(list(landmarks.values()), dtype=np.int32)
            hull = cv2.convexHull(hand_pts)
            cv2.fillConvexPoly(canvas.canvas, hull, (0, 0, 0))

        # ================= UI =================
        output = cv2.addWeighted(frame, 0.7, canvas.canvas, 0.3, 0)
        cv2.imshow("AirCanvas X", output)

        prev_state = state
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
