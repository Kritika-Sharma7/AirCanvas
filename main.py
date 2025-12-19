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
    (0, 0, 0)        # Black
]

PALETTE_X = 10
PALETTE_Y = 150
BOX_SIZE = 40

# ================= SHAPES =================
SHAPES = ["FREE", "LINE", "RECTANGLE", "CIRCLE", "TRIANGLE"]
SHAPE_BOX = 40
SHAPE_SELECT_THRESHOLD = 15
COLOR_SELECT_THRESHOLD = 15


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

    # ================= INIT MODULES =================
    tracker = HandTracker()
    classifier = GestureClassifier()
    fsm = GestureFSM()
    swipe = SwipeDetector()

    canvas = Canvas(frame.shape)
    stroke = StrokeEngine()
    shape_corrector = ShapeCorrector()

    prev_state = None
    current_color = COLORS[0]
    current_shape = "FREE"

    color_hold_counter = 0
    shape_hold_counter = 0

    SHAPE_BAR_X = frame.shape[1] - 60
    SHAPE_BAR_Y = 150

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ================= HAND TRACKING =================
        landmarks = tracker.process(frame)

        # ================= GESTURE =================
        gesture = classifier.classify(landmarks)
        state = fsm.update(gesture)

        # ================= PALM ERASER OVERRIDE =================
        if landmarks and classifier.is_palm_open(landmarks):
            state = "ERASE"

        # ================= SWIPE UNDO / REDO =================
        if landmarks and 8 in landmarks:
            swipe.update(landmarks[8])
            direction = swipe.detect()
            finger_count = classifier.count_fingers(landmarks)

            if direction:
                if finger_count == 2 and direction == "LEFT":
                    canvas.undo()
                elif finger_count == 2 and direction == "RIGHT":
                    canvas.redo()
                swipe.reset()

        # ================= TOOL SELECTION =================
        if state == "SELECT" and landmarks and 8 in landmarks:
            x, y = landmarks[8]

            # ---------- COLOR PALETTE ----------
            if x < frame.shape[1] // 2:
                hovering = False
                for i, color in enumerate(COLORS):
                    box_y = PALETTE_Y + i * (BOX_SIZE + 10)
                    if PALETTE_X < x < PALETTE_X + BOX_SIZE and box_y < y < box_y + BOX_SIZE:
                        hovering = True
                        color_hold_counter += 1
                        if color_hold_counter >= COLOR_SELECT_THRESHOLD:
                            current_color = color
                            color_hold_counter = 0
                        break
                if not hovering:
                    color_hold_counter = 0
                shape_hold_counter = 0

            # ---------- SHAPE BAR ----------
            else:
                hovering = False
                for i, shape in enumerate(SHAPES):
                    box_y = SHAPE_BAR_Y + i * (SHAPE_BOX + 10)
                    if SHAPE_BAR_X < x < SHAPE_BAR_X + SHAPE_BOX and box_y < y < box_y + SHAPE_BOX:
                        hovering = True
                        shape_hold_counter += 1
                        if shape_hold_counter >= SHAPE_SELECT_THRESHOLD:
                            current_shape = shape
                            shape_hold_counter = 0
                        break
                if not hovering:
                    shape_hold_counter = 0
                color_hold_counter = 0

        # ================= STROKE END =================
        if prev_state == "DRAW" and state != "DRAW" and stroke.current_stroke:
            if current_shape != "FREE":
                canvas.undo()

            # ❗ FIX: FREE means NO shape correction
            if current_shape == "FREE":
                result = None
            else:
                result = (current_shape, stroke.current_stroke)

            if result:
                shape, data = result
                pts = normalize_points(data)
                if pts is not None:
                    canvas.save()

                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

                    if shape == "LINE":
                        cv2.line(canvas.canvas, tuple(pts[0]), tuple(pts[-1]), current_color, 4)

                    elif shape == "RECTANGLE":
                        cv2.rectangle(canvas.canvas, (x_min, y_min), (x_max, y_max), current_color, 4)

                    elif shape == "CIRCLE":
                        r = max(x_max - x_min, y_max - y_min) // 2
                        cv2.circle(canvas.canvas, (cx, cy), r, current_color, 4)

                    elif shape == "TRIANGLE":
                        tri = np.array([(cx, y_min), (x_min, y_max), (x_max, y_max)])
                        cv2.polylines(canvas.canvas, [tri], True, current_color, 4)

            stroke.reset()

        # ================= FULL HAND ERASER =================
        if state == "ERASE" and landmarks:
            if prev_state != "ERASE":
                canvas.save()

            hand_pts = np.array(list(landmarks.values()), dtype=np.int32)
            hull = cv2.convexHull(hand_pts)
            cv2.fillConvexPoly(canvas.canvas, hull, (255, 255, 255))  # FIXED

        # ================= DRAWING =================
        if state == "DRAW" and landmarks and 8 in landmarks:
            if stroke.prev is None:
                canvas.save()
            stroke.draw(canvas.canvas, landmarks[8], current_color)

        # ================= UI =================
        for i, color in enumerate(COLORS):
            y = PALETTE_Y + i * (BOX_SIZE + 10)
            cv2.rectangle(frame, (PALETTE_X, y), (PALETTE_X + BOX_SIZE, y + BOX_SIZE), color, -1)
            if color == current_color:
                cv2.rectangle(frame, (PALETTE_X, y), (PALETTE_X + BOX_SIZE, y + BOX_SIZE), (255, 255, 255), 2)

        for i, shape in enumerate(SHAPES):
            box_y = SHAPE_BAR_Y + i * (SHAPE_BOX + 10)

            # Background
            cv2.rectangle(
                frame,
                (SHAPE_BAR_X, box_y),
                (SHAPE_BAR_X + SHAPE_BOX, box_y + SHAPE_BOX),
                (50, 50, 50),
                -1
            )

            cx = SHAPE_BAR_X + SHAPE_BOX // 2
            cy = box_y + SHAPE_BOX // 2

            # -------- SHAPE ICONS --------
            if shape == "LINE":
                cv2.line(frame, (cx-12, cy+12), (cx+12, cy-12), (255,255,255), 2)

            elif shape == "RECTANGLE":
                cv2.rectangle(frame, (cx-12, cy-12), (cx+12, cy+12), (255,255,255), 2)

            elif shape == "CIRCLE":
                cv2.circle(frame, (cx, cy), 12, (255,255,255), 2)

            elif shape == "TRIANGLE":
                pts = np.array([
                    (cx, cy-14),
                    (cx-14, cy+12),
                    (cx+14, cy+12)
                ])
                cv2.polylines(frame, [pts], True, (255,255,255), 2)

            elif shape == "FREE":
                cv2.putText(
                    frame,
                    "F",
                    (cx-7, cy+7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

            # -------- SELECTED HIGHLIGHT --------
            if shape == current_shape:
                cv2.rectangle(
                    frame,
                    (SHAPE_BAR_X, box_y),
                    (SHAPE_BAR_X + SHAPE_BOX, box_y + SHAPE_BOX),
                    (0, 255, 255),
                    2
                )


        cv2.putText(frame, f"STATE: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"TOOL: {current_shape}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        output = cv2.addWeighted(frame, 0.7, canvas.canvas, 0.3, 0)
        cv2.imshow("AirCanvas X", output)
        cv2.imshow("Whiteboard", canvas.canvas)

        prev_state = state

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
