import cv2
import numpy as np
from drawing.shape_corrector import ShapeCorrector
from vision.hand_tracking import HandTracker
from vision.gesture_classifier import GestureClassifier
from state.gesture_fsm import GestureFSM
from drawing.canvas import Canvas
from drawing.stroke_engine import StrokeEngine
# ---------------- COLOR PALETTE ----------------
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
def normalize_points(points):
    pts = np.array(points, dtype=np.int32)

    # If empty or single value → invalid
    if pts.size < 2:
        return None

    # If shape is (2,) → one point
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    # If shape is (N,1) → invalid
    if pts.shape[1] != 2:
        return None

    # Need at least 2 points to draw
    if pts.shape[0] < 2:
        return None

    return pts

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Read one frame to get shape for canvas
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        return
    # ---------------- SHAPE BAR CONFIG ----------------
    SHAPES = ["FREE", "LINE", "RECTANGLE", "CIRCLE", "TRIANGLE"]

    SHAPE_BAR_X = frame.shape[1] - 60
    SHAPE_BAR_Y = 150
    SHAPE_BOX = 40

    current_shape = "FREE"
    shape_hold_counter = 0
    SHAPE_SELECT_THRESHOLD = 15

    # Initialize modules
    tracker = HandTracker()
    classifier = GestureClassifier()
    fsm = GestureFSM()

    canvas = Canvas(frame.shape)
    stroke = StrokeEngine()
    shape_corrector = ShapeCorrector()
    current_color = COLORS[0]
    color_hold_counter = 0
    COLOR_SELECT_THRESHOLD = 15
    prev_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        # ==========================================================
        # STEP 1: HAND TRACKING
        # ==========================================================
        landmarks = tracker.process(frame)

        # ==========================================================
        # STEP 2: GESTURE CLASSIFICATION
        # ==========================================================
        gesture = classifier.classify(landmarks)

        # ==========================================================
        # STEP 3: FSM UPDATE
        # ==========================================================
        state = fsm.update(gesture)
 
        # ==========================================================
        # FIXED SELECTION LOGIC (COLOR vs SHAPE)
        # ==========================================================
        if state == "SELECT" and landmarks:
            x, y = landmarks[8]

            # -------- COLOR PALETTE (LEFT SIDE) --------
            if x < frame.shape[1] // 2:
                hovering_color = False

                for i, color in enumerate(COLORS):
                    box_y = PALETTE_Y + i * (BOX_SIZE + 10)

                    if PALETTE_X < x < PALETTE_X + BOX_SIZE and box_y < y < box_y + BOX_SIZE:
                        hovering_color = True
                        color_hold_counter += 1

                        if color_hold_counter >= COLOR_SELECT_THRESHOLD:
                            current_color = color
                            color_hold_counter = 0
                        break

                if not hovering_color:
                    color_hold_counter = 0

                shape_hold_counter = 0  # HARD RESET shape counter

            # -------- SHAPE BAR (RIGHT SIDE) --------
            else:
                hovering_shape = False

                for i, shape in enumerate(SHAPES):
                    box_y = SHAPE_BAR_Y + i * (SHAPE_BOX + 10)

                    if SHAPE_BAR_X < x < SHAPE_BAR_X + SHAPE_BOX and box_y < y < box_y + SHAPE_BOX:
                        hovering_shape = True
                        shape_hold_counter += 1

                        if shape_hold_counter >= SHAPE_SELECT_THRESHOLD:
                            current_shape = shape
                            shape_hold_counter = 0
                        break

                if not hovering_shape:
                    shape_hold_counter = 0

                color_hold_counter = 0  # HARD RESET color counter




        # ==========================================================
        # COLOR SELECTION LOGIC (FIXED)
        # ==========================================================
        # if state == "SELECT" and landmarks:
        #     x, y = landmarks[8]  # index fingertip
        #     hovering = False

        #     for i, color in enumerate(COLORS):
        #         box_y = PALETTE_Y + i * (BOX_SIZE + 10)

        #         if PALETTE_X < x < PALETTE_X + BOX_SIZE and box_y < y < box_y + BOX_SIZE:
        #             hovering = True
        #             color_hold_counter += 1

        #             if color_hold_counter >= COLOR_SELECT_THRESHOLD:
        #                 current_color = color
        #                 color_hold_counter = 0
        #                 break   # stop checking other boxes

        #     if not hovering:
        #         color_hold_counter = 0


        # ==========================================================
        # STEP 5: STROKE END DETECTION + SHAPE AUTO-CORRECTION
        # ==========================================================
        if prev_state == "DRAW" and state != "DRAW" and stroke.current_stroke:
            if current_shape != "FREE":
                canvas.undo()


            # STEP 5: Shape decision
            # STEP 5: Shape decision
            if current_shape == "FREE":
                # Do NOT force shape — draw as-is
                result = None
            else:
                result = (current_shape, stroke.current_stroke)


            # STEP 6: DRAW PERFECT SHAPES
            if result and current_shape!="FREE":
                shape, data = result
                pts = normalize_points(data)
                if pts is None:
                    stroke.current_stroke.clear()
                    stroke.reset()
                    continue

                x_min = int(pts[:, 0].min())
                y_min = int(pts[:, 1].min())
                x_max = int(pts[:, 0].max())
                y_max = int(pts[:, 1].max())

                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2


                if shape == "LINE":
                    cv2.line(
                        canvas.canvas,
                        tuple(pts[0]),
                        tuple(pts[-1]),
                        current_color,
                        4
                    )

                elif shape == "RECTANGLE":
                    cv2.rectangle(
                        canvas.canvas,
                        (x_min, y_min),
                        (x_max, y_max),
                        current_color,
                        4
                    )

                elif shape == "CIRCLE":
                    radius = max(x_max - x_min, y_max - y_min) // 2
                    cv2.circle(
                        canvas.canvas,
                        (cx, cy),
                        radius,
                        current_color,
                        4
                    )

                elif shape == "TRIANGLE":
                    triangle = np.array([
                        (cx, y_min),
                        (x_min, y_max),
                        (x_max, y_max)
                    ])
                    cv2.polylines(
                        canvas.canvas,
                        [triangle],
                        True,
                        current_color,
                        4
                    )

            stroke.current_stroke.clear()
            stroke.reset()


        # ==========================================================
        # STEP 6: NORMAL DRAWING
        # ==========================================================
        if state == "DRAW" and landmarks:
            if stroke.prev is None:
                canvas.save()
            stroke.draw(canvas.canvas, landmarks[8], current_color)

        # ==========================================================
        # DRAW COLOR PALETTE UI + HOLD FEEDBACK
        # ==========================================================
        for i, color in enumerate(COLORS):
            box_y = PALETTE_Y + i * (BOX_SIZE + 10)

            # Draw color box
            cv2.rectangle(
                frame,
                (PALETTE_X, box_y),
                (PALETTE_X + BOX_SIZE, box_y + BOX_SIZE),
                color,
                -1
            )

            # Highlight selected color
            if color == current_color:
                cv2.rectangle(
                    frame,
                    (PALETTE_X, box_y),
                    (PALETTE_X + BOX_SIZE, box_y + BOX_SIZE),
                    (255, 255, 255),
                    2
                )

            # -------- HOLD PROGRESS INDICATOR --------
            if state == "SELECT" and landmarks:
                x, y = landmarks[8]
                if PALETTE_X < x < PALETTE_X + BOX_SIZE and box_y < y < box_y + BOX_SIZE:
                    progress = int(
                        (color_hold_counter / COLOR_SELECT_THRESHOLD) * BOX_SIZE
                    )
                    cv2.rectangle(
                        frame,
                        (PALETTE_X, box_y + BOX_SIZE - progress),
                        (PALETTE_X + BOX_SIZE, box_y + BOX_SIZE),
                        (255, 255, 255),
                        -1
                    )

       # ==========================================================
       # DRAW SHAPE TOOLBAR + HOLD FEEDBACK
       # ==========================================================
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

                # Shape icon
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
                    cv2.putText(frame, "F", (cx-7, cy+7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Selected shape highlight
                if shape == current_shape:
                    cv2.rectangle(
                        frame,
                        (SHAPE_BAR_X, box_y),
                        (SHAPE_BAR_X + SHAPE_BOX, box_y + SHAPE_BOX),
                        (0, 255, 255),
                        2
                    )

                # -------- SHAPE HOLD PROGRESS --------
                if state == "SELECT" and landmarks:
                    x, y = landmarks[8]

                    if SHAPE_BAR_X < x < SHAPE_BAR_X + SHAPE_BOX and box_y < y < box_y + SHAPE_BOX:
                        progress = int(
                            (shape_hold_counter / SHAPE_SELECT_THRESHOLD) * SHAPE_BOX
                        )

                        cv2.rectangle(
                            frame,
                            (SHAPE_BAR_X, box_y + SHAPE_BOX - progress),
                            (SHAPE_BAR_X + SHAPE_BOX, box_y + SHAPE_BOX),
                            (0, 255, 255),
                            -1
                        )



        # ==========================================================
        # STEP 8: UI OVERLAYS
        # ==========================================================
        if gesture:
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # ==========================================================
        # STEP 9: FINAL RENDER
        # ==========================================================
        output = cv2.addWeighted(frame, 0.7, canvas.canvas, 0.3, 0)
        # ================= WHITEBOARD WINDOW =================
        cv2.imshow("Whiteboard", canvas.canvas)


        prev_state = state
        cv2.imshow("AirCanvas X", output)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
