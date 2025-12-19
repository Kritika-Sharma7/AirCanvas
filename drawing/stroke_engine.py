import cv2
from utils.filters import MovingAverage


class StrokeEngine:
    def __init__(self):
        self.prev = None
        self.smoother = MovingAverage()
        self.current_stroke = []
        self.thickness = 4   # Default stroke thickness

    def draw(self, canvas, point, color):
        # Smooth the point
        point = self.smoother.smooth(point)

        # Draw line if previous point exists
        if self.prev is not None:
            cv2.line(canvas, self.prev, point, color, self.thickness)

        self.prev = point
        self.current_stroke.append(point)

    def reset(self):
        self.prev = None
        self.current_stroke.clear()
        self.smoother.reset()
