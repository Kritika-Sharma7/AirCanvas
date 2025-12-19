import cv2
import math
from utils.filters import MovingAverage


class StrokeEngine:
    def __init__(self):
        self.prev = None
        self.smoother = MovingAverage()
        self.thickness = 4
        self.current_stroke = []

    def draw(self, canvas, point, color):
        point = self.smoother.smooth(point)

        if self.prev:
            cv2.line(canvas, self.prev, point, color, self.thickness)

        self.prev = point
        self.current_stroke.append(point)

    def reset(self):
        self.prev = None
        self.smoother.reset()
