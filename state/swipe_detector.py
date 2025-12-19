from collections import deque
import numpy as np
import time

class SwipeDetector:
    def __init__(self, window=8, threshold=80):
        self.points = deque(maxlen=window)
        self.last_trigger = 0
        self.cooldown = 1.0  # seconds
        self.threshold = threshold

    def update(self, point):
        self.points.append(point)

    def detect(self):
        if len(self.points) < 5:
            return None

        dx = self.points[-1][0] - self.points[0][0]
        now = time.time()

        if abs(dx) > self.threshold and (now - self.last_trigger) > self.cooldown:
            self.last_trigger = now
            return "RIGHT" if dx > 0 else "LEFT"

        return None

    def reset(self):
        self.points.clear()
