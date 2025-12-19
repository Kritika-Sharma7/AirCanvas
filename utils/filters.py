from collections import deque
import numpy as np


class MovingAverage:
    def __init__(self, size=5):
        self.size = size
        self.buffer = deque(maxlen=size)

    def smooth(self, point):
        self.buffer.append(point)
        avg = np.mean(self.buffer, axis=0)
        return int(avg[0]), int(avg[1])

    def reset(self):
        self.buffer.clear()
