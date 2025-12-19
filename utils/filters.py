from collections import deque
import numpy as np

<<<<<<< HEAD

=======
>>>>>>> 5db4b12 (RS Initial AirCanvas implementation with save feature)
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
