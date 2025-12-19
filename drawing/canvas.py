import numpy as np

class Canvas:
    def __init__(self, shape):
        self.canvas = np.ones(shape, dtype=np.uint8) * 255
        self.history = []

    def save(self):
        self.history.append(self.canvas.copy())

    def undo(self):
        if self.history:
            self.canvas = self.history.pop()

    def clear(self):
        self.canvas[:] = 255
