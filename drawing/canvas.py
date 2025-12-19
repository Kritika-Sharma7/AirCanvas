import numpy as np

class Canvas:
    def __init__(self, shape):
        # White canvas background
        self.canvas = np.ones(shape, dtype=np.uint8) * 255

        # History for undo / redo
        self.history = []
        self.redo_stack = []

    def save(self):
        """
        Save current canvas state.
        Any new action invalidates redo history.
        """
        self.history.append(self.canvas.copy())
        self.redo_stack.clear()

    def undo(self):
        """
        Undo last action.
        """
        if self.history:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.history.pop()

    def redo(self):
        """
        Redo last undone action.
        """
        if self.redo_stack:
            self.history.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()

    def clear(self):
        """
        Clear entire canvas and reset history.
        """
        self.canvas[:] = 255
        self.history.clear()
        self.redo_stack.clear()
