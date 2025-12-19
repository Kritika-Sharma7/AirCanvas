import numpy as np


class Canvas:
    def __init__(self, shape):
        # White background canvas
        self.canvas = np.ones(shape, dtype=np.uint8) * 255

        # Undo / Redo stacks
        self.history = []
        self.redo_stack = []

    # ==========================
    # Save state (before action)
    # ==========================
    def save(self):
        self.history.append(self.canvas.copy())
        self.redo_stack.clear()   # New action clears redo

    # ==========================
    # Undo last action
    # ==========================
    def undo(self):
        if self.history:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.history.pop()

    # ==========================
    # Redo undone action
    # ==========================
    def redo(self):
        if self.redo_stack:
            self.history.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()

    # ==========================
    # Clear entire canvas
    # ==========================
    def clear(self):
        self.canvas[:] = 255
        self.history.clear()
        self.redo_stack.clear()
