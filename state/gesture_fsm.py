class GestureFSM:
    def __init__(self):
        self.state = "IDLE"
        self.last_gesture = None
        self.count = 0
        self.threshold = 5

    def update(self, gesture):
        if gesture == self.last_gesture and gesture is not None:
            self.count += 1
        else:
            self.count = 0

        if self.count >= self.threshold:
            self.state = gesture

        self.last_gesture = gesture
        
        return self.state

