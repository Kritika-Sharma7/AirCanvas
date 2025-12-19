class GestureClassifier:
    def classify(self, lm):
        if not lm:
            return None

        index_up = lm[8][1] < lm[6][1]
        middle_up = lm[12][1] < lm[10][1]
        ring_up = lm[16][1] < lm[14][1]
        pinky_up = lm[20][1] < lm[18][1]

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "DRAW"

        if index_up and middle_up and not ring_up:
            return "SELECT"

        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "PAUSE"

        if index_up and middle_up and ring_up and pinky_up:
            return "COMMAND"

        return None
