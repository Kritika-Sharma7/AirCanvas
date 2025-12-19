class GestureClassifier:
    def classify(self, lm):
        """
        Classifies hand gesture into high-level intent.
        """
        if not lm:
            return None

        index_up = lm[8][1] < lm[6][1]
        middle_up = lm[12][1] < lm[10][1]
        ring_up = lm[16][1] < lm[14][1]
        pinky_up = lm[20][1] < lm[18][1]

        # ✏️ Draw: only index finger
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "DRAW"

        # 🎯 Select: index + middle
        if index_up and middle_up and not ring_up:
            return "SELECT"

        # ⏸ Pause: fist
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "PAUSE"

        # ⚙️ Command: open hand (without thumb)
        if index_up and middle_up and ring_up and pinky_up:
            return "COMMAND"

        return None

    def count_fingers(self, lm):
        """
        Counts number of raised fingers (excluding thumb).
        Used for swipe undo / redo.
        """
        if not lm:
            return 0

        fingers = [
            lm[8][1] < lm[6][1],    # Index
            lm[12][1] < lm[10][1],  # Middle
            lm[16][1] < lm[14][1],  # Ring
            lm[20][1] < lm[18][1],  # Pinky
        ]

        return sum(fingers)

    def is_palm_open(self, lm):
        """
        Detects fully open palm.
        Used for full-hand eraser.
        """
        if not lm:
            return False

        fingers = [
            lm[8][1] < lm[6][1],    # Index
            lm[12][1] < lm[10][1],  # Middle
            lm[16][1] < lm[14][1],  # Ring
            lm[20][1] < lm[18][1],  # Pinky
        ]

        return all(fingers)
