import cv2
import numpy as np

class ShapeCorrector:
    def detect_shape(self, points):
        if len(points) < 10:
            return None

        cnt = np.array(points, dtype=np.int32)

        peri = cv2.arcLength(cnt, True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # ---- SHAPE DETECTION ----
        if len(approx) == 2:
            return "LINE", approx

        elif len(approx) == 3:
            return "TRIANGLE", approx

        elif len(approx) == 4:
            return "RECTANGLE", approx

        elif len(approx) > 5:
            return "CIRCLE", cnt

        return None
