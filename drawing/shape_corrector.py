import cv2
import numpy as np
import math


class ShapeCorrector:
    def detect_shape(self, points):
        """
        Detects geometric shapes from freehand stroke points.
        Returns (shape_name, data) or None
        """

        # Need enough points for reliable detection
        if len(points) < 15:
            return None

        cnt = np.array(points, dtype=np.int32)

        # Ensure closed contour for area-based shapes
        if not np.array_equal(cnt[0], cnt[-1]):
            cnt = np.vstack([cnt, cnt[0]])

        # Reject tiny jitter strokes
        area = cv2.contourArea(cnt)
        if area < 800:
            return None

        # Perimeter and polygon approximation
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # ---------------- LINE ----------------
        if len(approx) == 2 or self._is_line(cnt):
            # Use first and last points for clean line
            line_pts = np.array([cnt[0], cnt[-1]])
            return "LINE", line_pts

        # ---------------- TRIANGLE ----------------
        if len(approx) == 3:
            return "TRIANGLE", approx

        # ---------------- RECTANGLE ----------------
        if len(approx) == 4 and self._is_rectangle(approx):
            return "RECTANGLE", approx

        # ---------------- CIRCLE ----------------
        if self._is_circle(cnt, area):
            return "CIRCLE", cnt

        return None

    # =====================================================
    # Helpers
    # =====================================================

    def _is_line(self, cnt):
        """
        Checks if points lie approximately on a straight line
        """
        vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = []

        for x, y in cnt:
            d = abs(vy * x - vx * y + x0 * y0 - y0 * x0)
            distances.append(d)

        return np.mean(distances) < 10

    def _is_rectangle(self, approx):
        """
        Checks if a 4-point polygon has ~90 degree angles
        """
        def angle(p1, p2, p3):
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p2 - p3)
            c = np.linalg.norm(p1 - p3)
            cos = (a * a + b * b - c * c) / (2 * a * b + 1e-6)
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))

        angles = []
        for i in range(4):
            angles.append(angle(
                approx[i - 1][0],
                approx[i][0],
                approx[(i + 1) % 4][0]
            ))

        return all(80 < a < 100 for a in angles)

    def _is_circle(self, cnt, area):
        """
        Compares contour area with enclosing circle area
        """
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi * r * r
        return abs(circle_area - area) / area < 0.3
