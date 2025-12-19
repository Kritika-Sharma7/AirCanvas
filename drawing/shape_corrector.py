import cv2
import numpy as np
<<<<<<< HEAD

class ShapeCorrector:
    def detect_shape(self, points):
        if len(points) < 10:
=======
import math

class ShapeCorrector:
    def detect_shape(self, points):
        if len(points) < 20:
>>>>>>> 5db4b12 (RS Initial AirCanvas implementation with save feature)
            return None

        cnt = np.array(points, dtype=np.int32)

<<<<<<< HEAD
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
=======
        # Ensure closed contour
        if not np.array_equal(cnt[0], cnt[-1]):
            cnt = np.vstack([cnt, cnt[0]])

        area = cv2.contourArea(cnt)
        if area < 1000:
            return None

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.015 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # -------- LINE --------
        if len(approx) == 2 or self._is_line(cnt):
            return "LINE", approx

        # -------- TRIANGLE --------
        if len(approx) == 3:
            return "TRIANGLE", approx

        # -------- RECTANGLE --------
        if len(approx) == 4 and self._is_rectangle(approx):
            return "RECTANGLE", approx

        # -------- CIRCLE --------
        if self._is_circle(cnt, area):
            return "CIRCLE", cnt

        return None

    def _is_line(self, cnt):
        [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = []
        for p in cnt:
            x, y = p
            d = abs(vy*x - vx*y + x0*y0 - y0*x0)
            distances.append(d)
        return np.mean(distances) < 10

    def _is_rectangle(self, approx):
        def angle(p1, p2, p3):
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p2 - p3)
            c = np.linalg.norm(p1 - p3)
            cos = (a*a + b*b - c*c) / (2*a*b + 1e-6)
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))

        angles = []
        for i in range(4):
            angles.append(angle(
                approx[i-1][0],
                approx[i][0],
                approx[(i+1)%4][0]
            ))

        return all(80 < a < 100 for a in angles)

    def _is_circle(self, cnt, area):
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi * r * r
        return abs(circle_area - area) / area < 0.25



# #GEMINI CORRECTION 
# import cv2
# import numpy as np
# import math

# class ShapeCorrector:
#     def detect_shape(self, points):
#         # Increased threshold: air drawing needs more points for a valid shape
#         if len(points) < 30: 
#             return None

#         # Convert to numpy array
#         cnt = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

#         # 1. Check if it's a LINE first (Linear Regression approach)
#         # Lines in air drawing rarely form "contours" with area
#         if self._is_line(cnt):
#             # For a line, just return the start and end points
#             line_pts = np.array([points[0], points[-1]]).reshape(-1, 1, 2)
#             return "LINE", line_pts

#         # 2. Geometry calculations for closed shapes
#         # Use Convex Hull to smooth out the jittery "inner" movements of the hand
#         hull = cv2.convexHull(cnt)
#         area = cv2.contourArea(hull)
#         perimeter = cv2.arcLength(hull, True)
        
#         if area < 1000:
#             return None

#         # Circularity check (The most reliable way for air-drawn circles)
#         # Formula: 4 * pi * Area / (Perimeter^2) -> 1.0 is a perfect circle
#         circularity = (4 * math.pi * area) / (perimeter**2)
        
#         if circularity > 0.75: # Loose threshold for shaky hands
#             return "CIRCLE", hull

#         # 3. Polygon Approximation (Triangle vs Rectangle)
#         # We use a slightly larger epsilon for air drawing
#         epsilon = 0.04 * perimeter 
#         approx = cv2.approxPolyDP(hull, epsilon, True)

#         if len(approx) == 3:
#             return "TRIANGLE", approx
        
#         if len(approx) == 4:
#             return "RECTANGLE", approx

#         return None

#     def _is_line(self, cnt):
#         # Flatten points for fitLine
#         points = cnt.reshape(-1, 2)
#         # Calculate the distance between start and end vs total path length
#         dist_start_end = np.linalg.norm(points[0] - points[-1])
#         total_path = sum(np.linalg.norm(points[i] - points[i-1]) for i in range(1, len(points)))
        
#         # If the hand moved in a mostly straight path from A to B
#         if dist_start_end / total_path > 0.85:
#             return True
            
#         # Secondary check: distance from the fitted line
#         vx, vy, x, y = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
#         # If the average deviation is very small, it's a line
#         return False
>>>>>>> 5db4b12 (RS Initial AirCanvas implementation with save feature)
