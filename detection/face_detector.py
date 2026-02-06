import cv2

class FaceDetector:
    """Simple face detector using OpenCV Haar cascade exclusively.

    This avoids depending on Mediapipe and provides a reliable fallback that
    works out-of-the-box with OpenCV installed via `opencv-python`.
    """
    def __init__(self, confidence=0.6):
        # Maintain compatibility with previous constructor signatures that
        # accepted a `confidence` argument; it's unused for Haar cascades.
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError('OpenCV Haar cascade could not be loaded')

    def detect(self, frame):
        """Detect faces and return list of (x, y, w, h) rectangles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
