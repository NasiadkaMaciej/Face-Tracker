import cv2
import logging
import numpy as np
from insightface.app import FaceAnalysis
import dlib

class FaceDetector:
    """Class for detecting faces in images using multiple detection methods."""
    
    def __init__(self, method='insightface'):
        """Initialize face detector with specified method."""
        self.method = method
        self.min_confidence = 0.5
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors based on selected method
        if method == 'insightface':
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace detector initialized")
        elif method == 'haar':
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            self.logger.info("Haar Cascade detector initialized")
        elif method == 'hog':
            self.hog_detector = dlib.get_frontal_face_detector()
            self.logger.info("HOG-based detector initialized")
        elif method == 'mediapipe':
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
            self.logger.info("MediaPipe detector initialized")
        else:
            self.logger.error(f"Unknown detection method: {method}, falling back to InsightFace")
            self.method = 'insightface'
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect_faces(self, image):
        """Detect faces in the input image and return list of face locations as (x, y, w, h) tuples."""
        if image is None:
            self.logger.error("Cannot detect faces in None image")
            return []
            
        faces = []
        
        try:
            if self.method == 'insightface':
                detections = self.face_app.get(image)
                for face in detections:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    faces.append((x1, y1, x2-x1, y2-y1))
            
            elif self.method == 'haar':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detections = self.haar_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for (x, y, w, h) in detections:
                    faces.append((x, y, w, h))
            
            elif self.method == 'hog':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detections = self.hog_detector(gray, 1)
                for detection in detections:
                    x = detection.left()
                    y = detection.top()
                    w = detection.right() - detection.left()
                    h = detection.bottom() - detection.top()
                    faces.append((x, y, w, h))
            
            elif self.method == 'mediapipe':
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_image)
                h, w, _ = image.shape
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append((x, y, width, height))
            if len(faces) != 0:
                self.logger.info(f"Detected {len(faces)} faces using {self.method} method")
            return faces
            
        except Exception as e:
            self.logger.error(f"Error during face detection with {self.method}: {str(e)}")
            return []