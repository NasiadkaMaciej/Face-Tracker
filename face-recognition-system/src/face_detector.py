import cv2
import logging
import numpy as np
from insightface.app import FaceAnalysis
import dlib  # Add dlib for HOG based detection

class FaceDetector:
    """Class for detecting faces in images using multiple detection methods."""
    
    def __init__(self, method='insightface'):
        """Initialize face detector with specified method."""
        self.method = method
        self.min_confidence = 0.5
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors based on selected method
        if method == 'insightface':
            # Initialize InsightFace
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace detector initialized")
        elif method == 'haar':
            # Haar cascade detector from OpenCV
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            self.logger.info("Haar Cascade face detector initialized")
        elif method == 'hog':
            # HOG-based detector from dlib
            self.hog_detector = dlib.get_frontal_face_detector()
            self.logger.info("HOG-based face detector initialized")
        elif method == 'mediapipe':
            # MediaPipe face detector
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
            self.logger.info("MediaPipe face detector initialized")
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
                # Use InsightFace for detection
                insightface_detections = self.face_app.get(image)
                
                # Convert InsightFace detections to (x, y, w, h) format
                for face in insightface_detections:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to (x, y, w, h) format
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    
                    faces.append((x, y, w, h))
                    
            elif self.method == 'haar':
                # Haar cascade detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                haar_detections = self.haar_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Add detected faces
                for (x, y, w, h) in haar_detections:
                    faces.append((x, y, w, h))
                    
            elif self.method == 'hog':
                # HOG-based detection with dlib
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dlib_detections = self.hog_detector(gray, 1)
                
                for detection in dlib_detections:
                    # Convert dlib rectangle to (x,y,w,h) format
                    x = detection.left()
                    y = detection.top()
                    w = detection.right() - detection.left()
                    h = detection.bottom() - detection.top()
                    faces.append((x, y, w, h))
                    
            elif self.method == 'mediapipe':
                # MediaPipe face detection
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    h, w, _ = image.shape
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append((x, y, width, height))
            
            self.logger.info(f"Detected {len(faces)} faces using {self.method} method")
            return faces
            
        except Exception as e:
            self.logger.error(f"Error during face detection with {self.method}: {str(e)}")
            return []