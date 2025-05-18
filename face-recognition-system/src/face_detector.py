import cv2
import logging
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    """Class for detecting faces in images using InsightFace."""
    
    def __init__(self, method='insightface', min_confidence=0.5):
        """Initialize face detector with specified method."""
        self.method = method
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        
        # Initialize InsightFace
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.logger.info("InsightFace detector initialized")
        
        # Keep legacy methods for compatibility
        if method == 'haar':
            # Haar cascade detector from OpenCV
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            self.logger.info("Haar Cascade face detector initialized as fallback")
    
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
                # Fallback to Haar cascade detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                haar_detections = self.haar_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Add detected faces
                for (x, y, w, h) in haar_detections:
                    faces.append((x, y, w, h))
            
            self.logger.info(f"Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            self.logger.error(f"Error during face detection: {str(e)}")
            return []