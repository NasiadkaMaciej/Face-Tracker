import cv2
import logging
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    """Class for detecting faces in images using InsightFace."""
    
    def __init__(self):
        """Initialize face detector using InsightFace."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize InsightFace detector
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.logger.info("InsightFace detector initialized")
    
    def detect_faces(self, image):
        """Detect faces in the input image and return the face objects and locations."""
        if image is None:
            self.logger.error("Cannot detect faces in None image")
            return []
            
        try:
            # Get face objects directly from InsightFace
            face_objects = self.face_app.get(image)
            
            # Also prepare standard format locations as (x, y, w, h)
            face_locations = []
            for face in face_objects:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                face_locations.append((x1, y1, x2-x1, y2-y1, face))
            
            if len(face_locations) > 0:
                self.logger.info(f"Detected {len(face_locations)} faces")
				
            return face_locations
            
        except Exception as e:
            self.logger.error(f"Error during face detection: {str(e)}")
            return []