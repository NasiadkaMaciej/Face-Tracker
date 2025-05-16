import cv2
import logging

class FaceDetector:
    """Class for detecting faces in images using various methods."""
    
    def __init__(self, method='hog', min_confidence=0.5):
        """Initialize face detector with specified method.
        
        'hog' or 'haar' detection method can be used with a confidence 
        threshold between 0.0-1.0.
        """
        self.method = method
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        
        # Initialize the selected detector
        if method == 'hog':
            # HOG-based detector from dlib
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.logger.info("HOG face detector initialized")
            
        elif method == 'haar':
            # Haar cascade detector from OpenCV
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.logger.info("Haar Cascade face detector initialized")
        else:
            self.logger.error(f"Unknown detection method: {method}")
            raise ValueError(f"Unknown detection method: {method}")
    
    def detect_faces(self, image):
        """Detect faces in the input image and return list of face locations as (x, y, w, h) tuples."""
        if image is None:
            self.logger.error("Cannot detect faces in None image")
            return []
            
        height, width = image.shape[:2]
        faces = []
        
        try:
            if self.method == 'hog':
                # HOG-based detection using dlib
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dlib_detections = self.detector(rgb_image, 1)
                
                # Convert dlib detections to OpenCV format
                for detection in dlib_detections:
                    x = detection.left()
                    y = detection.top()
                    w = detection.right() - x
                    h = detection.bottom() - y
                    faces.append((x, y, w, h))
                    
            elif self.method == 'haar':
                # Haar cascade detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                haar_detections = self.detector.detectMultiScale(
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