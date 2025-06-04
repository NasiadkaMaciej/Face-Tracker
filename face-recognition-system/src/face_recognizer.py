import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
import insightface
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os

class FaceRecognizer:
    def __init__(self, model_path=None, database_path=None, recognition_method='knn'):
        """Initialize face recognizer with model and database paths."""
        self.model_path = model_path
        self.database_path = Path(database_path) if database_path else None
        self.unknown_threshold = 0.6
        self.known_face_embeddings = []
        self.known_face_names = []
        self.logger = logging.getLogger(__name__)
        self.recognition_method = recognition_method
        self.model = None
        self.scaler = None
        
        # Initialize InsightFace for face embedding extraction
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # For methods that need a classifier
        self.classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(max_depth=5),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
            'svm': SVC(kernel='linear', probability=True)
        }
    
    def load_model(self):
        """Load the face recognition database and model if using ML methods."""
        try:
            # Check if database exists
            if self.database_path and self.database_path.exists():
                self.logger.info(f"Loading face database from {self.database_path}")
                
                # Load embeddings and names for all methods
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_embeddings = data["embeddings"]
                    self.known_face_names = data["names"]
                
                # For ML-based methods, load the trained model and scaler
                if self.recognition_method in self.classifiers:
                    model_path = self.database_path.parent / f"{self.database_path.stem}_{self.recognition_method}_model.pkl"
                    scaler_path = self.database_path.parent / f"{self.database_path.stem}_scaler.pkl"
                    
                    if model_path.exists() and scaler_path.exists():
                        self.model = joblib.load(model_path)
                        self.scaler = joblib.load(scaler_path)
                        self.logger.info(f"Loaded {self.recognition_method} model from {model_path}")
                    else:
                        self.logger.warning(f"Model or scaler files not found for {self.recognition_method}")
                        return False
                
                self.logger.info(f"Loaded {len(self.known_face_embeddings)} face embeddings")
                return True
            else:
                self.logger.warning("Face database not found or not specified")
                return False
        except Exception as e:
            self.logger.error(f"Error loading face recognition database: {str(e)}")
            return False

    def recognize_faces(self, frame, face_locations):
        """Identify people in detected faces."""
        recognized_faces = []
        
        # Use InsightFace to detect and get face embeddings
        faces = self.face_app.get(frame)
        
        # If face_locations were passed, we need to match them with InsightFace detections
        if face_locations and len(faces) > 0:
            for (x, y, w, h) in face_locations:
                best_match = None
                best_iou = 0
                
                for face in faces:
                    # Get bounding box from InsightFace (x1, y1, x2, y2)
                    bbox = face.bbox.astype(int)
                    fx1, fy1, fx2, fy2 = bbox
                    
                    # Calculate IoU between this face and the passed location
                    # Convert x,y,w,h to x1,y1,x2,y2
                    x1, y1, x2, y2 = x, y, x+w, y+h
                    
                    # Calculate intersection
                    intersection_x1 = max(x1, fx1)
                    intersection_y1 = max(y1, fy1)
                    intersection_x2 = min(x2, fx2)
                    intersection_y2 = min(y2, fy2)
                    
                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        box1_area = (x2 - x1) * (y2 - y1)
                        box2_area = (fx2 - fx1) * (fy2 - fy1)
                        union_area = box1_area + box2_area - intersection_area
                        iou = intersection_area / union_area
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_match = face
                
                if best_match and best_iou > 0.5:
                    # Use the matched face for recognition
                    name = self.recognize_face(best_match.embedding)
                    # Include the entire face object to access landmarks later
                    recognized_faces.append((x, y, w, h, name, best_match))
                else:
                    recognized_faces.append((x, y, w, h, "Unknown", None))
        else:
            # Use InsightFace detections directly
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2-x1, y2-y1
                
                name = self.recognize_face(face.embedding)
                # Include the face object
                recognized_faces.append((x, y, w, h, name, face))
                
        return recognized_faces

    def recognize_face(self, face_embedding):
        """Recognize a face embedding and return the name or 'Unknown'."""
        if len(self.known_face_embeddings) == 0:
            self.logger.warning("No known faces loaded, cannot recognize")
            return "Unknown"
        
        try:
            # ML-based methods (knn, naive_bayes, decision_tree, mlp, svm)
            if self.recognition_method in self.classifiers and self.model is not None:
                # Scale the embedding
                scaled_embedding = self.scaler.transform([face_embedding])
                
                # Get prediction probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(scaled_embedding)[0]
                    max_proba = np.max(proba)
                    
                    # Only trust prediction if probability is high enough
                    if max_proba > self.unknown_threshold:
                        prediction = self.model.predict(scaled_embedding)[0]
                        return prediction
                    return "Unknown"
                else:
                    # For models without probability estimation
                    prediction = self.model.predict(scaled_embedding)[0]
                    return prediction
            
            return "Unknown"
            
        except Exception as e:
            self.logger.error(f"Error recognizing face: {str(e)}")
            return "Error"
    
    def draw_facial_landmarks(self, frame, face_obj):
        """Draw facial landmarks (eyes, nose, mouth) on the frame."""
        if face_obj is None or not hasattr(face_obj, 'kps'):
            return
        
        try:
            # Get the landmarks
            landmarks = face_obj.kps
            
            if landmarks is not None and landmarks.shape[0] >= 5:
                # Define landmark colors
                colors = [
                    (255, 0, 0),    # Right eye (blue)
                    (0, 255, 0),    # Left eye (green)
                    (0, 255, 255),  # Nose (yellow)
                    (255, 0, 255),  # Right mouth (magenta)
                    (255, 0, 255)   # Left mouth (magenta)
                ]
                
                # Draw landmarks
                for i, (x, y) in enumerate(landmarks):
                    if i < len(colors):
                        cv2.circle(frame, (int(x), int(y)), 3, colors[i], -1)
                
                # Connect mouth corners with a line
                if landmarks.shape[0] >= 5:
                    # Draw mouth line
                    cv2.line(frame, 
                            (int(landmarks[3][0]), int(landmarks[3][1])), 
                            (int(landmarks[4][0]), int(landmarks[4][1])), 
                            (255, 0, 255), 2)
        except Exception as e:
            self.logger.error(f"Error drawing facial landmarks: {str(e)}")
    
    def mark_faces(self, frame, face_locations):
        """Mark known and unknown faces on the provided frame."""
        marked_frame = frame.copy()
        
        for face_data in face_locations:
            try:
                # Check if we have face recognition data with landmarks
                if len(face_data) == 6:
                    # If we have recognition data with face object (x, y, w, h, name, face_obj)
                    x, y, w, h, name, face_obj = face_data
                    
                    # Draw rectangle with color based on recognition status
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(marked_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw name label
                    cv2.rectangle(marked_frame, (x, y+h), (x+w, y+h+30), color, cv2.FILLED)
                    cv2.putText(marked_frame, name, (x+6, y+h+25), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Draw facial landmarks if available
                    self.draw_facial_landmarks(marked_frame, face_obj)
                    
                # For backward compatibility with old format (x, y, w, h, name)
                elif len(face_data) == 5:
                    x, y, w, h, name = face_data
                    
                    # Draw rectangle with color based on recognition status
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(marked_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw name label
                    cv2.rectangle(marked_frame, (x, y+h), (x+w, y+h+30), color, cv2.FILLED)
                    cv2.putText(marked_frame, name, (x+6, y+h+25), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            except Exception as e:
                self.logger.error(f"Error marking face: {str(e)}")
        
        return marked_frame
        
    def save_database(self, database_path=None):
        """Save face embeddings database."""
        save_path = Path(database_path) if database_path else self.database_path
        
        if not save_path:
            self.logger.error("No database path specified")
            return False
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "embeddings": self.known_face_embeddings,
                "names": self.known_face_names
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info(f"Face database saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving face database: {str(e)}")
            return False
    
    def add_face(self, name, face_embedding):
        """Add a face embedding to the database."""
        try:
            self.known_face_embeddings.append(face_embedding)
            self.known_face_names.append(name)
            return True
        except Exception as e:
            self.logger.error(f"Error adding face: {str(e)}")
            return False