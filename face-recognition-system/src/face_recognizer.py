import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
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
        self.unknown_threshold = 0.9
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
        
        if face_locations:
            for (x, y, w, h, face_obj) in face_locations:
                try:
                    # Use the embedding from the face object directly
                    name, prob = self.recognize_face(face_obj.embedding)
                    
                    # Include the entire face object to access landmarks later
                    recognized_faces.append((x, y, w, h, name, face_obj, prob))
                except Exception as e:
                    self.logger.error(f"Error recognizing face: {str(e)}")
                    recognized_faces.append((x, y, w, h, "Unknown", face_obj, 0.0))
        
        return recognized_faces

    def recognize_face(self, face_embedding):
        """Recognize a face embedding and return the name and probability."""
        if not self.known_face_embeddings:
            return "Unknown", 0.0
        
        try:
            # ML-based methods
            if self.recognition_method in self.classifiers and self.model is not None:
                # Scale the embedding
                scaled_embedding = self.scaler.transform([face_embedding])
                
                # Get prediction probabilities
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(scaled_embedding)[0]
                    max_proba = np.max(proba)
                    
                    if max_proba > self.unknown_threshold:
                        prediction = self.model.predict(scaled_embedding)[0]
                        return prediction, max_proba * 100
                return "Unknown", 0.0
            
            return "Unknown", 0.0

        except Exception as e:
            self.logger.error(f"Error recognizing face: {str(e)}")
            return "Error", 0.0
    
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
                # If we have recognition data with face object and probability (x, y, w, h, name, face_obj, prob)
                x, y, w, h, name, face_obj, prob = face_data
                
                # Format the name with probability
                if name != "Unknown":
                    label = f"{name} ({prob:.1f}%)"
                else:
                    label = name
                
                # Draw rectangle with color based on recognition status
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(marked_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name label with probability
                cv2.rectangle(marked_frame, (x, y+h), (x+w, y+h+30), color, cv2.FILLED)
                cv2.putText(marked_frame, label, (x+6, y+h+25), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                self.draw_facial_landmarks(marked_frame, face_obj)
                    
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