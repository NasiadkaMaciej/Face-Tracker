import cv2
import numpy as np
import pickle
import face_recognition
import logging
from pathlib import Path

class FaceRecognizer:
    def __init__(self, model_path=None, database_path=None, unknown_threshold=0.6):
        """Initialize face recognizer with model and database paths."""
        self.model_path = model_path
        self.database_path = Path(database_path) if database_path else None
        self.unknown_threshold = unknown_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the face recognition model and database."""
        try:
            # For face_recognition library, we don't need to load a specific model
            # But we need to load the database of known faces
            if self.database_path and self.database_path.exists():
                self.logger.info(f"Loading face database from {self.database_path}")
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                self.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
                return True
            else:
                self.logger.warning("Face database not found or not specified")
                return False
        except Exception as e:
            self.logger.error(f"Error loading face recognition model: {str(e)}")
            return False

    def recognize_faces(self, frame, face_locations):
        """Identify people in detected faces."""
        recognized_faces = []
        
        for (x, y, w, h) in face_locations:
            try:
                # Get face region in RGB format
                face_image = frame[y:y+h, x:x+w]
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Extract face features
                encodings = face_recognition.face_encodings(rgb_face)
                
                if not encodings:
                    self.logger.warning("Could not encode face")
                    recognized_faces.append((x, y, w, h, "Unknown"))
                else:
                    # Match face to database
                    face_encoding = encodings[0]
                    name = self.recognize_face(face_encoding)
                    recognized_faces.append((x, y, w, h, name))
                    
            except Exception as e:
                self.logger.error(f"Error recognizing face: {str(e)}")
                recognized_faces.append((x, y, w, h, "Error"))
            
        return recognized_faces

    def recognize_face(self, face_encoding):
        """Recognize a face encoding and return the name or 'Unknown'."""
        if len(self.known_face_encodings) == 0:
            self.logger.warning("No known faces loaded, cannot recognize")
            return "Unknown"
        
        try:
            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                # If the closest match is below threshold, return the name
                if face_distances[best_match_index] < self.unknown_threshold:
                    return self.known_face_names[best_match_index]
            
            return "Unknown"
            
        except Exception as e:
            self.logger.error(f"Error recognizing face: {str(e)}")
            return "Error"
    
    def mark_faces(self, frame, face_locations):
        """Mark known and unknown faces on the provided frame."""
        # Make a copy of the frame
        marked_frame = frame.copy()
        
        # Process each detected face
        for face_data in face_locations:
            try:
                if len(face_data) == 5:
                    # If we have recognition data (x, y, w, h, name)
                    x, y, w, h, name = face_data
                else:
                    # If we only have coordinates (x, y, w, h)
                    x, y, w, h = face_data
                    
                    # Extract face region and convert to RGB
                    face_image = frame[y:y+h, x:x+w]
                    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    
                    # Get face encoding
                    encodings = face_recognition.face_encodings(rgb_face)
                    
                    if not encodings:
                        name = "Unknown"
                    else:
                        # Get the first encoding and recognize face
                        face_encoding = encodings[0]
                        name = self.recognize_face(face_encoding)
                
                # Draw rectangle with color based on recognition status
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                cv2.rectangle(marked_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name label
                cv2.rectangle(marked_frame, (x, y+h), (x+w, y+h+30), color, cv2.FILLED)
                cv2.putText(marked_frame, name, (x+6, y+h+25), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
            except Exception as e:
                self.logger.error(f"Error marking face: {str(e)}")
        
        return marked_frame
        
    def save_database(self, database_path=None):
        """Save face encodings database."""
        # Use provided path, fallback to instance path if not provided
        save_path = Path(database_path) if database_path else self.database_path
        
        if not save_path:
            self.logger.error("No database path specified")
            return False
        
        try:
            # Create parent directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the database
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info(f"Face database saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving face database: {str(e)}")
            return False
    
    def add_face(self, name, face_encoding):
        """Add a face encoding to the database."""
        try:
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.logger.info(f"Added face for {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding face: {str(e)}")
            return False