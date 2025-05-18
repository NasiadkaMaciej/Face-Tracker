import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
import insightface
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, model_path=None, database_path=None, unknown_threshold=0.6):
        """Initialize face recognizer with model and database paths."""
        self.model_path = model_path
        self.database_path = Path(database_path) if database_path else None
        self.unknown_threshold = unknown_threshold
        self.known_face_embeddings = []
        self.known_face_names = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize InsightFace
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    def load_model(self):
        """Load the face recognition database."""
        try:
            if self.database_path and self.database_path.exists():
                self.logger.info(f"Loading face database from {self.database_path}")
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_embeddings = data["embeddings"]
                    self.known_face_names = data["names"]
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
        
        # Use InsightFace to detect and recognize faces
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
            # Calculate cosine similarity with known face embeddings
            similarities = []
            for emb in self.known_face_embeddings:
                similarity = cosine_similarity([face_embedding], [emb])[0][0]
                similarities.append(similarity)
            
            # Find the best match
            if similarities:
                best_match_index = np.argmax(similarities)
                similarity_score = similarities[best_match_index]
                
                # If similarity is above threshold, return the name
                if similarity_score > self.unknown_threshold:
                    return self.known_face_names[best_match_index]
            
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
                # Draw landmarks
                # InsightFace provides 5 points: left eye, right eye, nose, left mouth, right mouth
                for i, (x, y) in enumerate(landmarks):
                    color = (0, 255, 255)  # Yellow color for landmarks
                    
                    # Draw different colors based on feature type
                    if i == 0:  # Right eye (from the person's perspective)
                        color = (255, 0, 0)  # Blue
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    elif i == 1:  # Left eye
                        color = (0, 255, 0)  # Green
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    elif i == 2:  # Nose
                        color = (0, 255, 255)  # Yellow
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    elif i == 3:  # Right mouth corner
                        color = (255, 0, 255)  # Magenta
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    elif i == 4:  # Left mouth corner
                        color = (255, 0, 255)  # Magenta
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                
                # Connect mouth corners with a line
                if landmarks.shape[0] >= 5:
                    cv2.line(frame, 
                            (int(landmarks[3][0]), int(landmarks[3][1])), 
                            (int(landmarks[4][0]), int(landmarks[4][1])), 
                            (255, 0, 255), 2)
                    
                    # Add "Mouth" label at the center of mouth line
                    mouth_center_x = (landmarks[3][0] + landmarks[4][0]) // 2
                    mouth_center_y = (landmarks[3][1] + landmarks[4][1]) // 2
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
            self.logger.info(f"Added face for {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding face: {str(e)}")
            return False