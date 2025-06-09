import cv2
import numpy as np
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
import random
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# Include the classifiers we want to use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class FaceDataset:
    def __init__(self, dataset_path):
        """Initialize face dataset processor with the dataset directory path."""
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize InsightFace
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    def load_data(self, augment=True):
        """Load images, extract face embeddings, and get person names from the dataset.
        Returns a dictionary with face embeddings and corresponding names.
        If augment is True, apply data augmentation to increase dataset size.
        Unknown faces from the Unknown directory are automatically included.
        """
        self.logger.info(f"Loading data from {self.dataset_path}")
        
        # Dictionary to store results
        data = {
            "embeddings": [],
            "names": []
        }
        
        # List all person directories
        person_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(person_dirs)} persons in dataset")
        
        # Check for Unknown directory
        unknown_dir = self.dataset_path / "Unknown"
        has_unknown = unknown_dir.exists() and unknown_dir.is_dir()
        if has_unknown:
            self.logger.info("Found Unknown directory - will include in training")
        
        if not person_dirs:
            self.logger.warning(f"No person directories found in {self.dataset_path}")
            return data
        
        # Counters for detailed logging
        total_images_processed = 0
        total_faces_found = 0
        log_interval = 10
        
        # Process each person directory
        for person_dir in person_dirs:
            person_name = person_dir.name
            self.logger.info(f"Processing person: {person_name}")
            
            # Get all image files for this person
            image_files = [f for f in person_dir.glob("*") 
                         if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            
            if not image_files:
                self.logger.warning(f"No images found for person: {person_name}")
                continue
                
            self.logger.info(f"Found {len(image_files)} images for {person_name}")
            person_images_processed = 0
            person_faces_found = 0
            
            # Process each image
            for image_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        self.logger.warning(f"Could not load image: {image_file}")
                        continue
                    
                    # Store embeddings count before processing
                    prev_embeddings_count = len(data["embeddings"])
                    
                    # Process original image
                    self._process_image(image, person_name, data)
                    
                    # Count faces found in this image
                    faces_found = len(data["embeddings"]) - prev_embeddings_count
                    person_faces_found += faces_found
                    
                    # Optionally augment data
                    if augment:
                        augmented_images = self._augment_image(image)
                        aug_prev_count = len(data["embeddings"])
                        
                        for aug_img in augmented_images:
                            self._process_image(aug_img, person_name, data)
                        
                        # Count augmented faces
                        aug_faces_found = len(data["embeddings"]) - aug_prev_count
                        person_faces_found += aug_faces_found
                    
                    # Update counters
                    person_images_processed += 1
                    total_images_processed += 1
                    
                    # Log progress at intervals
                    if person_images_processed % log_interval == 0:
                        self.logger.info(f"Progress for {person_name}: {person_images_processed}/{len(image_files)} images processed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
            
            # Log completion for this person
            self.logger.info(f"Completed {person_name}: {person_images_processed} images processed, {person_faces_found} faces found")
            total_faces_found += person_faces_found
        
        self.logger.info(f"Dataset loaded: {len(data['embeddings'])} face embeddings from {len(set(data['names']))} persons")
        self.logger.info(f"Original images {total_images_processed} processed into {total_faces_found} faces")
        return data
    
    def _process_image(self, image, person_name, data):
        """Process a single image and add embeddings to the data dictionary."""
        # Detect faces using InsightFace
        faces = self.face_app.get(image)
        
        if not faces:
            return
            
        # Only use the first face (most prominent)
        # InsightFace already returns the embedding
        face = faces[0]
        embedding = face.embedding
        
        # Add embedding and name to results
        data["embeddings"].append(embedding)
        data["names"].append(person_name)
    
    def _augment_image(self, image):
        """Apply data augmentation techniques to an image."""
        augmented = []
        
        # Rotation (small angles)
        for angle in [-5, 5]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        # Brightness adjustment
        for alpha in [0.8, 1.2]:  # Darker and brighter
            brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append(brightened)
            
        # Add Gaussian blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        augmented.append(blurred)
        
        return augmented
        
    def train_models(self, database_path, methods=['knn', 'naive_bayes', 'decision_tree', 'mlp', 'svm']):
        """Train multiple recognition models on the dataset."""
        # Load the data
        data = self.load_data(augment=True)  # Use augmentation for better training
        
        if not data["embeddings"]:
            self.logger.error("No face embeddings found in dataset")
            return False
            
        # Get the base database path without extension
        base_path = Path(database_path).with_suffix('')
        
        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(data["embeddings"])
        y = data["names"]
        
        # Save the scaler
        scaler_path = f"{base_path}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"Saved feature scaler to {scaler_path}")
        
        # Dictionary of classifiers
        classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(max_depth=5),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
            'svm': SVC(kernel='linear', probability=True)
        }
        
        # Train and save each model
        for method in methods:
            if method in classifiers:
                try:
                    self.logger.info(f"Training {method} model...")
                    model = classifiers[method]
                    model.fit(X, y)
                    
                    # Save the model
                    model_path = f"{base_path}_{method}_model.pkl"
                    joblib.dump(model, model_path)
                    self.logger.info(f"Saved {method} model to {model_path}")
                except Exception as e:
                    self.logger.error(f"Error training {method} model: {str(e)}")
        
        # Also save the embeddings and names
        with open(database_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved face embeddings to {database_path}")
        
        return True