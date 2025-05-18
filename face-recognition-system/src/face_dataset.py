import cv2
import numpy as np
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
from tqdm import tqdm

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
    
    def load_data(self):
        """Load images, extract face embeddings, and get person names from the dataset.
        Returns a dictionary with face embeddings and corresponding names.
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
        
        if not person_dirs:
            self.logger.warning(f"No person directories found in {self.dataset_path}")
            return data
        
        # Process each person directory
        for person_dir in tqdm(person_dirs, desc="Processing faces"):
            person_name = person_dir.name
            self.logger.info(f"Processing person: {person_name}")
            
            # Get all image files for this person
            image_files = [f for f in person_dir.glob("*") 
                         if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            
            if not image_files:
                self.logger.warning(f"No images found for person: {person_name}")
                continue
                
            self.logger.info(f"Found {len(image_files)} images for {person_name}")
            
            # Process each image
            for image_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        self.logger.warning(f"Could not load image: {image_file}")
                        continue
                        
                    # Detect faces using InsightFace
                    faces = self.face_app.get(image)
                    
                    if not faces:
                        self.logger.warning(f"No face detected in {image_file}")
                        continue
                        
                    # Only use the first face (most prominent)
                    # InsightFace already returns the embedding
                    face = faces[0]
                    embedding = face.embedding
                    
                    # Add embedding and name to results
                    data["embeddings"].append(embedding)
                    data["names"].append(person_name)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
        
        self.logger.info(f"Dataset loaded: {len(data['embeddings'])} face embeddings from {len(set(data['names']))} persons")
        return data
    
    def add_person(self, name, images):
        """Add a new person to the dataset. Takes the person's name and a list of images.
        Returns the number of successfully added face images.
        """
        person_dir = self.dataset_path / name
        person_dir.mkdir(exist_ok=True)
        
        # Count existing images
        existing_images = len(list(person_dir.glob("*")))
        
        # Save new images
        count = 0
        for i, image in enumerate(images):
            try:
                # Detect faces using InsightFace
                faces = self.face_app.get(image)
                
                if not faces:
                    self.logger.warning(f"No face found in image {i+1}")
                    continue
                
                # If face found, save the image
                image_path = person_dir / f"{name}_{existing_images + count + 1}.jpg"
                cv2.imwrite(str(image_path), image)
                count += 1
                    
            except Exception as e:
                self.logger.error(f"Error adding image {i+1}: {str(e)}")
        
        self.logger.info(f"Added {count} images for {name}")
        return count