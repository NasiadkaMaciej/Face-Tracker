import os
import argparse
import cv2
from pathlib import Path
from collections import Counter

from src.face_dataset import FaceDataset
from src.face_recognizer import FaceRecognizer
from src.usb_camera import USBCamera
from src.utils import setup_logging, ensure_dir_exists

# Set up logger
logger = setup_logging("train")

def create_dataset_interactive(dataset_dir, camera_id=0):
    """Capture face images interactively using camera."""
    # Get name input
    person_name = input("Enter the person's name: ").strip()
    if not person_name:
        logger.error("Invalid name provided")
        return None, None
    
    # Create person directory
    person_dir = ensure_dir_exists(dataset_dir / person_name)
    logger.info(f"Using directory: {person_dir}")
    
    # Find the highest existing image number
    highest_num = 0
    existing_files = list(person_dir.glob(f"{person_name}_*.jpg"))
    for file in existing_files:
        try:
            # Extract the number from filename (person_name_X.jpg)
            num = int(file.stem.split('_')[-1])
            highest_num = max(highest_num, num)
        except (ValueError, IndexError):
            continue
    
    logger.info(f"Found {len(existing_files)} existing images for {person_name}")
    
    # Start camera
    logger.info(f"Initializing camera (device_id: {camera_id})...")
    camera = USBCamera(device_id=camera_id)
    
    if not camera.start():
        logger.error("Could not connect to camera")
        return None, None
    
    # Get camera properties
    test_frame = camera.get_frame()
    if test_frame is not None:
        height, width = test_frame.shape[:2]
        logger.info(f"Camera initialized successfully. Resolution: {width}x{height}")
    
    # Image capture loop
    images = []
    start_count = highest_num  # Start from the highest existing number
    count = start_count
    max_images = 50
    
    logger.info(f"Found {start_count} existing images. New images will start from #{start_count+1}")
    logger.info(f"Capturing {max_images} images for {person_name}. Press SPACE to capture, ESC to quit.")
    logger.info("Tip: Move your head to different positions/angles for better training results")
    
    try:
        while len(images) < max_images:
            # Get a frame and display it
            frame = camera.get_frame()
            if frame is None:
                logger.error("Could not read frame from camera")
                continue
            
            # Display the current frame
            cv2.imshow('Face Training - Press SPACE to capture, ESC to quit', frame)
            
            # Wait for key press (30ms delay for smooth video)
            key = cv2.waitKey(30) & 0xFF
            
            # If ESC pressed, quit
            if key == 27:  # ESC key
                logger.info(f"Capture stopped by user after {len(images)} images")
                break
                
            # If SPACE pressed, capture the current frame
            if key == 32:  # SPACE key
                # Increment counter first
                count += 1
                
                # Save the image
                img_path = person_dir / f"{person_name}_{count}.jpg"
                cv2.imwrite(str(img_path), frame)
                images.append(frame.copy())
                logger.info(f"Captured image {len(images)}/{max_images} (saved as {img_path.name})")
    finally:
        logger.info(f"Capture session completed. {len(images)} images captured")
        camera.release()
        cv2.destroyAllWindows()
        
    return person_name, images
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--interactive", action="store_true", help="Create dataset interactively using camera")
    parser.add_argument("--dataset", help="Path to the dataset directory")
    parser.add_argument("--output", help="Output path for the trained model")
    parser.add_argument("--camera-id", type=int, default=0, help="USB camera device ID (default: 0)")
    parser.add_argument("--augment", default=True, action="store_true", help="Apply data augmentation to increase dataset size")
    args = parser.parse_args()
    
    # Determine paths
    dataset_dir = Path(args.dataset) if args.dataset else Path("data/known_faces")
    
    # Interactive mode - create dataset
    if args.interactive:
        logger.info("=== Starting interactive dataset creation ===")
        # Pass the camera ID from args
        person_name, images = create_dataset_interactive(dataset_dir, args.camera_id)
        if person_name and images:
            logger.info(f"Successfully captured {len(images)} images for {person_name}")
        else:
            logger.warning("Interactive capture completed with no images")
    
    # Create face recognition database
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset path '{dataset_dir}' does not exist.")
        return
    
    logger.info("=== Starting database creation process ===")    
    success = process_and_save_database(dataset_dir, args.output, args.augment)
    
    if success:
        logger.info("=== Training completed successfully ===")
    else:
        logger.error("=== Training failed ===")
    
    return success

def process_and_save_database(dataset_path, output_path=None, augment=True):
    """Process dataset and create/save face recognition database."""
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Count persons and images
    person_dirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]
    logger.info(f"Found {len(person_dirs)} persons in dataset")
    
    total_images = 0
    for person_dir in person_dirs:
        person_name = person_dir.name
        images = list(person_dir.glob("*.jpg"))
        image_count = len(images)
        total_images += image_count
        logger.info(f"Person: {person_name} - {image_count} images")
    
    logger.info(f"Total images to process: {total_images}")
    
    # Initialize and load dataset
    logger.info("Initializing face feature extraction...")
    face_dataset = FaceDataset(dataset_path)
    
    # Data augmentation status
    if augment:
        logger.info("=== Data Augmentation Enabled ===")
        logger.info("The following augmentations will be applied to each image:")
        logger.info("  - Small rotations (+/- 5 degrees)")
        logger.info("  - Brightness variations (darker/brighter)")
        logger.info("  - Slight Gaussian blur")
        logger.info("  - Random noise addition")
        logger.info(f"This will multiply your dataset size by approximately 6x")
    else:
        logger.info("Data augmentation disabled. Processing original images only.")
        expected_embeddings = total_images
    
    logger.info("=== Starting Face Feature Extraction ===")
    logger.info("This process includes:")
    logger.info("  1. Loading each image")
    logger.info("  2. Detecting faces using InsightFace")
    logger.info("  3. Extracting facial embeddings (512-dimensional vectors)")
    if augment:
        logger.info("  4. Creating augmented variations of each image")
        logger.info("  5. Extracting embeddings from augmented images")
    
    logger.info("Extracting face features from images (this may take a while)...")
    
    # Process the dataset with or without augmentation
    data = face_dataset.load_data(augment=augment)
    
    logger.info(f"Face extraction completed")
    
    if not data["embeddings"]:
        logger.error("No face embeddings found in dataset")
        return False
    
    # Show statistics about processed data
    num_embeddings = len(data["embeddings"])
    num_persons = len(set(data["names"]))
    
    logger.info(f"Successfully extracted {num_embeddings} face embeddings from {num_persons} persons")
    
    # Count embeddings per person
    name_counts = Counter(data["names"])
    logger.info("Embeddings per person:")
    for name, count in name_counts.items():
        logger.info(f"  - {name}: {count} embeddings")
    
    # Create face recognizer and add faces
    # Use provided output_path if available, otherwise use default
    database_path = output_path if output_path else Path("data/models/face_recognition_database.pkl")
    logger.info(f"Creating face recognition database at: {database_path}")
    
    recognizer = FaceRecognizer(database_path=database_path)
    
    # Add each face to the recognizer
    logger.info(f"=== Building Face Recognition Database ===")
    logger.info(f"Adding {num_embeddings} face embeddings to the recognition database...")
    
    for i, (embedding, name) in enumerate(zip(data["embeddings"], data["names"])):
        recognizer.add_face(name, embedding)
    
    # Save the database with the correct path
    logger.info(f"Saving face recognition database to {database_path}")
    success = recognizer.save_database(database_path)
    
    if success:
        logger.info(f"Face recognition database saved successfully to {database_path}")
        logger.info(f"Database contains {num_embeddings} embeddings from {num_persons} persons")
        if augment:
            logger.info(f"Data augmentation increased dataset size from {total_images} to {num_embeddings} embeddings")
    else:
        logger.error("Failed to save face recognition database")

    logger.info("=== Training Machine Learning Models ===")
    success = face_dataset.train_models(database_path)
    if success:
        logger.info("ML models trained and saved successfully")
    else:
        logger.warning("Failed to train some ML models")
    
    return success


if __name__ == "__main__":
    main()