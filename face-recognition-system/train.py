import os
import argparse
import cv2
from pathlib import Path

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
    
    # Find the highest existing image number
    highest_num = 0
    for file in person_dir.glob(f"{person_name}_*.jpg"):
        try:
            # Extract the number from filename (person_name_X.jpg)
            num = int(file.stem.split('_')[-1])
            highest_num = max(highest_num, num)
        except (ValueError, IndexError):
            continue
    
    # Start camera
    camera = USBCamera(device_id=camera_id)
    
    if not camera.start():
        logger.error("Could not connect to camera")
        return None, None
    
    # Image capture loop
    images = []
    start_count = highest_num  # Start from the highest existing number
    count = start_count
    max_images = 100
    
    logger.info(f"Found {start_count} existing images. New images will start from #{start_count+1}")
    logger.info(f"Capturing {max_images} images for {person_name}. Press SPACE to capture, ESC to quit.")
    
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
    args = parser.parse_args()
    
    # Determine paths
    dataset_dir = Path(args.dataset) if args.dataset else Path("data/known_faces")
    
    # Interactive mode - create dataset
    if args.interactive:
        logger.info("Starting interactive dataset creation")
        # Pass the camera ID from args
        person_name, images = create_dataset_interactive(dataset_dir, args.camera_id)
        if person_name and images:
            logger.info(f"Adding {len(images)} images for {person_name} to dataset")
            face_dataset = FaceDataset(dataset_dir)
            face_dataset.add_person(person_name, images)
    
    # Create face recognition database
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset path '{dataset_dir}' does not exist.")
        return
        
    return process_and_save_database(dataset_dir, args.output) 

def process_and_save_database(dataset_path, output_path=None):
    """Process dataset and create/save face recognition database."""
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    face_dataset = FaceDataset(dataset_path)
    data = face_dataset.load_data()
    
    if not data["embeddings"]:
        logger.error("No face embeddings found in dataset")
        return False
        
    # Create face recognizer and add faces
    # Use provided output_path if available, otherwise use default
    database_path = output_path if output_path else Path("data/models/face_recognition_database.pkl")
    recognizer = FaceRecognizer(database_path=database_path)
    
	# Add each face to the recognizer
    for embedding, name in zip(data["embeddings"], data["names"]):
        recognizer.add_face(name, embedding)
    
    # Save the database with the correct path
    success = recognizer.save_database(database_path)
    if success:
        logger.info(f"Face recognition database saved to {database_path}")
    else:
        logger.error("Failed to save face recognition database")
    
    return success

if __name__ == "__main__":
    main()