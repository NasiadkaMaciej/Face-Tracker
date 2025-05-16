import cv2
import argparse
import time
import threading
from pathlib import Path

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.usb_camera import USBCamera
from src.utils import setup_logging

# Get logger
logger = setup_logging("recognize")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face recognition system")
    parser.add_argument("--detection-method", choices=["hog", "haar"], default="hog",
                        help="Face detection method to use")
    parser.add_argument("--database", help="Path to face recognition database")
    parser.add_argument("--output", help="Save output video to specified file")
    parser.add_argument("--camera-id", type=int, default=0, help="USB camera device ID (default: 0)")
    args = parser.parse_args()
    
    # Set up USB camera
    camera = USBCamera(device_id=args.camera_id)
    if not camera.start():
        logger.error(f"Failed to start USB camera (device ID: {args.camera_id})")
        return
    
    # Initialize face detector and recognizer
    face_detector = FaceDetector(method=args.detection_method)
    database_path = args.database or Path("data/models/face_recognition_database.pkl")
    face_recognizer = FaceRecognizer(database_path=database_path)
    if not face_recognizer.load_model():
        logger.warning("Could not load face database. Faces will not be recognized.")
    
    run_recognition_loop(camera, face_detector, face_recognizer)

def run_recognition_loop(camera, face_detector, face_recognizer):

    logger.info("Starting face recognition. Press 'q' to quit.")
    
    # Track performance
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 5
    
    # Store last detections
    last_faces = []
    
    # Control frame rate
    frame_delay = 1/30  # 15 FPS target
    
    # Processing thread control
    processing = False
    latest_frame = None
    latest_frame_lock = threading.Lock()
    
    def process_frame():
        nonlocal processing, latest_frame, last_faces
        
        try:
            # Grab the latest frame
            with latest_frame_lock:
                if latest_frame is None:
                    processing = False
                    return
                frame_to_process = latest_frame.copy()
                latest_frame = None  # Reset to indicate we've taken this frame
            
            # Process the frame
            faces = face_detector.detect_faces(frame_to_process)
            if faces:
                # Recognize faces
                faces = face_recognizer.recognize_faces(frame_to_process, faces)
                last_faces = faces  # Update results
        finally:
            # Always mark processing as done when finished
            processing = False
    
    try:
        while True:
            loop_start = time.time()
            
            # Always get frame for smooth video
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Could not capture frame")
                break
            
            # Update the latest frame for processing (skips any old frames)
            with latest_frame_lock:
                latest_frame = frame
            
            # Start processing if not already in progress
            if not processing:
                processing = True
                processing_thread = threading.Thread(target=process_frame)
                processing_thread.start()
            
            # Always display frame (with most recent detection results)
            marked_frame = face_recognizer.mark_faces(frame, last_faces) if last_faces else frame
            
            # Update fps counter
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                fps = frame_count / (time.time() - start_time)
                # Print FPS to terminal instead of rendering on frame
                print(f"\rFPS: {fps:.2f}", end="", flush=True)
            
            cv2.imshow('Face Recognition', marked_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Control frame rate to maintain 15 FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        cleanup(camera, frame_count, start_time)
        
def cleanup(camera, frame_count, start_time):
    camera.release()
    cv2.destroyAllWindows()
    
    # Log performance stats
    elapsed = time.time() - start_time
    if frame_count > 0:
        logger.info(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")

if __name__ == "__main__":
    main()