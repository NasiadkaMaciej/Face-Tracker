import cv2
import time
import argparse
import requests
import threading
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.usb_camera import USBCamera
from src.utils import setup_logging

# Get logger
logger = setup_logging("recognize")

# ESP32 servo controller URL
SERVO_URL = "http://192.168.4.1"
MAX_SPEED = 1000
MIN_SPEED = 42  # Minimum speed to move the servo
DEADZONE = 30   # Pixels from center where we don't move the camera

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face recognition and tracking system")
    parser.add_argument("--detection-method", 
                      choices=["insightface", "haar", "hog", "mediapipe"], 
                      default="insightface",
                      help="Face detection method to use")
    parser.add_argument("--recognition-method", 
                      choices=["cosine", "knn", "naive_bayes", "decision_tree", "mlp", "svm"], 
                      default="cosine",
                      help="Face recognition method to use")
    parser.add_argument("--database", help="Path to face recognition database")
    parser.add_argument("--camera-id", type=int, default=0, help="USB camera device ID (default: 0)")
    parser.add_argument("--target", default=None, help="Name of person to track (default: None - no specific tracking)")
    parser.add_argument("--servo-url", default="http://192.168.4.1", 
                      help="URL of the ESP32 servo controller")
    args = parser.parse_args()
    
    # Update global settings
    global SERVO_URL
    SERVO_URL = args.servo_url
    
    # Set up USB camera
    camera = USBCamera(device_id=args.camera_id)
    if not camera.start():
        logger.error(f"Failed to start USB camera (device ID: {args.camera_id})")
        return
    
    # Initialize face detector and recognizer
    face_detector = FaceDetector(method=args.detection_method)
    database_path = args.database or Path("data/models/face_recognition_database.pkl")
    face_recognizer = FaceRecognizer(database_path=database_path, 
                                  recognition_method=args.recognition_method)
    
    if not face_recognizer.load_model():
        logger.warning("Could not load face database. Faces will not be recognized.")
    
    logger.info(f"Using detection method: {args.detection_method}")
    logger.info(f"Using recognition method: {args.recognition_method}")
    
    run_tracking_loop(camera, face_detector, face_recognizer, args.target)

def calculate_servo_command(face_center_x, frame_width):
    """Calculate servo direction and speed based on face position."""
    # Calculate distance from center
    frame_center_x = frame_width // 2
    distance = face_center_x - frame_center_x
    
    # Determine if we're in the deadzone (close enough to center)
    if abs(distance) < DEADZONE:
        # Return current direction with speed 0 to actually stop the servo
        direction = "right" if distance >= 0 else "left"
        return direction, 0
    
    # Calculate direction
    direction = "left" if distance < 0 else "right"
    
    # Calculate speed based on distance
    max_distance = frame_width // 2  # Maximum possible distance from center
    
    # Linear scaling between MIN_SPEED and MAX_SPEED based on distance from center
    normalized_distance = min((abs(distance) - DEADZONE) / (max_distance - DEADZONE), 1.0)
    speed = int(MIN_SPEED + normalized_distance * (MAX_SPEED - MIN_SPEED))
    
    return direction, speed

def control_servo(direction, speed):
    """Send control command to ESP32 servo controller."""
    try:
        # When direction is None, send a stop command
        if direction is None:
            url = f"{SERVO_URL}/rotate?direction=stop&speed=0"
            requests.get(url, timeout=0.5)
            logger.info("Sent stop command to servo")
        else:
            url = f"{SERVO_URL}/rotate?direction={direction}&speed={speed}"
            requests.get(url, timeout=0.5)
            logger.info(f"Sent command: direction={direction}, speed={speed}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send servo command: {e}")

def run_tracking_loop(camera, face_detector, face_recognizer, target_name):
    logger.info(f"Starting tracking system. {'Tracking: ' + target_name if target_name else 'No specific person tracking enabled'}. Press 'q' to quit.")
    
    # Store target embedding for comparison with unknown faces
    target_embedding = None
    
    # Processing thread control
    processing = False
    latest_frame = None
    latest_frame_lock = threading.Lock()
    current_faces = []
    
    def process_frame():
        nonlocal processing, latest_frame, current_faces, target_embedding
        
        try:
            # Grab the latest frame
            with latest_frame_lock:
                if latest_frame is None:
                    processing = False
                    return
                frame_to_process = latest_frame.copy()
                frame_width = frame_to_process.shape[1]
                latest_frame = None
            
            # Detect and recognize faces
            faces = face_detector.detect_faces(frame_to_process)
            
            # Stop the servo if no faces detected
            if not faces:
                current_faces = []
                control_servo(None, 0)  # Explicitly stop servo when no faces detected
                processing = False
                return
                
            recognized_faces = face_recognizer.recognize_faces(frame_to_process, faces)
            current_faces = recognized_faces  # Update faces for display
            
            # If no specific target is set, just display faces without tracking
            if not target_name:
                processing = False
                return
            
            # Original tracking logic when target is specified
            target_face = None
            unknown_faces = []
            recognized_non_target_faces = []  # New list for recognized faces that aren't the target
            
            for x, y, w, h, name, face_obj in recognized_faces:
                if name == target_name:
                    # Store target face info but don't break, in case there are multiple matches
                    if target_face is None:  # Take the first match, or you could use another criterion
                        target_face = (x, y, w, h)
                        
                        # Store target embedding if not already saved
                        if target_embedding is None and face_obj is not None:
                            target_embedding = face_obj.embedding
                elif name == "Unknown":
                    # Store unknown faces for potential tracking
                    unknown_faces.append((x, y, w, h, face_obj))
                else:
                    # Store other recognized faces that aren't the target
                    recognized_non_target_faces.append((x, y, w, h, face_obj))
            
            # If target found, track it (highest priority)
            if target_face:
                x, y, w, h = target_face
                face_center_x = x + (w // 2)
                direction, speed = calculate_servo_command(face_center_x, frame_width)
                control_servo(direction, speed)
                
            # If target not found but unknown faces exist
            elif unknown_faces:
                if len(unknown_faces) == 1:
                    # Only one unknown face, track it
                    x, y, w, h, _ = unknown_faces[0]
                    face_center_x = x + (w // 2)
                    direction, speed = calculate_servo_command(face_center_x, frame_width)
                    control_servo(direction, speed)
                elif target_embedding is not None:
                    # Multiple unknown faces, find the most similar to target
                    best_match_idx = 0
                    highest_similarity = -1
                    
                    # Compare each unknown face with the target embedding
                    for i, (x, y, w, h, face_obj) in enumerate(unknown_faces):
                        if face_obj and face_obj.embedding is not None:
                            # Calculate similarity using cosine similarity
                            similarity = cosine_similarity([target_embedding], [face_obj.embedding])[0][0]
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match_idx = i
                    
                    # Track the most similar unknown face
                    x, y, w, h, _ = unknown_faces[best_match_idx]
                    face_center_x = x + (w // 2)
                    direction, speed = calculate_servo_command(face_center_x, frame_width)
                    control_servo(direction, speed)
                else:
                    # Just track the first unknown face if no target embedding
                    x, y, w, h, _ = unknown_faces[0]
                    face_center_x = x + (w // 2)
                    direction, speed = calculate_servo_command(face_center_x, frame_width)
                    control_servo(direction, speed)
            else:
                # No face to track, stop the servo
                control_servo(None, 0)
                
        finally:
            processing = False
    
    try:
        while True:
            # Get current frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Could not capture frame")
                time.sleep(0.1)
                continue
            
            # Update the latest frame for processing
            with latest_frame_lock:
                latest_frame = frame
            
            # Start processing if not already in progress
            if not processing:
                processing = True
                threading.Thread(target=process_frame).start()
            
            # Display the frame with marked faces
            display_frame = frame.copy()
            if current_faces:
                display_frame = face_recognizer.mark_faces(display_frame, current_faces)
            cv2.imshow('Face Recognition', display_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in tracking loop: {str(e)}")
    finally:
        # Stop the servo and clean up
        try:
            requests.get(f"{SERVO_URL}/rotate?direction=right&speed=0", timeout=0.5)
        except:
            pass
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()