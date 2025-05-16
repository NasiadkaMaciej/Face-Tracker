import cv2

class USBCamera:
    """Implementation for USB webcams."""
    
    def __init__(self, device_id=0, width=640, height=480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.capture = None
        
    def start(self):
        """Initialize and start the USB camera."""
        self.capture = cv2.VideoCapture(self.device_id)
        if not self.capture.isOpened():
            return False
            
        # Set resolution if specified
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True
        
    def get_frame(self):
        """Get a frame from the USB camera."""
        if self.capture is None:
            return None
            
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame
        
    def release(self):
        """Release camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None