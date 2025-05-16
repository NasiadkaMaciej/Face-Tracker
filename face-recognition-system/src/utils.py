import os
import logging
from pathlib import Path

def setup_logging(name="FaceRecognition"):
    """Configure and return a logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(name)

def ensure_dir_exists(directory_path):
    """Create directory if it doesn't exist."""
    path = Path(directory_path)
    path.mkdir(exist_ok=True, parents=True)
    return path