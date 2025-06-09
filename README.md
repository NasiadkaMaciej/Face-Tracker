# Face Recognition and Tracking System

This project implements a face recognition system with optional tracking capabilities. The system uses machine learning to identify individuals in real-time video feeds.

## Overview

The core system uses a camera to detect and recognize faces in real-time. It can be extended with a motorized camera mount for automatic tracking, keeping targetted faces centered in the frame.

## Features

- **High-Accuracy Face Detection**: Using InsightFace for reliable face detection
- **Multiple Recognition Methods**: KNN, Naive Bayes, Decision Tree, SVM, MLP
- **Interactive Training**: Simple camera-based training interface
- **Data Augmentation**: Enhances training data for better recognition
- **Performance Metrics**: Real-time FPS and recognition rate display
- **Optional Tracking**: Can be connected to a motorized mount (requires additional hardware)

## Installation

1. Install required Python packages:
```bash
pip install opencv-python numpy insightface scikit-learn requests tqdm onnxruntime
```

2. Create a `data/known_faces` directory to store training images.

3. For tracking functionality (optional):
```bash
cd esp32-servo-controller
pio run -t upload
```

## Training Process

### Overview

The training process involves capturing face images, extracting facial features, and training multiple machine learning models for comparison.

### Step-by-Step Process

1. **Data Collection**:
   - Images are captured using the train.py script with the `--interactive` flag
   - Each person gets their own directory in `data/known_faces/{person_name}`
   - Capture multiple images from different angles for better recognition

```bash
python train.py --interactive
```

2. **Face Feature Extraction**:
   - InsightFace detects faces in the training images
   - 512-dimensional embeddings are extracted for each face
   - These embeddings represent unique facial characteristics

3. **Data Augmentation**:
   - Original images are transformed to create additional training data:
   - Rotations (±5 degrees)
   - Brightness variations (80% and 120%)
   - Gaussian blur
   - This increases the dataset size by 5 times

4. **Model Training**:
   - Features are standardized using `StandardScaler`
   - Multiple classification models are trained:
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Decision Tree
     - Multi-layer Perceptron (MLP)
     - Support Vector Machine (SVM)

5. **Database Creation**:
   - Face embeddings and person names are saved to a database
   - Each model is saved separately along with the feature scaler

### Improving Recognition with Unknown Faces

As more unknown faces are added to the training set:
- Recognition confidence thresholds become more meaningful
- False positive rates decrease
- The system becomes more selective in identification

### Saved Files

After training, the following files are created in the `data/models/` directory:

- `face_recognition_database.pkl`: Primary database with embeddings and names
- `face_recognition_database_scaler.pkl`: StandardScaler for feature normalization
- Model files for each classifier (KNN, Naive Bayes, Decision Tree, MLP, SVM)

## Recognition and Tracking Process

### Main Recognition Loop

1. **Initialization**:
   - Camera connection is established
   - Face detector is initialized (InsightFace)
   - Recognition model is loaded based on selected method

2. **Frame Processing**:
   - Frames are captured from the camera
   - Processing happens in a separate thread for better performance
   - Performance metrics (FPS, recognition rate) are calculated

3. **Face Detection**:
   - InsightFace locates faces in each frame
   - Returns bounding boxes and face embeddings
   - Also provides facial landmarks (eyes, nose, mouth)

4. **Face Recognition**:
   - Face embeddings are passed to the selected ML model
   - Model predicts identity with confidence score
   - If confidence is below threshold, face is marked as "Unknown"

5. **Display**:
   - Recognized faces are marked with bounding boxes
   - Names and confidence scores are displayed
   - Facial landmarks are highlighted

### Recognition Methods Comparison

| Model | Configuration | Strengths | Weaknesses |
|-------|---------------|-----------|------------|
| K-Nearest Neighbors (KNN) | n_neighbors=5, weights='distance' | Simple, intuitive, works well with small datasets | Slow on large datasets, sensitive to irrelevant features |
| Naive Bayes | GaussianNB (default) | Fast training, handles high dimensionality well | Assumes feature independence (often violated in face data) |
| Decision Tree | max_depth=5 | Interpretable, handles non-linear data | Prone to overfitting, unstable with small data changes |
| Multi-Layer Perceptron (MLP) | hidden_layer_sizes=(100,), max_iter=300 | Captures complex patterns, non-linear | Sensitive to feature scaling, requires more data |
| Support Vector Machine (SVM) | kernel='linear', probability=True | Works well in high dimensions, robust | Slow training with large datasets, sensitive to parameters |

## Technical Details

### InsightFace Integration

InsightFace is used for both face detection and embedding extraction:

```python
# Initialization
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Detection and embedding extraction
faces = face_app.get(image)
for face in faces:
    embedding = face.embedding  # 512-dimensional vector
    bbox = face.bbox  # Bounding box coordinates
    landmarks = face.kps  # Facial landmarks
```

## Data Storage Format

The face recognition system stores face data in a pickle file with the following structure:

```python
{
    "embeddings": [array1, array2, ...],  # List of numpy arrays (512-dimensional face embeddings)
    "names": ["person1", "person2", ...]  # Corresponding person names
}
```

Each trained machine learning model is stored in a separate file with the naming convention:
```
face_recognition_database_<method>.pkl
```

Additionally, a scaler model is saved to standardize feature values:
```
face_recognition_database_scaler.pkl
```

## Usage

### Training Face Recognition

```bash
python train.py --interactive
```

Follow the prompts to capture face images. Move your head to different positions for better training results.

### Training on already existing files

```bash
python train.py
```

You can also use `--augment` parameter to augment photos (both interactive and existing)

### Running Face Recognition

```bash
python recognize.py --recognition-method knn --target "Person Name"
```

Command line options:
- `--recognition-method`: Choose from knn, naive_bayes, decision_tree, mlp, svm
- `--target`: Name of the person to track (optional)
- `--camera-id`: USB camera device ID (default: 0)

### Tracking Functionality (Optional)

If you've set up the hardware for tracking:

```bash
python recognize.py --recognition-method knn --target "Person Name" --servo-url http://192.168.4.1
```