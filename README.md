

# Face Recognition and Turntable Tracking System

An advanced, modular system for real-time face recognition and automatic face tracking using a motorized turntable. The project combines machine learning, computer vision, and embedded hardware to keep faces centered in the camera view by physically moving the camera mount.

---

## 🚀 Project Summary

This project detects and recognizes faces in real-time video streams using state-of-the-art machine learning models. It features a motorized turntable (servo-driven, ESP32-controlled) that automatically rotates to follow and center faces in the camera view. The system is designed for extensibility, performance, and ease of use, and demonstrates the integration of software and hardware for intelligent movement and tracking.

---

## 🏁 Quick Start

1. **Install dependencies:**
   ```bash
   pip install opencv-python numpy insightface scikit-learn requests tqdm onnxruntime
   ```
2. **Prepare data:**
   - Create a directory: `data/known_faces`
   - Add subfolders for each person, with their face images inside
3. **Train models:**
   ```bash
   python train.py --interactive
   ```
4. **Run recognition and tracking:**
   ```bash
   python recognize.py --recognition-method knn --target "Person Name"
   ```
5. *(Optional)* **Enable tracking hardware:**
   ```bash
   cd esp32-servo-controller
   pio run -t upload
   ```

---

## 📁 Project Structure


```
face-recognition-system/
   train.py, recognize.py         # Main scripts
   src/                          # Core modules (dataset, detector, recognizer, utils)
   data/
      known_faces/                # Training images (per person)
      ...                         # Model outputs, embeddings, etc.
esp32-servo-controller/         # Firmware for camera tracking hardware
```

---

## 🖼️ Hardware & Program Overview

| Electronic Schema | Turntable Photo |
|-------------------|----------------|
| ![Schema](https://nasiadka.pl/project/face-recognition-and-tracking-system/electronic_circuit.jpg) | ![Turntable](https://nasiadka.pl/project/face-recognition-and-tracking-system/main_picture.jpg) |

**Turntable Hardware:**
- The turntable is powered by a servo motor and controlled by an ESP32 microcontroller running a web server.
- The camera is mounted on the turntable, allowing it to rotate horizontally to follow faces.
- The ESP32 receives movement commands from the recognition software over Wi-Fi (HTTP requests), adjusting the turntable position in real time.

**Tracking Logic:**
1. The recognition software detects faces and identifies the target person in each frame.
2. The position of the target face is compared to the center of the camera frame.
3. If the face is off-center (outside a deadzone), the software calculates the direction and speed needed to re-center the face.
4. An HTTP command is sent to the ESP32 controller, which rotates the turntable accordingly.
5. The ESP32 servo controller receives the command, sets the servo speed and direction, and moves the turntable.
6. This closed-loop system keeps the target face centered automatically.


---

## ✨ Features

- **High-accuracy face detection** (InsightFace)
- **Multiple recognition methods:** KNN, Naive Bayes, Decision Tree, SVM, MLP
- **Interactive training** with camera
- **Data augmentation** (rotation, brightness, blur)
- **Real-time performance metrics** (FPS, recognition rate)
- **Automatic turntable movement:** Motorized camera mount rotates to follow faces
- **Closed-loop face tracking:** Keeps target centered using live feedback
- **ESP32-based servo controller:** Fast, wireless hardware integration

---

## 🧑‍💻 Training Process

1. **Data Collection:**
   - Use `train.py --interactive` to capture images with the camera, or use existing images.
   - Each person: `data/known_faces/{person_name}`
2. **Face Detection & Embedding:**
   - InsightFace detects faces and extracts 512D embeddings for each face.
3. **Data Augmentation:**
   - Images are rotated, brightness-adjusted, and blurred to increase dataset size and robustness.
4. **Model Training:**
   - Embeddings and names are collected.
   - Multiple classifiers (KNN, Naive Bayes, Decision Tree, MLP, SVM) are trained.
   - Models and scaler are saved for later use.

---

## 🕵️ Recognition & Tracking

1. **Camera & Initialization:**
   - USB camera is initialized and frames are captured.
   - Face detector and recognition model are loaded.
2. **Frame Processing:**
   - Frames are processed in real time, in a separate thread for performance.
   - Performance metrics (FPS, recognition rate) are calculated.
3. **Face Detection & Recognition:**
   - Faces are detected and embeddings extracted.
   - Each face is recognized using the selected model; confidence scores are calculated.
   - Bounding boxes, names, and facial landmarks are drawn on the frame.
4. **Turntable Tracking Logic:**
   - If a target person is specified, their position is tracked.
   - The position of the target face is compared to the center of the frame.
   - If the face is off-center (outside a deadzone), the direction and speed for the turntable are calculated.
   - An HTTP command is sent to the ESP32 controller to rotate the turntable and re-center the face.
   - The ESP32 receives the command and moves the servo accordingly.
   - This allows for hands-free, automatic following of people, ideal for security, robotics, or interactive installations.

---


---

## ⚙️ Command-Line Usage

**Train interactively:**
```bash
python train.py --interactive
```
**Train on existing files:**
```bash
python train.py
```
**Augment data:**
```bash
python train.py --augment
```
**Run recognition:**
```bash
python recognize.py --recognition-method knn --target "Person Name"
```
**Tracking (optional):**
```bash
python recognize.py --recognition-method knn --target "Person Name" --servo-url http://192.168.4.1
```

## 📄 License

This project is open source. See `LICENSE` for details.