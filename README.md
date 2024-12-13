# Object Detection with YOLO in Streamlit

This project demonstrates object detection using the YOLOv3 model within a Streamlit web application. 

**Features:**

**Video Upload:** Upload video files (MP4, AVI) for object detection.
* **YOLOv3 Model:** Utilizes the pre-trained YOLOv3 model for accurate object detection.
* **CUDA Acceleration:** Leverages CUDA for faster processing on GPU-enabled systems (optional).
* **Real-time Visualization:** Displays the video with detected objects in real-time within the Streamlit app.
* **Object Recognition:** Identifies and labels detected objects using the COCO dataset classes.

**Dependencies:**

* Streamlit
* OpenCV
* NumPy
* imutils
* CUDA (optional for GPU acceleration)

**Installation:**

1. **Install required libraries:**
   ```bash
   pip install streamlit opencv-python numpy imutils

2. # Install CUDA (optional)
* If you have a CUDA-enabled GPU, follow the instructions on the NVIDIA website to install the CUDA Toolkit.

* Download YOLOv3 files:
* Download yolov3.weights and yolov3.cfg from the official YOLO website or a reliable source.
* Place these files in the object_detection_streamlit directory.

* Download coco.names:
* Download coco.names from the COCO dataset repository.
* Place this file in the object_detection_streamlit directory.

# Run the Streamlit app:
    streamlit run app.py 

* Upload a video: Click the "Choose a video file" button and select the video you want to process.
* View results: The app will display the video with detected objects highlighted and labeled.

* Project Structure:

* app.py: The main Python file containing the Streamlit application code.
* yolov3.weights: The pre-trained YOLOv3 weights file.
* yolov3.cfg: The YOLOv3 model configuration file.
* coco.names: A text file containing the names of the object classes from the COCO dataset.

# Note:

* This project requires the YOLOv3 weights file (yolov3.weights) and configuration file (yolov3.cfg) to be in the same directory.
* For optimal performance, ensure you have a CUDA-enabled GPU and have installed the CUDA Toolkit.
* The imutils library provides helper functions for image and video processing.

# To improve this project:

* Consider adding options for adjusting confidence thresholds and Non-Maximum Suppression (NMS) parameters.
* Implement a more user-friendly interface with sliders or input fields for adjusting parameters.
* Explore more advanced object detection models like YOLOv5 or EfficientDet.
* Integrate with other Streamlit features like caching or progress bars for a smoother user experience.

# Open app.py in your preferred text editor
    nano app.py

# Paste the following content into app.py:
* echo "import streamlit as st
* import os
* import cv2
* import numpy as np
* import imutils
* import time

# Let's load the YOLO model first
* net = cv2.dnn.readNet(\"yolov3.weights\", \"yolov3.cfg\")
* net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # we used the CUDA backend for better performance
* net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Run on GPU if available

# Load the names of object classes from COCO dataset file in the directory(coco.names)
with open(\"coco.names\", \"r\") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the title for the app
* st.title(\"Object Detection Project for Deep Learning Class\")

# Let the user upload a video file
uploaded_file = st.file_uploader(\"Choose a video file\", type=[\"mp4\", \"avi\"])
if uploaded_file is not None:
    # Save the uploaded file in a temporary folder
    if not os.path.exists(\"temp\"):
        # Create the folder if it doesn't exist
        os.makedirs(\"temp\")
    " > app.py# Object-Detection-in-Pyhton
# Object-Detection-Coco-Sets
