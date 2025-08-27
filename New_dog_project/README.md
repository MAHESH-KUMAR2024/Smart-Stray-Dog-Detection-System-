# Dog Detection Using YOLOv8

## Project Overview
This project utilizes the [YOLOv8](https://github.com/ultralytics/ultralytics)
 model to detect dogs in a video. The model processes video frames in real-time, drawing bounding boxes around detected dogs and displaying their class labels.
![image showing dog detection](https://github.com/Omo-Rinsola/Dog-Detection-Using-YOLOv8/blob/main/Screenshot%202024-09-28%20at%2010.01.44.png)


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Reference](#Reference)

## Requirements
1. Python 
2. OpenCV
3. Ultralytics YOLO
4. PyTorch
5. NumPy

## Installation
To set up this project, you need to have Python installed on your machine. Follow the steps below to install the required libraries.

1. Clone the repository:
   ```bash
   git clone git@github.com:Omo-Rinsola/Dog-Detection-Using-YOLOv8.git
   cd Dog-Detection-Using-YOLOv8
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
## Usage
1. Place your video file (e.g., dogs.mp4) in the project directory.
2. Run the script to start detecting dogs in the video:
   ```bash
   python main.py
3. Press the Esc key to exit the video display.

## How It Works

The project uses OpenCV to handle video processing and the YOLOv8 model for object detection.
Each frame of the video is passed to the YOLO model, which identifies and draws bounding boxes around detected dogs.
The class label is displayed above each bounding box.

## Reference
[YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
