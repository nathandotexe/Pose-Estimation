# Pose Estimation & Gesture Recognition System

This project provides a **real-time pose estimation and hand gesture recognition system**. It can track full-body movements and create a skeletal representation while detecting predefined hand gestures such as 💕 Love, 👍 Thumbs Up, 👎 Thumbs Down, and counting the number of fingers raised.

## Features
- **Real-Time Full-Body Pose Estimation**: Tracks and visualizes body movements with a skeletal model.
- **Hand Gesture Recognition**:
  - Detects gestures like Love, Thumbs Up/Down.
  - Counts the number of fingers raised on each hand.
- **Interactive Visualization**: Overlays the skeletal model and gesture details onto the video feed.

## Requirements
### Software Requirements
- Python 3.7 or higher
- Required Python Libraries:
  - `mediapipe` – For pose and hand tracking.
  - `opencv-python` – For video processing and display.
  - `numpy` – For efficient numerical computations.
  - `matplotlib` – (Optional) For plotting and debugging.
  - `tkinter` – (Optional) For graphical user interface (GUI) enhancements.

### Hardware Requirements
- A webcam or compatible external camera.
- (Optional) GPU for enhanced real-time processing performance.
