# Autonomous Vehicle Robot System (AV-ROBOT)

This project implements a real-time autonomous vehicle perception system that combines lane detection and object detection using computer vision techniques. It utilizes multi-threading to process camera inputs efficiently for real-time applications.

## Features

- **Lane Detection**: Identifies and tracks lane markings on roads
- **Object Detection**: Uses YOLO (You Only Look Once) model to detect and track objects
- **Real-time Processing**: Multi-threaded architecture for simultaneous processing
- **Synchronized Output**: Combines lane and object detection results
- **Multiple Input Sources**: Works with webcams or video files
- **Frame Mirroring**: Option to flip camera input if needed
- **Video Output**: Can save processed results to video files

## Components

### Lane Detection
- Utilizes computer vision techniques to detect lane lines
- Implements perspective transform to get a bird's-eye view
- Applies color thresholding to identify lane markings
- Fits polynomials to lane lines
- Calculates road curvature and vehicle position
- Maintains tracking through challenging frames

### Object Detection
- Uses YOLOv5 model for detecting objects
- Implements object tracking with ByteTrack
- Provides confidence scores for detections
- Optimized for real-time processing

## Setup

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv5
- Trained model weights

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/AV_ROBOT_CAR_TURKEY.git
cd AV_ROBOT_CAR_TURKEY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the required model files:
   - YOLOv5 model at `object_and_depth_detection/models/best_v5_n.pt`

## Usage

### Running with default settings (webcam):
```bash
python main.py
```

### Using a specific video file:
```bash
python main.py --source lane_detection/project_video.mp4
```

### Flipping mirrored camera input:
```bash
python main.py --flip
```

### Saving output to a video file:
```bash
python main.py --output output_video.mp4
```

### Full command with all options:
```bash
python main.py --source 0 --flip --output output_video.mp4
```

## Command-line Arguments

- `--source`: Input source (0 for default webcam, or path to video file)
- `--flip`: Flag to horizontally flip the input frames (useful for mirrored cameras)
- `--output`: Path to save the output video file (optional)

## Architecture

The system uses a multi-threaded architecture:
1. **Main Thread**: Handles video capture and display
2. **Lane Detection Thread**: Processes frames for lane detection
3. **YOLO Detection Thread**: Processes frames for object detection

These threads communicate using a shared data structure with appropriate synchronization mechanisms to ensure thread safety.

## Performance

The system tracks and displays the performance metrics for each processing module:
- Lane detection FPS
- YOLO detection FPS

Performance depends on hardware capabilities. For optimal performance:
- Consider reducing resolution for faster processing
- Adjust YOLOv5 model size (nano, small, medium) based on your hardware
- Process every Nth frame if running on resource-constrained systems

## Advanced Configuration

For advanced users, you can modify:
- Camera calibration parameters in `main.py`
- Detection thresholds in processing functions
- Lane detection parameters in `lane_detection.py`
- YOLO confidence thresholds in `main.py`

## Troubleshooting

- **Camera not recognized**: Check your camera index (usually 0 for the default camera)
- **Poor lane detection**: Adjust the color thresholds in `lane_detection.py`
- **Slow performance**: Reduce resolution or use a smaller YOLO model
- **Video file not opening**: Ensure the file path is correct and the video codec is supported

## License

[Specify your license here]

## Contact

[Your contact information] 