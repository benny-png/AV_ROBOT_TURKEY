# Autonomous Vehicle Robot System (AV-ROBOT)

This project implements a real-time autonomous vehicle perception system that combines lane detection and object detection using computer vision techniques. It utilizes multi-threading and multiprocessing options to process camera inputs efficiently for real-time applications.

## Features

- **Lane Detection**:
    - Option 1 (Legacy): Utilizes traditional computer vision techniques.
    - Option 2 (New): Employs the `TwinLiteNetPlus` deep learning model (specifically the "nano" configuration) for more accurate and robust lane and drivable area segmentation.
- **Object Detection**: Uses YOLO (You Only Look Once) model (v6 nano) to detect and track objects.
- **Real-time Processing**:
    - `main.py`: Multi-threaded architecture for simultaneous processing.
    - `main_multiprocessing.py`: Multi-processing architecture for potentially improved performance on multi-core CPUs.
- **Synchronized Output**: Combines lane and object detection results.
- **Multiple Input Sources**: Works with webcams or video files.
- **Frame Mirroring**: Option to flip camera input if needed.
- **Video Output**: Can save processed results to video files.

## Components

### Lane Detection (New - TwinLiteNetPlus)
- Uses the `TwinLiteNetPlus` model, specifically the "nano" configuration, for efficient and accurate segmentation.
- The model is loaded from `lane_detection_2/pretrained/TwinLiteNetPlus/nano.pth`.
- Preprocesses input frames (resize to 640x640, normalize) for the model.
- Performs inference to get drivable area and lane line segmentations.
- Visualizes drivable areas in green and lane lines in red on the output frame.

### Lane Detection (Legacy)
- Utilizes computer vision techniques to detect lane lines.
- Implements perspective transform to get a bird's-eye view.
- Applies color thresholding to identify lane markings.
- Fits polynomials to lane lines.
- Calculates road curvature and vehicle position.
- Maintains tracking through challenging frames.

### Object Detection
- Uses YOLOv6 nano model (`object_and_depth_detection/models/best_v6_n.pt`) for detecting objects.
- Implements object tracking with ByteTrack.
- Provides confidence scores for detections.
- Optimized for real-time processing.

## Setup

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLO
- PyTorch (refer to `lane_detection_2/README.md` for specific version if using new lane detection)
- Trained model weights

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/AV_ROBOT_TURKEY-1.git # Replace with your actual repository URL
cd AV_ROBOT_TURKEY-1
```

2. Install dependencies:
   It's recommended to set up a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
# For the new lane detection, you might need specific PyTorch versions.
# Refer to lane_detection_2/README.md and install them, e.g.:
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Make sure you have the required model files:
   - YOLOv6 model at `object_and_depth_detection/models/best_v6_n.pt`
   - TwinLiteNetPlus nano model at `lane_detection_2/pretrained/TwinLiteNetPlus/nano.pth`

## Usage

You can run the system using either a multi-threading approach (`main.py`) or a multi-processing approach (`main_multiprocessing.py`). Both scripts accept the same command-line arguments.

### Running with default settings (webcam):
Using multi-threading:
```bash
python main.py
```
Using multi-processing:
```bash
python main_multiprocessing.py
```

### Using a specific video file:
```bash
python main.py --source path/to/your/video.mp4
# or
python main_multiprocessing.py --source path/to/your/video.mp4
```

### Flipping mirrored camera input:
```bash
python main.py --flip
# or
python main_multiprocessing.py --flip
```

### Saving output to a video file:
```bash
python main.py --output output_video.mp4
# or
python main_multiprocessing.py --output output_video.mp4
```

### Full command with all options:
```bash
python main.py --source 0 --flip --output output_video.mp4
# or
python main_multiprocessing.py --source 0 --flip --output output_video.mp4
```

## Command-line Arguments

- `--source`: Input source ('0' for default webcam, or path to video file). Default: '0'.
- `--flip`: Flag to horizontally flip the input frames (useful for mirrored cameras).
- `--output`: Path to save the output video file (optional). Default: ''.
- `--lane_task`: Specifies which lane detection output to visualize. Options: 'DA' (Drivable Area only), 'LL' (Lane Lines only), 'BOTH' (both Drivable Area and Lane Lines). Default: 'BOTH'.

## Architecture

The system offers two main execution modes:

1.  **`main.py` (Multi-threading)**:
    *   **Main Thread**: Handles video capture and display.
    *   **Lane Detection Thread**: Processes frames for lane detection using `TwinLiteNetPlus`.
    *   **YOLO Detection Thread**: Processes frames for object detection using YOLOv6.
    Threads communicate using a shared `SharedFrame` object with `threading.Lock` and `threading.Event` for synchronization.

2.  **`main_multiprocessing.py` (Multi-processing)**:
    *   **Main Process (Threaded)**: Handles video capture, display, and coordination of worker processes.
        *   A **Lane Detection Thread** within the main process manages a `multiprocessing.Pool` for lane detection.
        *   A **YOLO Detection Thread** within the main process manages a `multiprocessing.Pool` for object detection.
    *   **Lane Detection Worker Process**: Initializes and runs the `TwinLiteNetPlus` model in a separate process.
    *   **YOLO Detection Worker Process**: Initializes and runs the YOLOv6 model in a separate process.
    Communication between the main process threads and worker processes is managed via `multiprocessing.Pool` and the `SharedFrame` object (for frame data and events between main process threads).

This updated architecture aims to leverage multiple CPU cores more effectively for demanding tasks.

## Performance

The system tracks and displays the performance metrics (FPS) for each processing module.
Performance depends on hardware capabilities (CPU, GPU for PyTorch models).
- The `TwinLiteNetPlus` model will utilize CUDA if a compatible GPU and PyTorch version are available, otherwise it will run on the CPU.
- `main_multiprocessing.py` might offer better CPU utilization on multi-core systems.

For optimal performance:
- Ensure PyTorch is installed with CUDA support if you have an NVIDIA GPU.
- Consider reducing resolution for faster processing if needed (though models have fixed input sizes, this refers to capture resolution).
- Adjust YOLO model or `TwinLiteNetPlus` configuration (though "nano" is already the smallest) if performance is critical and accuracy can be traded.

## Advanced Configuration

- **YOLO Model**: The YOLO model path is defined in `main.py` and `main_multiprocessing.py`. You can switch to other compatible YOLO models.
- **Lane Model**: The `TwinLiteNetPlus` model path and configuration ("nano") are set in `main.py` and `main_multiprocessing.py`.
- Detection thresholds and other YOLO parameters can be adjusted in the `.track()` method call.
- Visualization colors and styles can be modified in the `process_lane_frame` function.

## Troubleshooting

- **Camera not recognized**: Check your camera index (usually '0' for the default camera).
- **Model loading errors**:
    - Ensure model files (`.pt`, `.pth`) are at the correct paths specified in the scripts.
    - Verify PyTorch installation and compatibility, especially if using a GPU. Check `lane_detection_2/README.md` for PyTorch requirements for `TwinLiteNetPlus`.
- **Slow performance**:
    - If using GPU, ensure CUDA drivers and PyTorch with CUDA support are correctly installed and being used.
    - Try `main_multiprocessing.py` on multi-core CPUs.
    - If still slow, consider a more powerful machine or further optimizing the processing pipeline (e.g., frame skipping, though not implemented by default).
- **Video file not opening**: Ensure the file path is correct and the video codec is supported by OpenCV.

## License

[Specify your license here]

## Contact

[Your contact information] 