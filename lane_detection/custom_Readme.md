
# Advanced Lane Lines Detection

This repository contains a robust lane detection system for autonomous vehicles, capable of identifying lane boundaries in various lighting and road conditions.

## Features

- Real-time lane detection on video streams or camera feeds
- Handles curved roads and varying lighting conditions
- Calculates road curvature radius and vehicle position
- Implements lane tracking with smoothing and recovery
- Visual feedback with overlaid lane area and metrics
- Option to save processed video output

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Installation

```bash
git clone https://github.com/yourusername/Advanced-Lane-Lines.git
cd Advanced-Lane-Lines
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python lane_detection.py
```

When prompted:
1. Enter a video file path or camera index (default is 0 for webcam)
2. Choose whether to save the processed output video
3. If saving, specify the output file path

Press 'q' to exit the application.

## How It Works

This lane detection pipeline:

1. **Camera Calibration**: Undistorts images using camera matrix and distortion coefficients
2. **Color Thresholding**: Extracts lane lines using HLS and LAB color spaces
3. **Perspective Transform**: Converts to bird's-eye view for easier lane detection
4. **Lane Detection**: Uses sliding window or polynomial search to identify lane pixels
5. **Polynomial Fitting**: Fits second-degree polynomial to lane lines
6. **Sanity Checking**: Validates detection with lane width and tracking history
7. **Smoothing**: Averages recent detections for stable output
8. **Metrics Calculation**: Computes lane curvature and vehicle position
9. **Visualization**: Projects detected lane onto the original image

## Customization

Adjust these parameters in the code to improve performance:

- Color thresholds in `color_threshold()`
- Perspective transform points in `perspective_transform()`
- Sliding window parameters in `find_lane_pixels()`
- Polynomial search margin in `search_around_poly()`
- Lane validation thresholds in `process_frame()`

## License

[MIT License](LICENSE)
