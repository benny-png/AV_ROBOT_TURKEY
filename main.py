import cv2
import numpy as np
import time
import threading
import argparse
from queue import Queue, Empty
from lane_detection.lane_detection import Camera, Line, process_frame
from ultralytics import YOLO

class SharedFrame:
    def __init__(self):
        self.frame = None
        self.processed_lane = None
        self.processed_yolo = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.lane_done_event = threading.Event()
        self.yolo_done_event = threading.Event()

def lane_detection_thread(shared_frame, camera, left_line, right_line, stop_event):
    """Thread function for lane detection processing"""
    lane_fps = 0
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # Wait for a new frame
        if shared_frame.new_frame_event.wait(timeout=0.1):
            shared_frame.new_frame_event.clear()
            
            with shared_frame.lock:
                if shared_frame.frame is None:
                    continue
                frame = shared_frame.frame.copy()
            
            # Process frame for lane detection
            processed = process_frame(frame, left_line, right_line, camera)
            
            # Calculate FPS
            frame_count += 1
            if (time.time() - start_time) > 1:
                lane_fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Add FPS text
            cv2.putText(processed, f"Lane FPS: {lane_fps:.1f}", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Store the processed frame
            with shared_frame.lock:
                shared_frame.processed_lane = processed
                shared_frame.lane_done_event.set()

def yolo_detection_thread(shared_frame, model, stop_event):
    """Thread function for YOLO object detection"""
    yolo_fps = 0
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # Wait for a new frame
        if shared_frame.new_frame_event.wait(timeout=0.1):
            # We don't clear the event here to allow the lane thread to also process
            
            with shared_frame.lock:
                if shared_frame.frame is None:
                    continue
                frame = shared_frame.frame.copy()
            
            # Process with YOLO
            results = model.track(source=frame, conf=0.1, iou=0.5, tracker="bytetrack.yaml")
            processed = results[0].plot()
            
            # Calculate FPS
            frame_count += 1
            if (time.time() - start_time) > 1:
                yolo_fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Add FPS text
            cv2.putText(processed, f"YOLO FPS: {yolo_fps:.1f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Store the processed frame
            with shared_frame.lock:
                shared_frame.processed_yolo = processed
                shared_frame.yolo_done_event.set()

def combine_frames(lane_frame, yolo_frame):
    """Combine lane detection and YOLO detection frames"""
    if lane_frame is None or yolo_frame is None:
        return None
    
    # Use lane detection as base and overlay YOLO detections
    # Find bounding boxes and overlay them on the lane frame
    # This is a simple implementation - just shows both side by side
    combined = np.hstack((lane_frame, yolo_frame))
    return combined

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combined Lane Detection and Object Detection')
    parser.add_argument('--source', type=str, default='0', 
                      help='Source for video input. Use 0 for webcam or provide a video file path')
    parser.add_argument('--flip', action='store_true', 
                      help='Flip the input frame horizontally')
    parser.add_argument('--output', type=str, default='', 
                      help='Output file path for saving video')
    args = parser.parse_args()
    
    # Load YOLO model
    #model = YOLO("object_and_depth_detection/models/best_v5_n.pt")

    #model.export(format="ncnn")

    model = YOLO("object_and_depth_detection/models/best_v5_n_ncnn_model")
    
    # Camera calibration data
    mtx = np.array([[1.43249747e+03, 0.00000000e+00, 6.75431644e+02],
                   [0.00000000e+00, 1.43065692e+03, 4.57012583e+02],
                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist = np.array([[ 0.01302033, 0.94139111, -0.00436593, 0.00480621, -3.33015441]])
    
    # Initialize camera and lines
    camera = Camera(mtx, dist)
    left_line = Line()
    right_line = Line()
    
    # Setup video capture
    source = args.source
    try:
        source = int(source)  # Try to convert to int for camera index
    except ValueError:
        pass  # Keep as string for file path
    
    cap = cv2.VideoCapture(source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer if output is specified
    output_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(args.output, fourcc, fps, (width*2, height))
    
    # Create shared data structure and events
    shared_frame = SharedFrame()
    stop_event = threading.Event()
    
    # Start processing threads
    lane_thread = threading.Thread(
        target=lane_detection_thread, 
        args=(shared_frame, camera, left_line, right_line, stop_event)
    )
    
    yolo_thread = threading.Thread(
        target=yolo_detection_thread, 
        args=(shared_frame, model, stop_event)
    )
    
    lane_thread.daemon = True
    yolo_thread.daemon = True
    
    lane_thread.start()
    yolo_thread.start()
    
    # Main loop - capture frames and coordinate processing
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream or error reading frame")
                break
            
            # Flip frame if requested
            if args.flip:
                frame = cv2.flip(frame, 1)  # 1 for horizontal flip
            
            # Share the new frame with processing threads
            with shared_frame.lock:
                shared_frame.frame = frame
                shared_frame.new_frame_event.set()
            
            # Wait for both threads to finish processing
            both_done = shared_frame.lane_done_event.wait(timeout=0.1) and shared_frame.yolo_done_event.wait(timeout=0.1)
            
            if both_done:
                with shared_frame.lock:
                    lane_result = shared_frame.processed_lane
                    yolo_result = shared_frame.processed_yolo
                    shared_frame.lane_done_event.clear()
                    shared_frame.yolo_done_event.clear()
                
                # Combine the results
                combined_result = combine_frames(lane_result, yolo_result)
                
                if combined_result is not None:
                    # Display combined frame
                    cv2.imshow('Combined Detection', combined_result)
                    
                    # Save frame if output is specified
                    if output_writer:
                        output_writer.write(combined_result)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        stop_event.set()
        lane_thread.join(timeout=1.0)
        yolo_thread.join(timeout=1.0)
        cap.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 