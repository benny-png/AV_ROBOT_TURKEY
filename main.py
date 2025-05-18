import cv2
import numpy as np
import time
import threading
import argparse
from queue import Queue, Empty
from ultralytics import YOLO
import torch
from lane_detection_2.model.model import TwinLiteNetPlus
import torch.backends.cudnn as cudnn

class SharedFrame:
    def __init__(self):
        self.frame = None
        self.processed_lane = None
        self.processed_yolo = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.lane_done_event = threading.Event()
        self.yolo_done_event = threading.Event()

def process_lane_frame(frame, lane_model, lane_task="BOTH", device="cuda", half=True):
    """Process a single frame with TwinLiteNetPlus model for lane detection"""
    # Prepare image for the model - resize to 640x640 and normalize
    original_h, original_w = frame.shape[:2]
    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        da_seg_out, ll_seg_out = lane_model(img_tensor)
    
    result_img = frame.copy()
    
    # Create base for color mask (640x640)
    color_area_resized = np.zeros((640, 640, 3), dtype=np.uint8)
    
    if lane_task == "DA" or lane_task == "BOTH":
        # Process drivable area output
        _, da_seg_mask = torch.max(da_seg_out, 1)
        da_seg_mask = da_seg_mask.int().cpu().numpy()[0] # Shape (640, 640)
        # Drivable area in green
        color_area_resized[da_seg_mask == 1] = [0, 255, 0]
    
    if lane_task == "LL" or lane_task == "BOTH":
        # Process lane line output
        _, ll_seg_mask = torch.max(ll_seg_out, 1)
        ll_seg_mask = ll_seg_mask.int().cpu().numpy()[0] # Shape (640, 640)
        # Lane lines in red
        color_area_resized[ll_seg_mask == 1] = [255, 0, 0]
    
    # Resize color mask back to original frame size
    color_seg_original_size = cv2.resize(color_area_resized, (original_w, original_h))
    
    # Overlay the mask onto the original frame
    # Ensure we only overlay where the mask is not black
    color_mask_overlay = np.any(color_seg_original_size > [0,0,0], axis=-1)
    result_img[color_mask_overlay] = result_img[color_mask_overlay] * 0.5 + color_seg_original_size[color_mask_overlay] * 0.5
    
    return result_img.astype(np.uint8)

def lane_detection_thread(shared_frame, lane_model, lane_task, device, half, stop_event):
    """Thread function for lane detection processing using TwinLiteNetPlus"""
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
            processed = process_lane_frame(frame, lane_model, lane_task, device, half)
            
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
    parser.add_argument('--lane_task', type=str, default='BOTH', choices=['DA', 'LL', 'BOTH'],
                      help='Lane detection task: DA (Drivable Area), LL (Lane Lines), BOTH. Default: BOTH')
    args = parser.parse_args()
    
    # Set device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"  # Use half precision only with CUDA
    
    # Load YOLO model
    yolo_model = YOLO("object_and_depth_detection/models/best_v6_n.pt")

    # Load TwinLiteNetPlus model with nano configuration
    lane_model = TwinLiteNetPlus(argparse.Namespace(config="nano"))
    lane_model.to(device)
    if half:
        lane_model.half()
    
    # Load pretrained weights
    lane_model.load_state_dict(torch.load("lane_detection_2/pretrained/TwinLiteNetPlus/nano.pth"))
    lane_model.eval()
    
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
        args=(shared_frame, lane_model, args.lane_task, device, half, stop_event)
    )
    
    yolo_thread = threading.Thread(
        target=yolo_detection_thread, 
        args=(shared_frame, yolo_model, stop_event)
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