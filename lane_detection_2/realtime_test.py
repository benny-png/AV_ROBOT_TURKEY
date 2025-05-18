import cv2
import torch
import argparse
import numpy as np
import time
from model.model import TwinLiteNetPlus # Assuming model.py is in the same directory or accessible

def process_frame_for_model(frame, device, img_size=640, half=False):
    """Prepares a single frame for TwinLiteNetPlus model input."""
    img_resized = cv2.resize(frame, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def visualize_output(original_frame, da_seg_out, ll_seg_out, lane_task="BOTH"):
    """Overlays drivable area and/or lane lines on the original frame."""
    original_h, original_w = original_frame.shape[:2]
    # Correctly get the spatial dimensions (height, width) of the model's output mask
    # da_seg_out has shape (batch_size, num_classes, height, width)
    # So, height is shape[2] and width is shape[3]
    output_h, output_w = da_seg_out.shape[2], da_seg_out.shape[3]

    result_img = original_frame.copy()
    
    # Create base for color mask (model output size)
    color_area_model_size = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    if lane_task == "DA" or lane_task == "BOTH":
        _, da_seg_mask = torch.max(da_seg_out, 1)
        da_seg_mask_np = da_seg_mask.int().cpu().numpy()[0] # Shape (output_h, output_w)
        color_area_model_size[da_seg_mask_np == 1] = [0, 255, 0] # Green for drivable area

    if lane_task == "LL" or lane_task == "BOTH":
        _, ll_seg_mask = torch.max(ll_seg_out, 1)
        ll_seg_mask_np = ll_seg_mask.int().cpu().numpy()[0] # Shape (output_h, output_w)
        # For lane lines, ensure they are distinctly visible, even if DA is also shown
        color_area_model_size[ll_seg_mask_np == 1] = [255, 0, 0] # Red for lane lines
        
    # Resize color mask back to original frame size for overlay
    color_seg_original_size = cv2.resize(color_area_model_size, (original_w, original_h))
    
    # Overlay the mask onto the original frame
    # Create a mask for non-black pixels in the overlay
    overlay_mask = np.any(color_seg_original_size > [0, 0, 0], axis=-1)
    result_img[overlay_mask] = cv2.addWeighted(
        result_img[overlay_mask], 0.5, 
        color_seg_original_size[overlay_mask], 0.5, 0
    )
    
    return result_img.astype(np.uint8)

def main(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    half = device_str == "cuda"  # Use half precision only with CUDA

    print(f"Using device: {device_str}")

    # Load TwinLiteNetPlus model
    # The 'config' argument for TwinLiteNetPlus is an object, not just a string.
    # We'll use argparse.Namespace to mimic this for the model constructor.
    model_args = argparse.Namespace(config=args.config)
    model = TwinLiteNetPlus(model_args)
    
    print(f"Loading model weights from: {args.weight}")
    try:
        model.load_state_dict(torch.load(args.weight, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the --weight path is correct and the model config matches the weights.")
        return

    model.to(device)
    if half:
        model.half()
    model.eval()

    # Setup video capture
    cap_source = args.source
    try:
        cap_source = int(cap_source)
    except ValueError:
        pass # Keep as string if it's a file path, though this script focuses on camera
    
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {cap_source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Optional: Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Optional: Set camera resolution

    fps = 0
    frame_count = 0
    start_time = time.time()

    print("Starting real-time lane detection. Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            original_frame_for_viz = frame.copy() # Keep a copy for visualization

            # Preprocess frame for the model
            input_tensor = process_frame_for_model(frame, device, args.img_size, half)

            # Run inference
            da_seg_out, ll_seg_out = model(input_tensor)
            
            # Visualize output
            processed_frame = visualize_output(original_frame_for_viz, da_seg_out, ll_seg_out, args.lane_task)

            # Calculate FPS
            frame_count += 1
            if (time.time() - start_time) >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, f"Task: {args.lane_task}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


            cv2.imshow('Real-time Lane Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Lane detection stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Lane Detection Test Script for TwinLiteNetPlus')
    parser.add_argument('--weight', type=str, default='pretrained/TwinLiteNetPlus/nano.pth', 
                        help='Path to model weights file (default: pretrained/TwinLiteNetPlus/nano.pth)')
    parser.add_argument('--config', type=str, default='nano', choices=['nano', 'small', 'medium', 'large'],
                        help='Model configuration type (default: nano). Must match the weights.')
    parser.add_argument('--source', type=str, default='0', 
                        help='Video source (camera index or video file path, default: 0 for webcam)')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Inference image size (pixels, default: 640)')
    parser.add_argument('--lane_task', type=str, default='BOTH', choices=['DA', 'LL', 'BOTH'],
                      help='Lane detection task to visualize: DA (Drivable Area), LL (Lane Lines), BOTH. Default: BOTH')
    
    args = parser.parse_args()
    main(args) 