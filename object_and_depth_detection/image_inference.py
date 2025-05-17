from ultralytics import YOLO
import cv2
import argparse
import os

def run_inference(model_path, image_path, save_output=False, output_dir="outputs"):
    # Load the model
    model = YOLO(model_path)
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run inference
    results = model(image, conf=0.4)
    
    # Process results
    annotated_image = results[0].plot()
    
    # Display detection info
    detections = results[0].boxes
    print(f"Found {len(detections)} objects")
    
    for i, detection in enumerate(detections):
        cls = int(detection.cls)
        conf = float(detection.conf)
        cls_name = results[0].names[cls]
        print(f"Detection {i+1}: {cls_name}, Confidence: {conf:.2f}")
    
    # Save output if requested
    if save_output:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{base_filename}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image to {output_path}")
    
    # Display the image
    cv2.imshow("YOLO Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image")
    parser.add_argument("--model", type=str, default="best_v2_n.pt", help="Path to the YOLO model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--save", action="store_true", help="Save the annotated image")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save output images")
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(args.model, args.image, args.save, args.output_dir) 