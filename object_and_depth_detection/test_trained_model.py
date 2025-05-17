from ultralytics import YOLO
import cv2
import time

# Load the model

model = YOLO("object_and_depth_detection/models/best_v5_n.pt")  # Load a custom trained model

# Set input source - 0 for default webcam
cap = cv2.VideoCapture("object_and_depth_detection/125mm mini portable traffic light.mp4")

# Optional: Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Process frames
frame_count = 0
start_time = time.time()
fps = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Calculate FPS
    frame_count += 1
    if (time.time() - start_time) > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
    
    # Process every frame (or skip frames by using frame_count % 2 == 0 for every other frame)
    results = model.track(source=frame, conf=0.4, iou=0.5, tracker="bytetrack.yaml")
    
    # Get the annotated frame
    annotated_frame = results[0].plot()
    
    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the annotated frame
    cv2.imshow("YOLO Detection", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()