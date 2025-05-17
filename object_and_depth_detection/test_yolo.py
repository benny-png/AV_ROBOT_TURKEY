from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on the source
results = model(source=0, stream=True)  # generator of Results objects

# Display results
for r in results:
    frame = r.orig_img  # Original frame
    annotated_frame = r.plot()  # Annotated frame with detections
    
    # Display the annotated frame
    cv2.imshow("YOLO Detection", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()