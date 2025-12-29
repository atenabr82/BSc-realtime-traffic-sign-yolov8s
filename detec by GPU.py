from ultralytics import YOLO
import cv2
import torch
import time

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(r"D:\Robotics projects\TrafficSignDetect\runs\train\traffic_signs_v1\weights\best.pt")
model.to(device)
model.fuse()

# video capture setup
cap = cv2.VideoCapture(0)

# Variables to calculate FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check camera index.")
        break

    # Start timer for FPS calculation
    curr_time = time.time()
    
    results = model.predict(
        frame,
        imgsz=480,
        half=True, 
        stream=True,
        verbose=False
    )

    for r in results:  
        annotated_frame = r.plot()
        
        # Calculate FPS
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO GTX 1650 Acceleration", annotated_frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()