from ultralytics import YOLO
import cv2
import time # Import time library for FPS calculation

# Load the model
model = YOLO(r"D:\Robotics projects\TrafficSignDetect\runs\train\traffic_signs_v1\weights\best.pt")
model.fuse()

# Initialize camera 
cap = cv2.VideoCapture(0) 
cap.set(3, 640)
cap.set(4, 480)

# Initialize variables for FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    # Formula: 1 / (Current_Time - Previous_Time)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model.predict(
        frame,
        imgsz=480,
        half=True,
        stream=True,
        verbose=False
    )

    for r in results:  
        annotated_frame = r.plot()
        
        # Display FPS text on the frame
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)