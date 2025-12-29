from ultralytics import YOLO
import cv2

model = YOLO(r"D:\Robotics projects\TrafficSignDetect\runs\train\traffic_signs_v1\weights\best.pt")
model.fuse()

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=480,
        half=True,
        stream=True,
        verbose=False
    )

    for r in results:  
        annotated_frame = r.plot()

        cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
