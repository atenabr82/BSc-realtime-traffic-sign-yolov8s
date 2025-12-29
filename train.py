from ultralytics import YOLO
import torch

# Select GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def run():
    torch.multiprocessing.freeze_support()

    # Using YOLOv8s pre-trained weights
    model = YOLO("yolov8s.pt").to(device)

    model.train(
        data=r"D:\Robotics projects\TrafficSignDetect\roboflow_traffic_signs\data.yaml",

        epochs=100,         # Best range for a dataset with ~14k images
        imgsz=640,
        batch=8,            # Optimal for GTX 1650
        workers=4,
        device=0,

        # Optimization settings (AdamW performs better than SGD here)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=20,        # Prevents overfitting with early stopping

        # augmentations for traffic sign datasets
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.6,
        mixup=0.05,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,

        cache=True,         # Speeds up training by caching images
        project="runs/train",
        name="traffic_signs_v1",
        exist_ok=True
    )

    # Validate model after training
    model.val()

if __name__ == "__main__":
    run()
