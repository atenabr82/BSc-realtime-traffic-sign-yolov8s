from ultralytics import YOLO
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YOLO(r"D:\Robotics projects\TrafficSignDetect\runs\train\traffic_signs_v1\weights\last.pt")
model.to(device)

model.train(
    data=r"D:\Robotics projects\TrafficSignDetect\roboflow_traffic_signs\data.yaml",
    epochs=100,       
    resume=True
)
