from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8m.pt") 

# Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the dataSet directory
dataSet_directory = os.path.join(script_directory, 'dataSet')
data_yaml_path = os.path.join(dataSet_directory, 'data.yaml')

model.train(data = data_yaml_path, epochs=100, patience=30, batch=32, imgsz=416)
print(f"train finish")

# result = model.predict("./yoloTest.png", save=True, conf=0.5)
# print(f"predict finish Result : {result}")