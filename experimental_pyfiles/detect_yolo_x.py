from ultralytics import YOLO
import os
import numpy as np

homedir = '/home/kgupta/data/registration_testing'
data_dir = 'x_2d_bigdataset_new_X_r3'
os.chdir(homedir)

# Load a pretrained YOLOv8n model
model = YOLO("runs/detect/yolov5_x_small/runs/train/v8_exp3_e300_correct/weights/best.pt")

with open(f'{data_dir}/full_Test.txt', 'r') as file:
    # Step 2: Read all lines into a list, stripping any trailing newline characters
    test_files = [line.strip() for line in file]

# Run inference on 'bus.jpg' with arguments
model.predict(test_files, save=True, imgsz=192, conf=0.25, iou=0.5, save_txt=True, save_conf=True, stream=True)
