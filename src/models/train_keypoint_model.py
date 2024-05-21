from ultralytics import YOLO

path_to_model = 'yolov8m-pose.pt'

model = YOLO(path_to_model)

model.train(data="data/interim/keypoints/data.yaml", epochs=100, imgsz=640, device=0, batch=16)