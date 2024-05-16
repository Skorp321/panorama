from ultralytics import YOLO

model_path = '/container_dir/models/yolov8m_goalkeeper_1280.pt'

model = YOLO(model_path)

model.export(format='engine', imgsz=1280, half=True, batch=3)