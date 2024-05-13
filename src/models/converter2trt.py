from ultralytics import YOLO


model = YOLO('models/yolov8m_goalkeeper_1280.pt')

model.export(format='engine')