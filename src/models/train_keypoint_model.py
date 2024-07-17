from ultralytics import YOLO

path_to_model = "yolov8m-pose.pt"

model = YOLO(path_to_model)

model.train(
    data="data/interim/keypoints/data.yaml",
    epochs=30,
    imgsz=1280,
    device=0,
    batch=4,
    close_mosaic=0,
)
