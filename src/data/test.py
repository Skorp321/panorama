import cv2
import ffmpegcv
from ultralytics import YOLO

vid_path = "/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4"
model_path = "/container_dir/models/yolov8m_goalkeeper_1280.pt"

cap = ffmpegcv.VideoCaptureNV(vid_path)
# cap = cv2.VideoCapture(vid_path)
model = YOLO(model_path)

while True:
    ret, frame = cap.read()
    if ret:
        print("proccesed")
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes.xyxy.detach().cpu().numpy():
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("frame", frame)
                cv2.waitKey(1)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        print("not worck(")
        break

cap.release()
