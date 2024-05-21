import cv2
import ffmpegcv
from tqdm.auto import tqdm
from ultralytics import YOLO
from tqdm.auto import tqdm

img_path = 'data/Swiss_vs_Slovakia-panoramic_video.mp4'
model_path = 'models/yolov8m_goalkeeper_1280.pt'

def get_crops(img, crop_width=1280, offset=0):
    """
    Generate crops from an image.
    Args:
        img (numpy.ndarray): The input image.
        crop_width (int, optional): The width of each crop. Defaults to 1280.
        offset (int, optional): The offset for each crop. Defaults to 0.
    Returns:
        Tuple[List[numpy.ndarray], List[int]]: A tuple containing a list of 
        cropped images and a list of corresponding coordinate offsets.
    """
    orig_width = img.shape[1]
    crops = []
    coord_offsets = []
    for c_idx in range(0, orig_width, crop_width):
        w_start = max(0, c_idx - offset)
        w_end = min(c_idx + crop_width, orig_width)
        if w_end <= w_start:
            continue
        crop = img[:, w_start:w_end]
        coord_offsets.append(w_start)
        crops.append(crop)

    return crops, coord_offsets

model = YOLO(model_path)
cap = cv2.VideoCapture(img_path)

with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame2', frame)
        crops, coord_offsets = get_crops(frame)
        
        for crop, coord_offset in zip(crops, coord_offsets):
            results = model(crop)
            for result in results:
                if len(result.boxes) != 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1 = x1.detach().cpu().numpy()
                        y1 = y1.detach().cpu().numpy()
                        x2 = x2.detach().cpu().numpy()
                        y2 = y2.detach().cpu().numpy()
                        x1 += coord_offset
                        x2 += coord_offset
                        
                        print(x1, y1, x2, y2)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.imshow('frame', frame)

cap.release()