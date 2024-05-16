import cv2
import ffmpegcv
from tqdm.auto import tqdm
from ultralytics import YOLO
from tqdm.auto import tqdm

img_path = '/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4'
model_path = '/container_dir/models/yolov8m_goalkeeper_1280.engine'

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
cap = ffmpegcv.VideoCaptureNV(img_path)

for frame in tqdm(cap):
    images, _ = get_crops(frame)
    output = model.predict(images, save=True, imgsz=1280, half=True, stream_buffer=True)
    #print(output[0])

cap.release()