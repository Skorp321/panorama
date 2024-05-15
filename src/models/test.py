import cv2
import ffmpegcv
from tqdm.auto import tqdm

img_path = '/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4'

cap = ffmpegcv.VideoCaptureNV(img_path)
ret, frame = cap.read()
print(frame)

cap.release()