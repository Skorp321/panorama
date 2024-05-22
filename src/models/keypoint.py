import cv2
from ultralytics import YOLO
import numpy as np
import os

cap = cv2.VideoCapture('/home/skorp321/Projects/panorama/data/Swiss_vs_Slovakia-panoramic_video.mp4')

ret, frame = cap.read()
cap.release()

model = YOLO('/home/skorp321/Projects/panorama/runs/pose/train8/weights/best.pt')

h_mat =  np.load('/home/skorp321/Projects/panorama/data/h_matrix_path.npy')
print(h_mat)
outputs = model(frame, save=True)[0]

print(outputs.keypoints.data)

