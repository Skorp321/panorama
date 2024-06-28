import pandas as pd
import numpy as np
import json
from ultralytics import YOLO
import cv2

file_path = "C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\data\\cvat\\Panoramic_video_football\\annotations\\person_keypoints_default.json"
img_path = "C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\data\\img2.jpg"
field_path = "C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\data\\field\\annotations\\person_keypoints_default.json"
field_img_pth = 'C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\data\\field.png'
detektor_path = 'C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\models\\yolov8m_goalkeeper.pt'

data = pd.DataFrame()

model = YOLO(
    "C:\\Users\\PotapovS\\Documents\\Projects\\panorama\\models\\yolov8m_keypoints.pt"
)
det_model = YOLO(detektor_path)

det = det_model(img_path, imgsz=1280)
res = model(img_path, imgsz=1280)

boxes = det[0].boxes.xywh.cpu().numpy()
keypoints = res[0].keypoints.data[0].cpu().numpy()

boxes = pd.DataFrame(boxes)
data['x'] = boxes[0] 
data['y']  = boxes[1] + boxes[3] / 2
data["x"] = data["x"].astype(int)
data["y"] = data["y"].astype(int)

with open(file_path) as f:
    keypoints_map_pos = json.load(f)

with open(field_path) as file:
    keypoints_field = json.load(file)

names = keypoints_map_pos["categories"][0]["keypoints"]
field_points = np.array(keypoints_field["annotations"][0]["keypoints"]).reshape(-1, 3)

field_df = pd.DataFrame(
    field_points,
    index=names,
    columns=["x", "y", "viz"],
)

df = pd.DataFrame(keypoints, index=names, columns=["x", "y", "conf"])
df_n = df[df["conf"] >= 0.75]
field_df_filtered = field_df[field_df.index.isin(df_n.index)]

img = cv2.imread(img_path)
img_field = cv2.imread(field_img_pth)

homog, mask = cv2.findHomography(df_n.values, field_df_filtered.values)   

pred_dst_pts = []                                                  # Initialize players tactical map coordiantes list
for pt in data.values:                                             # Loop over players frame coordiantes
    pt = np.append(np.array(pt), np.array([1]), axis=0)            # Covert to homogeneous coordiantes
    dest_point = np.matmul(homog, pt)                              # Apply homography transofrmation
    dest_point = 2*dest_point/dest_point[2]                        # Revert to 2D-coordiantes
    pred_dst_pts.append(dest_point[:2])                            # Update players tactical map coordiantes list
pred_dst_pts = np.array(pred_dst_pts)
    
for item in boxes.values:
    cv2.rectangle(img, (int(item[0]-item[2]/2), int(item[1]-item[3]/2)), (int(item[0]+item[2]/2), int(item[1]+item[3]/2)), (255,255,255), 1)
    
for item in pred_dst_pts:
    cv2.circle(img_field, (int(item[0]), int(item[1])), 6, (0, 0, 255), -(1))

cv2.imshow('field', img_field)
cv2.imshow("image", img)
cv2.waitKey(0)
