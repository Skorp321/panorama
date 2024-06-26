from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8m_keypoints.pt")
img = cv2.imread("data/Panoramic_video_football.mp4")
res = model.predict(img, imgsz=1280)

points = res[0].keypoints.xy[0].cpu().numpy()
#image_with_keypoints = cv2.drawKeypoints(img, points, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
for h_point in points:
    x, y = int(h_point[0]), int(h_point[1])
    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

cv2.imwrite('image.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()