from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_WEIGHTS/yolov8l.pt')
results = model("Images/3rd.jpg", show=True)
cv2.waitKey(0)