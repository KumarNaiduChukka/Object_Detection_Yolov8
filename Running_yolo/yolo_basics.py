<<<<<<< HEAD
from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_WEIGHTS/yolov8l.pt')
results = model("Images/3rd.jpg", show=True)
=======
from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_WEIGHTS/yolov8l.pt')
results = model("Images/3rd.jpg", show=True)
>>>>>>> 3950141e3a4e5b602793016cc6d72a2229629cf6
cv2.waitKey(0)