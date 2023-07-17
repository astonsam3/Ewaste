# !pip install opencv-python torch torchvision

# !git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5/

import cv2
import torch
import numpy as np
from PIL import Image

# Import YOLOv5 modules
from IPython.display import display
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import colors
from utils.torch_utils import select_device

# Define the path to your trained YOLOv5 model weights file
weights = 'E:\OneDrive\Documents\lekha/best.pt'

# Load YOLOv5 model
device = select_device('')
model = attempt_load(weights)
stride = int(model.stride.max())  # model stride
model = model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera index

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if frame is valid
    if not ret:
        break

    # Perform object detection on the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((640, 640))
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

    # Process the predictions
    if pred is not None and len(pred):
        img_h, img_w, _ = frame.shape

        # Display bounding boxes and labels
        for det in pred:
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1 = int(x1 * img_w / 640)
            y1 = int(y1 * img_h / 640)
            x2 = int(x2 * img_w / 640)
            y2 = int(y2 * img_h / 640)

            color = colors(int(cls))
            label = f'{int(cls)} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame with bounding boxes
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()