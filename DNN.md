# DNN

```python
import cv2
import time
import numpy as np

with open("./models/object_detection_classes_coco.txt", 'r') as f:
    class_names = f.read().split('\n')
print(class_names)

# Get a different colors for each of the classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv2.dnn.readNet(model='./models/frozen_inference_graph.pb',
                        config='./models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')
# Set backend and target
model.setPerferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPerferableTarget(cv2.dnn.DNN_TARGET_CUDA)
min_confidence_score = 0.4
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    imgHeight, imgWidth, channels = img.shape
    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104, 117, 123), swapRB=True)

    start = time.time()

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()

    end = time.time()
    fps = 1 / (end - start)
    # Run over each of the detections
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > min_confidence_score:
            class_id = detection[1]
            class_name = class_names[int(class_id) - 1]
            color = colors(int(class_id))

            bboxX = detection[3] * imgWidth
            bboxY = detection[4] * imgHeight
            bboxWidth = detection[5] * imgWidth
            bboxHeight = detection[6] * imgHeight

            cv2.rectangle(img, (int(bboxX), int(bboxY)), (int(bboxWidth), int(bboxHeight)), thickness=2, color=color)
            cv2.putText(img, class_name, (int(bboxX), int(bboxY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show FPS
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

