---
layout: default
title: Face, Hands, and 3D Detection
nav_order: 11
permalink: /face-detection/
description: Real-time face detection, face mesh landmarks, hand tracking, and objectron with MediaPipe.
---

```python
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results = face_detection.process(image)

        # Convert image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)
                print(id, detection)

                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("Face Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

```

## Land Marks

```python
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False

        # Process the image and find faces
        results = face_mesh.process(image)

        image.flags.writeable = True

        # Convert image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # print(face_landmarks)
                mp_draw.draw_landmarks(image, landmark_list=face_landmarks,
                                       connections=mp_face_mesh.FACEMESH_CONTOURS,
                                       landmark_drawing_spec=drawing_spec,
                                       connection_drawing_spec=drawing_spec)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("Face Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
```

## Hand Detection

```python
import cv2
import mediapipe as mp
import time

mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False

        # Process the image and find faces
        results = hands.process(image)

        image.flags.writeable = True

        # Convert image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print(hand_landmarks)
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("Face Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
```

## 3D

```python
import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name="Cup") as objectron:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False

        # Process the image and find faces
        results = objectron.process(image)

        image.flags.writeable = True

        # Convert image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                print(detected_object)
                mp_draw.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_draw.draw_axis(image, detected_object.rotation, detected_object.translation)
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("Objection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
```

