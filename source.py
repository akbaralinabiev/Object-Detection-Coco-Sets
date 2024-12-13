import streamlit as st
import os
import cv2
import numpy as np
import imutils

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

st.title("Object Detection Project for Deep Learning Class")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
if uploaded_file is not None:
    if not os.path.exists("temp"):
        os.makedirs("temp")

    temp_file_path = os.path.join("temp", uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    video = cv2.VideoCapture(temp_file_path)

    if not video.isOpened():
        st.error("Error: Could not open video file.")
    else:
        stframe = st.empty()

        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame = imutils.resize(frame, height=500)
            height, width = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
            net.setInput(blob)

            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

            outs = net.forward(output_layers)

            confidences = []
            boxes = []
            classIds = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = classes[classIds[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        video.release()
        os.remove(temp_file_path)
