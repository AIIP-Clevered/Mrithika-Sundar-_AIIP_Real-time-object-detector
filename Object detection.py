import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
net = cv2.dnn.readNet("C:/Users/mrith/Downloads/yolov4.weights", "C:/Users/mrith/Downloads/yolov4.cfg")
# loading yolo
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("C:/Users/Mani Mehrotra/Downloads/coco (1).names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
cap = None
running = False
def start_detection():
    global cap, running
    if not running:
        running = True
        cap = cv2.VideoCapture(0)
        # Run detect_objects() in a separate thread
        detection_thread = Thread(target=detect_objects)
        detection_thread.daemon = True  # Ensures the thread will close when the main program exits
        detection_thread.start()
def stop_detection():
    global cap, running
    running = False  # Stop the while loop in `detect_objects()`
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  
def detect_objects():
    global cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0) # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_detection()
    
root = tk.Tk()
root.title("Object Detection Interface")
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)
root.protocol("WM_DELETE_WINDOW", lambda: (stop_detection(), root.destroy()))
root.mainloop()
