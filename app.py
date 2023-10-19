from flask import Flask, render_template, request, Response
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import math
import time
import pygame




# Path to the alarm sound
path_alarm = "weights/alarm.wav"

# Initializing pygame
pygame.init()

# Loading the alarm sound
pygame.mixer.music.load(path_alarm)

# Function to play the alarm sound
def play_alarm_sound():
    try:
        # Check if music is not already playing
        if not pygame.mixer.music.get_busy():
            # Play the alarm sound
            pygame.mixer.music.play()

            # Allow the sound to play for a few seconds (adjust as needed)
            time.sleep(5)
    except Exception as e:
        print(f'Error playing alarm sound: {str(e)}')























model=YOLO("weights/best.pt")

# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]


classNames = ["GUN","GUN"]

app = Flask(__name__)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

# def process_frame(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     _, encoded_image = cv2.imencode('.jpg', frame)
#     return encoded_image.tobytes()


def process_frame(frame):
    results=model(frame,stream=True)
    if results:
        play_alarm_sound()

        
       

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            #print(x1, y1, x2, y2)
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
            #print(box.conf[0])
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]
            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            #print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)



    _, encoded_image = cv2.imencode('.jpeg', frame)
    return encoded_image.tobytes()





@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    frame_data = request.files['frame'].read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame)
    return Response(response=processed_frame, content_type='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
