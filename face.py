from PIL import Image 
import cv2
import torch
import os 
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import pyttsx3
import time

mtcnn = MTCNN(keep_all= False)
model = InceptionResnetV1(pretrained='vggface2').eval()

print('started')
name = 'unknown'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not accessible.")
    exit()

dic = []


pyttsx3.speak("testing")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))

recording = True
starting_time = time.time()

while recording:

    

    pyttsx3.speak("Enter e to enroll a face and r to compare two and q to quit: \n")
    key2 = input()
    ret, frame = cap.read()
    if ret is None:
        break
    height, width = frame.shape[:2]
    print(f"Using resolution: {width}x{height}")

    dic.append(frame)
    out.write(frame)


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)
    
    face = mtcnn(img)

    if face is not None:
        cv2.putText(frame, "Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        pyttsx3.speak("face is none exitting:\n")
    
    cv2.imshow('enroll face', frame)
    key = cv2.waitKey(1) & 0xFF
    

    key = key2

    if key == 'e' and face is not None:

        embedding = model(face.unsqueeze(0)).detach()

        os.makedirs('data',exist_ok=True)
        torch.save(embedding,'data/face_detec.pt')
        pyttsx3.speak("the face has been enrolled")
        continue

    if key == 'q':
        recording = False
        break



    if key == 'r':
        exis, img_frame = cap.read()
        if exis is None:
            break
        rgb2 = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

        imgR = Image.fromarray(rgb2)

        faceR = mtcnn(imgR)
        
        if faceR is not None:
            cv2.putText(img_frame, "Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('enrolled another face', img_frame)
        else:
            pyttsx3.speak("the face to be compared was not a face exiting:\n")
            break
        
        if faceR is not None:
            
            new_embedding = model(faceR.unsqueeze(0)).detach()

            torch.save(new_embedding, 'data/face_detec.pt2')


        saved_embedding = torch.load('data/face_detec.pt')

        similarity = F.cosine_similarity(saved_embedding, new_embedding)

        if similarity >= 0.75:
            if face is None:
                print("face is none\n {similarity}")
                continue
            pyttsx3.speak('it is a match congratulation')
        else:
            pyttsx3.speak('no match, recognized\n')
out.release()

cap.release()
cv2.destroyAllWindows()

