import cv2
import math
import argparse
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def detectAgeGender(image_path):
    frame = cv2.imread(image_path)
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        return frame

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    return resultImg

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_image = detectAgeGender(file_path)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)
        result_image.thumbnail((800, 600))
        imgtk = ImageTk.PhotoImage(image=result_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-32)', '(30-40)', '(40-50)', '(50-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20

root = tk.Tk()
root.title("Age and Gender Detection")

# Load background image
bg_image = Image.open("crowd.jpg")  # Change "crowd.jpg" to your image file path
bg_image = ImageTk.PhotoImage(bg_image)

# Create a canvas for background image
canvas = tk.Canvas(root, width=bg_image.width(), height=bg_image.height())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor="nw", image=bg_image)

# Create a frame for better organization
frame = tk.Frame(canvas, bg="white", padx=10, pady=10)
frame.place(relx=0.5, rely=0.5, anchor="center")

# Create a label for the image display
image_label = tk.Label(frame, text="Image Display Section", bg="white", font=("Times New Roman", 24, "bold"))
image_label.pack(pady=5)

panel = tk.Label(frame)
panel.pack(padx=10, pady=10)

# Add a button with custom styling
btn = tk.Button(frame, text="Select Image", command=select_image, bg="#007bff", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
btn.pack(pady=5)

root.mainloop()
