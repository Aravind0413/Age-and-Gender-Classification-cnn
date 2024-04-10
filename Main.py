from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import time
import cv2
import tensorflow as tf
from collections import namedtuple
from collections import defaultdict
from io import StringIO
from PIL import Image
import numpy as np
import winsound
import cv2 as cv
import math
import time
import argparse
main = tkinter.Tk()
main.title("Age Gender Classification Using CNN")
main.geometry("1000x600")

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

global filename
global detectionGraph
global msg


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


def loadModel():
    global detectionGraph
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')
    
    messagebox.showinfo("Training model loaded","Training model loaded")


   
def uploadVideo():
    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(0)
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
            cv.imwrite("age-gender-vido.jpg",frameFace)
        if cv.waitKey(1) & 0xFF == ord('q'):            
            exit()
    cap.release()
    cv.destroyAllWindows()
    print("time : {:.3f}".format(time.time() - t))
        


def uploadimage():
    global msg
    msg = ''
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    print('Reading from webcam.') 
    cap = cv.VideoCapture(filename)
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
            cv.imwrite("age-gender-out.jpg",frameFace)
        if cv2.waitKey(1) & 0xFF == ord('q'):            
            exit()
            
    cap.release()
    cv.destroyAllWindows()
    print("time : {:.3f}".format(time.time() - t))
        


def exit():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Age Gender Classification Using CNN')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Load & Generate CNN Model", command=loadModel)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

webcamButton = Button(main, text="Live Detection", command=uploadVideo)
webcamButton.place(x=50,y=150)
webcamButton.config(font=font1)

webcamButton = Button(main, text="Detect From Image", command=uploadimage)
webcamButton.place(x=50,y=200)
webcamButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=330,y=250)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
