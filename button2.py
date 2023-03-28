import pyttsx3
engine = pyttsx3.init()
voice = engine.getProperty('voices')
engine.setProperty('voice', ) #changing index changes voices but ony 0 and 1 are working here
engine.say("Loading")
engine.runAndWait()

import RPi.GPIO as GPIO
import time
import cv2
import face_recognition
import numpy as np
import os


BUTTON_PIN = 16
BUTTON_PIN2 = 21
Shut_DownPin = 12


GPIO.setmode(GPIO.BCM)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(Shut_DownPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


prev_button_state = GPIO.input(BUTTON_PIN)
prev_button_state2 = GPIO.input(BUTTON_PIN2)
prev_button_shut=GPIO.input(Shut_DownPin)



language='en'
classNames = []
classFile ="/home/cap/Desktop/coco.names"
with open(classFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')
  
configPath="/home/cap/Desktop/ssd_mobilenet_v3_large_coco_2022_01_14.pbtxt"
weightpath ="/home/cap/Desktop/frozen_inference_graph.pb"

# Load a sample picture and learn how to recognize it.
doaa_image = face_recognition.load_image_file("/home/cap/Desktop/doaa.jpg")
doaa_face_encoding = face_recognition.face_encodings(doaa_image)[0]
engine = pyttsx3.init()
engine.say("Loading Please Wait")
engine.runAndWait()
# Load a second sample picture and learn how to recognize it.
merihan_image = face_recognition.load_image_file("/home/cap/Desktop/merihan.jpeg")
merihan_face_encoding = face_recognition.face_encodings(merihan_image)[0]

kamel_image = face_recognition.load_image_file("/home/cap/Desktop/kamel.jpg")
kamel_face_encoding = face_recognition.face_encodings(kamel_image)[0]

darwish_image = face_recognition.load_image_file("/home/cap/Desktop/darwish.jpg")
darwish_face_encoding = face_recognition.face_encodings(darwish_image)[0]

toqa_image = face_recognition.load_image_file("/home/cap/Desktop/toqa.jpg")
toqa_face_encoding = face_recognition.face_encodings(toqa_image)[0]

dr_ali_image = face_recognition.load_image_file("/home/cap/Desktop/DR.ALI3.jpeg")
dr_ali_face_encoding = face_recognition.face_encodings(dr_ali_image)[0]
engine = pyttsx3.init()
engine = pyttsx3.init()
engine.say("Loading Please Wait")
engine.runAndWait()

mohsen_image = face_recognition.load_image_file("/home/cap/Desktop/mohsen.jpg")
mohsen_face_encoding = face_recognition.face_encodings(mohsen_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    doaa_face_encoding,
    merihan_face_encoding,
    kamel_face_encoding,
    darwish_face_encoding,
    toqa_face_encoding,
    dr_ali_face_encoding,
    mohsen_face_encoding
]
known_face_names = [
    "doaa",
    "merihan",
    "Doctor Mostafa Kamel",
    "Doctor Mohamed Darwish",
    "toqa",
    "Doctor ali hossein",
    "mohsen"
]
engine = pyttsx3.init()
engine.say("Device is activated")
engine.runAndWait()

def Shutdown():
    engine = pyttsx3.init()
    engine.say("Shutting down now")
    engine.runAndWait()
    os.system("sudo shutdown -h now")
    

def RecontionMode():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        video_capture = cv2.VideoCapture(0)
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

               
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        if (len(face_names)==0):
            engine=pyttsx3.init()
            engine.say("Can't detect anyone.")
            engine.runAndWait()
        else:    
            # Display the results
            for name in (face_names):
               
                engine=pyttsx3.init()
                engine.say(name)
                engine.runAndWait()
        break
        
        
    video_capture.release()


def ProcessGreyScaleForPhoto(path):
    #read image and process it to grey scale
    image = cv2.imread(path)
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    classIds, confs,d= net.detect(cv2image, confThreshold=.6) 
    return classIds,confs,cv2image

def CapturePhoto():
    path=""
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    if result:
        path="/home/cap/Desktop/image.jpg"
        cv2.imwrite(path, image)
        cam.release()
        return path


def FindPersons(classIds,confs):
    i=0
    numbers=""
    if len(classIds) !=0:
        for classId, confidence in zip(classIds.flatten(),confs.flatten()):
           if(classNames[classId-1]=='person'):
               i=i+1
    if(i>0):
        if(i>1):
            numbers="There are "+str(i)+" persons"
      
        elif (i==1):
            numbers="There is one person"  
    if(i==0):
            numbers="No One is here"
    engine = pyttsx3.init()
    engine.say(numbers)
    engine.runAndWait()

try:
    while True:
        time.sleep(0.01)
        button_state = GPIO.input(BUTTON_PIN)
        button_state2 = GPIO.input(BUTTON_PIN2)
        button_shut=  GPIO.input(Shut_DownPin)

        if button_state != prev_button_state:
            prev_button_state = button_state
            if button_state == GPIO.HIGH:
                   engine = pyttsx3.init()
                   engine.say("Counting mode is activating")
                   engine.runAndWait()
                  #detect model  
                   net=cv2.dnn_DetectionModel(weightpath,configPath)
                   net.setInputSize (320 , 230)
                   net.setInputScale(1.0 / 127.5)
                   net.setInputMean((127.5, 127.5, 127.5))
                   net.setInputSwapRB(True)
                   imgPath=CapturePhoto()
                   classIds,confs,img=ProcessGreyScaleForPhoto(imgPath)
                   FindPersons(classIds,confs)
        
        if button_state2 != prev_button_state2:
            prev_button_state2 = button_state2
            if button_state2 == GPIO.HIGH:
                engine = pyttsx3.init()
                engine.say("Recognition mode is activating")
                engine.runAndWait()
                RecontionMode()
        if button_shut != prev_button_shut:
            prev_button_shut = button_shut
            if button_shut == GPIO.HIGH:
                Shutdown()      
                
                
except KeyboardInterrupt:
    GPIO.cleanup()
    engine = pyttsx3.init()
    engine.say("Something not expected happened")
    engine.runAndWait()
    



