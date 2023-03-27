#All the imports go here
import numpy as np
import cv2
import winsound

#Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('eyes_cascade.xml')

#Variable store execution state
first_read = False

#Starting the video capture
cap = cv2.VideoCapture(1)
ret,img = cap.read()
cnt = 0
alerte = False
while(ret):
    ret,img = cap.read()
    #Coverting the recorded image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray,5,1,1)
    img2 = img.copy()
    #Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            #roi_face is face which is input to eye classifier
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            img2 = img
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
            for (x1,y1,w1,h1) in eyes:
                img2 = cv2.rectangle(roi_face_clr,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
            #Examining the length of eyes object for eyes
            if(len(eyes)>=2):
                #Check if program is running for detection    
                #cv2.sputText(img, "Eyes open!", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
                cnt = 0
            else:
                #print(cnt)
                #cv2.putText(img, "open your eyes!", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
                cnt += 1
                if(cnt >=27):
                    winsound.PlaySound("alerta.wav",winsound.SND_ASYNC)
                    print("1")
                    print("alarme declenche")
                    cnt = 0
                 
            
    else:
        cv2.putText(img,"No face detected",(100,100),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    #Controlling the algorithm with keys
    cv2.imshow('img',img)
    cv2.imshow('img2',img2)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()