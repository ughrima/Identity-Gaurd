import cv2
import face_recognition
import pickle
import numpy as np
import cvzone
import datetime
from datetime import datetime

print("loading encoded file...")
file=open("encodefile.p","rb")
encodelistknownwithids=pickle.load(file)
file.close()
encodelistknown,ids=encodelistknownwithids
# print(ids)
print("encoded file loaded")

def markattendance(name):
   now=datetime.now()
   current_date =now.strftime("%d-%m-%Y")
   with open(current_date+".csv","w+") as f:
      datalist=f.readlines()
      namelist=[]
      for line in datalist:
         entry=line.split(',')
         namelist.append(entry[0])
      if name not in namelist:
         now=datetime.now()
         time=now.strftime("%H:%M:%S")
         f.write(f'\n{name},{time}')

#save the data in dataframe then csv

data=cv2.CascadeClassifier("harrfile.xml")

webcam=cv2.VideoCapture(0)

# img=cv2.imread("raghav.jpeg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('webcam',img)
# cv2.waitKey()

while True:
    success,frame=webcam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    coordinates=data.detectMultiScale(gray)

    # for x,y,w,h in coordinates:
        #  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    facecurframe=face_recognition.face_locations(frame)
    encodecurframe=face_recognition.face_encodings(frame,facecurframe)
    
    for encoface, faceloc in zip(encodecurframe,facecurframe):
        matches=face_recognition.compare_faces(encodelistknown,encoface)
        facedis=face_recognition.face_distance(encodelistknown,encoface)

        matchindex=np.argmin(facedis)
        # print("match index",matchindex)

        for x,y,w,h in coordinates:
         cvzone.cornerRect(frame,[x,y,w,h],rt=0)  
        #  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
         if matches[matchindex]:
            # print("known face detected")
            name=ids[matchindex]
            cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)

    
    # encode=face_recognition.face_encodings(gray)
    # cv2.imshow('webcam',gray)

    # coordinates=data.detectMultiScale(gray)

    # for x,y,w,h in coordinates:
    #      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow('webcam',frame)
    key=cv2.waitKey(1)
    if (key==66) or (key==98):
        break

webcam.release()

print("end")



