import cv2
import os 
import pickle
import face_recognition
from os.path import join


pathlist = os.listdir("D:/Projects/attendance/images")
imglist = []
ids = []

for path in pathlist:
    imglist.append(cv2.imread(join("D:/Projects/attendance/images", path)))
    s=(os.path.splitext(path)[0])
    ids.append(s)

def encoding(imglist):
    encodelist=[]
    for img in imglist:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

print(ids)
print("encoding started")
encodelistknown=encoding(imglist)
encodelistknownwithids=[encodelistknown,ids]
print("encoding complete")

file=open("encodefile.p","wb")
pickle.dump(encodelistknownwithids,file)
file.close()
print("file saved")

