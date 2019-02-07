import face_recognition
from scipy import misc
import numpy as np
from skimage import transform
import os.path
import urllib.request
import time
import requests

START_MONTH = 11
START_DATE = 17
START_SLOT = 6
imageCounter = 12500


DESIRED_X = 32
DESIRED_Y = 21
DESIRED_SIZE = 24

FINAL_IMAGE_WIDTH = 64
FINAL_IMAGE_HEIGHT = 64

NAMES_PER_FILE = 100
TEMP_FILENAME = "temp.png"
monthOn = 0
dayOn = 0
daysInMonth = [31,29,31,30,31,30,31,31,30,31,30,31]
monthNames = ['january','february','march','april','may','june','july','august','september','october','november','december']

def getAvg(face, landmark):
    cum = np.zeros((2))
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum/len(face[landmark])

def getNorm(a):
    return (a-np.mean(a))/np.std(a)

for monthOn in range(START_MONTH,12):
    thisStartDate = 1
    if monthOn == START_MONTH:
        thisStartDate = START_DATE
    for dayOn in range(thisStartDate,daysInMonth[monthOn]+1):
        response = urllib.request.urlopen("https://www.famousbirthdays.com/"+monthNames[monthOn]+str(dayOn)+".html")
        pageSource = response.read().splitlines()
        lineOn = 0
        while str(pageSource[lineOn]) != "b'<div class=\"container people-list\">'":
            lineOn += 1

        thisStartSlot = 0
        if monthOn == START_MONTH and dayOn == START_DATE:
            thisStartSlot = START_SLOT
        for slotOn in range(0,48):
            while "class=\"face person-item\"" not in str(pageSource[lineOn]):
                lineOn += 1
            iul = str(pageSource[lineOn])
            pnl = str(pageSource[lineOn+4])
            imageURL = iul[iul.index("background: url(")+16:iul.index(") no-repeat center center")]
            personName = ""
            age = ""
            pnl_s = 2
            if pnl[pnl_s] == ' ':
               pnl_s += 1
            if "," in pnl:
                personName = pnl[pnl_s:pnl.index(",")]
                age = pnl[pnl.index(",")+2:-1]
            else:
                personName = pnl[pnl_s:pnl.index("(")-1]
                age = pnl[pnl.index("(")+1:pnl.index(")")]

            if slotOn < thisStartSlot or imageURL == 'https://www.famousbirthdays.com/faces/large-default.jpg' or personName == "Ronan Domingo"  or personName == "Glam And Gore" or personName == "Edith Piaf" or personName == "Lexi Marie":
                print(personName+" SKIPPED!")
            else:
            
                img_data = requests.get(imageURL).content
                with open("temp.png", 'wb') as handler:
                    handler.write(img_data)
                
                image_face_info = face_recognition.load_image_file(TEMP_FILENAME)
                face_landmarks = face_recognition.face_landmarks(image_face_info)

                image_numpy = misc.imread(TEMP_FILENAME)
                colorAmount = 0
                if len(image_numpy.shape) == 3:
                    nR = getNorm(image_numpy[:,:,0])
                    nG = getNorm(image_numpy[:,:,1])
                    nB = getNorm(image_numpy[:,:,2])
                    colorAmount = np.mean(np.square(nR-nG))+np.mean(np.square(nR-nB))+np.mean(np.square(nG-nB))
                if len(face_landmarks) == 1 and colorAmount >= 0.04: # We need there to only be one face in the image, AND we need it to be a colored image.
                    leftEyePosition = getAvg(face_landmarks[0],'left_eye')
                    rightEyePosition = getAvg(face_landmarks[0],'right_eye')
                    nosePosition = getAvg(face_landmarks[0],'nose_tip')
                    mouthPosition = getAvg(face_landmarks[0],'bottom_lip')
            
                    centralPosition = (leftEyePosition+rightEyePosition)/2
            
                    faceWidth = np.linalg.norm(leftEyePosition-rightEyePosition)
                    faceHeight = np.linalg.norm(centralPosition-mouthPosition)
                    if faceWidth >= faceHeight*0.7 and faceWidth <= faceHeight*1.5:

                        faceSize = (faceWidth+faceHeight)/2
            
                        toScaleFactor = faceSize/DESIRED_SIZE
                        toXShift = (centralPosition[0])
                        toYShift = (centralPosition[1])
                        toRotateFactor = np.arctan2(rightEyePosition[1]-leftEyePosition[1],rightEyePosition[0]-leftEyePosition[0])
            
                        rotateT = transform.SimilarityTransform(scale=toScaleFactor,rotation=toRotateFactor,translation=(toXShift,toYShift))
                        moveT = transform.SimilarityTransform(scale=1,rotation=0,translation=(-DESIRED_X,-DESIRED_Y))

                        outputArr = transform.warp(image=image_numpy,inverse_map=(moveT+rotateT))[0:FINAL_IMAGE_HEIGHT,0:FINAL_IMAGE_WIDTH]

                        misc.imsave("data/dataFace"+str(imageCounter)+".png",outputArr)
                        if imageCounter%NAMES_PER_FILE == 0:
                            f = open("names/name"+str(imageCounter)+".txt","w+")
                        fame = str(slotOn)
                        if monthOn == 1 and dayOn == 29:
                            fame *= 4
                        f.write(personName+","+fame+","+age+"\n")
                        if imageCounter%NAMES_PER_FILE == (NAMES_PER_FILE-1):
                            f.close()
                        print("DAY "+monthNames[monthOn]+" "+str(dayOn)+":  I just used person "+personName+" to create image number "+str(imageCounter))
                        imageCounter += 1
                time.sleep(0.5)
            lineOn += 1
