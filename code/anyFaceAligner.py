import face_recognition
from scipy import misc
import numpy as np
from skimage import transform
import os.path
import urllib.request
import time
import requests

DESIRED_X = 32
DESIRED_Y = 21
DESIRED_SIZE = 24

FINAL_IMAGE_WIDTH = 64
FINAL_IMAGE_HEIGHT = 64

INPUT_OUTPUTS = [["extraImages/hz.jpg","extraImages/hz_aligned.png"],
["extraImages/apple.png","extraImages/apple_aligned.png"],
["extraImages/gray_woman.jpg","extraImages/gwa.png"]]

def getAvg(face, landmark):
    cum = np.zeros((2))
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum/len(face[landmark])

def getNorm(a):
    return (a-np.mean(a))/np.std(a)


for i in range(len(INPUT_OUTPUTS)):
    image_face_info = face_recognition.load_image_file(INPUT_OUTPUTS[i][0])
    face_landmarks = face_recognition.face_landmarks(image_face_info)

    image_numpy = misc.imread(INPUT_OUTPUTS[i][0])
    colorAmount = 0
    imageSaved = False
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

            misc.imsave(INPUT_OUTPUTS[i][1],outputArr)
            imageSaved = True
    if imageSaved:
        print("Aligned face image ("+INPUT_OUTPUTS[i][1]+") saved successfully!")
    else:
        print("Face image ("+INPUT_OUTPUTS[i][1]+") failed. Either the image is grayscale, has no face, or the ratio of eye distance to mouth distance isn't close enough to 1.")
