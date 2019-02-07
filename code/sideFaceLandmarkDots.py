import face_recognition
from scipy import misc
import numpy as np

def getAvg(face, landmark):
    cum = np.zeros((2))
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum/len(face[landmark])

colors = [[255,255,0],[255,0,0],[0,255,255],[0,255,0],[128,255,0],[255,255,255],[255,0,255],[255,128,0],[128,0,255]]
filename = '/media/rob/Ma Book1/alignedCelebFaces/dottede3.jpg'
imagio = misc.imread(filename)
colorAmount = np.mean(np.square(imagio[:,:,0]-imagio[:,:,1]))+np.mean(np.square(imagio[:,:,0]-imagio[:,:,2]))+np.mean(np.square(imagio[:,:,1]-imagio[:,:,2]))
image = face_recognition.load_image_file(filename)
face_landmarks = face_recognition.face_landmarks(image)
co = 0
for landmark in face_landmarks[0]:
    center = (getAvg(face_landmarks[0], landmark))
    xcenter = int(round(center[0]))
    ycenter = int(round(center[1]))
    for x in range(xcenter-3,xcenter+3):
        for y in range(ycenter-3,ycenter+3):
            image[y,x] = colors[co]
    co += 1
        

misc.imsave('/media/rob/Ma Book1/alignedCelebFaces/dotted3.jpg',image)
