import face_recognition
import numpy as np
import os

from constants import *
from scipy import misc
from skimage import transform

DESIRED_X = 32
DESIRED_Y = 21
DESIRED_SIZE = 24

FINAL_IMAGE_WIDTH = 64
FINAL_IMAGE_HEIGHT = 64


def get_avg(face, landmark):
    cum = np.zeros(2)
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum / len(face[landmark])


def get_norm(a):
    return (a - np.mean(a)) / np.std(a)


extra_folder_contents = os.listdir(EXTRA_IMAGES_FOLDER)

for file_name in extra_folder_contents:
    if not file_name.endswith(tuple(IMAGE_EXTENSIONS)) or ALIGNED_TAG in file_name:
        continue
    aligned_file_name = os.path.splitext(file_name)[0] + ALIGNED_TAG + OUTPUT_EXTENSION
    if aligned_file_name in extra_folder_contents:
        print("Skipping {}...".format(aligned_file_name))
        continue
    image_file = "{}/{}".format(EXTRA_IMAGES_FOLDER, file_name)
    aligned_file = "{}/{}".format(EXTRA_IMAGES_FOLDER, aligned_file_name)
    image_face_info = face_recognition.load_image_file(image_file)
    face_landmarks = face_recognition.face_landmarks(image_face_info)

    image_numpy = misc.imread(image_file)
    colorAmount = 0
    imageSaved = False
    if len(image_numpy.shape) == 3:
        nR = get_norm(image_numpy[:, :, 0])
        nG = get_norm(image_numpy[:, :, 1])
        nB = get_norm(image_numpy[:, :, 2])
        colorAmount = np.mean(np.square(nR - nG)) + np.mean(np.square(nR - nB)) + np.mean(np.square(nG - nB))
    # We need there to only be one face in the image, AND we need it to be a colored image.
    if len(face_landmarks) == 1 and colorAmount >= 0.04:
        leftEyePosition = get_avg(face_landmarks[0], 'left_eye')
        rightEyePosition = get_avg(face_landmarks[0], 'right_eye')
        nosePosition = get_avg(face_landmarks[0], 'nose_tip')
        mouthPosition = get_avg(face_landmarks[0], 'bottom_lip')

        centralPosition = (leftEyePosition + rightEyePosition) / 2

        faceWidth = np.linalg.norm(leftEyePosition - rightEyePosition)
        faceHeight = np.linalg.norm(centralPosition - mouthPosition)
        if faceHeight * 0.7 <= faceWidth <= faceHeight * 1.5:
            faceSize = (faceWidth + faceHeight) / 2

            toScaleFactor = faceSize / DESIRED_SIZE
            toXShift = (centralPosition[0])
            toYShift = (centralPosition[1])
            toRotateFactor = np.arctan2(rightEyePosition[1] - leftEyePosition[1],
                                        rightEyePosition[0] - leftEyePosition[0])

            rotateT = transform.SimilarityTransform(scale=toScaleFactor, rotation=toRotateFactor,
                                                    translation=(toXShift, toYShift))
            moveT = transform.SimilarityTransform(scale=1, rotation=0, translation=(-DESIRED_X, -DESIRED_Y))

            outputArr = transform.warp(image=image_numpy, inverse_map=(moveT + rotateT))[0:FINAL_IMAGE_HEIGHT,
                        0:FINAL_IMAGE_WIDTH]

            misc.imsave(aligned_file, outputArr)
            imageSaved = True
    if imageSaved:
        print("Aligned face image ({}) saved successfully!".format(aligned_file))
    else:
        print("Face image ({}) failed. Either the image is grayscale, has no face, or the ratio of eye distance to "
              "mouth distance isn't close enough to 1.".format(aligned_file))
