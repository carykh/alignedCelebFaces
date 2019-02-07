import face_recognition
from scipy import misc
import numpy as np
from skimage import transform
import os.path

for i in range(1200):
    image_numpy = misc.imread('/media/rob/Ma Book1/mugshots/aligned/alignedFace'+str(i)+'.jpg')
    image_numpy = np.flip(image_numpy, axis=1)
    image_numpy = misc.imsave('/media/rob/Ma Book1/mugshots/aligned/alignedFace'+str(1200+i)+'.jpg',image_numpy)
    if i%100 == 0:
        print("done with "+str(i))
print("done af")
