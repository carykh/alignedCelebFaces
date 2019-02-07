from scipy import misc
import numpy as np

filename = '/media/rob/Ma Book1/alignedCelebFaces/data/dataFace3004.png'
bigimagio = np.zeros((84,84,3))
bigimagio.fill(255)
imagio = misc.imread(filename)
bigimagio[20:84,20:84,0:2] = 0
bigimagio[20:84,20:84,2:3] = imagio[:,:,2:3]

bigimagio[10:74,10:74,0:1] = 0
bigimagio[10:74,10:74,2:3] = 0
bigimagio[10:74,10:74,1:2] = imagio[:,:,1:2]

bigimagio[0:64,0:64,1:3] = 0
bigimagio[0:64,0:64,0:1] = imagio[:,:,0:1]
misc.imsave('/media/rob/Ma Book1/alignedCelebFaces/threech.jpg',bigimagio)
