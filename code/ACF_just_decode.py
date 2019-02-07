# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
import random
import math

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_COUNT = 13014
DENSE_SIZE = 300
learning_rate = 0.0002  # Used to be 0.001

inputs_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='targets')




### Encoder
conv0 = tf.layers.conv2d(inputs=inputs_, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x25
maxpool0 = tf.layers.max_pooling2d(conv0, pool_size=(2,2), strides=(2,2), padding='same')
# Now 32x32x25
conv1 = tf.layers.conv2d(inputs=maxpool0, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x40
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 16x16x40
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x60
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 8x8x60
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=240, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x80
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x80

maxpool3_flat = tf.reshape(maxpool3, [-1,4*4*240])

W_fc1 = weight_variable([4*4*240, 300])
b_fc1 = bias_variable([300])
tesy = tf.matmul(maxpool3_flat, W_fc1)
encoded = tf.nn.relu(tf.matmul(maxpool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([300, 4*4*240])
b_fc2 = bias_variable([4*4*240])
predecoded_flat = tf.nn.relu(tf.matmul(encoded, W_fc2) + b_fc2)

predecoded = tf.reshape(predecoded_flat, [-1,4,4,240])

### Decoder
upsample1 = tf.image.resize_images(predecoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 8x8x80
conv4 = tf.layers.conv2d(inputs=upsample1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x60
upsample2 = tf.image.resize_images(conv4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 16x16x60
conv5 = tf.layers.conv2d(inputs=upsample2, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x40
upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 32x32x40
conv6 = tf.layers.conv2d(inputs=upsample3, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x25
upsample4 = tf.image.resize_images(conv6, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 64x64x25
conv7 = tf.layers.conv2d(inputs=upsample4, filters=15, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x10


logits = tf.layers.conv2d(inputs=conv7, filters=3, kernel_size=(3,3), padding='same', activation=None)
#Now 64x64x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


print("made it here :) :) :)")
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess,  "/media/rob/Ma Book1/alignedCelebFaces/models/model27674.ckpt")

denseRep = np.load("denseArray.npy")
for i in range(10):
    dr = denseRep[i].reshape(1,300)
    #print(dr)
    reconstructedImage = sess.run([decoded], feed_dict={encoded: dr})
    ri_np = np.array(reconstructedImage).reshape((64,64,3))
    print(ri_np[30:32,30:32,:])
    misc.imsave('reconstruct'+str(i)+'.png',ri_np)
