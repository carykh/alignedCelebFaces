import glob
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from constants import *
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

learning_rate = 0.0000  # Used to be 0.001
inputs_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='targets')

""" Encoder """
conv0 = tf.layers.conv2d(inputs=inputs_, filters=120, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 64x64x25
maxpool0 = tf.layers.max_pooling2d(conv0, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 32x32x25
conv1 = tf.layers.conv2d(inputs=maxpool0, filters=160, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 32x32x40
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 16x16x40
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=200, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 16x16x60
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 8x8x60
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=240, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 8x8x80
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 4x4x80

maxpool3_flat = tf.reshape(maxpool3, [-1, 4 * 4 * 240])

W_fc1 = weight_variable([4 * 4 * 240, 300])
b_fc1 = bias_variable([300])
tesy = tf.matmul(maxpool3_flat, W_fc1)
encoded = tf.nn.relu(tf.matmul(maxpool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([300, 4 * 4 * 240])
b_fc2 = bias_variable([4 * 4 * 240])
predecoded_flat = tf.nn.relu(tf.matmul(encoded, W_fc2) + b_fc2)

predecoded = tf.reshape(predecoded_flat, [-1, 4, 4, 240])

""" Decoder """
upsample1 = tf.image.resize_images(predecoded, size=(8, 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 8x8x80
conv4 = tf.layers.conv2d(inputs=upsample1, filters=200, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 8x8x60
upsample2 = tf.image.resize_images(conv4, size=(16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 16x16x60
conv5 = tf.layers.conv2d(inputs=upsample2, filters=160, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 16x16x40
upsample3 = tf.image.resize_images(conv5, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 32x32x40
conv6 = tf.layers.conv2d(inputs=upsample3, filters=120, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 32x32x25
upsample4 = tf.image.resize_images(conv6, size=(64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 64x64x25
conv7 = tf.layers.conv2d(inputs=upsample4, filters=15, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 64x64x10


logits = tf.layers.conv2d(inputs=conv7, filters=3, kernel_size=(3, 3), padding='same', activation=None)
# Now 64x64x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

extra_folder_contents = os.listdir(EXTRA_IMAGES_FOLDER)
alignedImages = glob.glob("{}/*{}{}".format(EXTRA_IMAGES_FOLDER, ALIGNED_TAG, OUTPUT_EXTENSION))
preexistingEncodingFile = "denseArray27K.npy"
outputEncodingFile = "denseArray27K.npy"
preexistingNamesFile = "names/allNames.txt"

NEW_IMAGE_COUNT = len(alignedImages)
preexistingEncodings = np.load(preexistingEncodingFile)
OLD_IMAGE_COUNT, DENSE_SIZE = preexistingEncodings.shape
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "./models/model27674.ckpt")

outputEncodings = np.zeros((OLD_IMAGE_COUNT + NEW_IMAGE_COUNT, DENSE_SIZE))
outputEncodings[0:OLD_IMAGE_COUNT] = preexistingEncodings

for i, img in enumerate(alignedImages):
    imagio = misc.imread(img)
    imagio = imagio[:, :, 0:3].reshape((1, 64, 64, 3)) / 255.0
    file_name = os.path.basename(img)
    default_name = os.path.splitext(file_name)[0].replace(ALIGNED_TAG, '').replace('_', ' ').title()
    name = input("Name for {}: (default: {}) ".format(file_name, default_name)) or default_name
    denseRep = sess.run([encoded], feed_dict={inputs_: imagio, targets_: imagio})
    outputEncodings[OLD_IMAGE_COUNT + i] = np.array(denseRep)

    with open(preexistingNamesFile, "a") as myfile:
        myfile.write("{},custom,custom\n".format(name))
    print("{} has been added".format(name))

np.save(outputEncodingFile, outputEncodings)
