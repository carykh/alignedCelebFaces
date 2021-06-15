import html
import math
import pygame
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pygame.locals import *
from scipy import misc

eigenvalues = np.load("eigenvalues.npy")
eigenvectors = np.load("eigenvectors.npy")
eigenvectorInverses = np.linalg.pinv(eigenvectors)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_COUNT = 13016
DENSE_SIZE = 300
learning_rate = 0.0002  # Used to be 0.001
settings = np.zeros((DENSE_SIZE,))
approach_settings = np.zeros((2,))
approach_settings.fill(1)
denseData = np.load("denseArray27K.npy")
shouldICalculateImage = True

f = open('names/allNames.txt', 'r+')
allNames = f.read()
f.close()
allPeople = html.unescape(allNames).split('\n')
f = open('eigenvalueNames.txt', 'r+')
eigenvalueNames = f.read().split('\n')
f.close()
nearestPerson = 0

meanData = denseData.mean(axis=0)

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

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "models/model27674.ckpt")


def get_celeb_sliders(i):
    traits = denseData[i] - meanData
    return np.matmul(traits, eigenvectorInverses) / eigenvalues


celebSliders = np.zeros(denseData.shape)
for i in range(denseData.shape[0]):
    celebSliders[i] = get_celeb_sliders(i)


def calculate_image(settings):
    real_settings = meanData.copy()
    for i in range(DENSE_SIZE):
        real_settings += settings[i] * eigenvalues[i] * eigenvectors[i]
    real_settings = real_settings.reshape((1, DENSE_SIZE))
    reconstructed_image = sess.run([decoded], feed_dict={encoded: real_settings})
    ri_np = np.array(reconstructed_image).reshape((64, 64, 3))
    ri_np = np.swapaxes(ri_np, 0, 1)
    closest_celeb = np.argmin(np.linalg.norm((settings - celebSliders) * eigenvalues, axis=1))
    return ri_np * 255, closest_celeb


def create_special_children(index, parent_count, file_name):
    total_image = np.zeros((240 + 264 * parent_count, 300, 3))
    total_image.fill(255)

    parents = [-1] * parent_count

    child_settings = np.zeros((celebSliders[0].shape,))
    for i in range(0, parent_count):
        while parents[i] == -1 or int(allPeople[parents[i]].split(",")[1]) > 10:  # fame rank must be 5 or better
            parents[i] = np.random.randint(IMAGE_COUNT)
        parents[i] = 13015
        child_settings += celebSliders[parents[i]]
    child_settings /= parent_count

    for i in range(0, parent_count + 1):
        if i == parent_count:
            img, _ = calculate_image(child_settings)
        else:
            img, _ = calculate_image(celebSliders[parents[i]])
            # img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)

        total_image[24 + i * 264:216 + i * 264, 24:216] = misc.imresize(img, size=[192, 192], interp='nearest')

    blah = pygame.surfarray.make_surface(total_image)
    for i in range(0, parent_count + 1):
        name = "CHILD"
        if i < parent_count:
            name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        text_surface = font.render(name, 1, (0, 0, 0))
        blah.blit(text_surface, [24 + 264 * i, 220])
        if i < parent_count - 1:
            font = pygame.font.SysFont("Helvetica", 48)
            blah.blit(font.render('^', 1, (0, 0, 0)), [240 + 264 * i, 100])
    font = pygame.font.SysFont("Helvetica", 48)
    blah.blit(font.render('=', 1, (0, 0, 0)), [240 + 264 * (parent_count - 1), 100])
    pygame.image.save(blah, "spesh/" + file_name + "{:03d}".format(index) + ".png")


def create_child_grid(index, parent_count, file_name):
    total_image = np.zeros((264 + 264 * parent_count, 264 + 264 * parent_count, 3))
    total_image[264:, 264:, :] = 255

    parents = [-1] * parent_count
    for i in range(0, parent_count):
        parents[i] = np.random.randint(IMAGE_COUNT)
        img, _ = calculate_image(celebSliders[parents[i]])
        # img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)
        big_img = misc.imresize(img, size=[192, 192], interp='nearest')
        total_image[24 + (i + 1) * 264:216 + (i + 1) * 264, 24:216] = big_img
        total_image[24:216, 24 + (i + 1) * 264:216 + (i + 1) * 264] = big_img
        total_image[264 * (i + 1):264 * (i + 2), 264 * (i + 1):264 * (i + 2)] = [0, 255, 0]

    for i in range(0, parent_count):
        for j in range(0, parent_count):
            child_settings = (celebSliders[parents[i]] + celebSliders[parents[j]]) / 2
            img, _ = calculate_image(child_settings)
            total_image[24 + (i + 1) * 264:216 + (i + 1) * 264, 24 + (j + 1) * 264:216 + (j + 1) * 264] = misc.imresize(
                img, size=[192, 192], interp='nearest')
    blah = pygame.surfarray.make_surface(total_image)
    for i in range(0, parent_count):
        name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        text_surface = font.render(name, 1, (255, 255, 255))
        blah.blit(text_surface, [24 + 264 * (i + 1), 220])
        blah.blit(text_surface, [24, 220 + 264 * (i + 1)])
    pygame.image.save(blah, "spesh/{}{:03d}.png".format(file_name, index))


def create_family_tree(index, parent_count, file_name):
    total_image = np.zeros((264 * parent_count, 264 * parent_count, 3))
    total_image.fill(255)

    parents = [-1] * parent_count

    allSettings = np.zeros((parent_count, parent_count, celebSliders[0].shape[0]))
    for i in range(0, parent_count):
        parents[i] = np.random.randint(IMAGE_COUNT)
        allSettings[0, i] = celebSliders[parents[i]]
        img, _ = calculate_image(celebSliders[parents[i]])
        # img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)
        big_img = misc.imresize(img, size=[192, 192], interp='nearest')
        total_image[24 + i * 264:216 + i * 264, 40:232] = big_img

    for level in range(1, parent_count):
        for i in range(0, parent_count - level):
            allSettings[level, i] = (allSettings[level - 1, i] + allSettings[level - 1, i + 1]) * 0.5
            img, _ = calculate_image(allSettings[level, i])
            x_start = 24 + i * 264 + level * 132
            y_start = 40 + level * 264
            total_image[x_start:x_start + 192, y_start:y_start + 192] = misc.imresize(img, size=[192, 192],
                                                                                      interp='nearest')
            total_image[x_start + 92:x_start + 100, y_start - 32:y_start] = 0
            total_image[x_start:x_start + 192, y_start - 40:y_start - 32] = 0
            total_image[x_start:x_start + 8, y_start - 72:y_start - 40] = 0
            total_image[x_start + 184:x_start + 192, y_start - 72:y_start - 40] = 0

    blah = pygame.surfarray.make_surface(total_image)
    for i in range(0, parent_count):
        name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        text_surface = font.render(name, 1, (0, 0, 0))
        blah.blit(text_surface, [20 + 264 * i, 14])
    pygame.image.save(blah, "spesh/{}{:03d}.png".format(file_name, index))


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)

VISIBLE_COMPONENTS = 20

enteringName = False
isShiftPressed = False
enteredName = ""
frameTimer = 0
misspelledTimer = 0
scrollPosition = 0
stringToNameDict = {}

transitionTimes = np.zeros((2,))
transitionKeyFrames = np.zeros((2, DENSE_SIZE))

for i in range(len(allPeople)):
    line = allPeople[i]
    pieces = line.split(",")
    name = pieces[0]
    alpha_only_name = ''.join(x for x in name if (x.isalpha() or x == ' '))
    lower_name = alpha_only_name.lower()
    if len(lower_name) >= 1:
        stringToNameDict[lower_name] = i


def string_to_celeb(st):
    alpha_only_name = ''.join(x for x in st if (x.isalpha() or x == ' '))
    lower_name = alpha_only_name.lower()
    if lower_name not in stringToNameDict:
        return -1
    return stringToNameDict[lower_name]


oops_image = pygame.image.load("oops.png")
imagerect = oops_image.get_rect()

calculatedImage, nearestPerson = calculate_image(settings)


class Slider():
    def __init__(self, i, maxi, mini, x, y, w, h):
        self.maxi = maxi
        self.mini = mini
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.surf = pygame.surface.Surface((w, h))
        self.hit = False
        self.i = i
        self.font = pygame.font.SysFont("Helvetica", 16)

    def true_i(self):
        return self.i + scrollPosition

    def draw(self):
        j = self.true_i()
        eigen = "%.4f" % eigenvalues[j]
        name = "PCA #" + str(self.true_i() + 1) + " (" + eigen + ")"
        if j < len(eigenvalueNames) - 1:
            name = eigenvalueNames[j]

        txt_surf = self.font.render(name, 1, WHITE)
        txt_rect = txt_surf.get_rect(center=(self.w / 2, 13))

        s = 70
        if self.i % 2 + (self.i // 2) % 2 == 1:
            s = 100
        self.surf.fill((s, s, s))
        pygame.draw.rect(self.surf, (220, 220, 220), [10, 30, self.w - 20, 5], 0)
        for g in range(7):
            pygame.draw.rect(self.surf, (s + 50, s + 50, s + 50), [9 + (self.w - 20) / 6 * g, 40, 2, 5], 0)

        self.surf.blit(txt_surf, txt_rect)

        button_surf = pygame.surface.Surface((10, 20))
        button_surf.fill(TRANS)
        button_surf.set_colorkey(TRANS)
        pygame.draw.rect(button_surf, WHITE, [0, 0, 10, 20])

        surf = self.surf.copy()

        v = min(max(settings[j], -9999), 9999)
        pos = (10 + int((v - self.mini) / (self.maxi - self.mini) * (self.w - 20)), 33)
        button_rect = button_surf.get_rect(center=pos)
        surf.blit(button_surf, button_rect)
        button_rect.move_ip(self.x, self.y)

        screen.blit(surf, (self.x, self.y))

    def move(self):
        j = self.true_i()
        settings[j] = (pygame.mouse.get_pos()[0] - self.x - 10) / 130 * (self.maxi - self.mini) + self.mini
        if settings[j] < self.mini:
            settings[j] = self.mini
        if settings[j] > self.maxi:
            settings[j] = self.maxi


class ApproachSlider():
    def __init__(self, i, maxi, mini, x, y, w, h):
        self.maxi = maxi
        self.mini = mini
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.surf = pygame.surface.Surface((w, h))
        self.hit = False
        self.i = i
        self.font = pygame.font.SysFont("Helvetica", 16)

    def draw(self):
        if self.i == 0:
            st = "Go " + "%.1f" % (100 * approach_settings[self.i]) + "% the way to this celeb."
        else:
            st = "Speed of travel: " + "%.2f" % (100 * (1 - approach_settings[self.i])) + " frames"
        txt_surf = self.font.render(st, 1, WHITE)
        txt_rect = txt_surf.get_rect(center=(self.w / 2, 13))

        s = 70 + 30 * self.i
        self.surf.fill((s, s, s))
        pygame.draw.rect(self.surf, (220, 220, 220), [10, 35, self.w - 20, 5], 0)
        self.surf.blit(txt_surf, txt_rect)

        button_surf = pygame.surface.Surface((10, 30))
        button_surf.fill(TRANS)
        button_surf.set_colorkey(TRANS)
        pygame.draw.rect(button_surf, WHITE, [0, 0, 10, 30])

        surf = self.surf.copy()

        v = min(max(approach_settings[self.i], -9999), 9999)
        pos = (10 + int((v - self.mini) / (self.maxi - self.mini) * (self.w - 20)), 38)
        button_rect = button_surf.get_rect(center=pos)
        surf.blit(button_surf, button_rect)
        button_rect.move_ip(self.x, self.y)

        screen.blit(surf, (self.x, self.y))

    def move(self):
        approach_settings[self.i] = (pygame.mouse.get_pos()[0] - self.x - 10) / (self.w - 20) * (
                self.maxi - self.mini) + self.mini
        if approach_settings[self.i] < self.mini:
            approach_settings[self.i] = self.mini
        if approach_settings[self.i] > self.maxi:
            approach_settings[self.i] = self.maxi


def draw_buttons():
    enb_shade = 200
    if enteringName:
        enb_shade = math.sin(frameTimer * 0.06) * 40 + 200

    enter_name_button = pygame.surface.Surface((300, 120))
    pygame.draw.rect(enter_name_button, (enb_shade, enb_shade, enb_shade), [5, 5, 290, 110], 0)

    st = "[Enter celeb name]"
    if len(enteredName) >= 1:
        st = enteredName

    button_data = [
        [(0, 0), (300, 60), (230, 30, 30), 44, "RANDOMIZE", WHITE],
        [(0, 540), (300, 60), (30, 30, 230), 44, "GO TO MEAN", WHITE],
        [(800, 0), (300, 120), (230, 170, 30), 44, "INVERT", WHITE],
        [(300, 500), (500, 100), (0, 0, 0), 24, "Hey! You look like " + allPeople[nearestPerson].split(",")[0] + ".",
         WHITE],
        [(800, 120), (300, 120), (enb_shade, enb_shade, enb_shade), 30, st, BLACK],
        [(800, 360), (300, 120), (30, 170, 30), 44, "GO TO THEM", WHITE],
        [(800, 480), (300, 120), (30, 170, 30), 24, "GO TO RANDOM CELEB", WHITE]]

    for button in button_data:
        button_surface = pygame.surface.Surface(button[1])
        pygame.draw.rect(button_surface, button[2], [5, 5, button[1][0] - 10, button[1][1] - 10], 0)
        font = pygame.font.SysFont("Helvetica", button[3])
        b_text = font.render(button[4], 1, button[5])
        b_text_rect = b_text.get_rect(center=(button[1][0] / 2, button[1][1] / 2))
        button_surface.blit(b_text, b_text_rect)
        screen.blit(button_surface, button[0])

    if transitionTimes[0] >= 0:
        w = 290 * (frameTimer - transitionTimes[0]) / (transitionTimes[1] - transitionTimes[0])
        progress_bar_surface = pygame.surface.Surface((w, 25))
        progress_bar_surface.fill((0, 150, 0))
        screen.blit(progress_bar_surface, (805, 125))

    image_surface = pygame.surfarray.make_surface(calculatedImage)
    bigger = pygame.transform.scale(image_surface, (500, 500))
    screen.blit(bigger, (300, 0))

    if misspelledTimer >= 1:
        y = 5
        if misspelledTimer < 60:
            y = -115 + 120 * (0.5 + math.cos((misspelledTimer - 60) / 60.0 * math.pi) * 0.5)
        screen.blit(oops_image, (805, y))


# return misspelledTimer. how many frames the misspelled warning should show up. I know, it's weird and dumb.
def go_to_celeb(c):
    celeb_choice = string_to_celeb(enteredName)
    if c >= 0:
        celeb_choice = c
    if celeb_choice == -1:
        return 800
    else:
        slider_settings = celebSliders[celeb_choice]
        if approach_settings[1] == 1:
            for i in range(DENSE_SIZE):
                settings[i] += slider_settings[i] - settings[i] * approach_settings[0]
        else:
            transitionKeyFrames[0] = settings.copy()
            transitionKeyFrames[1] = settings.copy()
            for i in range(DENSE_SIZE):
                transitionKeyFrames[1, i] += slider_settings[i] - settings[i] * approach_settings[0]
            transitionTimes[0] = frameTimer - 1
            transitionTimes[1] = frameTimer - 1 + 100 * (1 - approach_settings[1])  # really bad magic numbers oh well
    return 0


pygame.init()
slides = []
for i in range(VISIBLE_COMPONENTS):
    eigen = "%.4f" % eigenvalues[i]
    slides.append(Slider(i, 3, -3, (i % 2) * 150, (i // 2) * 48 + 60, 150, 48))

approachSlides = []
for i in range(2):
    approachSlides.append(ApproachSlider(i, 1, 0, 800, 240 + 60 * i, 300, 60))

screen = pygame.display.set_mode((1100, 600))

running = True

# OPTIONAL SPECIAL CHILD CREATION
# create_special_children(0,1,"speshCaryD")
# for i in range(0,2):
#    create_special_children(i,2,"speshTwo")
# create_child_grid(0,6,"speshGrid")
# create_family_tree(0,12,"speshFamilyHuge")
# END OF OPTIONAL SPECIAL CHILD CREATION

while running:
    shouldICalculateImage = False
    frameTimer += 1
    misspelledTimer = max(0, misspelledTimer - 1)
    for event in pygame.event.get():
        # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
        if event.type == KEYDOWN:
            # If the Esc key has been pressed set running to false to exit the main loop
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                isShiftPressed = True
            elif event.key == K_ESCAPE:
                running = False
            elif enteringName:
                k = event.key
                isLetter = ord('a') <= k <= ord('z')
                if isLetter or k == ord('-') or k == ord(' ') or k == ord('\''):
                    ch = event.unicode
                    if isShiftPressed and isLetter:
                        ch = ch.upper()
                    enteredName = enteredName + ch
                if len(enteredName) >= 1 and (k == K_BACKSPACE or k == K_DELETE):
                    enteredName = enteredName[0:-1]
                if k == K_RETURN:
                    enteringName = False
                    misspelledTimer = go_to_celeb(-1)
                    shouldICalculateImage = True
        # Check for QUIT event; if QUIT, set running to false
        elif event.type == KEYUP:
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                isShiftPressed = False
        elif event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_loc = pygame.mouse.get_pos()
            if event.button == 4 or event.button == 5:
                dire = (event.button - 4.5) * 2
                if mouse_loc[0] < 300 and 60 <= mouse_loc[1] < 540:
                    i = (mouse_loc[0] // 150) + ((mouse_loc[1] - 60) // 48) * 2 + scrollPosition
                    settings[i] -= 0.2 * dire
                    shouldICalculateImage = True
                else:
                    scrollPosition = min(max(scrollPosition + 2 * int(dire), 0),
                                         denseData.shape[1] - VISIBLE_COMPONENTS)
                    for i in range(VISIBLE_COMPONENTS):
                        slides[i].val = settings[i + scrollPosition]
            else:
                enteringName = False
                if mouse_loc[0] < 300:
                    if mouse_loc[1] < 60:
                        for i in range(DENSE_SIZE):
                            settings[i] = np.random.normal(0, 1, 1)
                        shouldICalculateImage = True
                        enteredName = ""
                    elif mouse_loc[1] >= 540:
                        for i in range(DENSE_SIZE):
                            settings[i] = 0
                        shouldICalculateImage = True
                        enteredName = ""
                    else:
                        i = (mouse_loc[0] // 150) + ((mouse_loc[1] - 60) // 48) * 2
                        slides[i].hit = True
                elif mouse_loc[0] >= 800:
                    if mouse_loc[1] < 120:
                        for i in range(DENSE_SIZE):
                            settings[i] *= -1
                        shouldICalculateImage = True
                        misspelledTimer = 0
                        enteredName = ""
                    elif mouse_loc[1] < 240:
                        enteringName = True
                        misspelledTimer = 0
                        enteredName = ""
                    elif 240 <= mouse_loc[1] < 360:
                        i = ((mouse_loc[1] - 240) // 60)
                        approachSlides[i].hit = True
                    elif mouse_loc[1] >= 480:
                        c = np.random.randint(denseData.shape[0])
                        go_to_celeb(c)
                        shouldICalculateImage = True
                        enteredName = allPeople[c].split(",")[0]
                    elif mouse_loc[1] >= 360:
                        misspelledTimer = go_to_celeb(-1)
                        shouldICalculateImage = True

        elif event.type == pygame.MOUSEBUTTONUP:
            for s in slides:
                s.hit = False
            for a_s in approachSlides:
                a_s.hit = False

    if transitionTimes[0] >= 0:
        proportion_through = min(max((frameTimer - transitionTimes[0]) / (transitionTimes[1] - transitionTimes[0]), 0),
                                 1)
        if frameTimer >= transitionTimes[1]:
            proportion_through = 1
            transitionTimes[:] = -1

        settings = transitionKeyFrames[0] + proportion_through * (transitionKeyFrames[1] - transitionKeyFrames[0])
        shouldICalculateImage = True
    else:
        for s in slides:
            if s.hit:
                s.move()
                shouldICalculateImage = True
    for a_s in approachSlides:
        if a_s.hit:
            a_s.move()

    if shouldICalculateImage:
        calculatedImage, nearestPerson = calculate_image(settings)

    screen.fill(BLACK)
    for s in slides:
        s.draw()
    for a_s in approachSlides:
        a_s.draw()
    draw_buttons()

    pygame.display.flip()
